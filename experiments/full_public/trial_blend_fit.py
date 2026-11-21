"""Fit energy, force and full Hessian weights for independent kernel surfaces.

Two inner formula holdouts supply out-of-fold reaction errors. Distinct joint
DFT and CC fits add directions that cannot be reproduced by summing consecutive
residual fits. All 14 training NIST minima constrain the blend. Its stored
energy and Cartesian gradient equal the weighted sum including dispersion once.
"""
from functools import lru_cache
import json
import pickle
import numpy as np
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from experiments.full_public import kernel_trials as kt,trial_fit,trial_multiscale as ms
from experiments.full_public.trial_energy_blend import Components,molecular_weights
from experiments.full_public.trial_blend_shapes import BRANCHES
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_validation import run as validate
from experiments.full_public.trial_spectra import evaluate_trial
from src.forcefield import ForceField

WORK=kt.WORK
STAGE="p24-public"
OPTIONS=((.03,1.,.01),(.12,3.,.02),(.25,10.,.04),(.35,30.,.08))


@lru_cache(maxsize=42)
def read(stage,t,h,obs):
    if stage==14:
        choice=json.loads((WORK/"p14-public-settings.json").read_text())["branches"][t]["selected"]
        prefix=f"joint14-{t}-"+"-".join(f"{val:g}" for val in choice)
        return np.load(WORK/f"{prefix}-{obs}-{h}.npy")
    if stage==15:
        scale=json.loads((WORK/"p15-public-settings.json").read_text())["choices"][t]
        weights=np.load(WORK/f"fidelity-{t}-{h}-{scale:g}.npy")
        if obs=="parent":
            base=np.load(WORK/f"spin-energy-wb97xd3-{h if h>=0 else 'all'}.npy")
            return base+np.load(WORK/"joint14-pub.npy",mmap_mode="r")@weights
        base=np.load(WORK/"rad-prior.npy")[:,:,(0,1,-1).index(h)]
        return base+np.load(WORK/"joint14-rad.npy",mmap_mode="r").reshape(*base.shape,-1)@weights
    return ms.baseline(t,h,obs,f"p{stage}-public")


def columns(t,h,obs):
    return read(12,t,h,obs),np.stack([read(c,t,h,obs)-read(p,t,h,obs) for c,p in BRANCHES],axis=-1)


def learn(x,y,w,hx,hy,cm,pars):
    ridge,shape,radical=pars
    def loss(a):
        e=x@a-y
        den=np.hypot(e,2.)
        g=x.T@(w*e/den)+2*ridge*a
        value=np.dot(w,den)+ridge*np.sum(a*a)
        z=cm[:,:-1]@a-cm[:,-1]
        value+=shape*np.mean(z*z)
        g+=2*shape*(cm[:,:-1].T@z)/len(z)
        if len(hy):
            err=hx@a-hy
            dv=np.hypot(err,1.)
            value+=radical*dv.mean()
            g+=radical*(hx.T@(err/dv))/len(err)
        return value,g
    sol=minimize(loss,np.zeros(len(BRANCHES)),jac=True,method="L-BFGS-B",bounds=[(-1.2,2.)]*len(BRANCHES),options=dict(maxiter=300,ftol=1.e-12))
    if not sol.success:raise RuntimeError(sol.message)
    return sol.x


def run(t,meta,saddle):
    d=kt.equation_data(t)
    s,y,src,fold,kind,case=d
    eligible=trial_fit.selection_mask(d,t)
    cm=np.load(WORK/f"mixture24-{t}-shapes.npy")
    rows=[]; hrows=[]
    for h in (0,1):
        base,x=columns(t,h,"parent")
        use=(fold==h)&eligible&np.isin(src,kt.BARRIERS)
        w=np.zeros(use.sum())
        for k in np.unique(src[use]):
            take=src[use]==k
            w[take]=({"RDB7":.85,"BH9":.15}.get(k,1.))/take.sum()
        w*=np.where(kind[use]=="reaction",1.2,.9)
        rows.append(((s@x)[use]*kt.KCAL,(y-s@base)[use]*kt.KCAL,w))
        if t=="wb97xd3":
            base,modes=columns(t,h,"modes")
            take=(meta[:,0]==h)&saddle
            curv=lambda a:(a[take,3]+a[take,4]-2*a[take,0])/.004**2
            hrows.append((curv(modes),meta[take,2]-curv(base)))
        else:hrows.append((np.empty((0,len(BRANCHES))),np.empty(0)))
    dev={opt:[] for opt in OPTIONS}
    coefs={}
    for h in (0,1,-1):
        use=[1-h] if h>=0 else [0,1]
        x,yref,w=(np.concatenate([rows[i][j] for i in use]) for j in range(3))
        hx,hy=(np.concatenate([hrows[i][j] for i in use]) for j in range(2))
        if h<0:chosen=min(dev,key=lambda p:np.mean(dev[p]))
        for opt in OPTIONS if h>=0 else [chosen]:
            a=learn(x,yref,w/w.sum(),hx,hy,cm,opt)
            coefs[h,opt]=a
            if h>=0:
                xx,yy,ww=rows[h]
                dev[opt].append(float(np.average(abs(xx@a-yy),weights=ww)+.30*np.mean((cm[:,:-1]@a-cm[:,-1])**2)))
    for h in (0,1,-1):
        coef=coefs[h,chosen]
        b,x=columns(t,h,"parent")
        en=b+x@coef
        np.save(WORK/f"mix24-{t}-parent-{h}.npy",en)
        if t=="wb97xd3":
            b,x=columns(t,h,"modes")
            modes=b+x@coef
            np.save(WORK/f"mix24-{t}-modes-{h}.npy",modes)
    err=s@en-y
    mask=fold==-1
    np.savez_compressed(WORK/f"{STAGE}-{t}-equations.npz",source=src[mask],case=case[mask],kind=kind[mask],error_kcal=err[mask]*kt.KCAL)
    if t=="wb97xd3":np.save(WORK/f"{STAGE}-held-modes.npy",modes[meta[:,0]==-1])
    amounts=molecular_weights(BRANCHES,coef)
    model=Components(t,sorted(amounts))
    path=WORK/f"{STAGE}-{t}.npz"
    model.checkpoint(amounts,path)
    trained=ForceField.load(path,dispersion="d3")
    gs,_,_=kt.inputs()
    with (WORK/"ids.pkl").open("rb") as f:_,keys=pickle.load(f)
    for idx in np.random.default_rng(186).choice(len(keys),8,replace=False):
        g=gs[keys[idx]]
        energy,force=trained.predict(g.z,g.r,*g.state)
        np.testing.assert_allclose(energy,en[idx],rtol=0,atol=2.e-6)
        v=np.random.default_rng(idx).normal(size=g.r.shape)
        v/=np.linalg.norm(v)
        fd=(trained.energy(g.z,g.r-2.e-5*v,*g.state)-trained.energy(g.z,g.r+2.e-5*v,*g.state))/4.e-5
        np.testing.assert_allclose(np.sum(force*v),fd,atol=4.e-5)
    print("Verified full blended readout",t,chosen,flush=True)
    return dict(settings=chosen,weights=coef.tolist(),dev={str(k):v for k,v in dev.items()})


if __name__=="__main__":
    with threadpool_limits(2):
        meta=np.load(WORK/"rad-meta.npy")
        with (WORK/"radical-modes.pkl").open("rb") as f:saddle=np.array([a[1][0].split("/")[-1].startswith("ts") for a in pickle.load(f)])
        fits={t:run(t,meta,saddle) for t in kt.TARGETS}
        kt.write_json(WORK/f"{STAGE}-settings.json",dict(selected=STAGE,branches=dict.fromkeys(kt.TARGETS,"mix24"),fit=fits))
        compare(STAGE)
        validate(STAGE)
    with threadpool_limits(1):evaluate_trial(STAGE)
