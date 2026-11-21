"""Combine kernel corrections using out-of-fold energies and Hessian evidence.

Each column is the scalar energy added by one trained kernel stage. Learn its
strength from predictions on inner formulas withheld from that stage. Fitting
on the signed errors of all three observables preserves the thermochemical
cycle. Shared parents and D3 appear once in the saved potential.
"""
import pickle
from scipy.optimize import minimize
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.full_public import kernel_trials as kt, trial_fit, trial_multiscale as ms
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_validation import run as validate
from experiments.full_public.trial_spectra import evaluate_trial
from src.forcefield import ForceField

WORK=kt.WORK
STAGE="p23-public"
BRANCHES=((17,12),(18,17),(19,18),(20,12),(21,12),(22,12))
OPTIONS=((.04,.002),(.2,.012),(.8,.05))


def predictions(t,h,key):
    def pred(stage):
        return ms.baseline(t,h,key,f"p{stage}-public")
    base=pred(12)
    return base,np.stack([pred(stage)-pred(parent) for stage,parent in BRANCHES],axis=-1)


def optimize(x,y,w,curv,target,settings):
    """Smooth absolute energy loss and pseudo-Huber unstable-curvature error."""
    ridge,shape=settings
    def cost(a):
        residual=x@a-y
        norm=np.hypot(residual,2.0)
        loss=np.dot(w,norm)+ridge*np.dot(a,a)
        grad=x.T@(w*residual/norm)+2*ridge*a
        if len(target):
            err=curv@a-target
            denom=np.hypot(err,1.)
            loss+=shape*denom.mean()
            grad+=shape*(curv.T@(err/denom))/len(err)
        return loss,grad
    result=minimize(cost,np.ones(x.shape[1])*.35,method="L-BFGS-B",jac=True,bounds=[(-1.5,3.)]*x.shape[1],options=dict(ftol=1.e-13,maxiter=300))
    if not result.success:raise RuntimeError(result.message)
    return result.x


def missing_weights(h,dev):
    return dev[1-h] if h>=0 else np.concatenate(dev)


def one(t,meta,saddle):
    d=kt.equation_data(t)
    s,y,src,fold,kind,case=d
    primary=trial_fit.selection_mask(d,t)
    design=[];ref=[];weights=[];shape=[];hess=[]
    for h in (0,1):
        base,x=predictions(t,h,"parent")
        rows=(fold==h)&primary&np.isin(src,kt.BARRIERS)
        design.append((s@x)[rows]*kt.KCAL)
        ref.append((y-s@base)[rows]*kt.KCAL)
        # In the high-level branch the two public inner folds can contain very
        # different source proportions. Avoid giving BH9 the total RDB7 weight.
        scale=np.zeros(rows.sum())
        for label in np.unique(src[rows]):
            mask=src[rows]==label
            scale[mask]=({"RDB7":.85,"BH9":.15}.get(label,1.))/mask.sum()
        weights.append(scale*np.where(kind[rows]=="reaction",1.2,.9))
        if t=="wb97xd3":
            b,xm=predictions(t,h,"modes")
            ix=(meta[:,0]==h)&saddle
            fun=lambda a:(a[ix,3]+a[ix,4]-2*a[ix,0])/.004**2
            shape.append(fun(xm))
            hess.append(meta[ix,2]-fun(b))
        else:shape.append(np.empty((0,len(BRANCHES))));hess.append(np.empty(0))
    scores={str(opt):[] for opt in OPTIONS}
    selected={}
    for h in (0,1,-1):
        for opt in OPTIONS:
            take=lambda data:missing_weights(h,data)
            w=take(weights)
            c=optimize(take(design),take(ref),w/w.sum(),take(shape),take(hess),opt)
            if h>=0:
                e=design[h]@c-ref[h]
                scores[str(opt)].append(np.average(np.abs(e),weights=weights[h]))
            selected[h,opt]=c
    choice=min(OPTIONS,key=lambda opt:np.mean(scores[str(opt)]))
    for h in (0,1,-1):
        coef=selected[h,choice]
        base,x=predictions(t,h,"parent")
        en=base+x@coef
        np.save(WORK/f"mix23-{t}-parent-{h}.npy",en)
        if t=="wb97xd3":
            b,xm=predictions(t,h,"modes")
            modes=b+xm@coef
            np.save(WORK/f"mix23-{t}-modes-{h}.npy",modes)
    err=s@en-y;outer=fold==-1
    np.savez_compressed(WORK/f"{STAGE}-{t}-equations.npz",source=src[outer],case=case[outer],kind=kind[outer],error_kcal=err[outer]*kt.KCAL)
    if t=="wb97xd3":np.save(WORK/f"{STAGE}-held-modes.npy",modes[meta[:,0]==-1])
    model=ForceField.load(WORK/f"p12-public-{t}.npz",dispersion="d3")
    for (stage,prior),c in zip(BRANCHES,coef):
        child=ForceField.load(WORK/f"p{stage}-public-{t}.npz")
        parent=ForceField.load(WORK/f"p{prior}-public-{t}.npz")
        for basis in child.additional_bases[len(parent.additional_bases):]:
            basis.weights*=c
            model.additional_bases.append(basis)
    dest=WORK/f"{STAGE}-{t}.npz"
    model.save_with_base(kt.original.BASE,dest)
    model=ForceField.load(dest,dispersion="d3")
    z,r=trial_fit.sample()
    E,F=model.predict(z,r)
    np.testing.assert_allclose(E,en[trial_fit.sample.index],atol=2.e-7,rtol=0)
    v=np.random.default_rng(89).normal(size=r.shape)
    v/=np.linalg.norm(v)
    fd=(model.energy(z,r-2.e-5*v)-model.energy(z,r+2.e-5*v))/4.e-5
    np.testing.assert_allclose((F*v).sum(),fd,atol=3.e-5)
    print("OOF mixture",t,coef,flush=True)
    return dict(weights=coef.tolist(),settings=choice,development=scores)


def run():
    meta=np.load(WORK/"rad-meta.npy")
    with (WORK/"radical-modes.pkl").open("rb") as f:saddle=np.array([s[1][0].split("/")[-1].startswith("ts") for s in pickle.load(f)])
    fit={t:one(t,meta,saddle) for t in kt.TARGETS}
    kt.write_json(WORK/f"{STAGE}-settings.json",dict(selected=STAGE,branches=dict.fromkeys(kt.TARGETS,"mix23"),fit=fit))
    compare(STAGE)
    validate(STAGE)
    evaluate_trial(STAGE)


if __name__=="__main__":
    with threadpool_limits(1):run()
