"""Learn a common wB97X-D3 PES and a separate high-level energy correction.

RDB7 reports DFT and CC energies at the same structures. The DFT PES uses the
RDB7 DFT and radical energy differences. The high-level readout starts at that
PES and fits the CC and other published high-level differences. Each deployed
checkpoint takes geometry and electronic state only. The energy and force
anchors used here penalize *changes* to an already fitted potential.
"""
import json
import pickle

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from experiments.full_public import kernel_trials as kt, trial_joint_basis as joint
from experiments.full_public import trial_fit, trial_multiscale as ms
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_validation import run as validate
from experiments.reactivity.frequency_validation import TO_CM1
from src.forcefield import ForceField

WORK = kt.WORK
STAGE = "p15-public"
DFT = "wb97xd3"


def rdb_dft(high):
    """Pair every RDB7 equation with a second electronic reference by participant ID."""
    with (WORK / "rdb-dft-energies.pkl").open("rb") as f:
        labels = pickle.load(f)
    _, rows, _ = kt.inputs()
    obs = {(r[3],r[-1]): sum(sign*labels[name] for name,sign in r[4])
           for r in rows if r[0]=="high" and r[1]=="RDB7"}
    matrix, _, src, fold, kind, case = high
    use = src=="RDB7"
    values = np.array([obs[(n,k)] for n,k in zip(case[use],kind[use])])
    assert len(values)==len(obs)
    return matrix[use],values,src[use],fold[use],kind[use],case[use]


def p11_weights(h, size):
    settings = lambda i: json.loads((WORK / f"p{i}-public-settings.json").read_text())
    keys = (settings(8)["selected"], settings(10)["branches"][DFT])
    a,b = [np.load(WORK / f"{k}-{DFT}-weights-{h}.npy") for k in keys]
    n = len(joint.BASE)+12
    w = np.zeros(size)
    w[:n] = (a[:n]+b[:n])*.5
    w[n:n+1440] = a[n:]*.5
    w[n+1440:n+4000] = b[n:]*.5
    return w


def grouped(d, x, energies, h):
    train=(d[3]>=0)&(d[3]!=h)
    blocks=kt.grouped_moments(d,x,d[1]-d[0]@energies,train)
    combined=blocks.pop("other")
    combined+=50*blocks.pop("mid")
    combined+=42*blocks.pop("gap")
    return combined


def quadratic(gram, rows, target, scale):
    """Add weighted linear observations in blocks, keeping the shared memory bound."""
    for i in range(0,len(target),384):
        z=np.column_stack((rows[i:i+384],target[i:i+384]))
        gram+=scale*(z.T@z)


def solve(gram, anchor, strength, ridge):
    mat=gram[:-1,:-1]+strength*anchor
    mat.flat[::len(mat)+1]+=ridge
    return cho_solve(cho_factor(mat,check_finite=False),gram[:-1,-1],check_finite=False)


def write_model(t, weights, energies, d, rad_values):
    """Check serialized energies and conservative forces before scoring any checkpoint."""
    held=d[3]==-1
    model=ForceField.load(WORK / "spin-wb97xd3.npz",dispersion="d3")
    model.prediction_target=ForceField.load(WORK/f"p11-public-{t}.npz").prediction_target
    model.reactive_basis,rich,bond=joint.unpack(weights)
    model.additional_bases=[rich,bond]
    path=WORK/f"{STAGE}-{t}.npz"
    model.save_with_base(kt.original.BASE,path)
    g=ForceField.load(path,dispersion="d3")
    z,r=trial_fit.sample()
    e,f=g.predict(z,r)
    np.testing.assert_allclose(e,energies[trial_fit.sample.index],rtol=0,atol=2.e-7)
    v=np.random.default_rng(87).normal(size=r.shape)
    v/=np.linalg.norm(v)
    fd=(g.energy(z,r-2.e-5*v)-g.energy(z,r+2.e-5*v))/4.e-5
    np.testing.assert_allclose(fd,(v*f).sum(),atol=3.e-5)
    np.savez_compressed(WORK/f"{STAGE}-{t}-equations.npz",source=d[2][held],
        case=d[5][held],kind=d[4][held],error_kcal=((d[0]@energies-d[1])*kt.KCAL)[held])
    if rad_values is not None:
        np.save(WORK/f"{STAGE}-held-modes.npy",rad_values)


def run():
    pub=np.load(WORK/"joint14-pub.npy",mmap_mode="r")
    dim=pub.shape[1]
    rad=np.load(WORK/"joint14-rad.npy",mmap_mode="r").reshape(-1,7,dim)
    anchor=np.load(WORK/"joint14-anchor.npy",mmap_mode="r")
    rb=np.load(WORK/"rad-prior.npy")
    meta=np.load(WORK/"rad-meta.npy")
    with np.load(WORK/"rdb-dft-shapes.npz") as f:
        shapes={k:f[k] for k in f.files}
    with (WORK/"radical-modes.pkl").open("rb") as f:
        saddle=np.array([s[1][0].split("/")[-1].startswith("ts") for s in pickle.load(f)])
    dh=kt.equation_data("high")
    dr=kt.equation_data(DFT)
    ds=rdb_dft(dh)
    xh=dh[0]@pub
    xr=dr[0]@pub
    xs=xh[dh[2]=="RDB7"]
    development={t:{} for t in ("high",DFT)}
    chosen={}
    for h in (0,1,-1):
        col=h if h>=0 else 2
        base=np.load(WORK/f"spin-energy-wb97xd3-{h if h>=0 else 'all'}.npy")
        prior=p11_weights(h,dim)
        p_energy=base+pub@prior
        np.testing.assert_allclose(p_energy,ms.baseline(DFT,h,"parent"),rtol=0,atol=1e-6)
        p_rad=rb[:,:,col]+rad@prior
        # The radical parent is also the common prior on the singlet DFT data.
        grams=grouped(dr,xr,p_energy,h)
        grams+=grouped(ds,xs,p_energy,h)
        zz,yy=trial_fit.rad_rows(rad,p_rad,meta,np.where(saddle,3.,1.))
        train=(meta[:,0]>=0)&(meta[:,0]!=h)
        grams+=trial_fit.reduce_gram(zz,yy,np.repeat(train,3))
        del zz,yy
        # The first measurement is a directional force. The following four
        # are signed curvatures, including two middle modes across the spectrum.
        mask=(shapes["fold"]>=0)&(shapes["fold"]!=h)
        x=shapes["x"][mask]
        target=shapes["y"][mask]-shapes["base"][mask,:,col]-x@prior
        w=np.tile([2.,.45,.25,.25,.18],(len(x),1))
        w[:,0]*=shapes["stationary"][mask]
        quadratic(grams,(x*w[:,:,None]).reshape(-1,dim),(target*w).ravel(),3.)
        del x
        nist=joint.nist_constraints(DFT,h,dim)[:,:-1]
        grams[:-1,:-1]+=100*(nist.T@nist)
        options=(.08,.35,1.2)
        candidates={}
        for strength in options:
            delta=solve(grams,anchor,strength,2*strength)
            wt=prior+delta
            en=base+pub@wt
            modes=rb[:,:,col]+rad@wt
            candidates[strength]=wt
            if h>=0:
                e=dr[0]@en-dr[1]
                scores=ms.metrics(dr,e,(dr[3]==h))
                cm,sign=ms.saddle_metrics(modes,(meta[:,0]==h)&saddle,meta)
                val=.4*scores[0]+.3*(scores[1]+scores[2])+cm/350
                development[DFT].setdefault(strength,[]).append(float(val))
        # Use the same DFT hyperparameter for all folds in the high-level fit.
        # .35 is fixed in advance so an inner validation fold cannot pick its own parent.
        wt=candidates[.35]
        del grams
        en=base+pub@wt
        if h<0:
            chosen[DFT]=min(options,key=lambda s:np.mean(development[DFT][s]))
            best=candidates[chosen[DFT]]
            write_model(DFT,best,base+pub@best,dr,(rb[:,:,col]+rad@best)[meta[:,0]==-1])
        print("Completed matched DFT energy and curvature fit, fold",h,flush=True)
        hi=grouped(dh,xh,en,h)
        # Keep the DFT surface's measured shape while allowing energy differences
        # to move to the higher reference level. Each label retains its own method.
        hi[:-1,:-1]+=120*(nist.T@nist)
        xm=shapes["x"][mask]*w[:,:,None]
        xm=xm.reshape(-1,dim)
        hi[:-1,:-1]+=3*(xm.T@xm)
        del xm
        pri=trial_fit.selection_mask(dh,"high")
        opts=(.12,.6,2.0)
        selected=opts if h>=0 else [min(opts,key=lambda s:np.mean(development["high"][s]))]
        for a in selected:
            best=wt+solve(hi,anchor,a,3*a)
            e=base+pub@best
            if h>=0:
                scores=ms.metrics(dh,dh[0]@e-dh[1],(dh[3]==h)&pri)
                development["high"].setdefault(a,[]).append(.4*scores[0]+.3*(scores[1]+scores[2]))
            else:
                chosen["high"]=a
                write_model("high",best,e,dh,None)
            np.save(WORK/f"fidelity-high-{h}-{a:g}.npy",best)
        for a,weights in candidates.items():
            np.save(WORK/f"fidelity-wb97xd3-{h}-{a:g}.npy",weights)
        print("Completed high-level energy correction, fold",h,flush=True)
    kt.write_json(WORK/f"{STAGE}-settings.json",dict(selected=STAGE,choices=chosen,development=development))
    compare(STAGE)
    validate(STAGE)


if __name__=="__main__":
    with threadpool_limits(5):
        run()
