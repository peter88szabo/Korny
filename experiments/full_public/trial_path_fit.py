"""Fit a joint local potential on energy differences, projected forces and curvatures.

The objective uses the three actual benchmark observables. Iterative Huber
weights limit the influence of extreme training residuals on the MAE. Grambow
trajectory labels enter only the wB97X-D3 branch and never stand in for CC
energies. Four levels of trajectory, smoothness and curvature regularization
are compared exclusively on the inner formulas. The final energy still
contains one geometry-dependent scalar readout with conservative forces.
"""
import pickle
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits
from experiments.full_public import kernel_trials as kt, trial_path_features as paths, trial_fit, trial_multiscale as ms
from experiments.full_public.trial_isomers import gram as isomer_gram
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_validation import run as validate
from experiments.full_public.trial_spectra import evaluate_trial
from src.forcefield import ForceField

WORK=kt.WORK
TAG=paths.TAG
STAGE="p22-public"
SETTINGS=((.03,.02,.1,30.,8.),(.12,.04,1.,100.,8.),(.03,.02,1.,100.,4.),(.12,.10,3.,300.,12.))


def path_gram(h):
    x=np.load(WORK/f"{TAG}-steps.npy",mmap_mode="r")
    info=np.load(WORK/f"{TAG}-labels.npy")
    keep=np.load(WORK/f"{TAG}-eligible.npy")[:,(0,1,-1).index(h)]
    col=(0,1,-1).index(h)
    out=np.zeros((x.shape[-1]+1,)*2)
    ids=np.flatnonzero(keep)
    owner=info[:,10].astype(int)
    count=np.bincount(owner)
    # Cap the influence of long optimization paths. Retain every eligible step.
    for lo in range(0,len(ids),300):
        ix=ids[lo:lo+300]
        w=np.minimum(1.,np.sqrt(12./count[owner[ix]]))
        for j in (0,1):
            rhs=info[ix,j]-info[ix,(4 if j==0 else 7)+col]
            scale=w if j==0 else .35*w*info[ix,2]
            block=np.column_stack((x[ix,j],rhs))*scale[:,None]
            out+=block.T@block
    print("Included Grambow trajectory rows",h,len(ids),int(info[keep,2].sum()),flush=True)
    return out


def candidates():
    pub=np.load(WORK/f"{TAG}-main.npy",mmap_mode="r")[:len(np.load(WORK/"prior.npy",mmap_mode="r"))]
    rad=np.load(WORK/f"{TAG}-rad.npy",mmap_mode="r").reshape(-1,7,pub.shape[1])
    ef=np.load(WORK/f"{TAG}-anchor-ef.npy",mmap_mode="r")
    curv=np.load(WORK/f"{TAG}-anchor-curvature.npy",mmap_mode="r")
    rkhs=np.load(WORK/f"{TAG}-rkhs.npy",mmap_mode="r")
    meta=np.load(WORK/"rad-meta.npy")
    with (WORK/"radical-modes.pkl").open("rb") as f:
        saddle=np.array([s[1][0].split("/")[-1].startswith("ts") for s in pickle.load(f)])
    selections={}
    dev={t:{} for t in kt.TARGETS}
    names={}
    for ti,t in enumerate(kt.TARGETS):
        d=kt.equation_data(t)
        s,y,src,fold,kind,case=d
        primary=trial_fit.selection_mask(d,t)
        barrier=np.isin(src,kt.BARRIERS)
        x=s@pub
        for h in (0,1,-1):
            col=(0,1,-1).index(h)
            train=(fold>=0)&(fold!=h)
            base=ms.baseline(t,h,"parent","p12-public")
            residual=y-s@base
            # Balance sources using the same cap as established Korny fits.
            w=np.zeros(len(src))
            for source in np.unique(src[train]):
                keep=train&(src==source)
                w[keep]=min(1.,4500/keep.sum())
            aux=isomer_gram(t,h,pub,base)
            others=train&~barrier
            if others.any():
                z=np.column_stack((x[others],residual[others]))*np.sqrt(w[others])[:,None]
                aux+=z.T@z
                del z
            nist=np.load(WORK/f"{TAG}-nist-{3*ti+col}.npy").astype(float)
            ngram=nist.T@nist
            del nist
            if ti:
                bm=ms.baseline(t,h,"modes","p12-public")
                z,yr=trial_fit.rad_rows(rad,bm,meta,np.where(saddle,3.,1.))
                mask=np.repeat((meta[:,0]>=0)&(meta[:,0]!=h),3)
                aux+=trial_fit.reduce_gram(z,yr,mask)
                del z,yr
                pg=path_gram(h)
            ix=np.flatnonzero(train&barrier)
            # Optimizing forward and reverse separately treats an endpoint error
            # consistently regardless of the archive's chosen reaction direction.
            xb=x[ix]
            yb=residual[ix]
            base_weight=w[ix]*np.where(kind[ix]=="reaction",32.,28.)
            if h<0:
                selected=min(dev[t],key=lambda k:np.mean(dev[t][k]))
                selections[t]=selected
            for i,(reg,curvature,pathweight,shape,huber) in enumerate(SETTINGS):
                key=f"path22-{i}"
                if h<0 and key!=selected:continue
                fixed=aux+shape*ngram
                if ti:fixed+=pathweight*pg
                fixed[:-1,:-1]+=reg*ef+curvature*curv+2.*rkhs
                fixed.ravel()[::len(fixed)+1]+=.05
                rw=base_weight
                for iteration in range(3):
                    a=fixed[:-1,:-1].copy()
                    b=fixed[:-1,-1].copy()
                    for lo in range(0,len(ix),500):
                        sl=slice(lo,lo+500)
                        zw=xb[sl]*np.sqrt(rw[sl])[:,None]
                        a+=zw.T@zw
                        b+=xb[sl].T@(rw[sl]*yb[sl])
                    coef=cho_solve(cho_factor(a,overwrite_a=True,check_finite=False),b,check_finite=False)
                    del a
                    rw=base_weight*np.minimum(1.,huber/np.maximum(1.e-7,abs(xb@coef-yb)*kt.KCAL))
                en=base+pub@coef
                err=s@en-y
                if ti:
                    modes=bm+rad@coef
                np.save(WORK/f"{key}-{t}-weights-{h}.npy",coef)
                np.save(WORK/f"{key}-{t}-parent-{h}.npy",en)
                if ti:np.save(WORK/f"{key}-{t}-modes-{h}.npy",modes)
                if h>=0:
                    rx,fw,bw=ms.metrics(d,err,(fold==h)&primary)
                    value=.4*rx+.3*(fw+bw)
                    if ti:
                        imag,_=ms.saddle_metrics(modes,(meta[:,0]==h)&saddle,meta)
                        value+=imag/500.
                    dev[t].setdefault(key,[]).append(value)
                else:
                    mask=fold==-1
                    np.savez_compressed(WORK/f"{STAGE}-{t}-equations.npz",source=src[mask],case=case[mask],kind=kind[mask],error_kcal=err[mask]*kt.KCAL)
                    if ti:np.save(WORK/f"{STAGE}-held-modes.npy",modes[meta[:,0]==-1])
                    save(t,coef,en)
            print("Completed joint path fit",t,"fold",h,flush=True)
    kt.write_json(WORK/f"{STAGE}-settings.json",dict(selected=STAGE,branches=selections,development=dev,settings=SETTINGS))
    compare(STAGE)
    validate(STAGE)
    print("Scored public benchmark and saddle modes",flush=True)


def save(target,coef,energies):
    model=ForceField.load(WORK/f"p12-public-{target}.npz",dispersion="d3")
    offset=0
    for block,idx in paths.bases():
        block.weights=np.zeros(block.size)
        block.weights[idx]=coef[offset:offset+len(idx)]
        offset+=len(idx)
        model.additional_bases.append(block)
    assert offset==len(coef)
    location=WORK/f"{STAGE}-{target}.npz"
    model.save_with_base(kt.original.BASE,location)
    restored=ForceField.load(location,dispersion="d3")
    z,r=trial_fit.sample()
    e,f=restored.predict(z,r)
    np.testing.assert_allclose(e,energies[trial_fit.sample.index],rtol=0,atol=2.e-7)
    dx=np.random.default_rng(6219).normal(size=r.shape)
    dx/=np.linalg.norm(dx)
    derivative=(restored.energy(z,r-2.e-5*dx)-restored.energy(z,r+2.e-5*dx))/4.e-5
    np.testing.assert_allclose(np.sum(dx*f),derivative,atol=3.e-5)


if __name__=="__main__":
    with threadpool_limits(5):candidates()
    with threadpool_limits(1):evaluate_trial(STAGE)
