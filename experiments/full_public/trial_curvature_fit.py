"""Fit a reaction correction while preserving curvature throughout the PBE0 training set.

Every PBE0 training frame has a cached central geometry and two Cartesian
displacements. Their second difference constrains the change of the fitted PES
curvature along that displacement. This supplements the actual NIST and radical
derivative constraints and is independent of the held-out frequency panel.
"""
import argparse
import json
import pickle
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits
from experiments.full_public import kernel_trials as kt,trial_fit,trial_multiscale as ms
from experiments.full_public.trial_isomers import gram as isomers
from experiments.full_public.trial_constraints import nist_rows
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_validation import run as validate
from src.forcefield import ForceField
from src.bond_order_kernel import BondOrderKernel
from src.reactive_basis import ReactiveBasis
from src.message_kernel import MessageKernel
from src.periodic_kernel import PeriodicKernel
from src.orbital_moments import OrbitalMomentKernel
from src.atomic_states import AtomicStateKernel
from src.valence_competition import CompetitionKernel
from src.occupied_orbitals import OccupationKernel
from src.adaptive_spectrum import SpectralKernel

WORK=kt.WORK


def anchor_blocks(main,n,tag):
    """Compute separate energy/force and curvature Gram matrices in bounded slices."""
    from scipy.linalg.blas import dsyrk
    paths=[WORK/f"{tag}-anchor-{s}.npy" for s in ("ef","curvature")]
    if all(p.exists() for p in paths):
        return [np.load(p,mmap_mode="r") for p in paths]
    sources=np.load(WORK/"hdf-sources.npy")
    assert len(main)==n+3*len(sources)
    count=dict(zip(*np.unique(sources,return_counts=True)))
    w=np.sqrt([4500/count[s] for s in sources])[:,None]
    g=np.zeros((main.shape[1],main.shape[1]),order="F")
    c=np.zeros_like(g,order="F")
    for lo in range(0,len(sources),480):
        hi=min(lo+480,len(sources))
        center,plus,minus=main[n+3*lo:n+3*hi].reshape(hi-lo,3,-1).transpose(1,0,2)
        ef=np.vstack([center*w[lo:hi],(plus-minus)*(4/.002)*w[lo:hi]])
        # Symmetric displacements are +/-0.001 Å along a unit Cartesian vector.
        curvature=((plus-center)+(minus-center))/.001**2*w[lo:hi]
        g=dsyrk(1.,np.asfortranarray(ef.T),beta=1.,c=g,lower=1,overwrite_c=1)
        c=dsyrk(1.,np.asfortranarray(curvature.T),beta=1.,c=c,lower=1,overwrite_c=1)
    for mat,path in zip((g,c),paths):
        i,j=np.tril_indices(len(mat),-1)
        mat[j,i]=mat[i,j]
        np.save(path,mat)
    return g,c


def fit(stage):
    family, widths, parent = {
        17:("metric",(1.15,),"p12-public"),
        18:("density",(.7,1.35),"p17-public"),
        19:("message",(.7,1.35),"p18-public"),
        25:("periodic",(.85,1.35),"p24-public"),
        27:("orbital",(.8,1.5),"p25-public"),
        28:("state",(.8,1.5),"p27-public"),
        30:("competition",(.75,1.35),"p29-public"),
        33:("occupied",(.74,1.35),"p28-public"),
        34:("adaptive",(.85,1.3),"p28-public"),
        35:("frozen",(.85,1.3),"p28-public"),
        36:("orbital-bond",(.74,1.35),"p33-public"),
    }[stage]
    dest=f"p{stage}-public"
    n=len(np.load(WORK/"prior.npy",mmap_mode="r"))
    meta=np.load(WORK/"rad-meta.npy")
    with (WORK/"radical-modes.pkl").open("rb") as f:
        saddle=np.array([s[1][0].split("/")[-1].startswith("ts") for s in pickle.load(f)])
    trials={t:{} for t in kt.TARGETS}
    configs={}
    for width in widths:
        cls={"metric":BondOrderKernel,"density":ReactiveBasis,"message":MessageKernel,"periodic":PeriodicKernel,"orbital":OrbitalMomentKernel,"state":AtomicStateKernel,"competition":CompetitionKernel,"occupied":OccupationKernel,"orbital-bond":OccupationKernel,"adaptive":SpectralKernel,"frozen":SpectralKernel}[family]
        with np.load(WORK/f"{family}-basis.npz") as f: basis=cls.load(f)
        basis.length=width
        if stage==25: basis.degree=3 if width==.85 else 6
        prefix=f"{family}-{width:g}"
        main=np.load(WORK/f"{prefix}-main.npy",mmap_mode="r")
        rad=np.load(WORK/f"{prefix}-rad.npy",mmap_mode="r").reshape(-1,7,basis.size)
        pub=main[:n]
        g,c=anchor_blocks(main,n,prefix)
        old=ForceField.load(kt.OUT/"joint-pbe0-reaction.npz",dispersion="d3")
        rows=nist_rows(lambda z,r:basis.design(z,r,True)[1],old)[:,:-1]
        ng=rows.T@rows
        for t in kt.TARGETS:
            d=kt.equation_data(t)
            s,y,src,fold,kind,case=d
            primary=trial_fit.selection_mask(d,t)
            x=s@pub
            for h in (0,1,-1):
                base=ms.baseline(t,h,"parent",parent)
                residual=y-s@base
                block=kt.grouped_moments(d,x,residual,(fold>=0)&(fold!=h))
                a=block.pop("other")
                a+=48*block.pop("mid")
                a+=36*block.pop("gap")
                a+=isomers(t,h,pub,base)
                a[:-1,:-1]+=30*ng
                if t=="wb97xd3":
                    bm=ms.baseline(t,h,"modes",parent)
                    zz,yy=trial_fit.rad_rows(rad,bm,meta,np.where(saddle,3.,1.))
                    use=np.repeat((meta[:,0]>=0)&(meta[:,0]!=h),3)
                    a+=trial_fit.reduce_gram(zz,yy,use)
                    del zz,yy
                # Fix a nonzero curvature penalty throughout the sweep. Its
                # scale is in Å^4 relative to an energy-squared residual.
                settings=((.01,.0015),(.06,.009),(.3,.09)) if stage==25 else ((.02,.006),(.08,.025),(.25,.12))
                for reg,shape in settings:
                    key=f"p{stage}-{prefix}-r{reg:g}-c{shape:g}"
                    configs[key]=width
                    mt=a[:-1,:-1]+reg*g+shape*c
                    mt.flat[::len(mt)+1]+=.25
                    if stage==30 and t=="wb97xd3":
                        from experiments.full_public.trial_competition import strengthen
                        coef=strengthen(mt,a[:-1,-1],rad,bm,meta,saddle,(meta[:,0]>=0)&(meta[:,0]!=h))
                    else:
                        coef=cho_solve(cho_factor(mt,check_finite=False),a[:-1,-1],check_finite=False)
                    en=base+pub@coef
                    err=x@coef-residual
                    np.save(WORK/f"{key}-{t}-weights-{h}.npy",coef)
                    np.save(WORK/f"{key}-{t}-parent-{h}.npy",en)
                    if t=="wb97xd3":
                        modes=bm+rad@coef
                        np.save(WORK/f"{key}-{t}-modes-{h}.npy",modes)
                    if h>=0:
                        scores=ms.metrics(d,err,(fold==h)&primary)
                        score=.40*scores[0]+.30*(scores[1]+scores[2])
                        if t=="wb97xd3":
                            mh=(meta[:,0]==h)&saddle
                            cm,_=ms.saddle_metrics(modes,mh,meta)
                            score+=cm/350
                        trials[t].setdefault(key,[]).append(float(score))
                    else:
                        held=fold==-1
                        np.savez_compressed(WORK/f"{key}-{t}-equations.npz",source=src[held],
                            case=case[held],kind=kind[held],error_kcal=err[held]*kt.KCAL)
            print("Completed curvature-constrained fits",t,width,flush=True)
    selected={}
    import shutil
    for t,dev in trials.items():
        key=min(dev,key=lambda k:np.mean(dev[k]))
        selected[t]=key
        w=configs[key]
        with np.load(WORK/f"{family}-basis.npz") as f:basis=cls.load(f)
        basis.weights=np.load(WORK/f"{key}-{t}-weights--1.npy")
        basis.length=w
        if stage==25: basis.degree=3 if w==.85 else 6
        model=ForceField.load(WORK/f"{parent}-{t}.npz",dispersion="d3")
        model.additional_bases.append(basis)
        path=WORK/f"{dest}-{t}.npz"
        model.save_with_base(kt.original.BASE,path)
        restored=ForceField.load(path,dispersion="d3")
        z,r=trial_fit.sample()
        e,force=restored.predict(z,r)
        np.testing.assert_allclose(e,np.load(WORK/f"{key}-{t}-parent--1.npy")[trial_fit.sample.index],rtol=0,atol=2e-7)
        v=np.random.default_rng(437).normal(size=r.shape)
        v/=np.linalg.norm(v)
        fd=(restored.energy(z,r-2e-5*v)-restored.energy(z,r+2e-5*v))/4e-5
        np.testing.assert_allclose(fd,(force*v).sum(),atol=3.e-5)
        shutil.copyfile(WORK/f"{key}-{t}-equations.npz",WORK/f"{dest}-{t}-equations.npz")
        if t=="wb97xd3":
            np.save(WORK/f"{dest}-held-modes.npy",np.load(WORK/f"{key}-{t}-modes--1.npy")[meta[:,0]==-1])
    kt.write_json(WORK/f"{dest}-settings.json",dict(selected=dest,branches=selected,development=trials))
    compare(dest)
    validate(dest)
    print("Verified checkpoints and scored exact matched panel",selected,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",type=int,choices=(17,18,19,25,27,28,30,33,34,35,36))
    with threadpool_limits(5):fit(p.parse_args().stage)
