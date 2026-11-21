"""Use force-supervised RDB7 local energies as an extra coordinate of a bond kernel.

Distances between local environments now include their DFT-trained bond-energy
difference. Equally different geometries can therefore have very different
kernel similarities if one is energetically expensive. Each fold has its own
trajectory teacher. Computing the full nonlinear representation independently
for each fold keeps its development and validation labels out of the metric.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import pickle
import shutil
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from experiments.full_public import kernel_trials as kt, trial_metric as kernel, trial_sites, trial_aux, trial_fit, trial_multiscale as ms
from experiments.full_public.trial_curvature_fit import anchor_blocks
from experiments.full_public.trial_constraints import nist_rows
from experiments.full_public.trial_isomers import gram as isomer_gram
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_validation import run as validate
from experiments.full_public.trial_spectra import evaluate_trial
from src.bond_order_kernel import BondOrderKernel,METRIC
from src.forcefield import ForceField

WORK=kt.WORK
STAGE="p26-public"
PARENT="p24-public"
OPTIONS=((.02,.004,10.),(.10,.03,45.),(.4,.12,180.))


def build(h):
    with np.load(WORK/"metric-basis.npz") as f:b=BondOrderKernel.load(f)
    b.length=1.15
    b.energy_coefficients=np.load(WORK/"rdb-path-teacher"/f"teacher-{h}.npy")
    b.center_energies=np.zeros(b.size)
    for k,(start,stop) in enumerate(zip(b.offsets,b.offsets[1:])):
        if start==stop:continue
        z=(b.centers[start:stop]*METRIC)@b.transforms[k]
        g=np.exp(-.5*cdist(z,z,"sqeuclidean")/1.15**2)
        energy=g@b.energy_coefficients[start:stop]
        # Fix the coordinate scale from training centers, independently of all
        # reaction- and barrier-validation scores. The floor is in eV.
        scale=max(.8,.40*(np.percentile(energy,90)-np.percentile(energy,10)))
        b.center_energies[start:stop]=energy/scale
        b.energy_coefficients[start:stop]/=scale
    return b


def work(i):
    for j in range(i,min(i+80,len(points))):
        mmap[j]=kernel.values(*points[j])[0]


def make_cache(h,b,data,rad,n):
    global points,mmap
    tag=f"latent26-{h}"
    kernel.native(b)
    rng=np.random.default_rng(4291)
    for i in rng.choice(n,6,replace=False):
        z,r=data[i];v=rng.normal(size=r.shape);v/=np.linalg.norm(v)
        x,j=b.design(z,r,True)
        np.testing.assert_allclose(x,kernel.values(z,r)[0],atol=2.e-9)
        fd=(kernel.values(z,r-1.e-5*v)[0]-kernel.values(z,r+1.e-5*v)[0])/2.e-5
        np.testing.assert_allclose(j.T@v.ravel(),fd,atol=5.e-6)
    for side,frames in (("main",data),("rad",rad)):
        out=WORK/f"{tag}-{side}.npy"
        if out.exists() or (side=="main" and (WORK/f"{tag}-pub.npy").exists()):continue
        points=frames
        tmp=out.with_suffix(".pending")
        mmap=np.lib.format.open_memmap(tmp,mode="w+",dtype=float,shape=(len(points),b.size))
        with ProcessPoolExecutor(max_workers=5,mp_context=mp.get_context("fork")) as pool:
            for _ in pool.map(work,range(0,len(points),80),buffersize=12):pass
        mmap.flush();del mmap
        tmp.replace(out)
    pub=WORK/f"{tag}-pub.npy"
    if not pub.exists():
        path=WORK/f"{tag}-main.npy"
        full=np.load(path,mmap_mode="r")
        with threadpool_limits(4):anchor_blocks(full,n,tag)
        tmp=pub.with_suffix(".pending")
        with tmp.open("wb") as f:np.save(f,full[:n])
        tmp.replace(pub)
        del full
        path.unlink()  # The two exact Gram matrices retain all training-frame constraints.
    np.savez_compressed(WORK/f"{tag}-basis.npz",**b.fields())
    print("Verified and cached fold-specific trajectory metric",h,flush=True)


def run():
    gs,_,_=kt.inputs()
    with (WORK/"ids.pkl").open("rb") as f:_,keys=pickle.load(f)
    main=[(gs[n].z,gs[n].r) for n in keys]
    n=len(main)
    for _,z,r,v in trial_sites.hdf():main.extend([(z,r),(z,r+.001*v),(z,r-.001*v)])
    extra,off,_=trial_aux.read_frames()
    radpts=extra[off:]
    meta=np.load(WORK/"rad-meta.npy")
    with (WORK/"radical-modes.pkl").open("rb") as f:saddle=np.array([s[1][0].split("/")[-1].startswith("ts") for s in pickle.load(f)])
    data={t:kt.equation_data(t) for t in kt.TARGETS}
    primary={t:trial_fit.selection_mask(data[t],t) for t in kt.TARGETS}
    trials={t:{} for t in kt.TARGETS}
    selected={}
    for h in (0,1,-1):
        b=build(h)
        tag=f"latent26-{h}"
        with threadpool_limits(1):make_cache(h,b,main,radpts,n)
        pub=np.load(WORK/f"{tag}-pub.npy",mmap_mode="r")
        rad=np.load(WORK/f"{tag}-rad.npy",mmap_mode="r").reshape(-1,7,b.size)
        ef=np.load(WORK/f"{tag}-anchor-ef.npy",mmap_mode="r")
        curv=np.load(WORK/f"{tag}-anchor-curvature.npy",mmap_mode="r")
        pbe=ForceField.load(kt.OUT/"joint-pbe0-reaction.npz",dispersion="d3")
        nr=nist_rows(lambda z,r:b.design(z,r,True)[1],pbe)[:,:-1]
        ng=nr.T@nr
        for t in kt.TARGETS:
            d=data[t]
            s,y,src,fold,kind,case=d
            x=s@pub
            base=ms.baseline(t,h,"parent",PARENT)
            res=y-s@base
            inc=kt.grouped_moments(d,x,res,(fold>=0)&(fold!=h))
            obj=inc.pop("other")+50*inc.pop("mid")+42*inc.pop("gap")+isomer_gram(t,h,pub,base)
            if t=="wb97xd3":
                bm=ms.baseline(t,h,"modes",PARENT)
                z,rhs=trial_fit.rad_rows(rad,bm,meta,np.where(saddle,3.,1.))
                obj+=trial_fit.reduce_gram(z,rhs,np.repeat((meta[:,0]>=0)&(meta[:,0]!=h),3))
                del z,rhs
            if h<0:selected[t]=min(trials[t],key=lambda k:np.mean(trials[t][k]))
            for i,(ridge,shape,nist) in enumerate(OPTIONS):
                key=f"latent26-r{i}"
                if h<0 and key!=selected[t]:continue
                a=obj[:-1,:-1]+ridge*ef+shape*curv+nist*ng
                a.flat[::len(a)+1]+=.3
                weights=cho_solve(cho_factor(a,check_finite=False),obj[:-1,-1],check_finite=False)
                en=base+pub@weights
                error=s@en-y
                np.save(WORK/f"{key}-{t}-weights-{h}.npy",weights)
                np.save(WORK/f"{key}-{t}-parent-{h}.npy",en)
                if t=="wb97xd3":
                    modes=bm+rad@weights
                    np.save(WORK/f"{key}-{t}-modes-{h}.npy",modes)
                if h>=0:
                    re,fw,bw=ms.metrics(d,error,(fold==h)&primary[t])
                    score=.4*re+.3*(fw+bw)
                    if t=="wb97xd3":score+=ms.saddle_metrics(modes,(meta[:,0]==h)&saddle,meta)[0]/400.
                    trials[t].setdefault(key,[]).append(score)
                else:
                    outer=fold==-1
                    np.savez_compressed(WORK/f"{STAGE}-{t}-equations.npz",source=src[outer],case=case[outer],kind=kind[outer],error_kcal=error[outer]*kt.KCAL)
                    if t=="wb97xd3":np.save(WORK/f"{STAGE}-held-modes.npy",modes[meta[:,0]==-1])
                    model=ForceField.load(WORK/f"{PARENT}-{t}.npz",dispersion="d3")
                    b.weights=weights
                    model.additional_bases.append(b)
                    path=WORK/f"{STAGE}-{t}.npz"
                    model.save_with_base(kt.original.BASE,path)
                    restored=ForceField.load(path,dispersion="d3")
                    z,r=trial_fit.sample()
                    e,force=restored.predict(z,r)
                    np.testing.assert_allclose(e,en[trial_fit.sample.index],rtol=0,atol=2.e-7)
                    v=np.random.default_rng(7543).normal(size=r.shape);v/=np.linalg.norm(v)
                    fd=(restored.energy(z,r-2.e-5*v)-restored.energy(z,r+2.e-5*v))/4.e-5
                    np.testing.assert_allclose(fd,(force*v).sum(),atol=4.e-5)
            print("Completed trajectory-metric fits",h,t,flush=True)
    kt.write_json(WORK/f"{STAGE}-settings.json",dict(selected=STAGE,branches=selected,development=trials))
    compare(STAGE)
    validate(STAGE)
    with threadpool_limits(1):evaluate_trial(STAGE)


if __name__=="__main__":
    with threadpool_limits(5):run()
