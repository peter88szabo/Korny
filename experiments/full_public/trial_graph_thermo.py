"""Fit topology-dependent energy corrections to branch-specific energy residuals.

Single-bond neighborhoods stay locally flat across typical equilibrium lengths.
Continuous bond occupancies and closed-walk counts change as a bond forms or
breaks. The objective includes measured DFT curvatures for transition states,
endpoints and intermediate-frequency modes. A fraction of the other target's
*residual* may be transferred; each reference retains its own calibrated parent.
"""
import argparse
import pickle
import shutil
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from threadpoolctl import threadpool_limits
from experiments.full_public import kernel_trials as kt,trial_fit,trial_multiscale as ms
from experiments.full_public.trial_curvature_fit import anchor_blocks
from experiments.full_public.trial_constraints import nist_rows
from experiments.full_public.trial_isomers import gram as isomer_gram
from experiments.full_public.trial_fidelity import rdb_dft
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_validation import run as validate
from experiments.reactivity.frequency_validation import TO_CM1
from src.forcefield import ForceField
from src.bond_order_kernel import BondOrderKernel
from src.message_kernel import MessageKernel

WORK=kt.WORK
PARENT="p12-public"


def pack(d,x,base,h):
    y=d[1]-d[0]@base
    inc=kt.grouped_moments(d,x,y,(d[3]>=0)&(d[3]!=h))
    a=inc.pop("other")
    a+=50*inc.pop("mid")
    a+=45*inc.pop("gap")
    return a


def signed(curv,mass):
    return TO_CM1*np.sign(curv)*np.sqrt(abs(curv)/mass)


def run(stage):
    stage_name=f"p{stage}-public"
    family, kernel, widths = ("topology", MessageKernel, (.7,1.35)) if stage==20 else ("gated", BondOrderKernel, (1.15,1.15/np.sqrt(2)))
    # Each candidate uses the same reaction split. Test the weight of the
    # RDB curvature labels only on inner folds before scoring the public panel.
    settings = ((0.,.02,4.),(.15,.02,4.),(.4,.12,4.)) if stage==20 else ((0.,.02,.02),(0.,.02,.3),(.15,.02,.3),(.4,.12,.02))
    n=len(np.load(WORK/"prior.npy",mmap_mode="r"))
    meta=np.load(WORK/"rad-meta.npy")
    with (WORK/"radical-modes.pkl").open("rb") as f:
        saddle=np.array([s[1][0].split("/")[-1].startswith("ts") for s in pickle.load(f)])
    with np.load(WORK/"rdb-dft-shapes.npz") as f:
        measures={k:f[k] for k in ("fold","name","stationary","y","mass")}
    with np.load(WORK/"p12-rdb-shape-parents.npz") as f:
        np.testing.assert_array_equal(f["name"],measures["name"])
        shape_base=f["base"]
    refs=measures["y"]
    frequency=signed(refs[:,1:],measures["mass"][:,1:])
    datasets={t:kt.equation_data(t) for t in kt.TARGETS}
    low=rdb_dft(datasets["high"])
    primary={t:trial_fit.selection_mask(datasets[t],t) for t in kt.TARGETS}
    dev={t:{} for t in kt.TARGETS}
    config={}
    for width in widths:
        tag=f"{family}-{width:g}"
        with np.load(WORK/f"{family}-basis.npz") as f:basis=kernel.load(f)
        basis.length=width
        main=np.load(WORK/f"{tag}-main.npy",mmap_mode="r")
        pub=main[:n]
        rad=np.load(WORK/f"{tag}-rad.npy",mmap_mode="r").reshape(-1,7,basis.size)
        sx=np.load(WORK/f"{tag}-rdb-shapes.npy",mmap_mode="r")
        ef,curv=anchor_blocks(main,n,tag)
        old=ForceField.load(kt.OUT/"joint-pbe0-reaction.npz",dispersion="d3")
        tmp=nist_rows(lambda z,r:basis.design(z,r,True)[1],old)[:,:-1]
        ng=tmp.T@tmp
        xs={t:d[0]@pub for t,d in datasets.items()}
        xl=xs["high"][datasets["high"][2]=="RDB7"]
        for h in (0,1,-1):
            col=h if h>=0 else 2
            base={t:ms.baseline(t,h,"parent",PARENT) for t in kt.TARGETS}
            bm=ms.baseline("wb97xd3",h,"modes",PARENT)
            tr=(measures["fold"]>=0)&(measures["fold"]!=h)
            dftgram=pack(low,xl,base["high"],h)
            radical_gram=pack(datasets["wb97xd3"],xs["wb97xd3"],base["wb97xd3"],h)
            for t,ti in (("high",0),("wb97xd3",1)):
                d=datasets[t]
                core=pack(d,xs[t],base[t],h) if ti==0 else radical_gram.copy()
                aux=radical_gram if ti==0 else dftgram
                core+=2*isomer_gram(t,h,pub,base[t])
                core[:-1,:-1]+=55*ng
                # Curvature errors approximate frequency errors after division by
                # the reference frequency. A floor bounds soft-mode influence.
                qw=np.zeros_like(refs)
                qw[:,0]=3*measures["stationary"]
                qw[:,1:]=np.clip(195/np.maximum(abs(frequency),250),.08,.65)
                target=refs-shape_base[:,:,ti,col]
                z=np.column_stack(((sx[tr]*qw[tr,:,None]).reshape(-1,basis.size),(target[tr]*qw[tr]).ravel()))
                shapes=z.T@z
                del z
                if ti==1:
                    zz,yy=trial_fit.rad_rows(rad,bm,meta,np.where(saddle,4.,1.))
                    keep=np.repeat((meta[:,0]>=0)&(meta[:,0]!=h),3)
                    core+=2*trial_fit.reduce_gram(zz,yy,keep)
                    del zz,yy
                for transfer,ridge,shape_weight in settings:
                    key=f"p{stage}-{tag}-x{transfer:g}-r{ridge:g}"
                    if stage!=20: key+=f"-s{shape_weight:g}"
                    config[key]=width
                    a=core[:-1,:-1]+shape_weight*shapes[:-1,:-1]+transfer*aux[:-1,:-1]+ridge*ef+.009*curv
                    a.flat[::len(a)+1]+=.25
                    rhs=core[:-1,-1]+shape_weight*shapes[:-1,-1]+transfer*aux[:-1,-1]
                    coef=cho_solve(cho_factor(a,check_finite=False),rhs,check_finite=False)
                    del a
                    en=base[t]+pub@coef
                    err=d[0]@en-d[1]
                    np.save(WORK/f"{key}-{t}-weights-{h}.npy",coef)
                    np.save(WORK/f"{key}-{t}-parent-{h}.npy",en)
                    if ti==1:
                        modes=bm+rad@coef
                        np.save(WORK/f"{key}-{t}-modes-{h}.npy",modes)
                    if h>=0:
                        scores=ms.metrics(d,err,(d[3]==h)&primary[t])
                        use=measures["fold"]==h
                        predicted=shape_base[use,1:,ti,col]+sx[use,1:]@coef
                        cm=np.mean(abs(signed(predicted,measures["mass"][use,1:])-frequency[use]))
                        score=.4*scores[0]+.3*(scores[1]+scores[2])+cm/400
                        if ti==1:
                            imag,_=ms.saddle_metrics(modes,(meta[:,0]==h)&saddle,meta)
                            score+=imag/500
                        dev[t].setdefault(key,[]).append(float(score))
                    else:
                        held=d[3]==-1
                        np.savez_compressed(WORK/f"{key}-{t}-equations.npz",source=d[2][held],case=d[5][held],kind=d[4][held],error_kcal=err[held]*kt.KCAL)
            print("Completed",family,"fits",width,"fold",h,flush=True)
    selected={}
    for t,items in dev.items():
        key=min(items,key=lambda k:np.mean(items[k]))
        selected[t]=key
        with np.load(WORK/f"{family}-basis.npz") as f:basis=kernel.load(f)
        basis.length=config[key]
        basis.weights=np.load(WORK/f"{key}-{t}-weights--1.npy")
        parent=ForceField.load(WORK/f"{PARENT}-{t}.npz",dispersion="d3")
        parent.additional_bases.append(basis)
        path=WORK/f"{stage_name}-{t}.npz"
        parent.save_with_base(kt.original.BASE,path)
        model=ForceField.load(path,dispersion="d3")
        z,r=trial_fit.sample()
        e,force=model.predict(z,r)
        np.testing.assert_allclose(e,np.load(WORK/f"{key}-{t}-parent--1.npy")[trial_fit.sample.index],atol=1.e-7,rtol=0)
        v=np.random.default_rng(53).normal(size=r.shape);v/=np.linalg.norm(v)
        fd=(model.energy(z,r-2.e-5*v)-model.energy(z,r+2.e-5*v))/4.e-5
        np.testing.assert_allclose((force*v).sum(),fd,atol=3.e-5)
        shutil.copyfile(WORK/f"{key}-{t}-equations.npz",WORK/f"{stage_name}-{t}-equations.npz")
        if t=="wb97xd3":
            np.save(WORK/f"{stage_name}-held-modes.npy",np.load(WORK/f"{key}-{t}-modes--1.npy")[meta[:,0]==-1])
    kt.write_json(WORK/f"{stage_name}-settings.json",dict(selected=stage_name,branches=selected,development=dev))
    compare(stage_name);validate(stage_name)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",type=int,choices=(20,21),default=20)
    with threadpool_limits(5):run(parser.parse_args().stage)
