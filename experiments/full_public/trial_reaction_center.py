"""Use measured unstable-mode displacements to train continuous local bond gates.

The training label marks whether a pair stretches appreciably in a saddle's
unstable direction. Pairs in endpoints are negatives. Inference sees the local
geometry only. The gate multiplies an energy kernel and is differentiated when
evaluating forces. Both inner holdout folds are excluded from gate training.
"""
import hashlib
import pickle
import sqlite3
import numpy as np
from rdkit import Chem
from scipy.linalg import cho_factor,cho_solve
from scipy.special import expit
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from experiments.full_public import kernel_trials as kt,trial_metric as kernel,trial_sites,trial_aux
from experiments.reactivity.stationary_fit import internal_transform
from src.bond_order_kernel import BondOrderKernel,METRIC,environments
from src.bond_environment import ELEMENTS
from src.four_body import RADII

WORK=kt.WORK


def excerpts(db):
    """Keep entire held-out formulas and reused stationary geometries out of the pilot."""
    d=kt.equation_data("high")
    m=(d[2]=="RDB7")&(d[4]=="forward")
    cases={str(n).split("/")[-1]:h for n,h in zip(d[5][m],d[3][m])}
    records=list(db.execute("select name,reaction,coord_id from stationary where hessian is not null"))
    forbidden={i for n,c,i in records if cases[c]<2}
    seen=set()
    bins={True:[],False:[]}
    for n,c,i in records:
        if i in forbidden or i in seen:continue
        bins["/ts" in n].append(n)
        seen.add(i)
    for is_ts,items in bins.items():
        for n in sorted(items,key=lambda n:hashlib.sha256(n.encode()).digest())[:3200]:
            z,r,h=db.execute("select z,r,hessian from stationary where name=?",(n,)).fetchone()
            z=np.frombuffer(z,np.uint8);r=np.frombuffer(r,float).reshape(-1,3)
            if len(z)<2:continue
            vec=None
            if is_ts:
                h=np.frombuffer(h,float).reshape(r.size,r.size)
                mass=np.array([Chem.GetPeriodicTable().GetMostCommonIsotopeMass(int(s)) for s in z])
                t=internal_transform(mass,r)
                if not t.shape[1]:continue
                w,v=np.linalg.eigh(t.T@h@t)
                if w[0]>=0:continue
                vec=(t@v[:,0]).reshape(r.shape)
            yield z,r,vec
    with (WORK/"radical-modes.pkl").open("rb") as f:
        for fold,(name,z,r,state,stationary,modes) in pickle.load(f):
            if fold<2:continue
            is_ts=name.split("/")[-1].startswith("ts")
            yield z,r,modes[0][0] if is_ts and modes[0][1]<0 else None


def pilot():
    with np.load(WORK/"metric-basis.npz") as f:b=BondOrderKernel.load(f)
    xs=[[] for _ in range(10)];ys=[[] for _ in range(10)]
    counts=[0,0]
    with sqlite3.connect(f"file:{WORK/'rdb-dft-frames.sqlite'}?mode=ro",uri=True) as db:
        for z,r,v in excerpts(db):
            counts[int(v is not None)]+=1
            atom=np.flatnonzero(np.isin(z,ELEMENTS))
            a,c=np.triu_indices(len(atom),1);a,c=atom[a],atom[c]
            dr=r[c]-r[a]
            dist=np.linalg.norm(dr,axis=1)
            sc=np.array([RADII[int(s)]+RADII[int(t)] for s,t in zip(z[a],z[c])])
            mask=dist<1.95*sc
            if v is None:target=np.zeros(mask.sum())
            else:
                slope=abs(np.sum((v[c]-v[a])*dr,axis=1)[mask]/(dist[mask]*sc[mask]))
                target=np.asarray(slope>.45*slope.max(initial=1.e-12),float)
            for k,x,g,_,__,idx,___ in environments(z,r):
                take=g>.035
                if take.any():
                    xs[k].append(x[take]);ys[k].append(target[idx[take]])
    coeff=np.zeros(b.size);bias=np.full(10,-8.)
    rng=np.random.default_rng(515)
    for k in range(10):
        if not xs[k]: continue
        x=np.vstack(xs[k]); y=np.concatenate(ys[k])
        p=np.flatnonzero(y);n=np.flatnonzero(y==0)
        if not len(p) or not len(n):continue
        # Favor false-positive control without discarding the rare participating
        # bonds. The intercept matches the original empirical class prevalence.
        ip=rng.choice(p,min(len(p),5000),replace=False)
        im=rng.choice(n,min(len(n),25000),replace=False)
        ix=np.r_[ip,im]
        wt=np.r_[np.full(len(ip),len(p)/len(ip)),np.full(len(im),len(n)/len(im))]
        wt/=wt.mean()
        y=y[ix]
        sl=slice(*b.offsets[k:k+2])
        proj=b.transforms[k]
        sq=cdist((x[ix]*METRIC)@proj,(b.centers[sl]*METRIC)@proj,"sqeuclidean")
        design=np.column_stack((np.ones(len(ix)),np.exp((-.5/1.15**2)*sq)))
        beta=np.zeros(design.shape[1]);beta[0]=np.log(len(p)/len(n))
        for _ in range(16):
            prob=expit(design@beta)
            grad=design.T@(wt*(prob-y))
            grad[1:]+=.3*beta[1:]
            hess=design.T@((wt*prob*(1-prob))[:,None]*design)
            hess.flat[::len(hess)+1]+=.3
            step=cho_solve(cho_factor(hess),grad)
            # Bound the largest change of logit on an individual observation.
            beta-=step/max(1.,np.max(abs(design@step))/4.)
            if np.max(abs(step))<2.e-5:break
        bias[k]=beta[0];coeff[sl]=beta[1:]
    b.gate_coefficients=coeff;b.gate_bias=bias
    np.savez_compressed(WORK/"gated-basis.npz",**b.fields())
    print("Fitted bond participation gates using endpoints and saddles",counts,flush=True)
    return b


def prepare():
    basis=pilot()
    kernel.native(basis)
    gs,_,_=kt.inputs()
    with (WORK/"ids.pkl").open("rb") as f:_,keys=pickle.load(f)
    rng=np.random.default_rng(133)
    for i in rng.choice(len(keys),12,replace=False):
        g=gs[keys[i]]
        v=rng.normal(size=g.r.shape);v/=np.linalg.norm(v)
        for j,w in enumerate((1.15,1.15/np.sqrt(2))):
            basis.length=w
            raw,jac=basis.design(g.z,g.r,True)
            np.testing.assert_allclose(raw,kernel.values(g.z,g.r)[j],atol=1.e-9)
            fd=(kernel.values(g.z,g.r-1.e-5*v)[j]-kernel.values(g.z,g.r+1.e-5*v)[j])/2.e-5
            np.testing.assert_allclose(jac.T@v.ravel(),fd,atol=1.e-6)
    print("Verified native gated kernel and reaction-center forces",flush=True)
    data=[(gs[k].z,gs[k].r) for k in keys]
    for _,z,r,v in trial_sites.hdf(): data.extend([(z,r),(z,r+.001*v),(z,r-.001*v)])
    kernel.store("main",data,"gated")
    aux,nprivate,_=trial_aux.read_frames()
    kernel.store("rad",aux[nprivate:],"gated")


if __name__=="__main__":
    with threadpool_limits(2):prepare()
