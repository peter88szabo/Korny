"""Learn a geometry-only bond energy term from all eligible public RDB7 DFT paths.

Each structure supplies its energy difference to the same path's endpoint and,
when printed, a directional force. The same D3(BJ) contribution is subtracted
at both geometries and from the forces. The resulting bond energy represents
short-range chemistry for a subsequent high-level fit. Each observation keeps
its archive coordinates, charge, spin and original reaction split. Work is
checkpointed in independent blocks to survive an interrupted calculation.
"""
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import json
import multiprocessing as mp
import pickle
import sqlite3
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from threadpoolctl import threadpool_limits
from experiments.full_public import kernel_trials as kt,trial_metric as kernel,trial_fidelity
from experiments.full_public.data import fingerprint
from src.bond_order_kernel import BondOrderKernel
from src.dispersion.d3 import d3_energy_forces

WORK=kt.WORK
ROOT=WORK/"rdb-path-teacher"
BLOCK=384


def connect():
    global db
    db=sqlite3.connect(f"file:{WORK/'rdb-dft-frames.sqlite'}?mode=ro",uri=True)


def prepare_index():
    """Track repeated coordinates across all folds, including reserved optimization paths."""
    connect()
    gs,rows,_=kt.inputs()
    folds={}
    by_geometry=defaultdict(int)
    for t in kt.TARGETS:
        data=kt.equation_data(t)
        group={(s,c,k):int(h) for s,c,k,h in zip(data[2],data[5],data[4],data[3])}
        for target,s,_,case,terms,_,kind in rows:
            if target!=t:continue
            h=group[s,case,kind]
            bit=1<<(8 if h<0 else h)
            if s=="RDB7":folds[str(case).split("/")[-1]]=bit
            for name,_ in terms:by_geometry[name]|=bit
    signature=defaultdict(int)
    for key,mask in by_geometry.items():
        g=gs[key]
        signature[(fingerprint(g.z,g.r),*g.state)]|=mask
    meta=list(db.execute("select name,reaction,coord_id,z,r,q,m from stationary order by name"))
    final={}
    for name,rxn,idx,zb,rb,q,m in meta:
        z=np.frombuffer(zb,np.uint8); r=np.frombuffer(rb,float).reshape(-1,3)
        final[name]=(idx,z,q,m,signature[(fingerprint(z,r),q,m)]|folds[rxn])
    bits=[]; ids=[]; digests=[]
    for rid,name,buf in db.execute("select rowid,name,r from steps order by rowid"):
        idx,z,q,m,mask=final[name]
        sig=(fingerprint(z,np.frombuffer(buf,float).reshape(-1,3)),q,m)
        signature[sig]|=mask
        if not mask&256:
            bits.append(mask);ids.append(rid);digests.append(sig)
    masks=np.array([mask|signature[sig] for mask,sig in zip(bits,digests)],np.uint16)
    accepted=(masks&256)==0
    out=ROOT/'frames.npz'
    np.savez(out,rowid=np.asarray(ids)[accepted],exclude=masks[accepted])
    print("Indexed RDB7 path frames after removing validation overlap",int(accepted.sum()),flush=True)


def initialize():
    global mat,end_features,end_d3,rowids,exclusions,stationary
    connect()
    with np.load(WORK/"metric-basis.npz") as f:mat=BondOrderKernel.load(f)
    kernel.native(mat)
    with np.load(ROOT/"frames.npz") as f:rowids=f["rowid"];exclusions=f["exclude"]
    end_features=np.load(WORK/"metric-1.15-main.npy",mmap_mode="r")
    stationary={}
    for name,idx,z,r,e in db.execute("select name,coord_id,z,r,energy from stationary"):
        stationary[name]=(idx,np.frombuffer(z,np.uint8),np.frombuffer(r,float).reshape(-1,3),e)
    end_d3={}


def block(i):
    path=ROOT/f"block-{i:05}.npz"
    if path.exists():return
    idx=rowids[i*BLOCK:(i+1)*BLOCK]
    x=np.empty((len(idx),2,mat.size),np.float32)
    label=np.zeros((len(idx),4))
    # Query the inclusive span and then retain the exact vetted rowids.
    wanted=set(idx.tolist())
    rows=db.execute("select rowid,name,r,energy,force from steps where rowid between ? and ? order by rowid",(int(idx[0]),int(idx[-1])))
    j=0
    for rid,name,rb,e,fb in rows:
        if rid not in wanted:continue
        assert rid==idx[j]
        end,z,rend,eref=stationary[name]
        r=np.frombuffer(rb,float).reshape(-1,3)
        d3,df=d3_energy_forces(z,r)
        if name not in end_d3:end_d3[name]=d3_energy_forces(z,rend)[0]
        x[j,0]=kernel.values(z,r)[0]-end_features[end]
        label[j,0]=e-eref-(d3-end_d3[name])
        if fb is not None:
            force=np.frombuffer(fb,float).reshape(r.shape)
            v=force/max(np.linalg.norm(force),1.e-12)
            if np.linalg.norm(v)<.99:
                v=np.random.default_rng(rid).normal(size=r.shape);v/=np.linalg.norm(v)
            x[j,1]=(kernel.values(z,r-.001*v)[0]-kernel.values(z,r+.001*v)[0])/.002
            label[j,1]=np.sum((force-df)*v)
            label[j,2]=1
        else:x[j,1]=0
        label[j,3]=end
        j+=1
    assert j==len(idx) and np.isfinite(label).all() and np.isfinite(x).all()
    tmp=path.with_suffix(".pending")
    with tmp.open("wb") as handle:np.savez(handle,x=x,label=label,exclude=exclusions[i*BLOCK:i*BLOCK+j])
    tmp.replace(path)


def features():
    ROOT.mkdir(exist_ok=True)
    if not (ROOT/"frames.npz").exists():prepare_index()
    with np.load(ROOT/"frames.npz") as f:n=len(f["rowid"])
    tasks=[i for i in range((n+BLOCK-1)//BLOCK) if not (ROOT/f"block-{i:05}.npz").exists()]
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context("fork"),initializer=initialize) as pool:
        for i,_ in enumerate(pool.map(block,tasks,buffersize=16)):
            if (i+1)%100==0:print("Computed archived DFT features for",min(n,(i+1)*BLOCK),"frames in this run",flush=True)
    print("All archived RDB7 path columns complete",n,flush=True)


def train():
    """Fit each teacher without observations from its corresponding development fold."""
    from scipy.linalg.blas import dsyrk
    n=len(np.load(WORK/"prior.npy",mmap_mode="r"))
    mat=np.load(WORK/"metric-1.15-main.npy",mmap_mode="r")[:n]
    shape=(mat.shape[1]+1,)*2
    # Four mutually exclusive fold masks yield the three exact training Grams.
    sums=[np.zeros(shape,order="F") for _ in range(4)]
    counts=np.zeros((4,2),int)
    for file in sorted(ROOT.glob("block-*.npz")):
        with np.load(file) as d:
            x=d["x"];y=d["label"];exclude=d["exclude"]&3
        for bits in np.unique(exclude):
            use=exclude==bits
            counts[bits]+=[int(use.sum()),int(y[use,2].sum())]
            # Large uphill steps can dominate a fit to reaction-range energies.
            # Limit their leverage continuously while keeping each observation.
            e=np.column_stack((x[use,0],y[use,0]))
            e*=np.minimum(1.,np.sqrt(3./np.maximum(abs(y[use,0]),1.e-9)))[:,None]
            f=np.column_stack((x[use,1],y[use,1]))
            f*= (.35*y[use,2]*np.minimum(1.,np.sqrt(10./np.maximum(abs(y[use,1]),1.e-9))))[:,None]
            xw=np.asfortranarray(np.vstack([e,f]).T)
            sums[bits]=dsyrk(1.,xw,beta=1.,c=sums[bits],lower=1,overwrite_c=1)
    with (WORK/"ids.pkl").open("rb") as f:_,names=pickle.load(f)
    gs,_,_=kt.inputs()
    d3=np.array([d3_energy_forces(gs[k].z,gs[k].r)[0] for k in names])
    targets=(trial_fidelity.rdb_dft(kt.equation_data("high")),kt.equation_data("wb97xd3"))
    for h in (0,1,-1):
        select=[b for b in range(4) if h<0 or not b&(1<<h)]
        gram=sum(sums[b] for b in select)
        lower=np.tril_indices(len(gram),-1);gram[lower[::-1]]=gram[lower]
        # Fit absolute reaction, forward and reverse differences at wB97X-D3.
        # The two electronic reference sets contribute only at their common level.
        for d in targets:
            use=(d[3]>=0)&(d[3]!=h)&np.isin(d[2],["Grambow","RDB7"])
            x=d[0][use]@mat
            y=d[1][use]-d[0][use]@d3
            for i in range(0,len(y),384):
                s=slice(i,i+384)
                rows=np.column_stack((x[s],y[s]))
                gram+=3*(rows.T@rows)
        a=gram[:-1,:-1].copy()
        a.flat[::len(a)+1]+=1.
        coef=cho_solve(cho_factor(a),gram[:-1,-1])
        np.save(ROOT/f"teacher-{h}.npy",coef)
        print("Fitted RDB7 trajectory teacher",h,counts[select].sum(axis=0).tolist(),flush=True)
    kt.write_json(ROOT/"observations.json",dict(energy_rows=int(counts[:,0].sum()),force_rows=int(counts[:,1].sum())))


if __name__=="__main__":
    with threadpool_limits(1):features()
    with threadpool_limits(5):train()
