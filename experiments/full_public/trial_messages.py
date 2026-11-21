"""Construct sparse geometry kernels on local chemical environments through three bonds."""
import ctypes as ct
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pickle
import subprocess
from pathlib import Path
import numpy as np
from scipy.cluster.vq import vq
from scipy.linalg import block_diag
from threadpoolctl import threadpool_limits
from experiments.full_public import kernel_trials as kt,trial_sites,trial_aux
from experiments.reactivity.reaction_classes import lookup
from src.message_kernel import ELEMENTS,MessageKernel,graph

WORK=kt.WORK
LENGTHS=(.7,1.35)
FAMILY="message"


def prototypes(x,n,rng):
    """Select centers by distance-weighted sampling and refine them with Lloyd updates."""
    dist=np.sum((x-x.mean(axis=0))**2,axis=1)
    means=[]
    for _ in range(n):
        if dist.sum()<1.e-9:break
        j=rng.choice(len(x),p=dist/dist.sum())
        means.append(x[j])
        dist=np.minimum(dist,np.sum((x-x[j])**2,axis=1))
    c=np.array(means)
    for _ in range(4):
        ids=vq(x,c)[0]
        count=np.bincount(ids,minlength=len(c))
        accum=np.zeros_like(c)
        np.add.at(accum,ids,x)
        ok=count>0
        c[ok]=accum[ok]/count[ok,None]
    return c


def setup():
    global lib,types
    src=Path(__file__).with_suffix(".cpp")
    so=WORK/f"{FAMILY}-sites.so"
    if not so.exists() or so.stat().st_mtime<src.stat().st_mtime:
        subprocess.run(["c++","-O3","-std=c++17","-shared","-fPIC",str(src),"-o",str(so)],check=True)
    lib=ct.CDLL(str(so)).message_sites
    lib.argtypes=[ct.c_int]*3+[ct.c_void_p]*4
    lib.restype=None
    types=np.full(128,-1,np.int32)
    types[list(ELEMENTS)]=np.arange(4)


def raw(z,r,seeds):
    t=np.ascontiguousarray(types[z])
    rr=np.ascontiguousarray(r,float)
    m=seeds.shape[1]
    out=np.zeros((len(z),24+8*m+5*(FAMILY=="topology")))
    lib(len(z),m,int(FAMILY=="topology"),t.ctypes.data,rr.ctypes.data,seeds.ctypes.data,out.ctypes.data)
    return out


def learn(gs,rows,classes):
    classes.update(lookup())
    keys=trial_sites.candidates(gs,rows,classes)
    sites={s:[] for s in ELEMENTS}
    rng=np.random.default_rng(6117)
    for k in keys:
        g=gs[k]
        typ,x,*_=graph(g.z,g.r,topology=(FAMILY=="topology"))
        for i,s in enumerate(ELEMENTS):
            if (typ==i).any():sites[s].append(x[typ==i])
    def sample(items):
        x=np.vstack(items)
        return x if len(x)<=14000 else x[rng.choice(len(x),14000,replace=False)]
    seeds=np.ascontiguousarray([prototypes(sample(sites.pop(s)),24,rng) for s in ELEMENTS])
    sites={s:[] for s in ELEMENTS}
    for k in keys:
        g=gs[k]
        x=raw(g.z,g.r,seeds)
        for s in ELEMENTS:
            if (g.z==s).any():sites[s].append(x[g.z==s])
    proj,centers={},{}
    for s in ELEMENTS:
        x=sample(sites.pop(s))
        blocks=[]
        sections=[(x[:,:24],24,.009),(x[:,24:120],32,.01),(x[:,120:216],32,.018)]
        if FAMILY=="topology": sections.append((x[:,216:],5,.004))
        for arr,n,noise in sections:
            val,vec=np.linalg.eigh(np.cov(arr,rowvar=False))
            use=np.flatnonzero(val>1.e-7)[-n:]
            blocks.append(vec[:,use]/np.sqrt((val[use]+noise)*len(use)))
        proj[s]=block_diag(*blocks)
        centers[s]=prototypes(x@proj[s],1024,rng)
    model=MessageKernel(seeds,proj,centers,1.,np.zeros(sum(map(len,centers.values()))),topology=(FAMILY=="topology"))
    np.savez_compressed(WORK/f"{FAMILY}-basis.npz",**model.fields())
    return model


def batch(data):
    zs=np.concatenate([z for z,r in data])
    owner=np.repeat(np.arange(len(data)),[len(z) for z,r in data])
    x=np.vstack([raw(z,r,basis.seeds) for z,r in data])
    out=[np.zeros((len(data),basis.size)) for _ in LENGTHS]
    offset=0
    for s in ELEMENTS:
        c=basis.centers[s]
        mask=zs==s
        if mask.any():
            u=x[mask]@basis.projections[s]
            empty=np.sum(c*c,axis=1)
            d2=np.maximum(np.sum(u*u,axis=1)[:,None]+empty-2*u@c.T,0)
            groups,idx=np.unique(owner[mask],return_index=True)
            for ans,w in zip(out,LENGTHS):
                y=np.exp((-.5/w**2)*d2)-np.exp((-.5/w**2)*empty)
                ans[groups,offset:offset+len(c)]=np.add.reduceat(y,idx,axis=0)
        offset+=len(c)
    return out


def process(lo):
    hi=min(lo+48,len(frames))
    for arr,out in zip(dest,batch(frames[lo:hi])):arr[lo:hi]=out


def store(side,data):
    global frames,dest
    frames=data
    paths=[WORK/f"{FAMILY}-{w:g}-{side}.npy" for w in LENGTHS]
    dest=[np.lib.format.open_memmap(p.with_suffix(".pending"),mode="w+",dtype=float,shape=(len(data),basis.size)) for p in paths]
    with ProcessPoolExecutor(max_workers=5,mp_context=mp.get_context("fork")) as pool:
        for _ in pool.map(process,range(0,len(data),48),buffersize=12):pass
    for arr,p in zip(dest,paths):
        arr.flush()
        p.with_suffix(".pending").replace(p)
    print("Finished graph environment batch",side,len(data),flush=True)


def run():
    global basis
    setup()
    gs,rows,classes=kt.inputs()
    basis=learn(gs,rows,classes)
    rng=np.random.default_rng(252)
    with (WORK/"ids.pkl").open("rb") as f:_,keys=pickle.load(f)
    for i in rng.choice(len(keys),12,replace=False):
        g=gs[keys[i]]
        v=rng.normal(size=g.r.shape)
        v/=np.linalg.norm(v)
        vals=batch([(g.z,g.r),(g.z,g.r-1.e-5*v),(g.z,g.r+1.e-5*v)])
        for w,arr in zip(LENGTHS,vals):
            basis.length=w
            x,j=basis.design(g.z,g.r,True)
            np.testing.assert_allclose(x,arr[0],atol=1.e-9)
            np.testing.assert_allclose(v.ravel()@j,(arr[1]-arr[2])/2.e-5,atol=8.e-7)
    print("Verified graph readout and analytic Cartesian forces",basis.size,flush=True)
    data=[(gs[k].z,gs[k].r) for k in keys]
    for _,z,r,v in trial_sites.hdf():data.extend([(z,r),(z,r+.001*v),(z,r-.001*v)])
    store("main",data)
    aux,nprivate,_=trial_aux.read_frames()
    store("rad",aux[nprivate:])


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology",action="store_true")
    FAMILY="topology" if parser.parse_args().topology else "message"
    with threadpool_limits(1):run()
