"""Element-resolved angular densities with separate channels for forming and breaking bonds.

Radial distances use the sum of the two covalent radii. Five overlapping Gaussians
resolve local bond lengths and the directions of neighboring atoms enter through
the first two Legendre terms. Train-only atomic exemplars define the Matérn kernel.
"""
import ctypes as ct
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pickle
from pathlib import Path
import subprocess
import numpy as np
from scipy.linalg import block_diag
from scipy.cluster.vq import vq
from threadpoolctl import threadpool_limits
from experiments.full_public import kernel_trials as kt,trial_sites,trial_aux
from experiments.reactivity.reaction_classes import lookup
from src.local import descriptor_layout
from src.reactive_basis import ReactiveBasis

WORK=kt.WORK
ELEMENTS=(1,6,7,8)
RADII=np.array([.31,.76,.71,.66])
SHELLS=np.array([[.86,1.,1.22,1.60,2.15],[.09,.10,.15,.24,.36]])
SPEC=(5,2,1.9,2.7)
LENGTHS=(.7,1.35)


def setup():
    global lib,layout,raw_size,types
    n,l=SPEC[:2]
    widths,groups=descriptor_layout(len(ELEMENTS),n,l)
    raw_size=sum(widths)
    channels=n*len(ELEMENTS)
    layout=np.full((l,channels,channels),-1,np.int32)
    for i,(p,a,b) in enumerate(groups):
        layout[i,a,b]=p
    types=np.full(128,-1,np.int32)
    types[list(ELEMENTS)]=np.arange(len(ELEMENTS))
    so=WORK/"density-shells.so"
    src=Path(__file__).with_name("trial_descriptors.cpp")
    if not so.exists() or so.stat().st_mtime<src.stat().st_mtime:
        subprocess.run(["c++","-O3","-std=c++17","-shared","-fPIC",str(src),"-o",str(so)],check=True)
    lib=ct.CDLL(str(so)).density_sites
    lib.argtypes=[ct.c_int]*5+[ct.c_double]*2+[ct.c_void_p]*7
    lib.restype=None


def raw(z,r):
    t=np.ascontiguousarray(types[z])
    r=np.ascontiguousarray(r,float)
    out=np.zeros((len(z),raw_size))
    lib(len(z),len(ELEMENTS)*SPEC[0],*SPEC[:2],raw_size,*SPEC[2:],
        *(a.ctypes.data for a in (t,r,layout,out,RADII,SHELLS[0],SHELLS[1])))
    return out


def learn(geoms,rows,classes):
    classes.update(lookup())
    keys=trial_sites.candidates(geoms,rows,classes)
    raw_sites={s:[] for s in ELEMENTS}
    for name in keys:
        g=geoms[name]
        data=raw(g.z,g.r)
        for s in ELEMENTS:
            if (g.z==s).any():
                raw_sites[s].append(data[g.z==s])
    projections,rad,centers={},{},{}
    rng=np.random.default_rng(93741)
    for s in ELEMENTS:
        x=np.vstack(raw_sites.pop(s))
        if len(x)>16000:
            x=x[rng.choice(len(x),16000,replace=False)]
        blocks=[]
        for data,limit,noise in ((x[:,:20],20,.008),(x[:,20:],48,.006)):
            val,v=np.linalg.eigh(np.cov(data,rowvar=False))
            ix=np.flatnonzero(val>1.e-7)[-limit:]
            blocks.append(v[:,ix]/np.sqrt((val[ix]+noise)*len(ix)))
        projections[s]=block_diag(*blocks)
        rad[s]=blocks[0].shape[1]
        u=x@projections[s]
        dist=np.square(u-u.mean(axis=0)).sum(axis=1)
        chosen=[]
        for _ in range(768):
            if dist.sum()<1.e-8: break
            j=rng.choice(len(u),p=dist/dist.sum())
            chosen.append(u[j])
            dist=np.minimum(dist,np.sum((u-u[j])**2,axis=1))
        c=np.array(chosen)
        for _ in range(3):
            cluster=vq(u,c)[0]
            count=np.bincount(cluster,minlength=len(c))
            new=np.zeros_like(c)
            np.add.at(new,cluster,u)
            ix=count>0
            c[ix]=new[ix]/count[ix,None]
        centers[s]=c
    model=ReactiveBasis(ELEMENTS,SPEC,projections,centers,rad,1.,1.,
                        np.zeros(sum(map(len,centers.values()))),gaussian_shells=SHELLS)
    np.savez_compressed(WORK/"density-basis.npz",**model.fields())
    return model


def batch(data):
    """Use dense site-center products in small batches, then sum per molecule."""
    zs=np.concatenate([z for z,r in data])
    owners=np.repeat(np.arange(len(data)),[len(z) for z,r in data])
    arr=np.vstack([raw(z,r) for z,r in data])
    out=[np.zeros((len(data),basis.size)) for w in LENGTHS]
    offset=0
    for s in ELEMENTS:
        mask=zs==s
        c=basis.centers[s]
        if mask.any():
            x=arr[mask]@basis.projections[s]
            c2=np.sum(c*c,axis=1)
            d=np.sqrt(np.maximum(np.sum(x*x,axis=1)[:,None]+c2-2*(x@c.T),0))
            empty=np.sqrt(c2)
            for i,width in enumerate(LENGTHS):
                v=np.zeros_like(d)
                for lam in (.5*width,2*width):
                    t=np.sqrt(5)/lam*d
                    a=np.sqrt(5)/lam*empty
                    v+=.5*((1+t+t*t/3)*np.exp(-t)-(1+a+a*a/3)*np.exp(-a))
                ids,start=np.unique(owners[mask],return_index=True)
                out[i][ids,offset:offset+len(c)]=np.add.reduceat(v,start,axis=0)
        offset+=len(c)
    return out


def block(lo):
    hi=min(lo+64,len(frames))
    for arr,v in zip(dest,batch(frames[lo:hi])):
        arr[lo:hi]=v


def store(side,data):
    global frames,dest
    frames=data
    paths=[WORK/f"density-{w:g}-{side}.npy" for w in LENGTHS]
    dest=[np.lib.format.open_memmap(p.with_suffix(".pending"),mode="w+",dtype=float,shape=(len(data),basis.size)) for p in paths]
    with ProcessPoolExecutor(max_workers=5,mp_context=mp.get_context("fork")) as pool:
        for _ in pool.map(block,range(0,len(data),64),buffersize=12):pass
    for a,p in zip(dest,paths):
        a.flush()
        p.with_suffix(".pending").replace(p)
    print("Finished shell kernel geometry batch",side,len(data),flush=True)


def prepare():
    global basis
    setup()
    gs,rows,classes=kt.inputs()
    basis=learn(gs,rows,classes)
    with (WORK/"ids.pkl").open("rb") as f:
        _,keys=pickle.load(f)
    rng=np.random.default_rng(82)
    for i in rng.choice(len(keys),6,replace=False):
        g=gs[keys[i]]
        v=rng.normal(size=g.r.shape)
        v/=np.linalg.norm(v)
        values=batch([(g.z,g.r),(g.z,g.r-1.e-5*v),(g.z,g.r+1.e-5*v)])
        for w,arr in zip(LENGTHS,values):
            basis.length=w
            x,j=basis.design(g.z,g.r,True)
            np.testing.assert_allclose(x,arr[0],atol=1.e-9)
            np.testing.assert_allclose(j.T@v.ravel(),(arr[1]-arr[2])/2.e-5,atol=1.e-6)
    print("Verified radial and angular density energies and forces",basis.size,flush=True)
    data=[(gs[k].z,gs[k].r) for k in keys]
    for _,z,r,v in trial_sites.hdf():
        data.extend([(z,r),(z,r+.001*v),(z,r-.001*v)])
    store("main",data)
    aux,nprivate,_=trial_aux.read_frames()
    store("rad",aux[nprivate:])


if __name__=="__main__":
    with threadpool_limits(1):
        prepare()
