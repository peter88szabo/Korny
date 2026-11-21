"""Select training environments and prepare a periodic-group polynomial energy kernel."""
import ctypes as ct
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import pickle
import subprocess
import numpy as np
from scipy.linalg import block_diag
from threadpoolctl import threadpool_limits

from experiments.full_public import kernel_trials as kt, trial_sites, trial_aux
from experiments.full_public.trial_messages import prototypes
from experiments.reactivity.reaction_classes import lookup
from src.forcefield import ForceField
from src.local import descriptor_layout
from src.periodic_kernel import PeriodicKernel

WORK=kt.WORK
SHELLS=np.array([[.90,1.20,1.65,2.40],[.16,.22,.38,.59]])
CUTOFF=(2.8,3.8)
OPTIONS=((.85,3),(1.35,6))


def setup():
    global lib,layout,dim,groups,radii
    backbone=ForceField.load(WORK/"p12-public-high.npz")
    groups=backbone.groups.copy()
    groups[1]=0
    radii=backbone.radii
    widths,items=descriptor_layout(9,4,2)
    dim=sum(widths); channels=36
    layout=np.full((2,channels,channels),-1,np.int32)
    for k,(pos,a,b) in enumerate(items):layout[k,a,b]=pos
    src=Path(__file__).with_name("trial_descriptors.cpp")
    native=WORK/"periodic-sites.so"
    if not native.exists() or native.stat().st_mtime<src.stat().st_mtime:
        subprocess.run(["c++","-O3","-std=c++17","-shared","-fPIC",str(src),"-o",str(native)],check=True)
    lib=ct.CDLL(str(native)).periodic_sites
    lib.argtypes=[ct.c_int]*5+[ct.c_double]*2+[ct.c_void_p]*7
    lib.restype=None


def raw(z,r):
    t=np.ascontiguousarray(groups[z],np.int32)
    radius=np.ascontiguousarray(radii[z],float)
    xyz=np.ascontiguousarray(r,float)
    out=np.zeros((len(z),dim))
    lib(len(z),36,4,2,dim,*CUTOFF,*(a.ctypes.data for a in (t,xyz,layout,out,radius,SHELLS[0],SHELLS[1])))
    return out


def learn(gs,rows,classes):
    classes.update(lookup())
    keys=trial_sites.candidates(gs,rows,classes)
    sites={};rng=np.random.default_rng(38143)
    for k in keys:
        g=gs[k]; table=raw(g.z,g.r);types=groups[g.z]
        for group in np.unique(types):sites.setdefault(int(group),[]).append(table[types==group])
    centers={};proj={}
    for g in sorted(sites):
        x=np.vstack(sites.pop(g))
        if len(x)>6500:x=x[rng.choice(len(x),6500,replace=False)]
        blocks=[]
        for v,limit,floor in ((x[:,:36],24,.025),(x[:,36:],48,.04)):
            occupied=np.flatnonzero(np.max(abs(v),axis=0)>1.e-6)
            if len(x)>1 and len(occupied):
                c=np.cov(v[:,occupied],rowvar=False)
                val,axes=np.linalg.eigh(np.atleast_2d(c))
                use=np.flatnonzero(val>1.e-8)[-limit:]
                projection=np.zeros((v.shape[1],len(use)))
                projection[occupied]=axes[:,use]/np.sqrt((val[use]+floor)*max(1,len(use)))
            else:
                projection=np.empty((v.shape[1],0))
            blocks.append(projection)
        p=block_diag(*blocks)
        if p.shape[1]==0:continue
        u=x@p
        # A square-root frequency allocation gives uncommon groups a basis of
        # their own while retaining more centers for common organic sites.
        count=min(850,max(24,int(12*np.sqrt(len(u)))),len(u))
        c=prototypes(u,count,rng)
        if len(c):
            centers[g]=c;proj[g]=p
    b=PeriodicKernel(groups,radii,proj,centers,SHELLS,CUTOFF,.85,3,np.zeros(sum(map(len,centers.values()))))
    np.savez_compressed(WORK/"periodic-basis.npz",**b.fields())
    print("Training-only polynomial kernel centers per group",{k:len(v) for k,v in centers.items()},flush=True)
    return b


def batch(pts):
    types=np.concatenate([groups[z] for z,r in pts])
    owner=np.repeat(np.arange(len(pts)),[len(z) for z,r in pts])
    raw_sites=np.vstack([raw(z,r) for z,r in pts])
    out=[np.zeros((len(pts),basis.size)) for _ in OPTIONS]
    offset=0
    for key,c in basis.centers.items():
        take=types==key
        if take.any():
            p=basis.projections[key]
            # Nonzero columns depend on the neighbor elements actually present.
            data=raw_sites[take]
            active=np.flatnonzero(np.any(data!=0,axis=0))
            u=data[:,active]@p[active]
            who,ix=np.unique(owner[take],return_index=True)
            for dest,(ell,degree) in zip(out,OPTIONS):
                basis.length,basis.degree=ell,degree
                val=basis.values(u,c)[0]
                dest[who,offset:offset+len(c)]=np.add.reduceat(val,ix,axis=0)
        offset+=len(c)
    return out


def chunk(i):
    sl=slice(i,min(i+36,len(all_points)))
    for mat,v in zip(maps,batch(all_points[sl])):mat[sl]=v


def store(side,points):
    global maps,all_points
    all_points=points
    paths=[WORK/f"periodic-{ell:g}-{side}.npy" for ell,_ in OPTIONS]
    maps=[np.lib.format.open_memmap(p.with_suffix(".pending"),mode="w+",dtype=float,shape=(len(points),basis.size)) for p in paths]
    with ProcessPoolExecutor(max_workers=5,mp_context=mp.get_context("fork")) as pool:
        for _ in pool.map(chunk,range(0,len(all_points),36),buffersize=12):pass
    for mat,p in zip(maps,paths):mat.flush();p.with_suffix(".pending").replace(p)
    print("Computed polynomial kernels",side,len(points),flush=True)


def prepare():
    global basis
    setup()
    gs,rows,classes=kt.inputs()
    basis=learn(gs,rows,classes)
    with (WORK/"ids.pkl").open("rb") as f:_,names=pickle.load(f)
    rng=np.random.default_rng(518)
    samples=list(rng.choice(names,8,replace=False))
    samples+=next(([n] for n in names if 14 in gs[n].z),[])
    for name in samples:
        z,r=gs[name].z,gs[name].r
        v=rng.normal(size=r.shape);v/=np.linalg.norm(v)
        calc=batch([(z,r),(z,r-1.e-5*v),(z,r+1.e-5*v)])
        for arr,(ell,degree) in zip(calc,OPTIONS):
            basis.length,basis.degree=ell,degree
            x,j=basis.design(z,r,True)
            np.testing.assert_allclose(x,arr[0],atol=1.e-8)
            np.testing.assert_allclose(j.T@v.ravel(),(arr[1]-arr[2])/2.e-5,atol=2.e-6)
    print("Verified polynomial features and conservative forces including silicon",flush=True)
    pts=[(gs[n].z,gs[n].r) for n in names]
    for _,z,r,v in trial_sites.hdf():pts.extend([(z,r),(z,r+.001*v),(z,r-.001*v)])
    store("main",pts)
    aux,nprivate,_=trial_aux.read_frames()
    store("rad",aux[nprivate:])


if __name__=="__main__":
    with threadpool_limits(1):prepare()
