"""Learn a geometry metric from training energy differences, separately for each bond type.

An inexpensive pilot Gaussian fit estimates how sensitive each electronic surface
is to smooth coordination and bond-order inputs. The average outer product of
its descriptor gradients supplies the anisotropy of a larger set of sparse
kernels. All pilot labels and center candidates exclude both inner holdouts.
"""
import ctypes as ct
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pickle
import subprocess
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.cluster.vq import vq
from threadpoolctl import threadpool_limits

from experiments.full_public import kernel_trials as kt, trial_bond_order as bo, trial_sites, trial_aux
from src.bond_order_kernel import METRIC, environments, BondOrderKernel
from experiments.reactivity.reaction_classes import lookup

WORK=kt.WORK


def pilots():
    original=np.load(WORK/"bo-1.15-main.npy",mmap_mode="r")
    n=len(np.load(WORK/"prior.npy",mmap_mode="r"))
    weights=[]
    for t in kt.TARGETS:
        d=kt.equation_data(t)
        x=d[0]@original[:n]
        # Only these known reaction differences enter the pilot. Subtracting
        # a high-level parent here could indirectly introduce held labels.
        train=(d[3]>=2)&np.isin(d[2],kt.BARRIERS)
        gram=kt.grouped_moments(d,x,d[1],train)
        a=gram["gap"]+gram["mid"]
        mat=a[:-1,:-1].copy()
        mat.flat[::len(mat)+1]+=.25
        weights.append(cho_solve(cho_factor(mat),a[:-1,-1]))
    return weights


def choose(gs,rows,classes):
    with np.load(WORK/"bo-basis.npz") as f:
        old=BondOrderKernel.load(f)
    readouts=pilots()
    classes.update(lookup())
    pools=[[] for _ in range(10)]
    for key in trial_sites.candidates(gs,rows,classes):
        g=gs[key]
        for k,x,gate,*_ in environments(g.z,g.r):
            if (gate>.03).any():
                pools[k].append(x[gate>.03])
    transforms=[]
    centers=[]
    offsets=[0]
    rng=np.random.default_rng(91382)
    for k,pool in enumerate(pools):
        if not pool:
            transforms.append(np.eye(len(METRIC)))
            offsets.append(len(centers))
            continue
        data=np.vstack(pool)
        if len(data)>12000:
            data=data[rng.choice(len(data),12000,replace=False)]
        x=data*METRIC
        sl=slice(*old.offsets[k:k+2])
        c=old.centers[sl]*METRIC
        cov=np.zeros((len(METRIC),len(METRIC)))
        for coeff in readouts:
            part=np.zeros_like(cov)
            for i in range(0,len(x),192):
                dx=x[i:i+192,None]-c
                sim=np.exp(-.5*np.einsum("ncd,ncd->nc",dx,dx)/old.length**2)
                jac=-np.einsum("nc,ncd->nd",sim*coeff[sl],dx)/old.length**2
                part+=jac.T@jac
            cov+=part/max(np.trace(part),1.e-10)
        eig,vec=np.linalg.eigh(cov)
        # A nonzero isotropic floor protects directions absent from the pilots.
        scale=np.maximum(eig,0)**.35
        scale=np.sqrt(.25+.75*len(METRIC)*scale/max(scale.sum(),1.e-9))
        transform=(vec*scale)@vec.T
        transforms.append(transform)
        projected=x@transform
        # Sample by squared distance and then use a few Lloyd sweeps. Frequent
        # equilibrium geometries and isolated unusual reacting sites both keep centers.
        dist=np.sum((projected-projected.mean(axis=0))**2,axis=1)
        means=[]
        for _ in range(384):
            if dist.sum()<1.e-6:
                break
            j=rng.choice(len(dist),p=dist/dist.sum())
            means.append(projected[j])
            dist=np.minimum(dist,np.sum((projected-projected[j])**2,axis=1))
        means=np.array(means)
        if len(means):
            for _ in range(4):
                group=vq(projected,means)[0]
                count=np.bincount(group,minlength=len(means))
                new=np.zeros_like(means)
                np.add.at(new,group,projected)
                active=count>0
                means[active]=new[active]/count[active,None]
            centers.extend(np.linalg.solve(transform.T,means.T).T/METRIC)
        offsets.append(len(centers))
    basis=BondOrderKernel(centers,offsets,1.15,transforms=np.array(transforms))
    np.savez_compressed(WORK/"metric-basis.npz",**basis.fields())
    return basis


def native(basis):
    global centers,offsets,metric,lib,gate_args
    metric=np.ascontiguousarray(basis.transforms)
    offsets=np.ascontiguousarray(basis.offsets,np.int32)
    centers=np.ascontiguousarray(basis.centers*METRIC)
    for k,(start,end) in enumerate(zip(offsets,offsets[1:])):
        centers[start:end]=centers[start:end]@metric[k]
    src=Path(bo.__file__).with_suffix(".cpp")
    so=WORK/"metric-bonds.so"
    if not so.exists() or src.stat().st_mtime>so.stat().st_mtime:
        subprocess.run(["c++","-O3","-ffast-math","-std=c++17","-shared","-fPIC",str(src),"-o",str(so)],check=True)
    if basis.energy_coefficients is not None:
        gate_args=(np.ascontiguousarray(basis.energy_coefficients),np.ascontiguousarray(basis.center_energies))
        lib=ct.CDLL(str(so)).bond_order_energy_values
    elif basis.gate_coefficients is None:
        gate_args=()
        lib=ct.CDLL(str(so)).bond_order_metric_values
    else:
        gate_args=(np.ascontiguousarray(basis.gate_coefficients),np.ascontiguousarray(basis.gate_bias))
        lib=ct.CDLL(str(so)).bond_order_gated_values
    lib.argtypes=[ct.c_int]+[ct.c_void_p]*(7+len(gate_args))
    lib.restype=None


def values(z,r):
    z=np.ascontiguousarray(z,np.int32)
    r=np.ascontiguousarray(r,float)
    out=np.zeros((2,len(centers)))
    lib(len(z),*(a.ctypes.data for a in (z,r,centers,offsets,metric,*gate_args,out[0],out[1])))
    return out


def process(lo):
    for i in range(lo,min(lo+96,len(frames))):
        v=values(*frames[i])
        for j,arr in enumerate(dest):
            arr[i]=v[j]


def store(side,data,family="metric"):
    global frames,dest
    frames=data
    paths=[WORK/f"{family}-{w:g}-{side}.npy" for w in bo.LENGTHS]
    dest=[np.lib.format.open_memmap(p.with_suffix(".pending"),mode="w+",dtype=float,
        shape=(len(data),len(centers))) for p in paths]
    with ProcessPoolExecutor(max_workers=5,mp_context=mp.get_context("fork")) as pool:
        for _ in pool.map(process,range(0,len(data),96),buffersize=12):
            pass
    for arr,path in zip(dest,paths):
        arr.flush()
        path.with_suffix(".pending").replace(path)
    print("Cached",side,len(data),"frames",flush=True)


def prepare():
    gs,rows,classes=kt.inputs()
    basis=choose(gs,rows,classes)
    native(basis)
    with (WORK/"ids.pkl").open("rb") as f:
        _,keys=pickle.load(f)
    rng=np.random.default_rng(44)
    for i in rng.choice(len(keys),8,replace=False):
        g=gs[keys[i]]
        v=rng.normal(size=g.r.shape)
        v/=np.linalg.norm(v)
        for j,w in enumerate(bo.LENGTHS):
            basis.length=w
            x,df=basis.design(g.z,g.r,True)
            np.testing.assert_allclose(values(g.z,g.r)[j],x,atol=1.e-10)
            fd=(values(g.z,g.r-1.e-5*v)[j]-values(g.z,g.r+1.e-5*v)[j])/2.e-5
            np.testing.assert_allclose(v.ravel()@df,fd,atol=5.e-7)
    print("Verified metrics, both kernel lengths, and analytic forces",basis.size,flush=True)
    data=[(gs[k].z,gs[k].r) for k in keys]
    for _,z,r,v in trial_sites.hdf():
        data.extend([(z,r),(z,r+.001*v),(z,r-.001*v)])
    store("main",data)
    aux,nprivate,_=trial_aux.read_frames()
    store("rad",aux[nprivate:])


if __name__=="__main__":
    with threadpool_limits(1):
        prepare()
