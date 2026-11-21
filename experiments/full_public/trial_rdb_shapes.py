"""Independent signed-mode and gradient supervision from published RDB7 DFT Hessians.

Select structures deterministically by reaction ID and split, ahead of reading
their Hessians. Exclude reused fragments in another fold. Generate the target
and design in the *same* printed Cartesian frame and retain all three parent
predictions for cross-fitting. No new quantum calculation is needed.
"""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import sqlite3
import numpy as np
from rdkit import Chem
from experiments.full_public import kernel_trials as kt, trial_joint_basis as joint
from experiments.reactivity.stationary_fit import internal_transform
from src.forcefield import ForceField
from src.dispersion.d3 import d3_energy_forces

WORK = kt.WORK


def raw(z,r,q,m):
    """Energy features on a new frame in the joint fit's exact column order."""
    row = atom.design(z,r)[0]
    n=len(joint.BASE)+12
    b=row[n:].reshape(10,400)
    features=np.r_[row[:n],b[:,:144].ravel()*np.sqrt(400/144),b[:,144:].ravel()*np.sqrt(400/256),
                   rich.design(z,r)[0],order.design(z,r)[0]]
    base=spin.prior(z,r,q,m)[0]+d3_energy_forces(z,r)[0]
    x=spin.design(z,r,q,m)[0]
    return np.r_[features,base+weights@x]


def select(db, limit, held_limit):
    d=kt.equation_data("high")
    _,_,src,fold,kind,case=d
    split={str(c).split("/")[-1]:int(h) for c,h in zip(case[(src=="RDB7")&(kind=="forward")],fold[(src=="RDB7")&(kind=="forward")])}
    rows=db.execute("select name,reaction,coord_id,converged from stationary where hessian is not null")
    roles=defaultdict(set)
    records={}
    for name,rxn,idx,converged in rows:
        roles[idx].add(split[rxn])
        records.setdefault(idx,(name,rxn,converged))
    groups=defaultdict(list)
    for idx,(name,rxn,converged) in records.items():
        if len(roles[idx])!=1:
            continue
        h=split[rxn]
        saddle="/ts" in name
        groups[h,saddle].append((h,name,converged))
    selected=[]
    for (h,saddle),v in sorted(groups.items()):
        take=held_limit if h==-1 else limit
        selected+=sorted(v,key=lambda e:hashlib.sha256(e[1].encode()).digest())[:take]
    return selected


def rows(item):
    h,name,converged=item
    z,r,q,m,buf=connection.execute("select z,r,q,m,hessian from stationary where name=?",(name,)).fetchone()
    z=np.frombuffer(z,np.uint8)
    r=np.frombuffer(r,float).reshape(-1,3)
    if len(r)<2:
        return None
    hes=np.frombuffer(buf,float).reshape(r.size,r.size)
    mass=np.array([Chem.GetPeriodicTable().GetMostCommonIsotopeMass(int(s)) for s in z])
    t=internal_transform(mass,r)
    if not t.shape[1]:
        return None
    _,axes=np.linalg.eigh(t.T@hes@t)
    center=raw(z,r,q,m)
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8],"big"))
    direction=rng.normal(size=r.shape)
    direction/=np.linalg.norm(direction)
    step=.004
    force=(raw(z,r-step*direction,q,m)-raw(z,r+step*direction,q,m))/(2*step)
    out,ref,inertia=[force],[0.],[1.]
    for i in (0, axes.shape[1]//3, 2*axes.shape[1]//3, -1):
        mode=(t@axes[:,i]).reshape(r.shape)
        mode/=np.linalg.norm(mode)
        out.append((raw(z,r+step*mode,q,m)+raw(z,r-step*mode,q,m)-2*center)/step**2)
        ref.append(float(mode.ravel()@hes@mode.ravel()))
        inertia.append(float((mass[:,None]*mode**2).sum()))
    return h,name,converged,np.array(out),ref,inertia


def init_worker():
    global connection
    connection=sqlite3.connect(f"file:{WORK/'rdb-dft-frames.sqlite'}?mode=ro",uri=True)


def prepare():
    global atom,rich,order,spin,weights
    dim=np.load(WORK/"joint14-pub.npy",mmap_mode="r").shape[1]
    atom,rich,order=joint.unpack(np.zeros(dim))
    spin=ForceField.load(WORK/"spin-wb97xd3.npz")
    old=ForceField.load(kt.OUT/"joint-pbe0-reaction.npz")
    cfg=json.loads((WORK/"spin-choices.json").read_text())["wb97xd3"]["best"]
    weights=np.vstack([old.weights+np.load(WORK/(f"spin-wb97xd3-{h}-"+"-".join(map(str,cfg))+".npy")) for h in (0,1)]+[spin.weights])
    with sqlite3.connect(f"file:{WORK/'rdb-dft-frames.sqlite'}?mode=ro",uri=True) as db:
        jobs=select(db,72,60)
    data=[]
    with ProcessPoolExecutor(max_workers=5,mp_context=mp.get_context("fork"),initializer=init_worker) as pool:
        for ans in pool.map(rows,jobs,chunksize=3,buffersize=12):
            if ans is not None:
                data.append(ans)
    h,name,stationary,x,y,mass=map(np.asarray,zip(*data))
    np.savez(WORK/"rdb-dft-shapes.npz",fold=h,name=name,stationary=stationary,x=x[:,:,:-3],base=x[:,:,-3:],y=y,mass=mass)
    print("Verified signed modes and stationary directions",len(data),flush=True)


if __name__=="__main__":
    prepare()
