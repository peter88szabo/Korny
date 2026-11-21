"""Evaluate the cross-fitted p12 parents on matched RDB7 shape observations.

Share the backbone and all molecular features between the six fold/target
predictions. Cross-check each readout against the archived stationary ledger
before using its gradient or curvature as the prior of a new correction.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import pickle
import sqlite3
import numpy as np
from rdkit import Chem
from threadpoolctl import threadpool_limits
from experiments.full_public import kernel_trials as kt, trial_multiscale as ms, trial_fit
from experiments.full_public.trial_isomers import gram
from experiments.reactivity.stationary_fit import internal_transform
from src.forcefield import ForceField
from src.dispersion.d3 import d3_energy_forces

WORK=kt.WORK
TARGETS=("high","wb97xd3")
FOLDS=(0,1,-1)


def selected_coefficients():
    """Recover the actual chosen p12 readout using only the corresponding folds."""
    b,pub,_,rad,anchor,nist=trial_fit.get_features(("rich",1.,1.,256,False))
    rad=rad.reshape(-1,7,b.size)
    meta=np.load(WORK/"rad-meta.npy")
    with (WORK/"radical-modes.pkl").open("rb") as f:
        saddle=np.array([row[1][0].split("/")[-1].startswith("ts") for row in pickle.load(f)])
    result=[]
    for t in TARGETS:
        d=kt.equation_data(t)
        x=d[0]@pub
        for h in FOLDS:
            parent=ms.baseline(t,h,"parent")
            if h<0:
                w=np.load(WORK/f"ms-1-1-r2-s0.5-{t}-coef.npy")
            else:
                inc=kt.grouped_moments(d,x,d[1]-d[0]@parent,(d[3]>=0)&(d[3]!=h))
                inc["other"]+=gram(t,h,pub,parent)
                inc["other"][:-1,:-1]+=8*nist
                if t=="wb97xd3":
                    zz,yy=trial_fit.rad_rows(rad,ms.baseline(t,h,"modes"),meta,np.where(saddle,3.,1.))
                    inc["other"]+=trial_fit.reduce_gram(zz,yy,np.repeat((meta[:,0]>=0)&(meta[:,0]!=h),3))
                w=kt.solve(inc,anchor,32,64,.5,2.)
            np.testing.assert_allclose(parent+pub@w,ms.baseline(t,h,"parent","p12-public"),atol=1.e-7,rtol=0)
            result.append(w)
    np.save(WORK/"p12-rich-fold-vectors.npy",np.array(result))
    return np.array(result)


def configure():
    global parent,atom,rich,ws,wa,wr
    parent=ForceField.load(WORK/"p12-public-high.npz",dispersion="d3")
    atom=parent.reactive_basis
    rich=parent.additional_bases[0]
    ws,wa=[],[]
    wr=np.load(WORK/"p12-rich-fold-vectors.npy").T
    old=ForceField.load(kt.OUT/"joint-pbe0-reaction.npz")
    for t in TARGETS:
        cfg=json.loads((WORK/"spin-choices.json").read_text())[t]["best"]
        keys=(json.loads((WORK/"p8-public-settings.json").read_text())["selected"],
              json.loads((WORK/"p10-public-settings.json").read_text())["branches"][t])
        for h in FOLDS:
            if h<0:
                sw=ForceField.load(WORK/f"spin-{t}.npz").weights
            else:
                sw=old.weights+np.load(WORK/(f"spin-{t}-{h}-"+"-".join(map(str,cfg))+".npy"))
            ws.append(sw)
            a,b=[np.load(WORK/f"{k}-{t}-weights-{h}.npy") for k in keys]
            start=524
            wa.append(.5*np.r_[a[:start]+b[:start],np.column_stack([
                a[start:].reshape(10,144)*np.sqrt(400/144),b[start:].reshape(10,256)*np.sqrt(400/256)]).ravel()])
    ws=np.array(ws).T
    wa=np.array(wa).T


def energies(z,r,q,m):
    base=parent.prior(z,r,q,m)[0]+d3_energy_forces(z,r)[0]
    return base+parent.design(z,r,q,m)[0]@ws+atom.design(z,r)[0]@wa+rich.design(z,r)[0]@wr


def frames(name):
    z,r,q,m,buf=db.execute("select z,r,q,m,hessian from stationary where name=?",(name,)).fetchone()
    z=np.frombuffer(z,np.uint8);r=np.frombuffer(r,float).reshape(-1,3)
    mass=np.array([Chem.GetPeriodicTable().GetMostCommonIsotopeMass(int(s)) for s in z])
    h=np.frombuffer(buf,float).reshape(r.size,r.size)
    t=internal_transform(mass,r)
    _,axes=np.linalg.eigh(t.T@h@t)
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8],"big"))
    rand=rng.normal(size=r.shape)
    dirs=[rand]+[(t@axes[:,i]).reshape(r.shape) for i in (0,axes.shape[1]//3,2*axes.shape[1]//3,-1)]
    out=[(z,r)]
    for v in dirs:
        v=v/np.linalg.norm(v)
        out.extend([(z,r-.004*v),(z,r+.004*v)])
    return (q,m),out


def derivative(values):
    mid=values[0]
    minus,plus=values[1::2],values[2::2]
    return np.concatenate([((minus[:1]-plus[:1])/.008),
                           ((minus[1:]-mid)+(plus[1:]-mid))/.004**2],axis=0)


def job(name):
    st,pts=frames(name)
    val=np.array([energies(z,r,*st) for z,r in pts])
    return name,val[0],derivative(val)


def open_db():
    global db
    db=sqlite3.connect(f"file:{WORK/'rdb-dft-frames.sqlite'}?mode=ro",uri=True)


def run():
    if not (WORK/"p12-rich-fold-vectors.npy").exists():selected_coefficients()
    configure()
    with np.load(WORK/"rdb-dft-shapes.npz") as f:names=f["name"]
    with ProcessPoolExecutor(max_workers=5,mp_context=mp.get_context("fork"),initializer=open_db) as pool:
        data=list(pool.map(job,map(str,names),chunksize=3,buffersize=12))
    names,en,shape=map(np.asarray,zip(*data))
    with (WORK/"ids.pkl").open("rb") as f:ids,_=pickle.load(f)
    ix=np.array([ids[n] for n in names])
    for j,(t,h) in enumerate((t,h) for t in TARGETS for h in FOLDS):
        np.testing.assert_allclose(en[:,j],ms.baseline(t,h,"parent","p12-public")[ix],rtol=0,atol=2e-4)
    np.savez(WORK/"p12-rdb-shape-parents.npz",name=names,base=shape.reshape(len(names),5,2,3))
    print("Verified parent energies and evaluated derivatives",len(names),"structures, six readouts",flush=True)


if __name__=="__main__":
    with threadpool_limits(2):run()
