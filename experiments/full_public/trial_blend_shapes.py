"""Protect equilibrium forces and every internal Hessian entry in a multi-kernel blend."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.full_public.trial_energy_blend import Components
from experiments.full_public.trial_constraints import OUT,POINTS,MASSES,internal_transform,references,TO_CM1
from experiments.full_public.kernel_trials import WORK
from src.forcefield import ForceField

BRANCHES=((17,12),(18,17),(19,18),(20,12),(21,12),(22,12),(14,12),(15,12),(16,12),(13,12),(12,11))


def case(item):
    name,p=item
    z,mass=np.array([MASSES[a] for a in p["atoms"]]).T
    z=z.astype(int)
    r=np.asarray(p["positions_A"])
    freq=np.sort(p["frequencies_cm1"])
    t=internal_transform(mass,r)
    def jac(x):
        f=pieces.readout(z,x)
        cols=np.column_stack([f[:,pieces.stages.index(j)]-f[:,pieces.stages.index(i)] for j,i in BRANCHES])
        # The archived PBE0 wavenumbers lack eigenvectors. Orient the target
        # Hessian in the p12 mass-weighted internal eigenbasis.
        return np.column_stack((cols,parent.predict(z,x)[1].ravel()))
    a=jac(r)
    a[:,-1]*=-1
    h=np.stack([(jac(r-.001*v)-jac(r+.001*v))/.002 for v in np.eye(r.size).reshape(-1,*r.shape)])
    dyn=np.einsum("ai,abk,bj->ijk",t,(h+h.transpose(1,0,2))*.5,t,optimize=True)
    w,axes=np.linalg.eigh(dyn[:,:,-1])
    pp=np.einsum("ia,ijk,jb->abk",axes,dyn[:,:,:-1],axes,optimize=True)
    i,j=np.triu_indices(len(freq))
    rhs=np.zeros(len(i));rhs[i==j]=(freq/TO_CM1)**2-w
    scale=TO_CM1**2/(2*np.maximum(350.,(freq[i]+freq[j])/2)*100*np.sqrt(len(freq)))
    rows=np.column_stack((pp[i,j],rhs))*scale[:,None]*np.where(i==j,1.,np.sqrt(2.))[:,None]
    return np.vstack([a*6/np.sqrt(r.size),rows])


def run():
    global pieces,parent
    pts=json.loads(POINTS.read_text());pts.update(references())
    names=json.loads((OUT/"stationary.json").read_text())["training"]
    stages=sorted({v for b in BRANCHES for v in b})
    for target in ("high","wb97xd3"):
        pieces=Components(target,stages)
        parent=ForceField.load(WORK/f"p12-public-{target}.npz",dispersion="d3")
        with ProcessPoolExecutor(max_workers=5,mp_context=mp.get_context("fork")) as pool:
            rows=np.vstack(list(pool.map(case,[(name,pts[name]) for name in names],buffersize=8)))
        np.save(WORK/f"mixture24-{target}-shapes.npy",rows)
        print("Multi-kernel full Hessians on training minima",target,rows.shape,flush=True)


if __name__=="__main__":
    with threadpool_limits(1):run()
