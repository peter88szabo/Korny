"""Local energy smoothness and training-only full Hessian constraints for path fits."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp

import numpy as np
from scipy.linalg import block_diag
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from experiments.full_public import trial_path_features as paths, trial_parent_shapes as p12
from experiments.full_public.trial_constraints import POINTS, MASSES, internal_transform, references, TO_CM1, OUT
from src.bond_order_kernel import METRIC
from src.dispersion.d3 import d3_energy_forces

WORK=paths.WORK


def norm(banks):
    """Kernel Gram of the retained centers, giving the squared RKHS norm of the readout."""
    blocks=[]
    for (family,_,_), (basis,cols) in zip(paths.FAMILIES,banks):
        if family=="gated":
            groups=[(basis.centers[a:b]*METRIC)@basis.transforms[j] for j,(a,b) in enumerate(zip(basis.offsets,basis.offsets[1:]))]
        else:
            groups=list(basis.centers.values())
        start=0
        for values in groups:
            idx=cols[(cols>=start)&(cols<start+len(values))]-start
            start+=len(values)
            if not len(idx):continue
            dist=cdist(values[idx],values[idx])/basis.length
            if family=="density":
                mat=np.zeros_like(dist)
                for scale in (.5,2.):
                    v=np.sqrt(5)*dist/scale
                    mat+=.5*(1+v+v*v/3)*np.exp(-v)
            else:
                mat=np.exp(-.5*dist**2)
            blocks.append(mat)
    return block_diag(*blocks)


def cartesian(z,r,banks=None):
    """Compute all six cross-fitted parent forces from a single feature Jacobian."""
    banks = components if banks is None else banks
    m=p12.parent
    _,f=m.prior(z,r,0,1,True)
    f+=d3_energy_forces(z,r)[1]
    old=f.reshape(-1,1)+m.design(z,r,0,1,True)[1]@p12.ws
    old+=p12.atom.design(z,r,True)[1]@p12.wa+p12.rich.design(z,r,True)[1]@p12.wr
    return np.column_stack([*(b.design(z,r,True)[1][:,idx] for b,idx in banks),old])


def molecule(item,design=None,banks=None):
    design = cartesian if design is None else design
    banks = components if banks is None else banks
    name,point=item
    z,mass=np.array([MASSES[a] for a in point["atoms"]]).T
    z=z.astype(int)
    r=np.asarray(point["positions_A"])
    freq=np.sort(point["frequencies_cm1"])
    t=internal_transform(mass,r)
    size=sum(len(idx) for _,idx in banks)
    a=design(z,r)
    grams=[]
    # The off-diagonal Hessian terms protect mode mixing and bending. All
    # Cartesian rows and all mode pairs come from these 14 training molecules.
    h=np.stack([(design(z,r-.001*v)-design(z,r+.001*v))/.002 for v in np.eye(r.size).reshape(-1,*r.shape)])
    h=np.einsum("ai,abk,bj->ijk",t,(h+h.transpose(1,0,2))/2,t,optimize=True)
    i,j=np.triu_indices(len(freq))
    for k in range(a.shape[1]-size):
        w,axis=np.linalg.eigh(h[:,:,size+k])
        deriv=np.einsum("ia,ijk,jb->abk",axis,h[:,:,:size],axis,optimize=True)
        reference=np.zeros(len(i));diag=i==j
        reference[diag]=(freq/TO_CM1)**2-w
        scale=4*TO_CM1**2/(2*np.maximum(350.,(freq[i]+freq[j])/2)*120*np.sqrt(len(freq)))
        # Both symmetric entries contribute to the Frobenius error.
        scale*=np.where(diag,1.,np.sqrt(2.))
        vals=np.vstack([np.column_stack((a[:,:size],-a[:,size+k]))*(12/np.sqrt(r.size)),
                           np.column_stack((deriv[i,j],reference))*scale[:,None]])
        grams.append(vals.astype(np.float32))
    return name,grams


def run():
    global components
    components=paths.bases()
    np.save(WORK/f"{paths.TAG}-rkhs.npy",norm(components))
    p12.configure()
    points=json.loads(POINTS.read_text());points.update(references())
    names=json.loads((OUT/"stationary.json").read_text())["training"]
    grouped=[[] for _ in range(6)]
    with ProcessPoolExecutor(max_workers=3,mp_context=mp.get_context("fork")) as pool:
        for name,rows in pool.map(molecule,[(n,points[n]) for n in names],buffersize=4):
            for group,block in zip(grouped,rows):group.append(block)
    for group in grouped:
        assert len(group)==14
    for k,g in enumerate(grouped):
        np.save(WORK/f"{paths.TAG}-nist-{k}.npy",np.vstack(g))
    print("Computed complete training Hessian and stationary-force constraints",len(names),"molecules",len(np.vstack(grouped[0])),"rows per readout",flush=True)


if __name__=="__main__":
    with threadpool_limits(1):run()
