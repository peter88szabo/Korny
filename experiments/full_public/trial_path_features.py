"""Joint local kernel columns for equilibrium, reacting and off-equilibrium geometries.

Choose deterministic training-center subsets within each chemical element or
pair. Keep the full reaction-participation gate during inference even when the
linear readout uses only a subset of centers. Every archived optimization frame
shares its reaction's split. Each path energy is relative to its own stationary
structure, which eliminates the absolute electronic energy zero.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import sqlite3

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.full_public import trial_parent_shapes as parent
from experiments.full_public import trial_metric as bond, trial_messages as graph, trial_shells as shell
from experiments.full_public.trial_curvature_fit import anchor_blocks
from src.bond_order_kernel import BondOrderKernel
from src.message_kernel import MessageKernel
from src.reactive_basis import ReactiveBasis

WORK = parent.WORK
FAMILIES = (("density",1.35,ReactiveBasis),("topology",.7,MessageKernel),("gated",1.15,BondOrderKernel))
TAG = "path-joint-v1"


def bases():
    """Return model-space indices and bases in the same order as the design files."""
    output = []
    for family, width, model_class in FAMILIES:
        with np.load(WORK/f"{family}-basis.npz") as f:
            basis = model_class.load(f)
        basis.length = width
        if family == "gated":
            offsets = basis.offsets
        else:
            offsets = np.cumsum([0]+[len(c) for c in basis.centers.values()])
        # Retain local patterns even in sparsely populated element pairs.
        cols = np.concatenate([np.arange(i,j,3) for i,j in zip(offsets,offsets[1:])])
        output.append((basis,cols))
    return output


def stationary():
    banks = bases()
    count = sum(len(ids) for _,ids in banks)
    for side in ("main","rad"):
        maps = [np.load(WORK/f"{family}-{width:g}-{side}.npy",mmap_mode="r") for family,width,_ in FAMILIES]
        out = WORK/f"{TAG}-{side}.npy"
        if out.exists():
            continue
        new = np.lib.format.open_memmap(out.with_suffix(".pending"),mode="w+",dtype=float,shape=(len(maps[0]),count))
        for lo in range(0,len(new),260):
            sl=slice(lo,lo+260)
            new[sl] = np.column_stack([matrix[sl][:,ids] for (_,ids),matrix in zip(banks,maps)])
        new.flush()
        out.with_suffix(".pending").replace(out)
    main = np.load(WORK/f"{TAG}-main.npy",mmap_mode="r")
    n = len(np.load(WORK/"prior.npy",mmap_mode="r"))
    anchor_blocks(main,n,TAG)
    print("Completed joint energy, force and curvature anchors",main.shape,flush=True)


def prepare_paths():
    global selected, records, xs, aux, end_x, end_base
    from experiments.full_public import trial_multiscale as ms
    parent.configure()
    components=bases()
    shell.setup()
    shell.basis=components[0][0]
    graph.FAMILY="topology"
    graph.setup()
    graph.basis=components[1][0]
    bond.native(components[2][0])
    selected = [ids for _,ids in components]
    maps = [np.load(WORK/f"{family}-{width:g}-main.npy",mmap_mode="r") for family,width,_ in FAMILIES]
    n = len(np.load(WORK/"prior.npy",mmap_mode="r"))
    end_x = np.column_stack([m[:n][:,ix] for m,ix in zip(maps,selected)])
    end_base = np.column_stack([ms.baseline("wb97xd3",h,"parent","p12-public") for h in (0,1,-1)])
    with sqlite3.connect(f"file:{WORK/'grambow-steps.sqlite'}?mode=ro",uri=True) as db:
        records=list(db.execute("select p.coord_id,p.folds,p.z,p.q,p.m,s.r,s.de,s.force from paths p join steps s using(name) order by p.name,s.step"))
    dim=sum(map(len,selected))
    xp=WORK/f"{TAG}-steps.npy"
    ap=WORK/f"{TAG}-labels.npy"
    xs=np.lib.format.open_memmap(xp.with_suffix(".pending"),mode="w+",shape=(len(records),2,dim),dtype=np.float32)
    aux=np.lib.format.open_memmap(ap.with_suffix(".pending"),mode="w+",shape=(len(records),12),dtype=float)
    with ProcessPoolExecutor(max_workers=5,mp_context=mp.get_context("fork")) as pool:
        for j,_ in enumerate(pool.map(batch,range(0,len(records),20),buffersize=12)):
            if (j+1)%300==0:print("Completed archived radical geometries",(j+1)*20,flush=True)
    xs.flush()
    aux.flush()
    xp.with_suffix(".pending").replace(xp)
    ap.with_suffix(".pending").replace(ap)
    print("Completed archived path features",len(records),dim,flush=True)


def batch(start):
    """Evaluate center energies and force projections in the printed Cartesian frame."""
    points, ids, states, ref = [], [], [], []
    for k in range(start,min(start+20,len(records))):
        idx,fold,z,q,m,r,de,f=records[k]
        z=np.frombuffer(z,np.uint8)
        r=np.frombuffer(r,float).reshape(-1,3)
        valid=f is not None
        if valid:
            f=np.frombuffer(f,float).reshape(r.shape)
            norm=np.linalg.norm(f)
        else:
            norm=0.
        # The direction of the printed force gives a high-signal scalar equation.
        # At a stationary point use a reproducible direction and its projection.
        v=f/norm if norm>1.e-10 else np.random.default_rng(k).normal(size=r.shape)
        v/=np.linalg.norm(v)
        for pt in (r,r-.001*v,r+.001*v):
            points.append((z,pt))
            states.append((q,m))
        ids.append(idx)
        mask=sum(1<<int(h) for h in set(fold.split(",")))
        ref.append((de,norm,valid,mask))
    blocks=(shell.batch(points)[1],graph.batch(points)[0],np.array([bond.values(*p)[0] for p in points]))
    x=np.column_stack([b[:,ix] for b,ix in zip(blocks,selected)]).reshape(len(ids),3,-1)
    base=np.array([parent.energies(z,r,*s)[3:] for (z,r),s in zip(points,states)]).reshape(len(ids),3,3)
    sl=slice(start,start+len(ids))
    xs[sl,0]=x[:,0]-end_x[ids]
    xs[sl,1]=(x[:,1]-x[:,2])/.002
    aux[sl,:4]=ref
    aux[sl,4:7]=base[:,0]-end_base[ids]
    aux[sl,7:10]=(base[:,1]-base[:,2])/.002
    aux[sl,10]=ids
    aux[sl,11]=start+np.arange(len(ids))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step",choices=("stationary","paths"))
    with threadpool_limits(1 if (step:=parser.parse_args().step)=="paths" else 4):
        (prepare_paths if step=="paths" else stationary)()
