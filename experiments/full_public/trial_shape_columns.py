"""Materialize force and four signed Hessian columns in the archived RDB7 Cartesian frame."""
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.full_public import trial_parent_shapes as ps
from experiments.full_public import trial_messages as graph,trial_shells as shell,trial_metric as metric
from src.message_kernel import MessageKernel
from src.reactive_basis import ReactiveBasis
from src.bond_order_kernel import BondOrderKernel

WORK=ps.WORK


def task(name):
    _,pts=ps.frames(name)
    if family=="density":val=shell.batch(pts)
    elif family in ("metric","gated"):val=np.array([metric.values(*p) for p in pts]).transpose(1,0,2)
    else:val=graph.batch(pts)
    return [ps.derivative(v) for v in val]


def run(label):
    global family
    family=label
    with np.load(WORK/f"{family}-basis.npz") as f:
        if family=="density":
            shell.setup()
            shell.basis=ReactiveBasis.load(f)
            widths=shell.LENGTHS
        elif family in ("metric","gated"):
            b=BondOrderKernel.load(f)
            metric.native(b)
            widths=(1.15,1.15/np.sqrt(2))
        else:
            graph.FAMILY=family
            graph.setup()
            graph.basis=MessageKernel.load(f)
            widths=graph.LENGTHS
    with np.load(WORK/"rdb-dft-shapes.npz") as f:names=f["name"]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context("fork"),initializer=ps.open_db) as pool:
        result=list(pool.map(task,map(str,names),chunksize=2,buffersize=10))
    for j,w in enumerate(widths):
        np.save(WORK/f"{family}-{w:g}-rdb-shapes.npy",np.stack([v[j] for v in result]))
    print("Cached RDB7 derivative columns",family,len(names),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("family",choices=("topology","message","density","metric","gated"))
    with threadpool_limits(1):run(p.parse_args().family)
