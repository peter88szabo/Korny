"""Choose training bond environments, verify their derivatives, and cache design rows."""
import ctypes as ct
import multiprocessing as mp
import pickle
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

from src.bond_order_kernel import BondOrderKernel, environments, METRIC
from experiments.full_public import kernel_trials as kt, trial_sites, trial_aux
from experiments.reactivity.reaction_classes import lookup

WORK = kt.WORK
LENGTHS = (1.15, 1.15 / np.sqrt(2))


def choose(gs, rows, classes):
    """Farthest-point placement across training reaction classes and both sides."""
    classes.update(lookup())
    pool = [[] for _ in range(10)]
    keys = trial_sites.candidates(gs, rows, classes)
    for key in keys:
        g = gs[key]
        for k, feat, gate, *_ in environments(g.z, g.r):
            if (use := gate > .025).any():
                pool[k].append(feat[use])
    rng = np.random.default_rng(691)
    centers, offsets = [], [0]
    for data in pool:
        chosen = []
        if data:
            data = np.vstack(data)
            if len(data) > 8000:
                data = data[rng.choice(len(data), 8000, replace=False)]
            scaled = data * METRIC
            dist = ((scaled - scaled.mean(axis=0))**2).sum(axis=1)
            for _ in range(160):
                j = np.argmax(dist)
                if dist[j] < 1.e-6:
                    break
                chosen.append(data[j])
                dist = np.minimum(dist, np.sum((scaled-scaled[j])**2, axis=1))
        centers.extend(chosen)
        offsets.append(len(centers))
    basis = BondOrderKernel(centers, offsets, LENGTHS[0])
    np.savez_compressed(WORK/"bo-basis.npz", **basis.fields())
    print("Selected bond environments", np.diff(offsets).tolist(), flush=True)
    return basis


def evaluate(z, r):
    z = np.ascontiguousarray(z, dtype=np.int32)
    r = np.ascontiguousarray(r, dtype=float)
    out = np.zeros((2, len(centers)))
    lib(len(z), *(v.ctypes.data for v in (z,r,centers,offsets,out[0],out[1])))
    return out


def process(lo):
    for i in range(lo, min(lo+128, len(frames))):
        values = evaluate(*frames[i])
        for k, dest in enumerate(output):
            dest[i] = values[k]


def cache(side, data):
    global frames, output
    frames = data
    paths = [WORK/f"bo-{w:g}-{side}.npy" for w in LENGTHS]
    tmp = [p.with_suffix(".part.npy") for p in paths]
    output = [np.lib.format.open_memmap(p, mode="w+", dtype=float,
                shape=(len(data),len(centers))) for p in tmp]
    with ProcessPoolExecutor(max_workers=5, mp_context=mp.get_context("fork")) as pool:
        for _ in pool.map(process, range(0, len(frames),128), buffersize=12):
            pass
    for arr, p, dest in zip(output,tmp,paths):
        arr.flush()
        p.replace(dest)
    print("Completed bond-order design", side, len(data), flush=True)


def prepare():
    global lib, centers, offsets
    gs, rows, classes = kt.inputs()
    basis = choose(gs, rows, classes)
    centers = np.ascontiguousarray(basis.centers * METRIC)
    offsets = np.ascontiguousarray(basis.offsets, np.int32)
    src = Path(__file__).with_suffix(".cpp")
    so = WORK/"bond_order.so"
    subprocess.run(["c++", "-std=c++17", "-O3", "-ffast-math", "-fPIC", "-shared", str(src), "-o", str(so)], check=True)
    lib = ct.CDLL(str(so)).bond_order_values
    lib.argtypes = [ct.c_int]+[ct.c_void_p]*6
    lib.restype = None
    with (WORK/"ids.pkl").open("rb") as f:
        _, keys = pickle.load(f)
    rng = np.random.default_rng(32)
    for ix in rng.choice(len(keys), 16, replace=False):
        g = gs[keys[ix]]
        z, r = g.z, g.r
        direction = rng.normal(size=r.shape)
        direction /= np.linalg.norm(direction)
        for i,w in enumerate(LENGTHS):
            basis.length = w
            v,j = basis.design(z, r, True)
            np.testing.assert_allclose(evaluate(z,r)[i],v,atol=1e-10)
            deriv = (evaluate(z,r-1.e-5*direction)[i]-evaluate(z,r+1.e-5*direction)[i])/2.e-5
            np.testing.assert_allclose(deriv, direction.ravel()@j, atol=5e-7)
    print("Verified both lengths and Cartesian forces", flush=True)
    data = [(gs[k].z,gs[k].r) for k in keys]
    for _,z,r,v in trial_sites.hdf():
        data.extend([(z,r),(z,r+.001*v),(z,r-.001*v)])
    cache("main",data)
    aux,nprivate,_ = trial_aux.read_frames()
    cache("rad",aux[nprivate:])


if __name__ == "__main__":
    prepare()


def features(length, family="bo"):
    """Build the public energy/force and NIST shape anchors for one kernel length."""
    from .trial_constraints import nist_rows
    from src.forcefield import ForceField
    with np.load(WORK / f"{family}-basis.npz") as f:
        basis = BondOrderKernel.load(f)
    basis.length = length
    main = np.load(WORK/f"{family}-{length:g}-main.npy", mmap_mode="r")
    rad = np.load(WORK/f"{family}-{length:g}-rad.npy", mmap_mode="r")
    nbase = len(np.load(WORK/"prior.npy", mmap_mode="r"))
    sources = np.load(WORK/"hdf-sources.npy")
    per_source = {s:4500 / n for s,n in zip(*np.unique(sources,return_counts=True))}
    w = np.sqrt([per_source[s] for s in sources])[:, None]
    anchor = np.zeros((basis.size, basis.size))
    for i in range(0,len(sources),384):
        stop = min(i+384,len(sources))
        block = main[nbase+3*i:nbase+3*stop].reshape(-1,3,basis.size)
        en = block[:,0]*w[i:stop]
        force = (block[:,1]-block[:,2]) * (4/.002) * w[i:stop]
        anchor += en.T@en + force.T@force
    parent = ForceField.load(kt.OUT/"joint-pbe0-reaction.npz",dispersion="d3")
    rows = nist_rows(lambda z,r: basis.design(z,r,True)[1], parent)
    return basis,main[:nbase],None,rad,anchor,rows[:,:-1].T@rows[:,:-1]
