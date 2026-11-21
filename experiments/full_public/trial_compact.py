"""Fit a large, compactly supported sparse kernel without a dense center Gram.

Prototype selection uses eligible chemical formulas and the existing PBE0
training geometries. Each local support contains a fixed neighborhood of
prototypes and is differentiated at the cutoff. Sparse equation matrices
retain every public training reaction and HDF5 energy/force/curvature anchor.
"""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import pickle
import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
from threadpoolctl import threadpool_limits
from experiments.full_public import (
    kernel_trials as kt,
)
from experiments.full_public.trial_adaptive_spectrum import (
    tables,
    geometry_atoms,
    json_count,
)
from experiments.full_public.trial_spectra import evaluate_trial
from src.compact_kernel import CompactKernel
from src.reactive_basis import ReactiveBasis

WORK = kt.WORK
ROOT = WORK / "compact38"
STAGE = "p38-public"
PARENT = "p36-public"


def learn():
    geom = tables()
    n = len(np.load(WORK / "prior.npy", mmap_mode="r"))
    rng = np.random.default_rng(904)
    d = [kt.equation_data(t) for t in kt.TARGETS]
    denied = np.unique(np.concatenate([a[0][a[3] < 2].indices for a in d]))
    eligible = np.setdiff1d(
        np.unique(np.concatenate([a[0][a[3] >= 2].indices for a in d])), denied
    )
    records = len(np.load(WORK / "hdf-sources.npy"))
    hdf = n + 3 * rng.choice(records, 6500, replace=False)
    with np.load(WORK / "rich-basis.npz") as ar:
        base = ReactiveBasis.load(ar)
    centers = {}
    radii = {}
    for s, total in ((1, 4096), (6, 4096), (7, 2048), (8, 2048)):
        off, x = geom[s]
        parts = []
        for ids in (eligible, hdf):
            ix = geometry_atoms(off, ids)[0]
            if len(ix):
                parts.append(x[rng.choice(ix, min(len(ix), 50000), replace=False)])
        u = np.vstack(parts)
        # Remove nearly identical environments before allocating prototypes.
        # Repeated hydrogens in common groups would otherwise crowd out
        # geometries with partial or multiple bonds.
        _, idx = np.unique(np.round(u, 2), axis=0, return_index=True)
        u = u[np.sort(idx)]
        count = min(total, len(u))
        first = rng.choice(len(u), max(1, count // 2), replace=False)
        dist = cKDTree(u[first]).query(u, k=1)[0]
        dist[first] = 0.0
        extra = rng.choice(
            len(u), count - len(first), replace=False, p=dist / dist.sum()
        )
        c = u[np.r_[first, extra]]
        nearest = cKDTree(c).query(c, k=min(21, len(c)))[0][:, -1]
        centers[s] = c
        radii[s] = np.clip(1.40 * nearest, 0.16, 1.45)
    model = CompactKernel(
        base.species,
        base.spec,
        base.projections,
        centers,
        radii,
        np.zeros(sum(map(len, centers.values()))),
    )
    np.savez_compressed(ROOT / "basis.npz", **model.fields())
    print(
        "Compact prototypes by element",
        {k: len(c) for k, c in centers.items()},
        "total",
        model.size,
        flush=True,
    )
    return model


def encode(item):
    side, lo, hi = item
    f = ROOT / f"{side}-block-{lo}.npz"
    if f.exists():
        return
    counts = hi - lo
    ids = np.arange(lo + skip, hi + skip)
    cols = []
    for s, (off, design) in geoms.items():
        rows, owner, *_ = geometry_atoms(off, ids)
        cols.append(model.sparse(s, design[rows], owner, counts))
    a = sparse.hstack(cols, format="csr")
    a.eliminate_zeros()
    tmp = f.with_suffix(".pending")
    with tmp.open("wb") as fp:
        sparse.save_npz(fp, a, compressed=False)
    tmp.replace(f)


def consolidate(side, jobs):
    """Write a sparse matrix sequentially so large batches fit under the RAM cap."""
    count = 0
    for tag, lo, hi in jobs:
        with np.load(ROOT / f"{tag}-block-{lo}.npz") as z:
            count += len(z["indices"])
    n = jobs[-1][-1]
    arrays = {
        k: np.lib.format.open_memmap(
            ROOT / f"{side}-{k}.pending", mode="w+", dtype=t, shape=(size,)
        )
        for k, t, size in (
            ("indptr", np.int64, n + 1),
            ("indices", np.int32, count),
            ("data", float, count),
        )
    }
    offset = 0
    for tag, lo, hi in jobs:
        with np.load(ROOT / f"{tag}-block-{lo}.npz") as z:
            nnz = len(z["indices"])
            s = slice(offset, offset + nnz)
            arrays["data"][s] = z["data"]
            arrays["indices"][s] = z["indices"]
            arrays["indptr"][lo : hi + 1] = offset + z["indptr"].astype(np.int64)
            offset += nnz
    for k, a in arrays.items():
        a.flush()
        (ROOT / f"{side}-{k}.pending").replace(ROOT / f"{side}-{k}.npy")
    print("Sparse design", side, n, "geometries", count, "entries", flush=True)


def load_sparse(side):
    data, idx, ptr = [
        np.load(ROOT / f"{side}-{k}.npy", mmap_mode="r")
        for k in ("data", "indices", "indptr")
    ]
    return sparse.csr_matrix(
        (data, idx, ptr), shape=(len(ptr) - 1, model.size), copy=False
    )


def prepare():
    global model, geoms, skip
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "basis.npz").exists():
        with np.load(ROOT / "basis.npz") as f:
            model = CompactKernel.load(f)
    else:
        model = learn()
    for side in ("main", "rad"):
        if (ROOT / f"{side}-data.npy").exists():
            continue
        geoms = tables(side == "rad")
        skip = json_count() if side == "rad" else 0
        n = len(geoms[1][0]) - 1 - skip
        jobs = [(side, i, min(i + 384, n)) for i in range(0, n, 384)]
        with ProcessPoolExecutor(
            max_workers=5, mp_context=mp.get_context("fork")
        ) as pool:
            for i, _ in enumerate(pool.map(encode, jobs, buffersize=16)):
                if (i + 1) % 200 == 0:
                    print("Compact batches completed", side, (i + 1) * 384, flush=True)
        consolidate(side, jobs)
    gs, _, _ = kt.inputs()
    with (WORK / "ids.pkl").open("rb") as f:
        _, names = pickle.load(f)
    main = load_sparse("main")
    rng = np.random.default_rng(419)
    for ix in rng.choice(len(names), 8, replace=False):
        g = gs[names[ix]]
        val, j = model.design(g.z, g.r, True)
        np.testing.assert_allclose(val, main[ix].toarray()[0], atol=5.0e-9)
        v = rng.normal(size=g.r.shape)
        v /= np.linalg.norm(v)
        fd = (
            model.design(g.z, g.r - 1.0e-5 * v)[0]
            - model.design(g.z, g.r + 1.0e-5 * v)[0]
        ) / 2.0e-5
        np.testing.assert_allclose(j.T @ v.ravel(), fd, atol=2.0e-5)
    print("Verified sparse deployment values and Cartesian derivatives", flush=True)


def isomers(t, h, pub, base):
    """Use deterministic isomer differences at an identical formula and reference."""
    with (WORK / "isomer-observations.pkl").open("rb") as f:
        groups = pickle.load(f)["RDB7" if t == "high" else "Grambow"]
    rng = np.random.default_rng(310)
    left = []
    right = []
    rhs = []
    for entries in groups.values():
        rows = [(i, e - base[i]) for i, e, folds in entries if h not in folds]
        if len(rows) < 2:
            continue
        ids, val = map(np.asarray, zip(*rows))
        perm = rng.permutation(len(ids))
        other = np.roll(perm, 1)
        left.extend(ids[perm])
        right.extend(ids[other])
        rhs.extend(val[perm] - val[other])
    return (pub[left] - pub[right]) * 0.7, np.array(rhs) * 0.7


def fit():
    from experiments.full_public.trial_compact_solve import run

    run(model, load_sparse("main"), load_sparse("rad"))


if __name__ == "__main__":
    with threadpool_limits(1):
        prepare()
    with threadpool_limits(6):
        fit()
    with threadpool_limits(1):
        evaluate_trial(STAGE)
