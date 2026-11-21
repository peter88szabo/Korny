"""Evaluate new kernel columns in geometry blocks to bound resident memory."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from experiments.full_public.kernel_trials import WORK
from experiments.full_public.trial_sites import ACTIVE
from src.reactive_basis import ReactiveBasis


def label(tag, w, a):
    return f"{tag}-w{w:g}-a{a:g}"


def batch(bounds):
    lo, hi = bounds
    for s in ACTIVE:
        starts = offset[s][lo : hi + 1]
        sites = stored[s][starts[0] : starts[-1]]
        counts = np.diff(starts)
        present = counts > 0
        section = features[lo:hi, cols[s]]
        section[:] = 0.0
        if not len(sites):
            continue
        weights = np.ones(sites.shape[1])
        weights[basis.radial_dims[s] :] = basis.angular
        u = sites * weights
        c = basis.centers[s] * weights
        shift = (c * c).sum(axis=1)
        dist = np.sqrt(
            np.maximum(np.sum(u * u, axis=1)[:, None] + shift - 2 * u @ c.T, 0.0)
        )
        vals = np.zeros_like(dist)
        for width in (0.5 * basis.length, 2 * basis.length):
            factor = np.sqrt(5) / width
            t = dist * factor
            t0 = np.sqrt(shift) * factor
            vals += 0.5 * (
                (1 + t + t * t / 3) * np.exp(-t) - (1 + t0 + t0 * t0 / 3) * np.exp(-t0)
            )
        section[present] = np.add.reduceat(
            vals, starts[:-1][present] - starts[0], axis=0
        )
    return hi


def cache(tag, w, a, kind=None):
    global basis, stored, offset, features, cols
    path = WORK / (
        label(tag, w, a) + (f"-{kind}" if kind else "") + "-bank2-features.npy"
    )
    if path.exists():
        return path
    with np.load(WORK / f"{tag}-basis.npz") as b:
        basis = ReactiveBasis.load(b)
    basis.length = w
    basis.angular = a
    prefix = f"{tag}-{kind}" if kind else tag
    offset = {s: np.load(WORK / f"{prefix}-off-{s}.npy", mmap_mode="r") for s in ACTIVE}
    stored = {
        s: np.load(WORK / f"{prefix}-sites-{s}.npy", mmap_mode="r") for s in ACTIVE
    }
    n = len(offset[1]) - 1
    cols = {s: slice(i * 256, (i + 1) * 256) for i, s in enumerate(ACTIVE)}
    temp = path.with_suffix(".pending")
    features = np.lib.format.open_memmap(temp, mode="w+", dtype=float, shape=(n, 1024))
    with ProcessPoolExecutor(max_workers=5, mp_context=mp.get_context("fork")) as pool:
        for _ in pool.map(
            batch, [(i, min(n, i + 192)) for i in range(0, n, 192)], buffersize=12
        ):
            pass
    features.flush()
    del features
    temp.replace(path)
    print("Evaluated", n, "geometries for", path.name, flush=True)
    return path


if __name__ == "__main__":
    import sys
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        cache(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
