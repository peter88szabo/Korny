"""Prepare short-range reaction features from the 2021 single-electron energy fits.

For each element group, standardize donor and acceptor environment scores on
the Ions-16 training partition. The resulting smooth, distance-limited pair
features transfer the measured vertical energy data to neutral reactions too.
Atoms from validation molecules influence neither score nor scale.
"""

import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.full_public import kernel_trials as kt
from experiments.full_public import trial_aux, trial_sites
from experiments.full_public.trial_spin_population import CACHE
from experiments.full_public.trial_spin_population import ROOT as IONS
from experiments.full_public.trial_vertical_gaps import DEST as GAPS
from src.redox_environment import DIMENSION, RedoxEnvironmentKernel
from src.spin_local import DIM, GROUPS, SpinLocalKernel

ROOT = kt.OUT.parent / "paper_references/redox-environments"
BATCH = 256


def build():
    with np.load(GAPS / "weights.npz") as f:
        maps = np.stack(
            [f[k][:-2].reshape(2, GROUPS, DIM).sum(axis=0) for k in ("1", "-1")],
            axis=-1,
        )
    n = np.zeros(GROUPS)
    totals = np.zeros((GROUPS, 2))
    squared = totals.copy()
    for file in sorted(CACHE.glob("ions16-*.npz")):
        with np.load(file) as a:
            keep = a["part"] == 0
            x = a["x"][keep]
            t = a["types"][keep]
        for k in np.unique(t):
            v = x[t == k] @ maps[k]
            n[k] += len(v)
            totals[k] += v.sum(0)
            squared[k] += (v * v).sum(0)
    mean = totals / np.maximum(n, 1)[:, None]
    scale = np.sqrt(
        np.maximum(0.0, squared / np.maximum(n, 1)[:, None] - mean**2) + 0.15**2
    )
    with np.load(IONS / "spin-population.npz") as f:
        prior = SpinLocalKernel.load(f)
    s = RedoxEnvironmentKernel(
        prior.groups,
        prior.radii,
        maps / scale[:, None, :],
        -mean / scale,
        np.zeros(DIMENSION),
    )
    np.savez_compressed(ROOT / "basis.npz", **s.fields())
    return s


def block(job):
    name, lo = job
    path = ROOT / f"{name}-{lo:07d}.npy"
    if path.is_file():
        return
    values = np.vstack(
        [basis.design(z, r)[0] for z, r in frames[name][lo : lo + BATCH]]
    )
    with path.with_suffix(".pending").open("wb") as f:
        np.save(f, values)
    path.with_suffix(".pending").replace(path)


def prepare():
    global basis, frames
    ROOT.mkdir(exist_ok=True)
    basis = build()
    g, _, _ = kt.inputs()
    with (kt.WORK / "ids.pkl").open("rb") as f:
        _, ids = pickle.load(f)
    geom = [(g[k].z, g[k].r) for k in ids]
    for _, z, r, v in trial_sites.hdf():
        geom.extend([(z, r), (z, r + 0.001 * v), (z, r - 0.001 * v)])
    rad, nprivate, _ = trial_aux.read_frames()
    frames = {"main": geom, "rad": rad[nprivate:]}
    dims = {}
    jobs = []
    for key, rows in frames.items():
        dims[key] = len(rows)
        jobs.extend((key, i) for i in range(0, len(rows), BATCH))
    with ProcessPoolExecutor(mp_context=mp.get_context("fork"), max_workers=2) as pool:
        for i, _ in enumerate(pool.map(block, jobs, buffersize=6), 1):
            if i % 200 == 0:
                print("Encoded local redox feature blocks", i, len(jobs), flush=True)
    for key, num in dims.items():
        path = ROOT / f"{key}.npy"
        arr = np.lib.format.open_memmap(
            path.with_suffix(".pending"), "w+", float, (num, DIMENSION)
        )
        for lo in range(0, num, BATCH):
            arr[lo : lo + BATCH] = np.load(ROOT / f"{key}-{lo:07d}.npy")
        arr.flush()
        del arr
        path.with_suffix(".pending").replace(path)
    kt.write_json(ROOT / "rows.json", dims)
    print("Ready local donor-acceptor reaction geometry features", dims, flush=True)


if __name__ == "__main__":
    with threadpool_limits(1):
        prepare()
