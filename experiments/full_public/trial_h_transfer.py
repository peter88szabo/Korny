"""Fit Chebyshev and Krawtchouk corrections on smooth H-transfer connectivity.

Shared coordinate preparation keeps both experiments on exactly the same
geometries. Feature files store double precision because the force and
curvature anchors use symmetric displacements as small as 0.001 angstrom.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import pickle

import numpy as np

from experiments.full_public import kernel_trials as kt, trial_aux, trial_sites
from experiments.full_public.trial_operator_kernels import scalar_coefficients
from src.forcefield import ForceField
from src.hydrogen_transfer import HydrogenTransferKernel, POWERS

ROOT = kt.WORK / "hydrogen-transfer86"
ANGULAR = kt.WORK / "hydrogen-transfer-angular88"
STAGES = {"chebyshev": (73, 86), "krawtchouk": (82, 87)}


def bases(angular=False):
    parent = ForceField.load(kt.WORK / "p68-public-high.npz")
    result = {}
    for family in STAGES:
        coeff = np.eye(5)
        if family == "krawtchouk":
            modes = scalar_coefficients(family)
            for n in range(1, 5):
                # The balanced finite binomial chain gives a degree-n mode
                # of definite parity. Normalize at full occupancy and remove
                # only eigensolver roundoff from the forbidden coefficients.
                c = modes[n - 1, n % 2 : n + 1 : 2]
                coeff[n] = 0
                coeff[n, n % 2 : n + 1 : 2] = c / c.sum()
        window = np.cos(np.deg2rad([160.0, 110.0])) if angular else None
        result[family] = HydrogenTransferKernel(
            parent.radii, family, coeff, np.zeros(21 * len(POWERS)), window
        )
    return result


def dual_features(points, models):
    """Evaluate geometry once and both mode families on all triples in a batch."""
    first = next(iter(models.values()))
    result = {k: np.zeros((len(points), b.size)) for k, b in models.items()}
    data = []
    for index, (z, r) in enumerate(points):
        c = first.contacts(z, r)
        if c is not None:
            x, w, _, _ = first.coordinates(c)
            data.append((np.full(len(w), index), c[0], c[1], x, w))
    if data:
        owner, pair, symmetric, x, w = (np.concatenate(v) for v in zip(*data))
        cols = pair[:, None] * len(POWERS) + np.arange(len(POWERS))
        for k, b in models.items():
            vals, _ = b.modes(x, symmetric)
            np.add.at(result[k], (owner[:, None], cols), w[:, None] * vals)
    return result


def _batch(lo):
    return lo, dual_features(frames[lo : lo + 96], models)


def cache_geometries(side, points, directory):
    """Publish complete feature banks atomically for a repeatable training run."""
    global frames
    frames = points
    files = {k: directory / f"{side}-{k}.npy" for k in models}
    if all(p.exists() for p in files.values()):
        return
    mats = {
        k: np.lib.format.open_memmap(
            p.with_suffix(".pending"),
            mode="w+",
            dtype=float,
            shape=(len(points), models[k].size),
        )
        for k, p in files.items()
    }
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("fork")) as pool:
        for lo, block in pool.map(_batch, range(0, len(points), 96), buffersize=12):
            for k, v in block.items():
                mats[k][lo : lo + len(v)] = v
    for k, p in files.items():
        mats[k].flush()
        p.with_suffix(".pending").replace(p)
    frames = None
    print("Completed both H-transfer designs", side, len(points), flush=True)


def prepare(angular=False):
    global models
    directory = ANGULAR if angular else ROOT
    directory.mkdir(exist_ok=True)
    models = bases(angular)
    if not all((directory / f"main-{k}.npy").exists() for k in models):
        gs, _, _ = kt.inputs()
        with (kt.WORK / "ids.pkl").open("rb") as f:
            _, names = pickle.load(f)
        points = [(gs[k].z, gs[k].r) for k in names]
        for _, z, r, d in trial_sites.hdf():
            points.extend([(z, r), (z, r + 0.001 * d), (z, r - 0.001 * d)])
        cache_geometries("main", points, directory)
        del gs, points
    if not all((directory / f"radical-{k}.npy").exists() for k in models):
        points, start, _ = trial_aux.read_frames()
        cache_geometries("radical", points[start:], directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("prepare", "chebyshev", "krawtchouk"))
    parser.add_argument(
        "--angular",
        action="store_true",
        help="Test the 110–160 degree H-transfer window",
    )
    args = parser.parse_args()
    if args.step == "prepare":
        prepare(args.angular)
    else:
        from experiments.full_public.h_transfer_sources import fit

        fit(args.step, args.angular)
