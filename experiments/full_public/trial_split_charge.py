"""Compare a variational charge-response correction with the frozen p68 parent."""

import argparse
import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments.full_public import kernel_trials as kt, trial_aux, trial_sites
from experiments.full_public.trial_path_directions import fit_surfaces
from src.forcefield import ForceField
from src.split_charge import SplitChargeResponse

ROOT = kt.WORK / "split-charge72"


def batch(lo):
    return lo, np.array([basis.design(z, r)[0] for z, r in frames[lo : lo + 64]])


def encode(name, geometries):
    global frames
    dest = ROOT / f"{name}.npy"
    if not dest.is_file():
        frames = geometries
        pending = dest.with_suffix(".pending")
        mat = np.lib.format.open_memmap(
            pending, mode="w+", dtype=float, shape=(len(frames), basis.size)
        )
        with ProcessPoolExecutor(
            max_workers=2, mp_context=mp.get_context("fork")
        ) as pool:
            for lo, x in pool.map(
                batch, range(0, len(frames), 64), chunksize=2, buffersize=8
            ):
                mat[lo : lo + len(x)] = x
        mat.flush()
        del mat
        pending.replace(dest)
        frames = None
    return np.load(dest, mmap_mode="r")


def run(morse=False):
    global basis, ROOT
    ROOT = kt.WORK / ("morse-bends84" if morse else "split-charge72")
    ROOT.mkdir(exist_ok=True)
    parent = ForceField.load(kt.WORK / "p68-public-high.npz")
    groups = parent.groups.copy()
    groups[1] = 0
    if morse:
        from src.morse_bends import MorseBendBasis

        basis = MorseBendBasis(groups, parent.radii, np.zeros(342))
    else:
        basis = SplitChargeResponse(groups, parent.radii, np.zeros(105))
    gs, _, _ = kt.inputs()
    with (kt.WORK / "ids.pkl").open("rb") as f:
        _, names = pickle.load(f)
    main = [(gs[k].z, gs[k].r) for k in names]
    for _, z, r, d in trial_sites.hdf():
        main.extend([(z, r), (z, r + 0.001 * d), (z, r - 0.001 * d)])
    mat = encode("main", main)
    del main, gs
    geometries, start, _ = trial_aux.read_frames()
    rad = encode("rad", geometries[start:]).reshape(-1, 7, basis.size)
    del geometries
    stage, key = (84, "morse84") if morse else (72, "charge72")
    fit_surfaces(
        [(basis, None)],
        mat,
        rad,
        {"response_features": basis.size},
        f"p{stage}-public",
        key,
        parent_stage="p68-public",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--morse-bends", action="store_true")
    run(parser.parse_args().morse_bends)
