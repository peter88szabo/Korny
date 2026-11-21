"""Prepare conservative kernel rows for the observed change of force across a saddle."""

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments.full_public import trial_path_forces as force_fit
from experiments.full_public import trial_periodic as kernel
from experiments.full_public.audit_irc_curvature import path_chord

ROOT = force_fit.SOURCE.parent / "multiscale-path66"


def encode(rx):
    """Differentiate each scalar basis function along the measured force secant."""
    _, data = path_chord(rx)
    if data is None:
        return None
    z, positions, forces, unit_displacement, chord_length = data
    step = 0.0005
    points = [(z, r + sign * step * unit_displacement)
              for r in positions for sign in (-1, 1)]
    design = np.column_stack([a[:, force_fit.cols] for a in kernel.batch(points)])
    left, right = (design[1::2] - design[::2]) / (2 * step)
    label = -np.sum((forces[1] - forces[0]) * unit_displacement) / chord_length
    return np.r_[(right - left) / chord_length, label]


def prepare():
    ROOT.mkdir(exist_ok=True)
    dest = ROOT / "training-curvatures.npz"
    if dest.is_file():
        return
    records = json.loads((force_fit.SOURCE / "snapshot.json").read_text())
    train = [r["rxn"] for r in records if not r["heldout"]]
    force_fit.initialize()
    mat = np.zeros((len(train), 2 * len(force_fit.cols) + 1))
    use = np.zeros(len(train), dtype=bool)
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context("fork")) as pool:
        for i, result in enumerate(pool.map(encode, train, chunksize=4, buffersize=8)):
            if result is not None:
                mat[i] = result
                use[i] = True
            if (i + 1) % 1000 == 0:
                print("Prepared training saddle curvatures", i + 1, len(train), flush=True)
    pending = dest.with_suffix(".pending")
    with pending.open("wb") as stream:
        np.savez(stream, ids=train, rows=mat, use=use)
    pending.replace(dest)


def transfer_to_public():
    """Calibrate energy, force-training and curvature-training increments on Grambow."""
    from experiments.full_public.trial_irc_transfer import transfer
    from experiments.full_public.trial_measured_curvature import load_basis
    from experiments.full_public.trial_path_scales import OPTIONS

    readouts = []
    records = None
    for stage in (61, 64, 67):
        model_root = ROOT.parent / f"multiscale-path{stage}"
        manifest = json.loads((model_root / "snapshot.json").read_text())
        if records is not None and manifest != records:
            raise ValueError("Native kernel increments require identical training partitions")
        records = manifest
        with np.load(model_root / "native.npz") as archive:
            readouts.append(archive["weights"][:, 0])
    energy, force, curvature = readouts
    increments = np.column_stack((energy, force - energy, curvature - force))
    banks = []
    for (width, degree), columns in zip(OPTIONS, np.split(increments, len(OPTIONS)), strict=True):
        basis, indices = load_basis()
        basis.length, basis.degree = width, degree
        banks.append((basis, indices, columns))
    transfer(["energy", "force_training_increment", "curvature_fit_increment"],
             banks, records, "p68-public", "p49-public", "p49-public")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transfer", action="store_true")
    if parser.parse_args().transfer:
        transfer_to_public()
    else:
        prepare()
