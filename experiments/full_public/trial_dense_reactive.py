"""Fit the full two-resolution kernel directly on Korny's target reactions.

P61 compresses 1,996 coefficients into a shared native energy and a handful
of family energies. This controlled alternative fits every coefficient on the
target reference. It also uses all eligible RDB19-Rad endpoint gaps and saddle
heights through a separate, regularized source-specific readout.
"""

import argparse
import json

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from experiments.full_public import kernel_trials as kt
from experiments.full_public.trial_all_transfers import ROOT as JOINT
from experiments.full_public.trial_irc_transfer import IRC
from experiments.full_public.trial_measured_curvature import load_basis
from experiments.full_public.trial_path_directions import compact_banks, fit_surfaces
from experiments.full_public.trial_path_scales import OPTIONS

ROOT = kt.WORK / "target-dense62"
NATIVE = IRC.parent / "multiscale-path61"


def inputs():
    _, selected = load_basis()
    p = len(selected)
    n = len(np.load(kt.WORK / "prior.npy", mmap_mode="r"))
    sources = np.load(kt.WORK / "hdf-sources.npy")
    files = {}
    for side in ("main", "rad"):
        files[side] = [
            np.load(kt.WORK / f"periodic-{width:g}-{side}.npy", mmap_mode="r")
            for width, _ in OPTIONS
        ]
        out = ROOT / f"{side}.npy"
        if out.exists():
            continue
        rows = n if side == "main" else len(files[side][0])
        tmp = out.with_suffix(".pending")
        result = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=float, shape=(rows, 2 * p)
        )
        for lo in range(0, rows, 256):
            sl = slice(lo, min(lo + 256, rows))
            result[sl] = np.column_stack([a[sl][:, selected] for a in files[side]])
        result.flush()
        del result
        tmp.replace(out)
    if not (ROOT / "hdf-shape.npy").exists():
        # The old within-resolution Grams are exact. Stream just the missing
        # cross terms so two kernels can cancel at the same PBE0 geometry.
        shape = np.zeros((2 * p, 2 * p))
        for j, (width, _) in enumerate(OPTIONS):
            sl = slice(j * p, (j + 1) * p)
            for factor, tag in ((0.05, "ef"), (0.012, "curvature")):
                g = np.load(
                    kt.WORK / f"periodic-{width:g}-anchor-{tag}.npy", mmap_mode="r"
                )
                shape[sl, sl] += factor * g[np.ix_(selected, selected)]
        counts = dict(zip(*np.unique(sources, return_counts=True)))
        weights = np.sqrt([4500 / counts[s] for s in sources])[:, None]
        cross = np.zeros((p, p))
        for lo in range(0, len(sources), 160):
            hi = min(lo + 160, len(sources))
            terms = []
            for a in files["main"]:
                x, v, m = (
                    a[n + 3 * lo : n + 3 * hi][:, selected]
                    .reshape(hi - lo, 3, p)
                    .transpose(1, 0, 2)
                )
                w = weights[lo:hi]
                terms.append(
                    np.vstack(
                        (
                            np.sqrt(0.05) * w * x,
                            np.sqrt(0.05) * w * (v - m) * (4 / 0.002),
                            np.sqrt(0.012) * w * ((v - x) + (m - x)) / 0.001**2,
                        )
                    )
                )
            cross += terms[0].T @ terms[1]
        shape[:p, p:] = cross
        shape[p:, :p] = cross.T
        pending = ROOT / "hdf-shape.pending"
        with pending.open("wb") as stream:
            np.save(stream, shape)
        pending.replace(ROOT / "hdf-shape.npy")
    return n, selected


def source_constraint():
    records = json.loads((NATIVE / "snapshot.json").read_text())
    rows = []
    for r in records:
        if r["heldout"]:
            continue
        rx = r["rxn"]
        with (
            np.load(NATIVE / (rx + ".npz")) as f,
            np.load(JOINT / "irc" / (rx + ".npz")) as a,
        ):
            x = f["rows"][:2].astype(float)
            prior = a["prior"]
            ref = a["reference"]
        # Every source row is compared at its original stationary coordinates.
        # Verify orientation against independently loaded endpoint references.
        np.testing.assert_allclose(
            x[:, -1],
            (ref[2] - ref[0], ref[1] - (ref[2] + ref[0]) / 2),
            atol=1e-6,
            rtol=0,
        )
        x[:, -1] -= [prior[2] - prior[0], prior[1] - (prior[2] + prior[0]) / 2]
        rows.append(x)
    a = np.vstack(rows)
    g = a[:, :-1].T @ a[:, :-1]
    rhs = a[:, :-1].T @ a[:, -1]
    # Eliminate a distinct B3LYP-D3(BJ) offset in coefficient space:
    # min_d ||A(w+d)-y||² + tau ||d||². The source and target Hamiltonians
    # retain independent coefficients, relative energy origins and references.
    tau = 2.0
    fac = cho_factor(g + tau * np.eye(len(g)))
    gram = tau * cho_solve(fac, g)
    out = np.zeros((len(g) + 1,) * 2)
    out[:-1, :-1] = 0.5 * (gram + gram.T)
    out[:-1, -1] = out[-1, :-1] = tau * cho_solve(fac, rhs)
    return out, len(rows)


def run(robust=False):
    ROOT.mkdir(exist_ok=True)
    _, selected = inputs()
    n = len(selected)
    print(
        "Prepared dense target features with both scales and the cross-resolution curvature Gram",
        2 * n,
        flush=True,
    )
    gram, paths = source_constraint()
    bases, _ = compact_banks(np.zeros((2 * n, 1)), 1)
    banks = [(b, None) for b, _ in bases]
    main = np.load(ROOT / "main.npy", mmap_mode="r")
    rad = np.load(ROOT / "rad.npy", mmap_mode="r").reshape(-1, 7, 2 * n)
    info = {"native_training_paths": paths, "columns_per_resolution": n}
    fit_surfaces(
        banks,
        main,
        rad,
        info,
        "p65-public" if robust else "p62-public",
        "robust65" if robust else "dense62",
        np.load(ROOT / "hdf-shape.npy"),
        gram,
        huber_kcal=5.0 if robust else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robust", action="store_true")
    run(parser.parse_args().robust)
