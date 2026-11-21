"""Fit a charge-conserving local model on the original AIMNet2 parent.

Every training atom contributes an absolute Hirshfeld-charge target. The
linear coefficients measure charge transfer on three smooth distance shells,
with an additional ionic response. A whole-formula holdout tests geometry and
composition transfer. Each block is read twice (fit and validation), keeping
memory use independent of the size of the HDF5 file.
"""

import argparse
import ctypes
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
from rdkit import Chem
from threadpoolctl import threadpool_limits

from experiments.full_public import aimnet_parent as parent

FIELDS = ("numbers", "coord", "charge", "charges")
WIDTH = 4
PT = Chem.GetPeriodicTable()
RADII = np.r_[0.0, [PT.GetRcovalent(i) for i in range(1, 119)]]


def library():
    src = Path(__file__).with_name("parent_charges.cpp")
    so = parent.ROOT / "parent-charges.so"
    if not so.exists() or src.stat().st_mtime > so.stat().st_mtime:
        pending = so.with_suffix(".pending")
        subprocess.run(
            [
                "c++",
                "-O3",
                "-std=c++17",
                "-shared",
                "-fPIC",
                str(src),
                "-o",
                str(pending),
            ],
            check=True,
        )
        pending.replace(so)
    func = ctypes.CDLL(str(so)).charge_basis
    func.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 7
    func.restype = None
    return func


def features(fn, z, r, q):
    """Compact per-batch design. Sum across atoms is zero for every learned column."""
    elements = np.unique(z)
    ix = np.zeros(119, np.int32)
    ix[elements] = np.arange(len(elements))
    col = (WIDTH * elements.astype(int)[:, None] + np.arange(WIDTH)).ravel()
    x = np.zeros((*z.shape, len(col)))
    base = np.empty(z.shape)
    buffers = [
        np.ascontiguousarray(z, np.uint8),
        np.ascontiguousarray(r, np.float32),
        np.ascontiguousarray(q, np.int8),
        RADII,
        ix,
        base,
        x,
    ]
    fn(*z.shape, len(col), *(a.ctypes.data for a in buffers))
    return col, x, base


def run(archive, tag):
    fn = library()
    blocked = parent.validation_formulas()
    root = parent.ROOT
    funcs = 119 * WIDTH
    # The public HDF5 is sufficiently large that checkpoints between atom-count
    # groups avoid rereading completed groups after a power interruption.
    work = root / f"charges-{tag}"
    work.mkdir(exist_ok=True)
    with h5py.File(archive, "r") as h:
        metadata = {
            "source": archive.name,
            "groups": sorted(h),
            "size": archive.stat().st_size,
            "reference": "Hirshfeld charges, closed-shell wB97M/def2-TZVPP",
            "heldout_formulas": blocked.tolist(),
        }
        stamp = work / "source.json"
        if stamp.exists() and json.loads(stamp.read_text()) != metadata:
            raise ValueError("Source or validation split changed")
        stamp.write_text(json.dumps(metadata))
        for group in sorted(h):
            checkpoint = work / f"gram-{group}.npz"
            if checkpoint.exists():
                continue
            g = np.zeros((funcs, funcs))
            rhs = np.zeros(funcs)
            ntrain = nheld = 0
            for _, lo, v in parent.blocks(h, FIELDS, [group], rows=512):
                val = parent.heldout(v["numbers"], blocked)
                ntrain += int(np.count_nonzero(~val))
                nheld += int(val.sum())
                if np.all(val):
                    continue
                z, r, q, y = (v[k][~val] for k in FIELDS)
                if np.any((np.sum(z, axis=1) - q) % 2):
                    raise ValueError("Closed-shell parity failed")
                cols, x, base = features(fn, z, r, q)
                if lo == 0:
                    np.testing.assert_allclose(x.sum(axis=1), 0, atol=2e-12)
                    np.testing.assert_allclose(base.sum(axis=1), q, atol=2e-12)
                a = x.reshape(-1, len(cols))
                target = (y - base).ravel()
                g[np.ix_(cols, cols)] += a.T @ a
                rhs[cols] += a.T @ target
            with checkpoint.with_suffix(".pending").open("wb") as f:
                np.savez_compressed(f, g=g, rhs=rhs, ntrain=ntrain, nheld=nheld)
            checkpoint.with_suffix(".pending").replace(checkpoint)
            print(
                f"Fitted charge group {group} with {ntrain:,} training and {nheld:,} validation geometries",
                flush=True,
            )
        g = np.eye(funcs) * 0.1
        rhs = np.zeros(funcs)
        ntr = nte = 0
        for file in sorted(work.glob("gram-*.npz")):
            with np.load(file) as s:
                g += s["g"]
                rhs += s["rhs"]
                ntr += int(s["ntrain"])
                nte += int(s["nheld"])
        coeff = np.linalg.solve(g, rhs)
        np.save(work / "weights.npy", coeff)
        err = 0.0
        baseline = 0.0
        ng = 0
        na = 0
        worst = 0.0
        for _, _, v in parent.blocks(h, FIELDS, rows=512):
            mask = parent.heldout(v["numbers"], blocked)
            if not np.any(mask):
                continue
            z, r, q, y = (v[k][mask] for k in FIELDS)
            cols, x, base = features(fn, z, r, q)
            pred = base + x @ coeff[cols]
            worst = max(worst, float(np.max(abs(pred.sum(axis=1) - q))))
            err += np.abs(y - pred).sum()
            baseline += np.abs(y - base).sum()
            ng += len(q)
            na += z.size
        if ng != nte or not na:
            raise ValueError("Invalid held-out atom count")
        result = {
            "training_geometries": ntr,
            "heldout_geometries": ng,
            "heldout_atoms": na,
            "mae_e": err / na,
            "baseline_mae_e": baseline / na,
            "max_charge_conservation_error_e": worst,
        }
        (work / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        print("Held-out atom charge fit", result, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=parent.ARCHIVE)
    p.add_argument("--tag", default="full")
    args = p.parse_args()
    with threadpool_limits(2):
        run(args.archive, args.tag)
