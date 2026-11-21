"""Fit a dispersion-free ωB97M energy/force readout on the AIMNet2 archive.

The radial features are the covalent-radius-scaled Gaussians of Korny's
existing reaction surface. Element offsets and charge terms absorb the source
energy zero. The energy loss uses eV per atom. Two seeded unit projections of
each full Cartesian force vector estimate the componentwise squared force
loss. The held-out pass then evaluates every Cartesian force component.
"""

import argparse
import ctypes
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
from threadpoolctl import threadpool_limits

from experiments.full_public import aimnet_parent as parent
from experiments.full_public import kernel_trials as kt
from src.forcefield import ForceField

SPECIES = np.array([1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 33, 34, 35, 53])
FIELDS = ("numbers", "coord", "charge", "energy", "forces")
CENTERS = np.array([0.75, 1.0, 1.25, 1.55, 2.05, 2.75, 3.5])
FORCE_SCALE = 2.0


class PairDesign:
    """Expose the compact source design and the exact existing Korny feature indices."""

    def __init__(self):
        self.model = ForceField.load(kt.WORK / "p43-public-high.npz")
        g = self.model.groups
        self.radius = np.ascontiguousarray(self.model.radii, float)
        pairs = np.maximum(g[:, None], g[None, :])
        pairs = pairs * (pairs + 1) // 2 + np.minimum(g[:, None], g[None, :])
        self.pair_ids = np.unique(pairs[np.ix_(SPECIES, SPECIES)])
        self.index = np.full((119, 119), -1, np.int32)
        for i, k in enumerate(self.pair_ids):
            self.index[pairs == k] = 7 * i
        self.atomic = np.full(119, -1, np.int32)
        n = 7 * len(self.pair_ids)
        self.atomic[SPECIES] = np.arange(n, n + len(SPECIES))
        self.width = n + len(SPECIES) + 2
        grid = np.arange(6 * 1024 + 2) / 1024.0
        u = (grid[:, None] - CENTERS) / 0.22
        ff = np.exp(-0.5 * u * u)
        self.table = np.ascontiguousarray(np.column_stack((ff, -u * ff / 0.22)))
        src = Path(__file__).with_name("parent_potential.cpp")
        dest = parent.ROOT / "parent-potential.so"
        if not dest.exists() or src.stat().st_mtime > dest.stat().st_mtime:
            temp = dest.with_suffix(".pending")
            subprocess.run(
                [
                    "c++",
                    "-O3",
                    "-std=c++17",
                    "-fPIC",
                    "-shared",
                    str(src),
                    "-o",
                    str(temp),
                ],
                check=True,
            )
            temp.replace(dest)
        self.fn = ctypes.CDLL(str(dest)).parent_pair
        self.fn.argtypes = [ctypes.c_int] * 6 + [ctypes.c_void_p] * 13
        self.fn.restype = None

    def design(self, data, offset, coeff=None):
        z, r, q, e, f = (data[k] for k in FIELDS)
        if np.any(self.atomic[z] < 0) or np.any((np.sum(z, axis=1) - q) % 2):
            raise ValueError("Unexpected species or spin in the source")
        n, a = z.shape
        nd = 2 if coeff is None else 0
        x = np.zeros((n, nd + 1, self.width))
        y = np.zeros((n, nd + 1))
        out = np.zeros((n, a, 3)) if coeff is not None else np.empty(1)
        buffers = [
            np.ascontiguousarray(z, np.uint8),
            np.ascontiguousarray(r, np.float32),
            np.ascontiguousarray(q, np.int8),
            np.ascontiguousarray(f, np.float32),
            np.ascontiguousarray(e, float),
            self.index,
            self.atomic,
            self.radius,
            self.table,
            np.zeros(self.width) if coeff is None else coeff,
            x,
            y,
            out,
        ]
        self.fn(
            n,
            a,
            self.width,
            nd,
            offset,
            int(coeff is not None),
            *(v.ctypes.data for v in buffers),
        )
        return x, y, out


def run(path, tag):
    design = PairDesign()
    blocked = parent.validation_formulas()
    work = parent.ROOT / f"potential-{tag}"
    work.mkdir(exist_ok=True)
    totals = [0, 0]
    with h5py.File(path, "r") as h:
        identity = {
            "archive": path.name,
            "bytes": path.stat().st_size,
            "groups": sorted(h),
            "protected_formulas": blocked.tolist(),
        }
        marker = work / "source.json"
        if marker.exists() and json.loads(marker.read_text()) != identity:
            raise ValueError("Incompatible auxiliary cache")
        marker.write_text(json.dumps(identity))
        first = True
        for group in sorted(h):
            out = work / f"gram-{group}.npz"
            if out.exists():
                continue
            gram = np.zeros((design.width, design.width))
            rhs = np.zeros(design.width)
            ntr = nte = 0
            for _, lo, data in parent.blocks(h, FIELDS, [group], rows=1024):
                held = parent.heldout(data["numbers"], blocked)
                x, y, _ = design.design(data, lo)
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    raise ValueError(f"Invalid energy/force record {group}:{lo}")
                if first:
                    z, r = data["numbers"][0], data["coord"][0].astype(float)
                    row, _ = design.model.design(z, r, int(data["charge"][0]), 1, False)
                    pos = 1 + len(design.model.elements)
                    exact = row[pos : pos + 315].reshape(45, 7)[design.pair_ids].ravel()
                    np.testing.assert_allclose(
                        x[0, 0, : len(exact)], exact, atol=3.0e-6, rtol=0
                    )
                    rng = np.random.default_rng(1329)
                    coeff = rng.normal(size=design.width)
                    single = {k: v[:1].copy() for k, v in data.items()}
                    _, _, f = design.design(single, lo, coeff)
                    v = rng.normal(size=r.shape)
                    v /= np.linalg.norm(v)
                    delta = 1.0e-3
                    # Reference coordinates are float32. Use the actual rounded
                    # displacement for the force comparison at this precision.
                    plus = (r + delta * v).astype(np.float32)
                    minus = (r - delta * v).astype(np.float32)
                    e = []
                    for rr in (minus, plus):
                        single["coord"] = rr[None]
                        a, _, _ = design.design(single, lo, coeff)
                        e.append(a[0, 0] @ coeff)
                    np.testing.assert_allclose(
                        (e[0] - e[1]) / (2 * delta),
                        np.sum(f[0] * (plus - minus)) / (2 * delta),
                        atol=6.0e-4,
                        rtol=0,
                    )
                    first = False
                x = x[~held]
                y = y[~held]
                ntr += len(y)
                nte += int(held.sum())
                if not len(y):
                    continue
                weight = np.array(
                    [
                        1.0 / int(group),
                        FORCE_SCALE / np.sqrt(2.0),
                        FORCE_SCALE / np.sqrt(2.0),
                    ]
                )
                a = (x * weight[None, :, None]).reshape(-1, design.width)
                v = (y * weight).ravel()
                gram += a.T @ a
                rhs += a.T @ v
            with out.with_suffix(".pending").open("wb") as f:
                np.savez_compressed(f, gram=gram, rhs=rhs, ntr=ntr, nte=nte)
            out.with_suffix(".pending").replace(out)
            print("Fitted bare energy and force group", group, ntr, nte, flush=True)
        g = np.eye(design.width) * 1.0e-7
        rhs = np.zeros(design.width)
        for fn in sorted(work.glob("gram-*.npz")):
            with np.load(fn) as a:
                g += a["gram"]
                rhs += a["rhs"]
                totals[0] += int(a["ntr"])
                totals[1] += int(a["nte"])
        weights = np.linalg.solve(g, rhs)
        np.savez(
            work / "weights.npz",
            weights=weights,
            pair_ids=design.pair_ids,
            species=SPECIES,
        )
        sums = np.zeros(4)
        natom = 0
        ngeom = 0
        for _, lo, data in parent.blocks(h, FIELDS, rows=1024):
            mask = parent.heldout(data["numbers"], blocked)
            if not np.any(mask):
                continue
            val = {k: v[mask] for k, v in data.items()}
            x, y, f = design.design(val, lo, weights)
            err = (x[:, 0] @ weights) - y[:, 0]
            ferr = f - val["forces"]
            sums += np.array(
                [
                    abs(err).sum(),
                    abs(err).sum() / val["numbers"].shape[1],
                    abs(ferr).sum(),
                    (ferr * ferr).sum(),
                ]
            )
            natom += val["numbers"].size
            ngeom += len(err)
        if ngeom != totals[1]:
            raise ValueError("Holdout count changed")
        stats = {
            "level": "bare wB97M/def2-TZVPP",
            "training_geometries": totals[0],
            "validation_geometries": ngeom,
            "validation_atoms": natom,
            "total_energy_mae_eV": sums[0] / ngeom,
            "per_atom_energy_mae_eV": sums[1] / ngeom,
            "force_component_mae_eV_A": sums[2] / (3 * natom),
            "force_component_rmse_eV_A": np.sqrt(sums[3] / (3 * natom)),
        }
        kt.write_json(work / "metrics.json", stats)
        print("Bare energy/force validation", stats, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=parent.ARCHIVE)
    p.add_argument("--tag", default="full")
    args = p.parse_args()
    with threadpool_limits(3):
        run(args.archive, args.tag)
