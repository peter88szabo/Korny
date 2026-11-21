"""Read equilibrium X-H bond lengths and stretching force constants from archived RDB7 Hessians."""

from collections import Counter
import hashlib
import pickle
import sqlite3
import numpy as np
from rdkit import Chem
from experiments.full_public import kernel_trials as kt
from experiments.full_public.trial_messages import prototypes
from src.bond_order_kernel import environments
from src.bond_environment import PAIRS, ELEMENTS
from src.four_body import RADII
from src.local_stretch import StretchKernel, FEATURE_SCALE

WORK = kt.WORK
ROOT = WORK / "stretch39"
KEYS = (PAIRS.index((1, 6)), PAIRS.index((1, 7)), PAIRS.index((1, 8)))


def measurements():
    d = kt.equation_data("high")
    # Hold out entire benchmark and inner selection formulas. A shared molecule
    # inherits all its occurrences, including any held-out reaction equation.
    deny = set(map(int, d[0][d[3] < 2].indices))
    records = []
    seen = set()
    counts = Counter()
    with sqlite3.connect(
        f"file:{WORK / 'rdb-dft-frames.sqlite'}?mode=ro", uri=True
    ) as db:
        for name, idx, bz, br, bh in db.execute(
            "select name,coord_id,z,r,hessian from stationary where converged=1 and hessian is not null and name not like '%/ts%' order by name"
        ):
            if idx in seen or idx in deny:
                continue
            seen.add(idx)
            z = np.frombuffer(bz, np.uint8)
            r = np.frombuffer(br, float).reshape(-1, 3)
            h = np.frombuffer(bh, float).reshape(r.size, r.size)
            keep = np.flatnonzero(np.isin(z, ELEMENTS))
            a, b = np.triu_indices(len(keep), 1)
            a, b = keep[a], keep[b]
            d = np.linalg.norm(r[b] - r[a], axis=1)
            scale = np.array(
                [RADII[int(i)] + RADII[int(j)] for i, j in zip(z[a], z[b])]
            )
            use = d < 1.95 * scale
            a, b = a[use], b[use]
            bonds = []
            for k, x, gate, _, _, indices, _ in environments(z, r):
                if k not in KEYS:
                    continue
                for feat, edge in zip(x, indices):
                    if not 0.84 < 1 + feat[0] < 1.16:
                        continue
                    i, j = int(a[edge]), int(b[edge])
                    if z[i] != 1:
                        i, j = j, i
                    length = np.linalg.norm(r[i] - r[j])
                    axis = (r[i] - r[j]) / length
                    masses = np.array(
                        [
                            Chem.GetPeriodicTable().GetMostCommonIsotopeMass(int(s))
                            for s in (z[i], z[j])
                        ]
                    )
                    v = np.zeros_like(r)
                    v[i] = axis * masses[1] / masses.sum()
                    v[j] = -axis * masses[0] / masses.sum()
                    spring = float(v.ravel() @ h @ v.ravel())
                    if not np.isfinite(spring) or spring <= 0:
                        continue
                    bonds.append((k, feat[1:], length, spring, i, j))
                    counts[str(PAIRS[k])] += 1
            if bonds:
                records.append((name, idx, z, r, bonds))
    with (ROOT / "measurements.pkl").open("wb") as f:
        pickle.dump(records, f)
    kt.write_json(
        ROOT / "measurements.json", dict(geometries=len(records), bonds=counts)
    )
    print(
        "Measured equilibrium stretches",
        len(records),
        "structures",
        dict(counts),
        flush=True,
    )
    return records


def basis(records):
    rng = np.random.default_rng(3913)
    centers = {}
    lengths = {}
    diagnostics = {}
    for k in KEYS:
        x = []
        y = []
        hold = []
        for name, idx, z, r, bonds in records:
            # Compare equilibrium-length interpolation on separate molecular formulas.
            formula = tuple(sorted(map(int, z)))
            h = (
                int.from_bytes(
                    hashlib.sha256(str(formula).encode()).digest()[:4], "big"
                )
                % 5
                == 0
            )
            for pair, env, l, spring, *_ in bonds:
                if k == pair:
                    x.append(env)
                    y.append(l)
                    hold.append(h)
        x = np.array(x)
        y = np.array(y)
        hold = np.array(hold)
        xx = x[~hold]
        if len(xx) > 14000:
            xx = xx[rng.choice(len(xx), 14000, replace=False)]
        centers[k] = (
            prototypes(xx * FEATURE_SCALE, min(120, len(xx)), rng) / FEATURE_SCALE
        )
        proto = centers[k]

        def readout(data):
            raw = []
            for lo in range(0, len(data), 512):
                delta = (data[lo : lo + 512, None, :] - proto) * FEATURE_SCALE
                w = np.exp(-0.5 * np.einsum("bcd,bcd->bc", delta, delta) / 0.80**2)
                raw.append(w / (0.002 + w.sum(axis=1))[:, None])
            return np.vstack(raw) if raw else np.empty((0, len(proto)))

        a = readout(x[~hold])
        prior = float(np.median(y[~hold]))
        g = a.T @ a + 0.45 * np.eye(len(proto))
        lengths[k] = np.clip(
            prior + np.linalg.solve(g, a.T @ (y[~hold] - prior)),
            np.percentile(y[~hold], 0.25) - 0.025,
            np.percentile(y[~hold], 99.75) + 0.025,
        )
        sample = readout(x[hold])
        val = prior + sample @ (lengths[k] - prior)
        diagnostics[str(PAIRS[k])] = dict(
            n=int(hold.sum()),
            mae_A=float(abs(val - y[hold]).mean()) if len(val) else None,
            centers=len(proto),
        )
    b = StretchKernel(centers, lengths, 0.11, np.zeros(sum(map(len, centers.values()))))
    np.savez_compressed(ROOT / "basis.npz", **b.fields())
    kt.write_json(ROOT / "length-validation.json", diagnostics)
    print(
        "Bond-length diagnostic on reserved training formulas", diagnostics, flush=True
    )


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True)
    if (ROOT / "measurements.pkl").exists():
        with (ROOT / "measurements.pkl").open("rb") as f:
            records = pickle.load(f)
    else:
        records = measurements()
    basis(records)
