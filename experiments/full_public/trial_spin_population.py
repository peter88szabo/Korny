"""Train charge-state-specific spin redistribution on published atomic labels.

Use Ions-16 doublet α/β NBO populations and original Korny training radical
centers for neutral doublets. Reserve every ChEMBL-20 formula, every existing
Korny validation formula and an independent fraction of the remaining
Ions-16 formulas. ChEMBL-20 measures transfer to a separately published set.
The neutral radical target identifies a Lewis radical center. Its one-hot
labels have a different physical interpretation from signed NBO populations.
"""

import json
import multiprocessing as mp
import pickle
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from experiments.full_public import aimnet_parent as ref
from experiments.full_public import kernel_trials as kt
from src.forcefield import ForceField
from src.spin_local import DIM, GROUPS, SpinLocalKernel

ROOT = kt.OUT.parent / "AIMNet_NSE_2021"
CACHE = ROOT / "spin-sites"
WIDTH = GROUPS * DIM


def init():
    global empty
    ff = ForceField.load(kt.WORK / "p43-public-high.npz")
    groups = ff.groups.copy()
    groups[1] = 0
    empty = SpinLocalKernel(
        groups,
        ff.radii,
        np.zeros((3, GROUPS, DIM)),
        np.zeros((3, GROUPS, DIM)),
        np.zeros(WIDTH),
    )


def external(job):
    source, group, lo = job
    dest = CACHE / f"{source}-{group}-{lo:06d}.npz"
    if dest.exists():
        return
    with h5py.File(ROOT / (source + ".h5"), "r") as h:
        g = h[group]
        sl = slice(lo, lo + 480)
        z, q, m, r, y, ids = (
            g[k][sl]
            for k in ("numbers", "charge", "mult", "coord", "charges", "mol_id")
        )
    keep = m == 2
    z, q, r, y, ids = (v[keep] for v in (z, q, r, y, ids))
    if len(q):
        if np.any(abs(q) != 1):
            raise ValueError("Unrecognized published doublet charge")
        if np.max(abs(y.sum(axis=(1, 2)) - q)) > 0.002:
            raise ValueError("NBO charges disagree with molecular charge")
        spin = (y[:, :, 1] - y[:, :, 0]).astype(float)
        if np.max(abs(spin.sum(axis=1) - 1.0)) > 0.002:
            raise ValueError("NBO spin difference disagrees with multiplicity")
        types = empty.groups[z]
        values = np.empty((*z.shape, DIM), np.float32)
        loc = {}
        for i, key in enumerate(ids):
            n = loc.get(key)
            if (
                n is None
                or not np.array_equal(r[i], r[n])
                or not np.array_equal(z[i], z[n])
            ):
                values[i] = empty.sites(z[i], r[i])[1]
                loc[key] = i
            else:
                values[i] = values[n]
        held = ref.heldout(z, protected)
        part = np.full(len(z), 2) if source == "chembl20" else held.astype(int)
    else:
        types = np.empty((0, int(group[1:])), np.uint8)
        values = np.empty((*types.shape, DIM), np.float32)
        spin = np.empty(types.shape)
        part = np.empty(0, int)
    with dest.with_suffix(".pending").open("wb") as f:
        np.savez_compressed(
            f,
            types=types.astype(np.uint8),
            x=values,
            y=spin.astype(np.float32),
            q=q,
            part=part.astype(np.uint8),
        )
    dest.with_suffix(".pending").replace(dest)


def neutral():
    with (kt.WORK / "atom-state-labels.pkl").open("rb") as f:
        records = pickle.load(f)
    group = defaultdict(list)
    for _, split, z, r, q, m, fields in records:
        if m != 2 or q != 0:
            continue
        y = fields[:, 1]
        if not np.isclose(y.sum(), 1.0) or np.any(y < 0):
            continue
        types, x, _ = empty.sites(z, r)
        group[len(z)].append((types, x, y, 0 if split == "fit" else 3))
    for size, records in group.items():
        types, x, y, part = map(np.asarray, zip(*records, strict=True))
        dest = CACHE / f"neutral-{size:03d}.npz"
        with dest.with_suffix(".pending").open("wb") as f:
            np.savez_compressed(
                f,
                types=types.astype(np.uint8),
                x=x.astype(np.float32),
                y=y.astype(np.float32),
                q=np.zeros(len(y), np.int8),
                part=part.astype(np.uint8),
            )
        dest.with_suffix(".pending").replace(dest)


def load_sources():
    global protected
    init()
    CACHE.mkdir(exist_ok=True)
    blocked = set(ref.validation_formulas())
    jobs = []
    for source in ("chembl20", "ions16"):
        with h5py.File(ROOT / (source + ".h5"), "r") as h:
            for k in sorted(h):
                if source == "chembl20":
                    blocked.update(ref.formula_keys(h[k]["numbers"][:]))
                jobs.extend(
                    (source, k, lo) for lo in range(0, len(h[k]["charge"]), 480)
                )
    protected = np.array(sorted(blocked), np.uint64)
    stamp = CACHE / "protected-formulas.json"
    if stamp.is_file() and json.loads(stamp.read_text()) != protected.tolist():
        raise ValueError("Formula exclusions changed")
    stamp.write_text(json.dumps(protected.tolist()))
    neutral()
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context("fork")) as pool:
        for i, _ in enumerate(pool.map(external, jobs, buffersize=8)):
            if (i + 1) % 200 == 0:
                print(
                    "Encoded published atomic-spin data blocks",
                    i + 1,
                    len(jobs),
                    flush=True,
                )


def batches():
    for file in sorted(CACHE.glob("*.npz")):
        with np.load(file) as data:
            if len(data["q"]):
                yield tuple(data[k] for k in ("types", "x", "y", "q", "part"))


def probabilities(t, x, w):
    v = np.einsum("nad,nad->na", x, w[t])
    v -= v.max(axis=1, keepdims=True)
    p = np.exp(v)
    return p / p.sum(axis=1, keepdims=True)


def train():
    load_sources()
    lhs = np.zeros((3, GROUPS, DIM, DIM))
    rhs = np.zeros((3, GROUPS, DIM))
    counts = Counter()
    for t, x, y, q, part in batches():
        for row in range(3):
            take = (q + 1 == row) & (part == 0)
            if not np.any(take):
                continue
            ts = t[take]
            xs = x[take]
            ys = np.maximum(0, y[take]) + 0.05 / x.shape[1]
            logp = np.log(ys / ys.sum(axis=1, keepdims=True))
            counts[row] += len(xs)
            for g in np.unique(ts):
                use = ts == g
                a = xs[use].astype(float)
                lhs[row, g] += a.T @ a
                rhs[row, g] += a.T @ logp[use]
    logits = np.zeros_like(rhs)
    for k, g in np.ndindex(3, GROUPS):
        # A weak prior for unseen group scores gives a small positive
        # allocation without changing the molecular total.
        prior = np.zeros(DIM)
        prior[0] = -4.0
        logits[k, g] = np.linalg.solve(
            lhs[k, g] + 4 * np.eye(DIM), rhs[k, g] + 4 * prior
        )
    # Weighted centering allows signed polarization while conserving total
    # spin exactly. Fit it only to the actual signed NBO labels for ions.
    gram = np.tile(np.eye(WIDTH)[None, :, :] * 3.0, (3, 1, 1))
    cross = np.zeros((3, WIDTH))
    for t, x, y, q, part in batches():
        for k in (0, 2):
            select = (q + 1 == k) & (part == 0)
            for subset in np.array_split(
                np.flatnonzero(select), max(1, (select.sum() + 23) // 24)
            ):
                if not len(subset):
                    continue
                ts = t[subset]
                xs = x[subset]
                ys = y[subset]
                p = probabilities(ts, xs, logits[k])
                # Build a compact basis only for chemical groups in this block.
                groups, inv = np.unique(ts, return_inverse=True)
                idx = (DIM * groups.astype(int)[:, None] + np.arange(DIM)).ravel()
                site = np.zeros((*ts.shape, len(groups), DIM))
                mol, atom = np.indices(ts.shape)
                site[mol, atom, inv.reshape(ts.shape)] = xs
                site = site.reshape(*ts.shape, -1)
                design = p[:, :, None] * (
                    site - np.einsum("na,nad->nd", p, site)[:, None, :]
                )
                a = design.reshape(-1, len(idx))
                gram[k][np.ix_(idx, idx)] += a.T @ a
                cross[k, idx] += a.T @ (ys - p).ravel()
    pol = np.zeros_like(logits)
    for k in (0, 2):
        pol[k] = cho_solve(cho_factor(gram[k]), cross[k]).reshape(GROUPS, DIM)
    model = SpinLocalKernel(empty.groups, empty.radii, logits, pol, np.zeros(WIDTH))
    dest = ROOT / "spin-population.npz"
    np.savez_compressed(dest, **model.fields())
    totals = defaultdict(lambda: np.zeros(7))
    splits = Counter()
    for t, x, y, q, part in batches():
        for k in range(3):
            for which in (0, 1, 2, 3):
                use = (q + 1 == k) & (part == which)
                splits[k, which] += int(use.sum())
                if which == 0 or not np.any(use):
                    continue
                ts = t[use]
                xs = x[use]
                ys = y[use]
                p = probabilities(ts, xs, logits[k])
                h = np.einsum("nad,nad->na", xs, pol[k, ts])
                pred = p * (1.0 + h - (p * h).sum(axis=1, keepdims=True))
                np.testing.assert_allclose(pred.sum(1), 1.0, atol=3.0e-12)
                v = totals[k, which]
                v += np.array(
                    [
                        len(p),
                        p.size,
                        abs(pred - ys).sum(),
                        abs(1.0 / p.shape[1] - ys).sum(),
                        np.count_nonzero(pred.argmax(1) == ys.argmax(1)),
                        np.count_nonzero(ys < -0.01),
                        np.sum((pred - ys) ** 2),
                    ]
                )
    info = []
    for (k, s), (n, atoms, err, base, top, neg, sq) in totals.items():
        info.append(
            {
                "system_charge": k - 1,
                "split": {
                    1: "Ions-16 formula holdout",
                    2: "ChEMBL-20 independent",
                    3: "neutral radical centers",
                }[s],
                "geometries": int(n),
                "atoms": int(atoms),
                "mae_e": err / atoms,
                "uniform_mae_e": base / atoms,
                "rmse_e": float(np.sqrt(sq / atoms)),
                "top1_fraction": top / n,
                "negative_atoms": int(neg),
            }
        )
    kt.write_json(
        ROOT / "spin-validation.json",
        {
            "training_rows_by_charge": {str(k - 1): n for k, n in counts.items()},
            "metrics": info,
        },
    )
    print("Spin-projection validation", info, flush=True)


if __name__ == "__main__":
    with threadpool_limits(2):
        train()
