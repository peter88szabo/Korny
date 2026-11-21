"""Fit a one-step spin redistribution and test its contribution to doublet barriers.

First learn atom-to-atom spin transfer from Ions-16 NBO populations and the
training neutral Lewis centers. Then fit an additive conservative energy to the
ordinary reaction and derivative labels. Every singlet feature is identically
zero. Both fits enforce the existing molecular/formula validation exclusions.
"""

import argparse
import pickle
from collections import Counter, defaultdict

import h5py
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from experiments.full_public import kernel_trials as kt
from experiments.full_public import trial_fit
from experiments.full_public import trial_multiscale as ms
from experiments.full_public import trial_spin_population as populations
from experiments.full_public import trial_spin_transfer as transfer
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_isomers import gram as isomers
from experiments.full_public.trial_redox_fit import external
from experiments.full_public.trial_spectra import evaluate_trial
from experiments.full_public.trial_validation import run as validate
from src.forcefield import ForceField
from src.spin_exchange import PAIRS, RADII, SpinExchangeKernel, population_design
from src.spin_local import SpinLocalKernel

WORK = kt.WORK
ROOT = populations.ROOT / "neighbor-spin55"
STAGE = "p55-public"
PARENT = "p49-public"


def atom_records(basis, training):
    """Stream only the requested split; verify cached rows against original HDF5."""
    for source in ("ions16", "chembl20"):
        if training and source == "chembl20":
            continue
        with h5py.File(populations.ROOT / (source + ".h5"), "r") as h:
            for group in sorted(h):
                g = h[group]
                for lo in range(0, len(g["charge"]), 480):
                    dest = populations.CACHE / f"{source}-{group}-{lo:06d}.npz"
                    with np.load(dest) as f:
                        part = f["part"]
                        use = np.flatnonzero((part == 0) if training else (part != 0))
                        if not len(use):
                            continue
                        types, x, spin, charge = (
                            f[k][use] for k in ("types", "x", "y", "q")
                        )
                    sl = slice(lo, lo + 480)
                    take = np.flatnonzero(g["mult"][sl] == 2)[use]
                    numbers, positions, ref = (
                        g[k][sl][take] for k in ("numbers", "coord", "charges")
                    )
                    np.testing.assert_array_equal(types, basis.groups[numbers])
                    np.testing.assert_array_equal(charge, g["charge"][sl][take])
                    np.testing.assert_allclose(
                        spin, ref[:, :, 1] - ref[:, :, 0], atol=1e-6
                    )
                    yield from zip(
                        numbers,
                        positions,
                        charge,
                        types,
                        x,
                        spin,
                        part[use],
                        strict=True,
                    )
    with (WORK / "atom-state-labels.pkl").open("rb") as f:
        neutral = pickle.load(f)
    for _, split, z, r, q, m, fields in neutral:
        y = fields[:, 1]
        if (
            q
            or m != 2
            or (split == "fit") != training
            or not np.isclose(y.sum(), 1.0)
            or np.any(y < 0)
        ):
            continue
        types, x, _ = basis.sites(z, r)
        yield z, r, q, types, x, y, 0 if training else 3


def train_atoms():
    ROOT.mkdir(exist_ok=True)
    with np.load(populations.ROOT / "spin-population.npz") as f:
        local = SpinLocalKernel.load(f)
    width = PAIRS * len(RADII)
    grams = np.zeros((3, width + 1, width + 1))
    counts = Counter()
    pending = defaultdict(list)

    def flush(k):
        if pending[k]:
            a = np.vstack(pending[k])
            grams[k] += a.T @ a
            pending[k].clear()

    for z, r, q, types, x, y, _ in atom_records(local, True):
        k = local.charge_channel(q)
        s, _ = local.population(types, x, q, 2)
        a = population_design(types, z, r, local.radii, s)
        pending[k].append(np.column_stack((a, y - s)))
        counts[int(q)] += 1
        if len(pending[k]) == 128:
            flush(k)
    for k in range(3):
        flush(k)

    def solve(g, prior, ridge):
        return cho_solve(
            cho_factor(g[:-1, :-1] + ridge * np.eye(width)), g[:-1, -1] + ridge * prior
        )

    # NBO populations share a reference convention across charged molecules.
    # Neutral Lewis sites are categorical; shrink their independent readout
    # toward the ion-trained one instead of mixing the two target definitions.
    shared = solve(grams[0] + grams[2], np.zeros(width), 4.0)
    exchange = np.stack([solve(g, shared, 8.0) for g in grams]).reshape(3, PAIRS, -1)
    basis = SpinExchangeKernel(
        local.groups,
        local.radii,
        local.logits,
        local.polarization,
        np.zeros(local.size),
        exchange,
    )
    metrics = defaultdict(lambda: np.zeros(6))
    for z, r, q, types, x, y, part in atom_records(local, False):
        s, _ = local.population(types, x, q, 2)
        ds, _ = basis.redistribution(z, r, types, s, q)
        np.testing.assert_allclose(ds.sum(), 0.0, atol=1e-10)
        pred = s + ds
        metrics[int(part), int(q)] += [
            len(s),
            1,
            np.abs(s - y).sum(),
            np.abs(pred - y).sum(),
            int(s.argmax() == y.argmax()),
            int(pred.argmax() == y.argmax()),
        ]
    names = {1: "Ions-16 held formulas", 2: "ChEMBL-20", 3: "neutral radical centers"}
    out = [
        {
            "split": names[p],
            "charge": q,
            "geometries": int(n),
            "atoms": int(na),
            "old_mae": ea / na,
            "new_mae": eb / na,
            "old_top1": aa / n,
            "new_top1": ab / n,
        }
        for (p, q), (na, n, ea, eb, aa, ab) in sorted(metrics.items())
    ]
    np.savez_compressed(ROOT / "basis.npz", **basis.fields())
    kt.write_json(
        ROOT / "populations.json", {"training_by_charge": counts, "held": out}
    )
    print("Completed atomic spin fit", dict(counts), out, flush=True)


def fit():
    with np.load(ROOT / "basis.npz") as f:
        basis = SpinExchangeKernel.load(f)

    def encode(rows):
        return np.vstack([basis.design_state(z, r, q, m)[0] for z, r, q, m in rows])

    fit_doublets(basis, encode, "exchange55", STAGE, ROOT)


def fit_doublets(basis, encode, key, stage, root):
    """Calibrate a doublet-only local kernel under common energy and shape losses."""
    structures = transfer.geometries()
    pub = encode(structures["public"])
    rad = encode(structures["radical"]).reshape(-1, 7, basis.size)
    meta = np.load(WORK / "rad-meta.npy")
    saddle = np.array([r[1][0].split("/")[-1].startswith("ts") for r in transfer.modes])
    ext = external(basis, basis.design_state)
    summary = {}
    for t in kt.TARGETS:
        data = kt.equation_data(t)
        s, y, src, fold, kind, case = data
        base = ms.baseline(t, -1, "parent", PARENT)
        g = kt.grouped_moments(data, s @ pub, y - s @ base, fold >= 0)
        eq = (
            g["other"]
            + 96 * g["mid"]
            + 36 * g["gap"]
            + isomers(t, -1, pub, base)
            + ext[t][0]
        )
        if t == "wb97xd3":
            bm = ms.baseline(t, -1, "modes", PARENT)
            a, b = trial_fit.rad_rows(rad, bm, meta, np.where(saddle, 3.0, 1.0))
            eq += trial_fit.reduce_gram(a, b, np.repeat(meta[:, 0] >= 0, 3))
        w = cho_solve(cho_factor(eq[:-1, :-1] + 4 * np.eye(basis.size)), eq[:-1, -1])
        basis.weights = w
        ff = ForceField.load(WORK / f"{PARENT}-{t}.npz", dispersion="d3")
        ff.additional_bases.append(basis)
        path = WORK / f"{stage}-{t}.npz"
        ff.save_with_base(kt.original.BASE, path)
        rebuilt = ForceField.load(path, dispersion="d3")
        pred = base + pub @ w
        for m in (1, 2):
            i = next(
                i
                for i, (z, r, q, mult) in enumerate(structures["public"])
                if mult == m and 5 < len(z) < 14
            )
            z, r, q, m = structures["public"][i]
            energy, force = rebuilt.predict(z, r, q, m)
            np.testing.assert_allclose(energy, pred[i], atol=5e-6, rtol=0)
            u = np.random.default_rng(i).normal(size=r.shape)
            u /= np.linalg.norm(u)
            fd = (
                rebuilt.energy(z, r - 2e-5 * u, q, m)
                - rebuilt.energy(z, r + 2e-5 * u, q, m)
            ) / 4e-5
            np.testing.assert_allclose(fd, (u * force).sum(), atol=8e-5, rtol=0)
        held = fold < 0
        np.save(WORK / f"{key}-{t}-parent--1.npy", pred)
        np.savez_compressed(
            WORK / f"{stage}-{t}-equations.npz",
            source=src[held],
            case=case[held],
            kind=kind[held],
            error_kcal=((s @ pred - y) * kt.KCAL)[held],
        )
        if t == "wb97xd3":
            val = bm + rad @ w
            np.save(WORK / f"{key}-{t}-modes--1.npy", val)
            np.save(WORK / f"{stage}-held-modes.npy", val[meta[:, 0] < 0])
        summary[t] = {}
        for name, observations in ext[t][1].items():
            if len(observations):
                rows = np.asarray(observations)
                errors = rows[..., :-1] @ w - rows[..., -1]
                summary[t][name] = (np.abs(errors).mean(0) * kt.KCAL).tolist()
    kt.write_json(root / "external-validation.json", summary)
    kt.write_json(
        WORK / f"{stage}-settings.json",
        {
            "selected": stage,
            "parent": PARENT,
            "branches": dict.fromkeys(kt.TARGETS, key),
        },
    )
    compare(stage)
    validate(stage)
    print("Completed reaction validation", stage, flush=True)
    evaluate_trial(stage)
    print("Completed relaxed frequency validation", stage, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("atoms", "fit"))
    if parser.parse_args().step == "atoms":
        train_atoms()
    else:
        fit()
