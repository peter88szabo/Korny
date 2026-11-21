"""Transfer three optimized-geometry energy surfaces with curvature constraints.

Learn separate auxiliary readouts for PM6, HF/3-21G and r²SCAN-3c; each has
its own geometry distribution and elemental energy reference. Fit the primary
Korny energy only to its established references. Transferable features are
differences between auxiliary kernel readouts, evaluated at a single geometry.
They have analytic forces, require no QM calculation and accept no product ID.
"""

import argparse
import json
import shutil
import sqlite3
from collections import defaultdict

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from experiments.full_public import amk_correction_data as archive
from experiments.full_public import kernel_trials as kt
from experiments.full_public import trial_network_hessians as network
from experiments.full_public import trial_periodic as periodic
from experiments.full_public.paired_baseline_audit import (
    fit_ridges,
    paired_intervals,
    read_pickle,
)
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_constraints import nist_rows
from experiments.full_public.trial_isomers import gram as isomer_gram
from experiments.full_public.trial_measured_curvature import load_basis
from experiments.full_public.trial_multiscale import baseline
from experiments.full_public.trial_spectra import evaluate_trial
from experiments.full_public.trial_validation import run as validate
from src.forcefield import ForceField

WORK = kt.WORK
METHODS = ("pm6", "hf", "r2scan")


def atom_counts(z):
    return np.array([np.count_nonzero(z == n) for n in (1, 6, 7, 8)], int)


def complete_sides(reactant, product, counts, ts):
    """Identify independently scoreable barrier directions in a fragmented archive."""
    return [
        bool(side) and np.array_equal(counts[side].sum(0), counts[ts])
        for side in (reactant, product)
    ]


def deduplicate(keys, values):
    """Give one vote per state and geometry; skip conflicting repeated labels."""
    groups = defaultdict(list)
    for i, k in enumerate(keys):
        groups[k].append(i)
    keep, y = [], []
    for indices in groups.values():
        target = values[indices]
        if np.ptp(target) <= 0.001:
            keep.append(indices[0])
            y.append(target.mean())
    return np.array(keep, dtype=int), np.array(y)


def source_fit():
    periodic.setup()
    periodic.basis, cols = load_basis()
    cols = cols[::3]
    periodic.OPTIONS = ((0.85, 3),)
    weights, report = {}, {}
    with sqlite3.connect(f"file:{archive.DATABASE}?mode=ro", uri=True) as db:
        for method in METHODS:
            entries = db.execute(
                """SELECT case_id,role,split,q,m,fp,z,r,published_baseline_eV
                FROM frames WHERE dataset='RDB7' AND method=? AND split IN ('train','validation')
                ORDER BY frame""",
                (method,),
            ).fetchall()
            if not entries:
                raise ValueError(f"No usable rows for {method}")
            xyz = [
                (np.frombuffer(v[6], "u1"), np.frombuffer(v[7], "<f8").reshape(-1, 3))
                for v in entries
            ]
            if any((v[3], v[4]) != (0, 1) for v in entries):
                raise ValueError(
                    "Additional electronic states require a state-specific auxiliary readout"
                )
            x = np.empty((len(entries), len(cols)))
            for lo in range(0, len(x), 32):
                block = periodic.batch(xyz[lo : lo + 32])[0]
                x[lo : lo + len(block)] = block[:, cols]
            counts = np.array([atom_counts(z) for z, r in xyz])
            y = np.array([v[-1] for v in entries])
            train = np.array([i for i, v in enumerate(entries) if v[2] == "train"])
            uniq, target = deduplicate([entries[i][5] for i in train], y[train])
            use = train[uniq]
            weights[method] = fit_ridges(x[use], target, counts[use])[1e-2]
            # Score each direction only when the corresponding archived endpoint
            # contains all atoms of the one saddle. Some products store just
            # one of the fragments indicated in the original RDB7 ledger.
            reactions = defaultdict(lambda: defaultdict(list))
            for i, (case, role, split, *_) in enumerate(entries):
                if split == "validation":
                    reactions[case][role].append(i)
            errors = defaultdict(list)
            res = x @ weights[method] - y
            for roles in reactions.values():
                ts = roles["transition_state"]
                if len(ts) != 1:
                    continue
                t = ts[0]
                sides = [roles["reactant"], roles["product"]]
                complete = complete_sides(*sides, counts, t)
                for side, ok, name in zip(
                    sides, complete, ("forward", "reverse"), strict=True
                ):
                    if ok:
                        errors[name].append(res[t] - res[side].sum())
                if all(complete):
                    errors["reaction"].append(res[sides[1]].sum() - res[sides[0]].sum())
            report[method] = {
                "training_frames": len(train),
                "distinct_training_geometries": len(use),
                "training_reaction_ids": len({entries[i][0] for i in train}),
                "validation": {
                    k: {"n": len(v), "mae_kcal": float(np.mean(np.abs(v)) * kt.KCAL)}
                    for k, v in errors.items()
                },
            }
            print("Fitted deposited baseline", method, report[method], flush=True)
    # The reference-energy offset for each element cancels from balanced
    # chemistry. Remove it on the training data for *each* auxiliary target.
    ids, _names = read_pickle("ids.pkl")
    dft = defaultdict(list)
    for name, e in read_pickle("rdb-dft-energies.pkl").items():
        dft[ids[name]].append(e)
    ix, counts, dft_y, delta = [], [], [], []
    for (formula, _), records in read_pickle("isomer-observations.pkl")["RDB7"].items():
        for i, cc, folds in records:
            if -1 in folds:
                raise ValueError("Held geometry in training observations")
            if dft[i] and np.ptp(dft[i]) <= 0.001:
                ix.append(i)
                counts.append(atom_counts(np.asarray(formula)))
                dft_y.append(np.mean(dft[i]))
                delta.append(cc - dft_y[-1])
    raw = np.load(WORK / "periodic-0.85-main.npy", mmap_mode="r")
    x = raw[np.ix_(ix, cols)]
    reference = fit_ridges(x, np.asarray(dft_y), np.asarray(counts))[1e-2]
    cc = fit_ridges(x, np.asarray(delta), np.asarray(counts))[1e-2]
    ff = ForceField.load(WORK / "p50-public-high.npz", dispersion="off")
    modes = ff.additional_bases[-1].weights[cols]
    projection = np.column_stack(
        (cc, modes, *(weights[k] - reference for k in METHODS))
    )
    np.savez_compressed(
        archive.ROOT / "auxiliary.npz", projection=projection, columns=cols
    )
    kt.write_json(archive.ROOT / "source-fit.json", report)


def solve_candidate(control=False):
    """Fit against the same electronic, force and vibrational labels in both arms."""
    stage = "p54-control" if control else "p54-public"
    with np.load(archive.ROOT / "auxiliary.npz") as f:
        projection, cols = f["projection"], f["columns"]
    if control:
        projection = projection[:, :2]
    n = projection.shape[1]
    data = kt.equation_data("high")
    s, y, src, fold, kind, case = data
    ref = baseline("high", -1, "parent", "p49-public")
    raw = np.load(WORK / "periodic-0.85-main.npy", mmap_mode="r")
    pub = np.empty((len(ref), n))
    for lo in range(0, len(ref), 256):
        hi = min(lo + 256, len(ref))
        pub[lo:hi] = raw[lo:hi][:, cols] @ projection
    x = s @ pub
    train = fold >= 0
    # Express each feature in units of its training-only RMS displacement in
    # reaction energy. This makes the ridge comparable across auxiliary levels.
    scale = np.sqrt(np.mean(x[train & (src == "RDB7")] ** 2, axis=0)).clip(1e-8)
    x /= scale
    pub /= scale
    projection /= scale
    blocks = kt.grouped_moments(data, x, y - s @ ref, train)
    eq = (
        blocks["other"]
        + 96 * blocks["mid"]
        + 36 * blocks["gap"]
        + isomer_gram("high", -1, pub, ref)
    )
    for tag, factor in (("ef", 0.12), ("curvature", 0.045)):
        g = np.load(WORK / f"periodic-0.85-anchor-{tag}.npy", mmap_mode="r")
        eq[:n, :n] += factor * projection.T @ g[np.ix_(cols, cols)] @ projection
    basis, _ = load_basis()
    model = ForceField.load(WORK / "p49-public-high.npz", dispersion="d3")
    norm = nist_rows(
        lambda z, r: basis.design(z, r, True)[1][:, cols] @ projection, model
    )
    eq += 20 * norm.T @ norm
    manifest = json.loads((network.DEST / "manifest.json").read_text())["records"]
    # The p50 cache contains true reference-minus-p49 forces and Hessian
    # responses. Transform its training rows into this small feature space.
    for info in manifest:
        if info["split"] != "train":
            continue
        with np.load(network.ROOT / info["file"]) as f:
            a = f["rows"].astype(float)
        z = np.column_stack((a[:, : len(cols)] @ projection, a[:, len(cols)]))
        eq += z.T @ z
    coeff = cho_solve(cho_factor(eq[:n, :n] + 8 * np.eye(n)), eq[:n, n])
    basis.weights = np.zeros(basis.size)
    basis.weights[cols] = projection @ coeff
    model.additional_bases.append(basis)
    model.save_with_base(kt.original.BASE, WORK / f"{stage}-high.npz")
    pred = ref + pub @ coeff
    geoms, _, _ = kt.inputs()
    _, names = read_pickle("ids.pkl")
    deployed = ForceField.load(WORK / f"{stage}-high.npz", dispersion="d3")
    for spin in (1, 2, 3):
        i = next(
            i
            for i, k in enumerate(names)
            if geoms[k].state[1] == spin and 5 < len(geoms[k].z) < 14
        )
        g = geoms[names[i]]
        e, force = deployed.predict(g.z, g.r, *g.state)
        np.testing.assert_allclose(e, pred[i], rtol=0, atol=5e-6)
        u = np.random.default_rng(i).normal(size=g.r.shape)
        u /= np.linalg.norm(u)
        fd = (
            deployed.energy(g.z, g.r - 2e-5 * u, *g.state)
            - deployed.energy(g.z, g.r + 2e-5 * u, *g.state)
        ) / 4e-5
        np.testing.assert_allclose(np.sum(u * force), fd, rtol=0, atol=8e-5)
    held = ~train
    np.save(WORK / f"{stage}-high-parent--1.npy", pred)
    np.savez_compressed(
        WORK / f"{stage}-high-equations.npz",
        source=src[held],
        case=case[held],
        kind=kind[held],
        error_kcal=((s @ pred - y) * kt.KCAL)[held],
    )
    for suffix in ("wb97xd3.npz", "wb97xd3-equations.npz", "held-modes.npy"):
        shutil.copyfile(WORK / ("p49-public-" + suffix), WORK / (stage + "-" + suffix))
    kt.write_json(
        WORK / f"{stage}-settings.json",
        {
            "selected": stage,
            "branches": {"high": stage, "wb97xd3": "spin49"},
            "features": [
                "original_CC_minus_DFT",
                "network_mode_response",
                *(() if control else METHODS),
            ],
            "amplitudes": coeff.tolist(),
            "training_scales": scale.tolist(),
        },
    )
    compare(stage)
    validate(stage)
    kt.write_json(
        archive.ROOT / f"{stage}-minus-p49-intervals.json", paired_intervals(stage)
    )
    evaluate_trial(stage)
    print("Completed reoptimized structure fit", stage, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("step", choices=("source", "fit"))
    if p.parse_args().step == "source":
        source_fit()
    else:
        solve_candidate(control=True)
        solve_candidate()
