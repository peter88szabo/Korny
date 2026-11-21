"""Fit short-range donor–acceptor features to both reactive surfaces.

The new basis sees coordinates and local chemistry only. Atomic scores were
trained independently on the vertical IP and EA of ions. The reactive readout
uses the original training reaction gaps and midpoint barriers, the actual
coupled-cluster channels, NIST training minima, molecular forces and measured
radical curvatures. No validation labels select model parameters.
"""

import json
import pickle
from collections import defaultdict

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from experiments.full_public import kernel_trials as kt
from experiments.full_public import trial_fit
from experiments.full_public import trial_multiscale as ms
from experiments.full_public import trial_spin_extent as extent
from experiments.full_public import trial_spin_transfer as transfer
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_constraints import nist_rows
from experiments.full_public.trial_curvature_fit import anchor_blocks
from experiments.full_public.trial_isomers import gram as isomers
from experiments.full_public.trial_redox_environments import ROOT
from experiments.full_public.trial_spectra import evaluate_trial
from experiments.full_public.trial_spin_population import ROOT as IONS
from experiments.full_public.trial_validation import run as validate
from src.forcefield import ForceField
from src.redox_environment import RedoxEnvironmentKernel
from src.spin_local import SpinLocalKernel

WORK = kt.WORK
PARENT = "p49-public"
STAGE = "p51-public"


def external(basis, state_design=None):
    scan, labels = extent.scans()
    network, channels, refs = extent.network()
    states = scan + network
    names = [s[0] for s in states]
    if names != json.loads((extent.ROOT / "ids.json").read_text()):
        raise ValueError("Changed geometry order")
    if state_design is None:
        fx = np.vstack([basis.design(z, r)[0] for _, z, r, *_ in states])
    else:
        fx = np.vstack([state_design(z, r, q, m)[0] for _, z, r, q, m, *_ in states])
    with np.load(IONS / "spin-population.npz") as f:
        spin_model = SpinLocalKernel.load(f)
    spin_features, m = transfer.compressed("external", len(states), spin_model)
    raw = np.vstack(
        [
            np.load(extent.ROOT / f"block-{lo:06d}.npz")["rows"]
            for lo in range(0, len(states), 48)
        ]
    )
    coeff = json.loads((WORK / "p48-public-settings.json").read_text())["coefficients"]
    mix = json.loads((WORK / "p49-public-settings.json").read_text())["fit"]
    ix = {n: i for i, n in enumerate(names)}
    by = {}
    for t in kt.TARGETS:
        ff = ForceField.load(WORK / f"{PARENT}-{t}.npz", dispersion="d3")
        c = np.array(coeff[t])
        base = raw[:, 9] + spin_features @ ff.additional_bases[-1].weights
        delta = raw[:, 8] + raw[:, :8] @ c - raw[:, 9]
        for j, label in [(2, "mix_doublet"), (3, "mix_triplet")]:
            base += delta * (m == j) * mix[t][label]
        # Geometry caches contain the high-level baseline for network states
        # and the radical baseline for scans. Check one of each through the
        # ordinary deployed calculator, in the matching electronic branch.
        i = next(k for k, s in enumerate(states) if s[-1] == t and s[4] > 1)
        _, z, r, q, spin, *_ = states[i]
        np.testing.assert_allclose(base[i], ff.energy(z, r, q, spin), atol=5e-6, rtol=0)
        g = np.zeros((basis.size + 1,) * 2)
        groups = defaultdict(list)
        if t == "high":

            def side(names, base=base):
                ids = [ix[n] for n in names]
                return np.r_[
                    fx[ids].sum(0), sum(refs[n] for n in names) - base[ids].sum()
                ]

            for c in channels:
                a, b, h = side(c["reactant"]), side(c["product"]), side([c["ts"]])
                gap, mid = b - a, h - (a + b) / 2
                if c["split"] == "train":
                    g += 20 * np.outer(gap, gap) + 32 * np.outer(mid, mid)
                else:
                    key = (
                        "separated"
                        if len(c["reactant"]) + len(c["product"]) > 2
                        else "connected"
                    )
                    groups[key].append([gap, mid, h - a, h - b])
        else:
            # Preserve energy variation already learned on the distinct
            # auxiliary DFT scan levels as an anchor during this redox test.
            train = defaultdict(list)
            for n, e, mult, sys, method, split in labels:
                if split == "train":
                    train[sys, method].append(fx[ix[n]])
                elif split == "validation":
                    groups["benzopinacol"].append(np.r_[fx[ix[n]], e - base[ix[n]]])
            for rows in train.values():
                a = np.array(rows) - np.mean(rows, axis=0)
                g[:-1, :-1] += (40 / len(a)) * (a.T @ a)
            center = next(
                ix[n]
                for n, e, m, sys, method, part in labels
                if part == "validation" and m == 1 and n.endswith("/0")
            )
            truth = next(e for n, e, m, sys, method, part in labels if ix[n] == center)
            groups["benzopinacol"] = (
                np.array(groups["benzopinacol"])
                - np.r_[fx[center], truth - base[center]]
            )
        by[t] = (g, groups)
    return by


def fit():
    with np.load(ROOT / "basis.npz") as f:
        basis = RedoxEnvironmentKernel.load(f)
    external_sets = external(basis)
    main = np.load(ROOT / "main.npy", mmap_mode="r")
    n = len(np.load(WORK / "prior.npy", mmap_mode="r"))
    pub = main[:n]
    ef, curv = anchor_blocks(main, n, "redox51")
    rad = np.load(ROOT / "rad.npy", mmap_mode="r").reshape(-1, 7, basis.size)
    meta = np.load(WORK / "rad-meta.npy")
    with (WORK / "radical-modes.pkl").open("rb") as f:
        saddle = np.array(
            [r[1][0].split("/")[-1].startswith("ts") for r in pickle.load(f)]
        )
    geom, _, _ = kt.inputs()
    with (WORK / "ids.pkl").open("rb") as f:
        _, names = pickle.load(f)
    metrics = {}
    for t in kt.TARGETS:
        ff = ForceField.load(WORK / f"{PARENT}-{t}.npz", dispersion="d3")
        s, y, src, fold, kind, case = d = kt.equation_data(t)
        base = ms.baseline(t, -1, "parent", PARENT)
        x = s @ pub
        g = kt.grouped_moments(d, x, y - s @ base, fold >= 0)
        ext, held = external_sets[t]
        G = g["other"] + 96 * g["mid"] + 36 * g["gap"] + isomers(t, -1, pub, base) + ext
        G[:-1, :-1] += 0.12 * ef + 0.045 * curv
        nr = nist_rows(lambda z, r: basis.design(z, r, True)[1], ff)
        G += 20.0 * nr.T @ nr
        if t == "wb97xd3":
            bm = ms.baseline(t, -1, "modes", PARENT)
            a, b = trial_fit.rad_rows(rad, bm, meta, np.where(saddle, 3.0, 1.0))
            G += trial_fit.reduce_gram(a, b, np.repeat(meta[:, 0] >= 0, 3))
        lhs = G[:-1, :-1].copy()
        lhs.flat[:: len(lhs) + 1] += 4.0
        w = cho_solve(cho_factor(lhs), G[:-1, -1])
        basis.weights = w.copy()
        ff.additional_bases.append(basis)
        target = WORK / f"{STAGE}-{t}.npz"
        ff.save_with_base(kt.original.BASE, target)
        deployed = ForceField.load(target, dispersion="d3")
        predicted = base + pub @ w
        for spin in (1, 2, 3):
            i = next(
                i
                for i, n in enumerate(names)
                if geom[n].state[1] == spin and 4 < len(geom[n].z) < 12
            )
            mol = geom[names[i]]
            z, r, st = mol.z, mol.r, mol.state
            e, f = deployed.predict(z, r, *st)
            np.testing.assert_allclose(e, predicted[i], atol=4e-6, rtol=0)
            u = np.random.default_rng(i).normal(size=r.shape)
            u /= np.linalg.norm(u)
            fd = (
                deployed.energy(z, r - 2e-5 * u, *st)
                - deployed.energy(z, r + 2e-5 * u, *st)
            ) / 4e-5
            np.testing.assert_allclose((u * f).sum(), fd, atol=7e-5, rtol=0)
        np.save(WORK / f"redox51-{t}-parent--1.npy", predicted)
        use = fold == -1
        np.savez_compressed(
            WORK / f"{STAGE}-{t}-equations.npz",
            source=src[use],
            case=case[use],
            kind=kind[use],
            error_kcal=(s @ predicted - y)[use] * kt.KCAL,
        )
        if t == "wb97xd3":
            modes = bm + rad @ w
            np.save(WORK / f"redox51-{t}-modes--1.npy", modes)
            np.save(WORK / f"{STAGE}-held-modes.npy", modes[meta[:, 0] == -1])
        metrics[t] = {}
        for name, rows in held.items():
            a = np.asarray(rows)
            metrics[t][name] = {
                "n": len(rows),
                "p49_mae_kcal": (np.abs(a[..., -1]).mean(0) * kt.KCAL).tolist(),
                "p51_mae_kcal": (
                    np.abs(a[..., :-1] @ w - a[..., -1]).mean(0) * kt.KCAL
                ).tolist(),
            }
    kt.write_json(ROOT / "external-validation.json", metrics)
    kt.write_json(
        WORK / f"{STAGE}-settings.json",
        {
            "selected": STAGE,
            "branches": dict.fromkeys(kt.TARGETS, "redox51"),
            "parent": PARENT,
        },
    )
    compare(STAGE)
    validate(STAGE)
    evaluate_trial(STAGE)
    print("Completed short-range redox-environment fit", metrics, flush=True)


if __name__ == "__main__":
    with threadpool_limits(1):
        fit()
