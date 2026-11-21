"""Fit reactive energy differences and measured vibrational responses on the p49 surface.

Each AutoMeKin DFT structure contributes its stationary Cartesian forces and
two Hessian-vector products. One displacement is the softest signed mode,
including the negative mode at a saddle. The other samples every remaining
mode in a deterministic mixture. The full force responses constrain motion
perpendicular to these displacements too. Geometries, charge and multiplicity
suffice for the deployed scalar energy and analytic forces.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import pickle
import sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from experiments.full_public import kernel_trials as kt
from experiments.full_public import trial_fit
from experiments.full_public import trial_multiscale as ms
from experiments.full_public.automekin_modes import DEST
from experiments.full_public.paper_data import DFT, OUT
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_constraints import nist_rows
from experiments.full_public.trial_isomers import gram as isomer_gram
from experiments.full_public.trial_measured_curvature import load_basis
from experiments.full_public.trial_spectra import evaluate_trial
from experiments.full_public.trial_validation import run as validate
from experiments.reactivity.frequency_validation import TO_CM1
from src.forcefield import ForceField

ROOT = OUT / "network-hessians50"
PARENT = "p49-public"
STAGE = "p50-public"
STEP = 0.002


def setup():
    global models, basis, cols
    models = [
        ForceField.load(kt.WORK / f"{PARENT}-{t}.npz", dispersion="d3")
        for t in kt.TARGETS
    ]
    basis, selected = load_basis()
    cols = selected[::3]


def mode_rows(name, r, freq, mu, mass, features, width):
    """Apply the same signed force and curvature operators to any energy columns.

    Columns after width contain frozen parent energies. Convert their forces
    and projected curvatures to source-minus-parent targets. With only basis
    columns this function returns the design rows for an additive correction.
    """
    center, force = features(r)
    observed = force.copy()
    observed[:, width :] *= -1
    blocks = [2.0 * observed / np.sqrt(r.size)]
    # The directions are mass normalized in the archived Cartesian frame.
    # H_cart = M U^T diag(lambda) U M. Use the reported signed frequencies
    # and isotope masses to put every force response in eV/angstrom^2.
    eig = freq * np.abs(freq) / TO_CM1**2
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
    )
    mixture = rng.choice([-1.0, 1.0], len(freq)) / np.maximum(400, np.abs(freq))
    mixture[np.argmin(freq)] = 0.0
    diag = None
    for direction in (mu[np.argmin(freq)], mixture @ mu):
        u = direction / np.linalg.norm(direction)
        cart = u.reshape(r.shape)
        _, plus = features(r + STEP * cart)
        _, minus = features(r - STEP * cart)
        response = (minus - plus) / (2 * STEP)
        ref = ((mu @ (mass * u) * eig) @ mu) * mass
        if diag is None:
            diag = np.r_[u @ response, u @ ref, np.sum(mass * u**2)]
        response[:, width :] = ref[:, None] - response[:, width :]
        # All measured internal modes observe each perturbed force vector.
        # Give the soft and bending directions a meaningful relative weight.
        scale = TO_CM1**2 / (2 * np.maximum(450, np.abs(freq)) * 140)
        blocks.append((mu @ response) * (scale / np.sqrt(len(freq)))[:, None])
    rows = np.vstack(blocks)
    return rows, center, diag, force


def record(info):
    dest = ROOT / info["file"]
    if dest.is_file():
        return
    with np.load(DEST / info["file"]) as a:
        z, r, freq = a["z"], a["r_A"], a["freq_cm1"]
        mu = a["modes"].reshape(len(freq), r.size)
        mass = np.repeat(a["mass_u"], 3)
        q, spin = int(a["charge"]), int(a["mult"])

    def features(pos):
        val, jac = basis.design(z, pos, True)
        ev, forces = zip(*(ff.predict(z, pos, q, spin) for ff in models), strict=True)
        return np.r_[val[cols], ev], np.column_stack(
            (jac[:, cols], *(f.ravel() for f in forces))
        )

    rows, center, diag, force = mode_rows(info["id"], r, freq, mu, mass, features, len(cols))
    with dest.with_suffix(".pending").open("wb") as f:
        np.savez_compressed(
            f,
            rows=rows.astype(np.float32),
            energy=center,
            soft=diag,
            forces=force.astype(np.float32),
        )
    dest.with_suffix(".pending").replace(dest)


def prepare():
    ROOT.mkdir(exist_ok=True)
    rec = json.loads((DEST / "manifest.json").read_text())["records"]
    setup()
    with ProcessPoolExecutor(mp_context=mp.get_context("fork"), max_workers=3) as pool:
        for i, _ in enumerate(pool.map(record, rec, buffersize=9), 1):
            if i % 250 == 0:
                print(
                    "Encoded published stationary force and mode responses",
                    i,
                    len(rec),
                    flush=True,
                )
    print("All network force and mode rows ready", len(rec), flush=True)


def network_channels(rec, centers, surface):
    """Join each DFT or CC channel to the coordinates actually used for its label."""
    T = kt.TARGETS.index(surface)
    energies = {}
    channels = []
    if surface == "high":
        from experiments.full_public.trial_spin_extent import network

        states, channels, references = network()
        for n, z, r, q, spin, _, _ in states:
            key = n.split("/")[-1]
            shape = ROOT / (n.split("/")[1] + "_" + key + ".npz")
            src = DEST / shape.name
            row = None
            if shape.is_file() and src.is_file():
                with np.load(src) as a:
                    same = (
                        int(a["charge"]) == q
                        and int(a["mult"]) == spin
                        and np.array_equal(a["z"], z)
                        and np.allclose(a["r_A"], r, atol=1e-7, rtol=0)
                    )
                if same:
                    with np.load(shape) as a:
                        row = a["energy"]
            if row is None:
                val = basis.design(z, r)[0][cols]
                energies[n] = np.r_[
                    val, references[n] - models[T].energy(z, r, q, spin)
                ]
            else:
                energies[n] = np.r_[
                    row[: len(cols)], references[n] - row[len(cols) + T]
                ]
    else:
        with sqlite3.connect(
            f"file:{OUT / 'electronic.sqlite'}?mode=ro", uri=True
        ) as db:
            table = dict(
                db.execute(
                    "select id,energy_eV from geometries where source='AutoMeKin' and method=?",
                    (DFT,),
                )
            )
            for d in rec:
                n = d["id"]
                a = centers[n]
                energies[n] = np.r_[a[: len(cols)], table[n] - a[len(cols) + T]]
            for sys, n, r, ts, p in db.execute(
                "select system,channel,reactant,ts,product from channels where source='AutoMeKin' and method=? and product is not null",
                (DFT,),
            ):
                channels.append(
                    {
                        "channel": n,
                        "system": sys,
                        "reactant": [r],
                        "ts": ts,
                        "product": [p],
                        "split": "validation" if sys == "cBD-CN" else "train",
                    }
                )
    out = defaultdict(list)
    for c in channels:
        names = c["reactant"] + [c["ts"]] + c["product"]
        if any(n not in energies for n in names):
            continue
        r = sum(energies[n] for n in c["reactant"])
        p = sum(energies[n] for n in c["product"])
        h = energies[c["ts"]]
        kind = "separated" if len(names) > 3 else "connected"
        out[c["system"], c["split"], kind].append(
            [p - r, h - (r + p) / 2, h - r, h - p]
        )
    return {k: np.array(v) for k, v in out.items()}


def fit():
    setup()
    rec = json.loads((DEST / "manifest.json").read_text())["records"]
    n = len(cols)
    centers, external, mode_checks = {}, {}, defaultdict(list)
    # Accumulate sufficient statistics per training network. Enrollment and
    # mode selection were fixed before seeing the held network's errors.
    systems = sorted({info["system"] for info in rec if info["split"] == "train"})
    for info in rec:
        with np.load(ROOT / info["file"]) as f:
            centers[info["id"]] = f["energy"]
            for j, t in enumerate(kt.TARGETS):
                if info["split"] == "train":
                    a = f["rows"].astype(float)
                    a = np.column_stack((a[:, :n], a[:, n + j]))
                    key = t, info["system"]
                    if key not in external:
                        external[key] = np.zeros((n + 1, n + 1))
                    external[key] += a.T @ a
                elif info["split"] == "validation":
                    v = f["soft"]
                    force = f["forces"]
                    mode_checks[t, info["role"]].append(
                        (v, np.column_stack((force[:, :n], force[:, n + j])))
                    )
    src = np.load(kt.WORK / "periodic-0.85-main.npy", mmap_mode="r")
    rad_src = np.load(kt.WORK / "periodic-0.85-rad.npy", mmap_mode="r")
    meta = np.load(kt.WORK / "rad-meta.npy")
    with (kt.WORK / "radical-modes.pkl").open("rb") as f:
        saddles = np.array(
            [r[1][0].split("/")[-1].startswith("ts") for r in pickle.load(f)]
        )

    def subset(raw, count):
        out = np.empty((count, n))
        for lo in range(0, count, 256):
            out[lo : lo + 256] = raw[lo : min(count, lo + 256)][:, cols]
        return out

    raw_public = subset(src, len(np.load(kt.WORK / "prior.npy", mmap_mode="r")))
    rad = subset(rad_src, len(rad_src)).reshape(-1, 7, n)
    anchor = np.zeros((n, n))
    for term, weight in (("ef", 0.12), ("curvature", 0.045)):
        g = np.load(kt.WORK / f"periodic-0.85-anchor-{term}.npy", mmap_mode="r")
        anchor += weight * g[np.ix_(cols, cols)]
    settings, results = {}, {}
    geom, _, _ = kt.inputs()
    with (kt.WORK / "ids.pkl").open("rb") as f:
        _, names = pickle.load(f)
    for j, t in enumerate(kt.TARGETS):
        ff = models[j]
        d = kt.equation_data(t)
        s, y, source, fold, kind, case = d
        parent = ms.baseline(t, -1, "parent", PARENT)
        pub = raw_public[: len(parent)]
        pieces = kt.grouped_moments(d, s @ pub, y - s @ parent, fold >= 0)
        eq = (
            pieces["other"]
            + 96 * pieces["mid"]
            + 36 * pieces["gap"]
            + isomer_gram(t, -1, pub, parent)
        )
        eq[:n, :n] += anchor
        norm = nist_rows(lambda z, r: basis.design(z, r, True)[1][:, cols], ff)
        eq += 20.0 * (norm.T @ norm)
        if t == "wb97xd3":
            old = ms.baseline(t, -1, "modes", PARENT)
            a, b = trial_fit.rad_rows(rad, old, meta, np.where(saddles, 3.0, 1.0))
            eq += trial_fit.reduce_gram(a, b, np.repeat(meta[:, 0] >= 0, 3))

        channels = network_channels(rec, centers, t)
        engrams = {k: np.zeros_like(eq) for k in systems}
        for (sys, split, _), ar in channels.items():
            if split == "train":
                engrams[sys] += (
                    20 * ar[:, 0, :].T @ ar[:, 0, :] + 32 * ar[:, 1, :].T @ ar[:, 1, :]
                )

        def solve(strength, omit=None, eq=eq, engrams=engrams, t=t):
            train = eq.copy()
            for sys in systems:
                if sys != omit:
                    train += engrams[sys] + strength * external[t, sys]
            train[:n, :n].flat[:: n + 1] += 5.0
            return cho_solve(cho_factor(train[:n, :n]), train[:n, n])

        # Select by changes from the frozen parent on left-out training
        # networks, balancing each network and both observable types equally.
        # Parent energies may have seen these networks. The inherited cBD-CN
        # validation and the original public panel cannot select the strength.
        opts = {}
        for strength in (0.1, 1.0, 10.0, 100.0):
            losses = []
            for sys in systems:
                v = np.r_[solve(strength, sys), -1.0]
                ratios = [
                    (v @ g @ v) / max(g[-1, -1], 1e-12)
                    for g in (engrams[sys], external[t, sys])
                ]
                losses.append(float(np.mean(ratios)))
            opts[strength] = float(np.mean(losses))
        strength = min(opts, key=opts.get)
        w = solve(strength)
        settings[t] = {
            "strength": strength,
            "network_calibration_scores": opts,
            "columns": n,
        }
        e = parent + pub @ w
        key = "network50"
        np.save(kt.WORK / f"{key}-{t}-parent--1.npy", e)
        use = fold == -1
        np.savez_compressed(
            kt.WORK / f"{STAGE}-{t}-equations.npz",
            source=source[use],
            kind=kind[use],
            case=case[use],
            error_kcal=(s @ e - y)[use] * kt.KCAL,
        )
        if t == "wb97xd3":
            modes = old + rad @ w
            np.save(kt.WORK / f"{key}-{t}-modes--1.npy", modes)
            np.save(kt.WORK / f"{STAGE}-held-modes.npy", modes[meta[:, 0] == -1])
        new, _ = load_basis()
        new.weights = np.zeros(new.size)
        new.weights[cols] = w
        ff.additional_bases.append(new)
        dest = kt.WORK / f"{STAGE}-{t}.npz"
        ff.save_with_base(kt.original.BASE, dest)
        deployed = ForceField.load(dest, dispersion="d3")
        for spin in (1, 2, 3):
            idx = next(
                i
                for i, k in enumerate(names)
                if geom[k].state[1] == spin and 5 < len(geom[k].z) < 14
            )
            sample = geom[names[idx]]
            z, r, st = sample.z, sample.r, sample.state
            en, force = deployed.predict(z, r, *st)
            np.testing.assert_allclose(en, e[idx], atol=4e-6, rtol=0)
            u = np.random.default_rng(idx).normal(size=r.shape)
            u /= np.linalg.norm(u)
            fd = (
                deployed.energy(z, r - 2e-5 * u, *st)
                - deployed.energy(z, r + 2e-5 * u, *st)
            ) / 4e-5
            np.testing.assert_allclose(np.sum(u * force), fd, atol=8e-5, rtol=0)
        stats = {}
        for (sys, split, ch), ar in channels.items():
            if split == "validation":
                err = ar[:, :, :n] @ w - ar[:, :, n]
                stats[ch] = {
                    "n": len(ar),
                    "p49_mae_kcal": (abs(ar[:, :, n]).mean(0) * kt.KCAL).tolist(),
                    "p50_mae_kcal": (abs(err).mean(0) * kt.KCAL).tolist(),
                }
        for (target, role), rows in mode_checks.items():
            if target != t:
                continue
            modes = []
            force_abs = [[], []]
            for v, f in rows:
                p = v[n + j]
                ref = v[-2]
                mass = v[-1]
                k = np.array([p, p + v[:n] @ w, ref])
                modes.append(np.sign(k) * np.sqrt(abs(k) / mass) * TO_CM1)
                force_abs[0].extend(abs(f[:, -1]))
                force_abs[1].extend(abs(f[:, -1] + f[:, :n] @ w))
            modes = np.array(modes)
            stats[role] = {
                "structures": len(modes),
                "soft_direction_mae_cm1": np.abs(modes[:, :2] - modes[:, 2:])
                .mean(0)
                .tolist(),
                "sign_agreement": ((modes[:, :2] < 0) == (modes[:, 2:] < 0))
                .mean(0)
                .tolist(),
                "force_mae_eV_A": np.mean(force_abs, axis=1).tolist(),
            }
        results[t] = stats
        print("Completed network force-response fit", t, settings[t], stats, flush=True)
    kt.write_json(ROOT / "external-validation.json", results)
    kt.write_json(
        kt.WORK / f"{STAGE}-settings.json",
        {
            "selected": STAGE,
            "branches": dict.fromkeys(kt.TARGETS, "network50"),
            "fit": settings,
        },
    )
    compare(STAGE)
    validate(STAGE)
    evaluate_trial(STAGE)
    print("Network Hessian fit and common validation complete", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("step", choices=("prepare", "fit"))
    with threadpool_limits(1):
        if p.parse_args().step == "prepare":
            prepare()
        else:
            fit()
