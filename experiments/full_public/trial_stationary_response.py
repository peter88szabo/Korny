"""Fit force vectors and signed Hessian-vector products at reactive stationary points.

At a saddle, a single negative eigenvalue along the reaction path is expected.
Fitting only its diagonal projection leaves spurious soft orthogonal modes
unconstrained. The stored DFT Hessians also describe how *all* atomic forces
change along an internal displacement. Use the softest two modes and a bend
and stretch, including the off-diagonal response in each case.

The calibration starts from p37 and retains its reaction and PBE0 constraints.
RDB7 DFT shapes are auxiliary observations for its higher-level energy surface.
Grambow geometries, Hessians and energies share the same electronic level.
"""

import hashlib
import multiprocessing as mp
import pickle
import re
import sqlite3
import tarfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from rdkit import Chem
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from experiments.full_public import (
    kernel_trials as kt,
)
from experiments.full_public import (
    trial_fit,
)
from experiments.full_public import (
    trial_multiscale as ms,
)
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_constraints import nist_rows
from experiments.full_public.trial_isomers import gram as isomer_gram
from experiments.full_public.trial_measured_curvature import load_basis
from experiments.full_public.trial_spectra import evaluate_trial
from experiments.full_public.trial_stationary_audit import run as audit
from experiments.full_public.trial_validation import run as validate
from experiments.reactivity.frequency_validation import TO_CM1
from experiments.reactivity.radical_curvatures import ARCHIVE, final_hessian
from experiments.reactivity.stationary_fit import internal_transform
from src.forcefield import ForceField

PARENT = "p37-public"
STAGE = "p40-public"
WORK = kt.WORK
ROOT = WORK / "stationary-response40"


def select_training():
    """Rank convergence-verified training points by ID, independently of their errors."""
    taken = set()
    eligible = {}
    for target in kt.TARGETS:
        s, _, _, fold, *_ = kt.equation_data(target)
        taken.update(s[fold == -1].indices.tolist())
        eligible[target] = set(s[fold >= 0].indices.tolist())
    with (WORK / "ids.pkl").open("rb") as handle:
        ids, _ = pickle.load(handle)
    selected = defaultdict(list)
    with sqlite3.connect(
        f"file:{WORK / 'rdb-dft-frames.sqlite'}?mode=ro", uri=True
    ) as db:
        for name, idx in db.execute(
            "select name,coord_id from stationary where converged=1 and hessian is not null and length(z) between 5 and 24"
        ):
            if idx in eligible["high"] and idx not in taken:
                selected["high", "/ts" in name].append((name, idx))
    with (WORK / "radical-modes.pkl").open("rb") as handle:
        for fold, (name, z, _, state, converged, _) in pickle.load(handle):
            match = re.search(r"/rxn(\d+)/(p|r|ts)\d+\.log$", name)
            if (
                not match
                or not converged
                or fold < 0
                or state != (0, 2)
                or not 5 <= len(z) <= 24
            ):
                continue
            idx = ids.get(f"Grambow/{match[1]}/{match[2]}")
            if idx in eligible["wb97xd3"] and idx not in taken:
                selected["wb97xd3", match[2] == "ts"].append((name, idx))

    names = {}
    for (target, saddle), choices in selected.items():
        used = set()
        for name, idx in sorted(
            choices, key=lambda row: hashlib.sha256(row[0].encode()).digest()
        ):
            if idx in used:
                continue
            used.add(idx)
            names[name] = (target, saddle)
            if len(used) == 144:
                break
    jobs = []
    with sqlite3.connect(
        f"file:{WORK / 'rdb-dft-frames.sqlite'}?mode=ro", uri=True
    ) as db:
        for name, (target, saddle) in names.items():
            if target == "high":
                z, r, q, m, h = db.execute(
                    "select z,r,q,m,hessian from stationary where name=?", (name,)
                ).fetchone()
                z = np.frombuffer(z, np.uint8)
                r = np.frombuffer(r, float).reshape(-1, 3)
                jobs.append(
                    (
                        target,
                        name,
                        saddle,
                        z,
                        r,
                        (q, m),
                        np.frombuffer(h, float).reshape(r.size, r.size),
                    )
                )
    with tarfile.open(ARCHIVE, "r|gz") as arc:
        for member in arc:
            if member.name in names and names[member.name][0] == "wb97xd3":
                z, r, st, h = final_hessian(
                    arc.extractfile(member).read().decode(errors="replace")
                )
                jobs.append(
                    ("wb97xd3", member.name, names[member.name][1], z, r, st, h)
                )
    if len(jobs) != len(names):
        raise ValueError("Selected Hessian missing from archive")
    return jobs


def stationary_rows(z, r, hessian, gradients):
    """Project signed force changes onto the eigenvectors of the archived Hessian.

    Each column except the last is a conservative basis force. The last is
    the fixed parent force, converted here into the response still to fit.
    """
    mass = np.array(
        [Chem.GetPeriodicTable().GetMostCommonIsotopeMass(int(a)) for a in z]
    )
    t = internal_transform(mass, r)
    values, axes = np.linalg.eigh(t.T @ hessian @ t)
    freq = TO_CM1 * np.sign(values) * np.sqrt(abs(values))
    directions = t @ axes
    # Distinct reference directions only, including two low-frequency axes to
    # penalize false instabilities transverse to a reaction path.
    modes = np.unique([0, 1, np.argmin(abs(freq - 900.0)), len(values) - 1])

    center = gradients(r)
    center[:, -1] *= -1
    rows = [3.0 * center / np.sqrt(r.size)]
    for j in modes:
        norm = np.linalg.norm(directions[:, j])
        step = (0.0015 / norm) * directions[:, j].reshape(r.shape)
        dynamic = (
            directions.T @ (gradients(r - step) - gradients(r + step)) * (norm / 0.003)
        )
        dynamic[:, -1] *= -1
        dynamic[j, -1] += values[j]
        # Around a wavenumber, transform curvature errors to approximately
        # frequency errors. A finite scale treats soft and imaginary modes stably.
        scale = TO_CM1**2 / (
            2 * np.maximum(350.0, 0.5 * (abs(freq) + abs(freq[j]))) * 150.0
        )
        scale *= 0.6 / np.sqrt(len(modes))
        scale[j] *= 1.5
        rows.append(dynamic * scale[:, None])
    return np.vstack(rows)


def response(item):
    target, name, saddle, z, r, state, hessian = item
    path = ROOT / (
        target + "-" + hashlib.sha256(name.encode()).hexdigest()[:20] + ".npz"
    )
    if path.exists():
        return

    def gradients(pos):
        return np.column_stack(
            (
                basis.design(z, pos, True)[1][:, columns],
                parents[target].predict(z, pos, *state)[1].ravel(),
            )
        )

    rows = stationary_rows(z, r, hessian, gradients)
    comp = str(tuple(sorted(map(int, z)))).encode()
    fold = int.from_bytes(hashlib.sha256(comp).digest()[:4], "big") % 4
    with path.with_suffix(".pending").open("wb") as handle:
        np.savez(
            handle,
            rows=rows.astype(np.float32),
            fold=fold,
            saddle=saddle,
            name=name,
        )
    path.with_suffix(".pending").replace(path)


def train(target):
    b, idx = load_basis()
    s, y, src, fold, kind, case = d = kt.equation_data(target)
    n = s.shape[1]
    pub = np.load(WORK / "periodic-0.85-main.npy", mmap_mode="r")[:n, idx].copy(
        order="C"
    )
    base = ms.baseline(target, -1, "parent", PARENT)
    x = s @ pub
    train = fold >= 0
    block = kt.grouped_moments(d, x, y - s @ base, train)
    eq = (
        block["other"]
        + 48 * block["mid"]
        + 36 * block["gap"]
        + isomer_gram(target, -1, pub, base)
    )
    ff = ForceField.load(WORK / f"{PARENT}-{target}.npz", dispersion="d3")
    nr = nist_rows(lambda z, r: b.design(z, r, True)[1][:, idx], ff)
    eq += 18 * nr.T @ nr
    for suffix, scale in [("ef", 0.12), ("curvature", 0.045)]:
        full = np.load(WORK / f"periodic-0.85-anchor-{suffix}.npy", mmap_mode="r")
        eq[:-1, :-1] += scale * full[np.ix_(idx, idx)]
    # Accumulate by formula before consulting candidate strengths. Formula
    # holdouts assess the added calibration on the fixed parent surface.
    blocks = np.zeros((4, len(idx) + 1, len(idx) + 1))
    counts = np.zeros((4, 2), int)
    for path in sorted(ROOT.glob(target + "-*.npz")):
        with np.load(path) as f:
            a = f["rows"].astype(float)
            h = int(f["fold"])
            blocks[h] += a.T @ a
            counts[h, int(f["saddle"])] += 1
    total = blocks.sum(axis=0)

    def solve(observations, strength):
        g = eq + strength * observations
        g.flat[:: len(g) + 1] += 0.8
        return cho_solve(cho_factor(g[:-1, :-1]), g[:-1, -1])

    scores = {}
    for strength in (0.3, 3.0, 30.0, 120.0):
        scores[strength] = []
        for held_block in blocks:
            coef = solve(total - held_block, strength)
            v = np.r_[coef, -1.0]
            scores[strength].append(float(v @ held_block @ v / held_block[-1, -1]))
    chosen = min(scores, key=lambda k: np.mean(scores[k]))
    weights = solve(total, chosen)
    en = base + pub @ weights
    key = "stationary40"
    np.save(WORK / f"{key}-{target}-parent--1.npy", en)
    b.weights[:] = 0.0
    b.weights[idx] = weights
    ff.additional_bases.append(b)
    path = WORK / f"{STAGE}-{target}.npz"
    ff.save_with_base(kt.original.BASE, path)
    restored = ForceField.load(path, dispersion="d3")
    z, r = trial_fit.sample()
    energy, force = restored.predict(z, r)
    np.testing.assert_allclose(energy, en[trial_fit.sample.index], atol=2.0e-6, rtol=0)
    dr = np.random.default_rng(991).normal(size=r.shape)
    dr /= np.linalg.norm(dr)
    estimate = (
        restored.energy(z, r - 2.0e-5 * dr) - restored.energy(z, r + 2.0e-5 * dr)
    ) / 4.0e-5
    np.testing.assert_allclose(estimate, np.sum(force * dr), atol=5.0e-5, rtol=0)
    held = fold == -1
    np.savez_compressed(
        WORK / f"{STAGE}-{target}-equations.npz",
        source=src[held],
        case=case[held],
        kind=kind[held],
        error_kcal=(s @ en - y)[held] * kt.KCAL,
    )
    if target == "wb97xd3":
        rad = np.load(WORK / "periodic-0.85-rad.npy", mmap_mode="r")[:, idx]
        modes = ms.baseline(target, -1, "modes", PARENT) + (rad @ weights).reshape(
            -1, 7
        )
        np.save(WORK / f"{key}-{target}-modes--1.npy", modes)
        np.save(
            WORK / f"{STAGE}-held-modes.npy",
            modes[np.load(WORK / "rad-meta.npy")[:, 0] == -1],
        )
    print(
        "Fitted stationary force and curvature corrections",
        target,
        counts.sum(axis=0),
        chosen,
        flush=True,
    )
    return {
        "strength": chosen,
        "development": scores,
        "counts_by_formula_fold": counts.tolist(),
    }


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True)
    with threadpool_limits(1):
        basis, columns = load_basis()
        parents = {
            t: ForceField.load(WORK / f"{PARENT}-{t}.npz", dispersion="d3")
            for t in kt.TARGETS
        }
        jobs = select_training()
        print(
            "Selected training Hessians and equilibrium forces", len(jobs), flush=True
        )
        with ProcessPoolExecutor(
            max_workers=4, mp_context=mp.get_context("fork")
        ) as pool:
            for j, _ in enumerate(pool.map(response, jobs, buffersize=12)):
                if (j + 1) % 96 == 0:
                    print(
                        "Completed reactive Hessian-vector features",
                        j + 1,
                        len(jobs),
                        flush=True,
                    )
    with threadpool_limits(3):
        fits = {t: train(t) for t in kt.TARGETS}
        kt.write_json(
            WORK / f"{STAGE}-settings.json",
            {
                "selected": STAGE,
                "branches": dict.fromkeys(kt.TARGETS, "stationary40"),
                "fits": fits,
            },
        )
        compare(STAGE)
        validate(STAGE)
    with threadpool_limits(1):
        evaluate_trial(STAGE)
        audit(STAGE)
