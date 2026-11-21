"""One matched linear refit of p37 with a doubled transition-state midpoint weight.

Rebuild the final solve for each of p37's two reference levels, including all
unchanged derivative observations and regularizers. The high-level readout is
the measured-QM7-X curvature calibration; the radical readout is p36's occupied
bond kernel. Reconstruct their published coefficients at the original weights
before scoring one predeclared alternative. All earlier coefficients are frozen.
"""

import pickle

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from experiments.full_public import (
    kernel_trials as kt,
)
from experiments.full_public import (
    trial_fit,
)
from experiments.full_public import trial_measured_curvature as measured
from experiments.full_public import (
    trial_multiscale as ms,
)
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_constraints import nist_rows
from experiments.full_public.trial_isomers import gram as isomer_gram
from experiments.full_public.trial_spectra import evaluate_trial
from experiments.full_public.trial_validation import run as validate
from src.forcefield import ForceField
from src.occupied_orbitals import OccupationKernel

WORK = kt.WORK
STAGE = "p42-public"
PARENT = "p37-public"
ORIGINAL_MID = 48.0
NEW_MID = 96.0
GAP = 36.0


def run(target):
    high = target == "high"
    ff = ForceField.load(WORK / f"{PARENT}-{target}.npz", dispersion="d3")
    head = ff.additional_bases[-1]
    d = kt.equation_data(target)
    s, y, src, fold, kind, case = d
    n = s.shape[1]
    parent = "p36-public" if high else "p33-public"
    base = ms.baseline(target, -1, "parent", parent)
    if high:
        _basis, idx = measured.load_basis()
        prefix = "periodic-0.85"
        pub = np.load(WORK / f"{prefix}-main.npy", mmap_mode="r")[:n, idx].copy(
            order="C"
        )
        np.testing.assert_array_equal(
            head.weights[np.setdiff1d(np.arange(head.size), idx)], 0.0
        )
    else:
        assert (
            isinstance(head, OccupationKernel)
            and head.bond_populations
            and head.length == 0.74
        )
        idx = np.arange(head.size)
        prefix = "orbital-bond-0.74"
        pub = np.load(WORK / f"{prefix}-main.npy", mmap_mode="r")[:n]
    x = s @ pub
    block = kt.grouped_moments(d, x, y - s @ base, fold >= 0)
    other = block["other"] + isomer_gram(target, -1, pub, base)
    fixed = np.zeros_like(other)
    for suffix, mult in [
        ("ef", 0.12 if high else 0.02),
        ("curvature", 0.045 if high else 0.006),
    ]:
        a = np.load(WORK / f"{prefix}-anchor-{suffix}.npy", mmap_mode="r")
        fixed[:-1, :-1] += mult * a[np.ix_(idx, idx)]
    if high:
        for f in sorted((WORK / "measured-curvature37").glob("*.npz")):
            with np.load(f) as ar:
                rows = ar["rows"].astype(float)
            fixed += 8.0 * (rows.T @ rows)
    else:
        old = ForceField.load(kt.OUT / "joint-pbe0-reaction.npz", dispersion="d3")
        nr = nist_rows(lambda z, r: head.design(z, r, True)[1], old)[:, :-1]
        fixed[:-1, :-1] += 30.0 * (nr.T @ nr)
        meta = np.load(WORK / "rad-meta.npy")
        with (WORK / "radical-modes.pkl").open("rb") as f:
            saddle = np.asarray(
                [item[1][0].split("/")[-1].startswith("ts") for item in pickle.load(f)]
            )
        rad = np.load(WORK / f"{prefix}-rad.npy", mmap_mode="r").reshape(
            -1, 7, head.size
        )
        bm = ms.baseline(target, -1, "modes", parent)
        for lo in range(0, len(meta), 128):
            sl = slice(lo, lo + 128)
            mat, yy = trial_fit.rad_rows(
                rad[sl], bm[sl], meta[sl], np.where(saddle[sl], 3.0, 1.0)
            )
            active = np.repeat(meta[sl, 0] >= 0, 3)
            fixed += trial_fit.reduce_gram(mat, yy, active)

    def fit(mid):
        energy = other + GAP * block["gap"] + mid * block["mid"]
        lhs = energy[:-1, :-1] + fixed[:-1, :-1]
        lhs.flat[:: len(lhs) + 1] += 0.6 if high else 0.25
        rhs = (0.30 if high else 1.0) * energy[:-1, -1] + fixed[:-1, -1]
        w = cho_solve(cho_factor(lhs), rhs)
        if high:
            move = x[(fold >= 0) & np.isin(src, kt.BARRIERS)] @ w * kt.KCAL
            w *= min(1.0, 1.1 / max(np.sqrt(np.mean(move**2)), 1.0e-12))
        return w

    control = fit(ORIGINAL_MID)
    np.testing.assert_allclose(control, head.weights[idx], atol=2.0e-5, rtol=3.0e-5)
    np.testing.assert_allclose(
        base + pub @ control,
        ms.baseline(target, -1, "parent", PARENT),
        atol=2.0e-6,
        rtol=0,
    )
    print(
        "Reproduced p37 linear coefficients and geometry energies", target, flush=True
    )
    coeff = fit(NEW_MID)
    head.weights[idx] = coeff
    energy = base + pub @ coeff
    key = "midpoint42"
    np.save(WORK / f"{key}-{target}-parent--1.npy", energy)
    path = WORK / f"{STAGE}-{target}.npz"
    ff.save_with_base(kt.original.BASE, path)
    restored = ForceField.load(path, dispersion="d3")
    z, r = trial_fit.sample()
    e, f = restored.predict(z, r)
    np.testing.assert_allclose(e, energy[trial_fit.sample.index], atol=2e-6, rtol=0)
    dr = np.random.default_rng(21).normal(size=r.shape)
    dr /= np.linalg.norm(dr)
    fd = (restored.energy(z, r - 2e-5 * dr) - restored.energy(z, r + 2e-5 * dr)) / 4e-5
    np.testing.assert_allclose(np.sum(f * dr), fd, atol=5.0e-5)
    held = fold == -1
    np.savez_compressed(
        WORK / f"{STAGE}-{target}-equations.npz",
        source=src[held],
        case=case[held],
        kind=kind[held],
        error_kcal=(s @ energy - y)[held] * kt.KCAL,
    )
    if not high:
        modes = bm + rad @ coeff
        np.save(WORK / f"{key}-{target}-modes--1.npy", modes)
        np.save(WORK / f"{STAGE}-held-modes.npy", modes[meta[:, 0] == -1])
    print("Completed controlled midpoint reweighting", target, flush=True)


if __name__ == "__main__":
    with threadpool_limits(3):
        for t in kt.TARGETS:
            run(t)
        kt.write_json(
            WORK / f"{STAGE}-settings.json",
            {
                "selected": STAGE,
                "branches": dict.fromkeys(kt.TARGETS, "midpoint42"),
                "control": PARENT,
                "old_mid": ORIGINAL_MID,
                "new_mid": NEW_MID,
                "gap": GAP,
            },
        )
        compare(STAGE)
        validate(STAGE)
    with threadpool_limits(1):
        evaluate_trial(STAGE)
