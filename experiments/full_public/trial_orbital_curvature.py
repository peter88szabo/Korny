"""Recalibrate orbital-moment energy contributions against measured reactive Hessians.

The parent uses a Gaussian on high powers of a geometry-dependent hopping
matrix. An accurate relative energy can coexist with excessive Cartesian
curvature where those moments change quickly. Give each element group two
kernel widths and fit its amplitude to the existing reaction, force and
Hessian observations. Frozen internal prototype coefficients limit this
calibration to a small number of physically interpretable degrees of freedom.
"""

import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import numpy as np
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
from experiments.full_public import trial_stationary_response as reference
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_constraints import nist_rows
from experiments.full_public.trial_curvature_fit import anchor_blocks
from experiments.full_public.trial_isomers import gram as isomer_gram
from experiments.full_public.trial_spectra import evaluate_trial
from experiments.full_public.trial_stationary_audit import run as audit
from experiments.full_public.trial_validation import run as validate
from src.forcefield import ForceField
from src.orbital_moments import OrbitalMomentKernel, environments

WORK = kt.WORK
ROOT = WORK / "orbital-curvature41"
WIDTHS = (0.8, 1.5)
STAGE = "p41-public"
PARENT = reference.PARENT


class OrbitalReadout:
    """The public energy, Cartesian force and serialized model all use the same columns."""

    def __init__(self, target):
        self.ff = ForceField.load(WORK / f"{PARENT}-{target}.npz", dispersion="d3")
        self.index, self.basis = next(
            (i, b)
            for i, b in enumerate(self.ff.additional_bases)
            if isinstance(b, OrbitalMomentKernel)
        )
        b = self.basis
        if self.ff.spin_readouts:
            raise ValueError(
                "A spin-specific readout requires splitting the orbital coefficients"
            )
        self.keys = list(b.centers)
        self.offsets = np.cumsum([0] + [len(c) for c in b.centers.values()])
        self.n = len(self.keys)
        with np.load(WORK / "orbital-basis.npz") as ar:
            stored = OrbitalMomentKernel.load(ar)
        for k in self.keys:
            np.testing.assert_array_equal(b.centers[k], stored.centers[k])
            np.testing.assert_array_equal(b.projections[k], stored.projections[k])

    def design(self, z, r, forces=False):
        b = self.basis
        raw, jac = environments(z, r, b.groups, b.radii, forces)
        values = np.zeros((len(WIDTHS), self.n))
        df = np.zeros((r.size, len(WIDTHS), self.n)) if forces else None
        for a, k in enumerate(self.keys):
            take = b.groups[z] == k
            if not take.any():
                continue
            p = b.projections[k]
            delta = (raw[take] @ p)[:, None, :] - b.centers[k]
            d2 = np.einsum("akd,akd->ak", delta, delta)
            zero = np.einsum("kd,kd->k", b.centers[k], b.centers[k])
            weights = b.weights[self.offsets[a] : self.offsets[a + 1]]
            if forces:
                der = p.T @ jac[take].reshape(take.sum(), raw.shape[1], r.size)
            for j, w in enumerate(WIDTHS):
                kernel = np.exp(-0.5 * d2 / w**2)
                values[j, a] = (kernel - np.exp(-0.5 * zero / w**2)).sum(
                    axis=0
                ) @ weights
                if forces:
                    slope = np.einsum("ak,akd->ad", kernel * weights, delta) / w**2
                    df[:, j, a] = np.einsum("ad,adq->q", slope, der)
        return values.ravel(), df.reshape(r.size, -1) if forces else None

    def compress(self, side):
        """Read each mapped training array in bounded slices and reduce the basis by element."""
        parts = []
        for w in WIDTHS:
            full = np.load(WORK / f"orbital-{w:g}-{side}.npy", mmap_mode="r")
            out = np.zeros((len(full), self.n))
            for lo in range(0, len(full), 256):
                block = full[lo : lo + 256]
                for a, (l, r) in enumerate(zip(self.offsets, self.offsets[1:])):
                    out[lo : lo + len(block), a] = (
                        block[:, l:r] @ self.basis.weights[l:r]
                    )
            parts.append(out)
            del full
        return np.hstack(parts)

    def deploy(self, weights, path):
        """Merge the correction at the parent's width, retaining one copy of each map."""
        parent_weights = self.basis.weights.copy()
        for w, scales in zip(WIDTHS, weights.reshape(len(WIDTHS), self.n)):
            delta = parent_weights * np.repeat(scales, np.diff(self.offsets))
            if w == self.basis.length:
                self.basis.weights = parent_weights + delta
            else:
                basis = deepcopy(self.basis)
                basis.length = w
                basis.weights = delta
                self.ff.additional_bases.append(basis)
        self.ff.save_with_base(kt.original.BASE, path)


def archive_rows(item):
    target, name, _, z, r, _, hessian = item
    key = target + "-" + hashlib.sha256(name.encode()).hexdigest()[:20] + ".npz"
    path = ROOT / key
    if path.exists():
        return

    def gradients(pos):
        return np.column_stack(
            (models[target].design(z, pos, True)[1], np.zeros(r.size))
        )

    rows = reference.stationary_rows(z, r, hessian, gradients)
    with np.load(reference.ROOT / key) as stored:
        if str(stored["name"]) != name:
            raise ValueError("Stationary response cache has different structure ID")
        # The parent is identical. Reuse its residual calculated in the exact
        # same frame, reference mode order and displaced Cartesian coordinates.
        rows[:, -1] = stored["rows"][:, -1]
        fold = int(stored["fold"])
    with path.with_suffix(".pending").open("wb") as f:
        np.savez(f, gram=rows.T @ rows, fold=fold)
    path.with_suffix(".pending").replace(path)


def train(target):
    model = models[target]
    s, y, src, fold, kind, case = d = kt.equation_data(target)
    n = s.shape[1]
    main = model.compress("main")
    pub = main[:n]
    x = s @ pub
    base = ms.baseline(target, -1, "parent", PARENT)
    residual = y - s @ base
    blocks = kt.grouped_moments(d, x, residual, fold >= 0)
    eq = (
        blocks["other"]
        + 48 * blocks["mid"]
        + 36 * blocks["gap"]
        + isomer_gram(target, -1, pub, base)
    )
    nr = nist_rows(lambda z, r: model.design(z, r, True)[1], model.ff)
    eq += 8 * nr.T @ nr
    ef, curvature = anchor_blocks(main, n, f"orbital41-{target}")
    eq[:-1, :-1] += 0.04 * ef + 0.012 * curvature
    shape = np.zeros((4, *eq.shape))
    for fn in sorted(ROOT.glob(target + "-*.npz")):
        with np.load(fn) as f:
            shape[int(f["fold"])] += f["gram"]

    def fit(sh, strength):
        g = eq + strength * sh
        # Each readout multiplier is dimensionless. Ridge and the observed
        # reaction equations regularize distant element groups weakly sampled in Hessians.
        g.flat[:: len(g) + 1] += 22.0
        return cho_solve(cho_factor(g[:-1, :-1]), g[:-1, -1])

    scores = {}
    for st in (0.3, 3.0, 30.0, 120.0):
        scores[st] = []
        for held in shape:
            v = np.r_[fit(shape.sum(axis=0) - held, st), -1.0]
            scores[st].append(float(v @ held @ v / held[-1, -1]))
    chosen = min(scores, key=lambda v: np.mean(scores[v]))
    coeff = fit(shape.sum(axis=0), chosen)
    energy = base + pub @ coeff
    path = WORK / f"{STAGE}-{target}.npz"
    model.deploy(coeff, path)
    deployed = ForceField.load(path, dispersion="d3")
    z, r = trial_fit.sample()
    e, f = deployed.predict(z, r)
    np.testing.assert_allclose(e, energy[trial_fit.sample.index], atol=2.0e-6, rtol=0)
    dr = np.random.default_rng(25).normal(size=r.shape)
    dr /= np.linalg.norm(dr)
    fd = (
        deployed.energy(z, r - 0.00002 * dr) - deployed.energy(z, r + 0.00002 * dr)
    ) / 0.00004
    np.testing.assert_allclose(fd, (f * dr).sum(), atol=5.0e-5)
    use = fold == -1
    np.save(WORK / f"orbital41-{target}-parent--1.npy", energy)
    np.savez_compressed(
        WORK / f"{STAGE}-{target}-equations.npz",
        case=case[use],
        source=src[use],
        kind=kind[use],
        error_kcal=(s @ energy - y)[use] * kt.KCAL,
    )
    if target == "wb97xd3":
        # Compression must use the frozen parent weights, preceding deployment.
        raw = OrbitalReadout(target).compress("rad")
        modes = ms.baseline(target, -1, "modes", PARENT) + (raw @ coeff).reshape(-1, 7)
        np.save(WORK / f"orbital41-{target}-modes--1.npy", modes)
        np.save(
            WORK / f"{STAGE}-held-modes.npy",
            modes[np.load(WORK / "rad-meta.npy")[:, 0] == -1],
        )
    print(
        "Calibrated orbital curvature",
        target,
        chosen,
        "training shape ratios",
        scores[chosen],
        flush=True,
    )
    return {
        "strength": chosen,
        "dev": scores,
        "groups": model.keys,
        "widths": WIDTHS,
        "coefficients": coeff.tolist(),
    }


if __name__ == "__main__":
    ROOT.mkdir(exist_ok=True)
    with threadpool_limits(1):
        models = {t: OrbitalReadout(t) for t in kt.TARGETS}
        jobs = reference.select_training()
        for target in kt.TARGETS:
            item = next(i for i in jobs if i[0] == target)
            z, r = item[3:5]
            model = models[target]
            original = model.basis
            val, jac = model.design(z, r, True)
            j = WIDTHS.index(original.length)
            a, df = original.design(z, r, True)
            np.testing.assert_allclose(
                val.reshape(2, -1)[j].sum(), a @ original.weights, atol=1.0e-9
            )
            np.testing.assert_allclose(
                jac.reshape(r.size, 2, -1)[:, j].sum(axis=1),
                df @ original.weights,
                atol=2.0e-8,
            )
        with ProcessPoolExecutor(
            max_workers=2, mp_context=mp.get_context("fork")
        ) as pool:
            for i, _ in enumerate(pool.map(archive_rows, jobs, buffersize=6)):
                if (i + 1) % 144 == 0:
                    print(
                        "Completed reactive orbital curvature rows",
                        i + 1,
                        len(jobs),
                        flush=True,
                    )
        results = {t: train(t) for t in kt.TARGETS}
        kt.write_json(
            WORK / f"{STAGE}-settings.json",
            {
                "selected": STAGE,
                "branches": dict.fromkeys(kt.TARGETS, "orbital41"),
                "fits": results,
            },
        )
        compare(STAGE)
        validate(STAGE)
        evaluate_trial(STAGE)
        audit(STAGE)
