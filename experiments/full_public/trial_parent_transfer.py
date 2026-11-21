"""Transfer AIMNet2 bare-ωB97M pair curves into Korny's two existing energy surfaces.

One learned amplitude per valence-group pair controls a radial curve. Atomic
energy offsets from the auxiliary method cancel in balanced reactions and are
excluded. Energy differences, bare PBE0 derivative anchors, NIST minima and
radical Hessian observations calibrate the transfer on original training data.
The correction is merged into existing weights so analytic forces and D3
evaluation remain available from the ordinary force-field interface.
"""

import argparse
import pickle

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from experiments.full_public import aimnet_parent as parent
from experiments.full_public import kernel_trials as kt
from experiments.full_public import trial_aux, trial_fit
from experiments.full_public import trial_multiscale as ms
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_constraints import nist_rows
from experiments.full_public.trial_isomers import gram as isomers
from experiments.full_public.trial_parent_potential import CENTERS
from experiments.full_public.trial_spectra import evaluate_trial
from experiments.full_public.trial_validation import run as validate
from src.forcefield import ForceField

WORK = kt.WORK


class Curves:
    def __init__(self, tag):
        with np.load(parent.ROOT / f"potential-{tag}/weights.npz") as data:
            self.pair_ids = data["pair_ids"]
            self.spline = data["weights"][: 7 * len(self.pair_ids)].reshape(-1, 7)
        model = ForceField.load(WORK / "p43-public-high.npz")
        self.g = model.groups
        self.r = model.radii
        self.slot = np.full(45, -1, int)
        self.slot[self.pair_ids] = np.arange(len(self.pair_ids))
        self.cols = (
            1
            + len(model.elements)
            + (self.pair_ids[:, None] * 7 + np.arange(7)).ravel()
        )
        self.projection = (
            np.repeat(np.eye(len(self.pair_ids)), 7, axis=0)
            * self.spline.ravel()[:, None]
        )

    def design(self, z, r, gradient=False):
        i, j = np.triu_indices(len(z), 1)
        a, b = self.g[z[i]], self.g[z[j]]
        big = np.maximum(a, b)
        pair = self.slot[big * (big + 1) // 2 + np.minimum(a, b)]
        take = pair >= 0
        i, j, pair = i[take], j[take], pair[take]
        d = r[i] - r[j]
        dist = np.linalg.norm(d, axis=1)
        scale = self.r[z[i]] + self.r[z[j]]
        t = (dist[:, None] / scale[:, None] - CENTERS) / 0.22
        v = np.exp(-0.5 * t * t) * self.spline[pair]
        row = np.bincount(pair, v.sum(axis=1), minlength=len(self.pair_ids))
        if not gradient:
            return row
        factor = (v * t).sum(axis=1) / (0.22 * dist * scale)
        f = np.zeros((len(z), 3, len(self.pair_ids)))
        np.add.at(f, (i, slice(None), pair), d * factor[:, None])
        np.add.at(f, (j, slice(None), pair), -d * factor[:, None])
        return row, f.reshape(r.size, -1)


def run(tag):
    stage, key, prior = (
        ("p46-public", "parent46", "p44-public")
        if tag == "pilot"
        else ("p47-public", "parent47", "p44-public")
    )
    curves = Curves(tag)
    # The original public design stores exactly these pair features; the
    # 91,156-geometry Gram contains energy and all Cartesian force components.
    raw = np.load(WORK / "base-x.npy", mmap_mode="r")
    pub = np.empty((len(raw), len(curves.pair_ids)))
    for lo in range(0, len(raw), 512):
        block = raw[lo : lo + 512]
        pub[lo : lo + len(block)] = block[:, curves.cols] @ curves.projection
    gram = np.load(kt.OUT / "joint-gram.npy", mmap_mode="r")
    anchor = (
        curves.projection.T @ gram[np.ix_(curves.cols, curves.cols)] @ curves.projection
    )
    pts, npvt, meta = trial_aux.read_frames()
    rad = np.vstack([curves.design(z, r) for z, r in pts[npvt:]]).reshape(
        -1, 7, len(curves.pair_ids)
    )
    with (WORK / "radical-modes.pkl").open("rb") as f:
        saddle = np.array(
            [r[1][0].split("/")[-1].startswith("ts") for r in pickle.load(f)]
        )
    geom, _, _ = kt.inputs()
    with (WORK / "ids.pkl").open("rb") as f:
        _, names = pickle.load(f)
    results = {}
    for target in kt.TARGETS:
        s, y, src, fold, kind, case = d = kt.equation_data(target)
        x = s @ pub
        base = ms.baseline(target, -1, "parent", prior)
        g = kt.grouped_moments(d, x, y - s @ base, fold >= 0)
        eq = g["other"] + 96 * g["mid"] + 36 * g["gap"] + isomers(target, -1, pub, base)
        eq[:-1, :-1] += 0.24 * anchor
        model = ForceField.load(WORK / f"{prior}-{target}.npz", dispersion="d3")
        nz = nist_rows(lambda z, r: curves.design(z, r, True)[1], model)
        eq += 15.0 * nz.T @ nz
        if target == "wb97xd3":
            bm = ms.baseline(target, -1, "modes", prior)
            a, v = trial_fit.rad_rows(rad, bm, meta, np.where(saddle, 3.0, 1.0))
            eq += trial_fit.reduce_gram(a, v, np.repeat(meta[:, 0] >= 0, 3))
        lhs = eq[:-1, :-1].copy()
        lhs.flat[:: len(lhs) + 1] += 16.0
        w = cho_solve(cho_factor(lhs), eq[:-1, -1])
        # Both default and spin-specific legacy readouts refer to the common
        # radial block. A copied readout needs the same coefficient increment.
        delta = curves.projection @ w
        model.weights = model.weights.copy()
        model.weights[curves.cols] += delta
        for spin, (old, extra) in list(model.spin_readouts.items()):
            replacement = old.copy()
            replacement[curves.cols] += delta
            model.spin_readouts[spin] = (replacement, extra)
        dest = WORK / f"{stage}-{target}.npz"
        model.save_with_base(kt.original.BASE, dest)
        e = base + pub @ w
        ff = ForceField.load(dest, dispersion="d3")
        for m in (1, 2, 3):
            i = next(
                i
                for i, n in enumerate(names)
                if geom[n].state[1] == m and 4 <= len(geom[n].z) <= 10
            )
            z, r, qm = geom[names[i]].z, geom[names[i]].r, geom[names[i]].state
            np.testing.assert_allclose(pub[i], curves.design(z, r), atol=1e-8, rtol=0)
            en, force = ff.predict(z, r, *qm)
            np.testing.assert_allclose(en, e[i], atol=3.0e-6, rtol=0)
            v = np.random.default_rng(i).normal(size=r.shape)
            v /= np.linalg.norm(v)
            fd = (
                ff.energy(z, r - 2.0e-5 * v, *qm) - ff.energy(z, r + 2.0e-5 * v, *qm)
            ) / 4.0e-5
            np.testing.assert_allclose((force * v).sum(), fd, atol=6.0e-5, rtol=0)
        np.save(WORK / f"{key}-{target}-parent--1.npy", e)
        h = fold == -1
        np.savez_compressed(
            WORK / f"{stage}-{target}-equations.npz",
            case=case[h],
            source=src[h],
            kind=kind[h],
            error_kcal=(s @ e - y)[h] * kt.KCAL,
        )
        if target == "wb97xd3":
            modes = bm + rad @ w
            np.save(WORK / f"{key}-{target}-modes--1.npy", modes)
            np.save(WORK / f"{stage}-held-modes.npy", modes[meta[:, 0] == -1])
        results[target] = w.tolist()
        print("Completed matched data transfer", stage, target, flush=True)
    kt.write_json(
        WORK / f"{stage}-settings.json",
        {
            "selected": stage,
            "branches": dict.fromkeys(kt.TARGETS, key),
            "source": tag,
            "parent": prior,
            "pair_ids": curves.pair_ids.tolist(),
            "amplitudes": results,
        },
    )
    compare(stage)
    validate(stage)
    with threadpool_limits(1):
        evaluate_trial(stage)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", choices=["pilot", "full"], default="full")
    args = p.parse_args()
    with threadpool_limits(2):
        run(args.tag)
