"""Transfer conserved atom-resolved spin populations to reaction surfaces.

A molecule can have extensive ordinary bond energy and a localized spin
contribution at the same time. Each new energy feature weights a local
chemical environment by a signed spin population. Fit these contributions
alongside the p48 size correction on the original reaction, gradient and
Hessian constraints, the verified AutoMeKin channels and training spin scans.
Reserve ChEMBL-20, the six radical path holdouts, the cBD-CN network and
benzopinacol from this energy fit.
"""

import argparse
import json
import multiprocessing as mp
import pickle
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from experiments.full_public import kernel_trials as kt
from experiments.full_public import trial_aux, trial_fit
from experiments.full_public import trial_multiscale as ms
from experiments.full_public import trial_spin_extent as extent
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_spin_population import ROOT as IONS
from experiments.full_public.trial_validation import run as validate
from src.forcefield import ForceField
from src.spin_local import DIM, GROUPS, SpinLocalKernel

WORK = kt.WORK
ROOT = extent.DATA / "local-spin49"
PARENT = "p44-public"
STAGE = "p49-public"
BATCH = 128
D = DIM * GROUPS
TARGET_DFT = "wb97xd3"


def blank():
    ff = ForceField.load(WORK / f"{PARENT}-high.npz")
    g = ff.groups.copy()
    g[1] = 0
    return SpinLocalKernel(
        g, ff.radii, np.zeros((3, GROUPS, DIM)), np.zeros((3, GROUPS, DIM)), np.zeros(D)
    )


def geometries():
    global modes, meta, ext, labels, links, refs, names
    geom, _, _ = kt.inputs()
    with (WORK / "ids.pkl").open("rb") as f:
        _, names = pickle.load(f)
    public = [(geom[n].z, geom[n].r, *geom[n].state) for n in names]
    pts, npvt, meta = trial_aux.read_frames()
    with (WORK / "radical-modes.pkl").open("rb") as f:
        modes = pickle.load(f)
    rs = np.repeat(np.array([r[1][3] for r in modes]), 7, axis=0)
    radical = [
        (z, r, int(q), int(m)) for (z, r), (q, m) in zip(pts[npvt:], rs, strict=True)
    ]
    scan, labels = extent.scans()
    network, links, refs = extent.network()
    ext = scan + network
    new = [(z, r, q, m) for _, z, r, q, m, _, _ in ext]
    return {"public": public, "radical": radical, "external": new}


def encode(job):
    kind, lo = job
    dest = ROOT / f"{kind}-{lo:06d}.npz"
    if dest.is_file():
        return
    types = []
    site = []
    ptr = [0]
    states = []
    for z, r, q, m in rows[kind][lo : lo + BATCH]:
        states.append((q, m))
        if m > 1:
            t, x, _ = basis.sites(z, r)
            types.append(t)
            site.append(x)
            ptr.append(ptr[-1] + len(z))
        else:
            ptr.append(ptr[-1])
    with dest.with_suffix(".pending").open("wb") as f:
        np.savez_compressed(
            f,
            t=np.concatenate(types).astype(np.uint8)
            if types
            else np.empty(0, np.uint8),
            x=np.vstack(site) if site else np.empty((0, DIM)),
            ptr=ptr,
            states=states,
        )
    dest.with_suffix(".pending").replace(dest)


def prepare():
    global rows, basis
    ROOT.mkdir(exist_ok=True)
    basis = blank()
    rows = geometries()
    # The raw features have no fitted parameters, so this cache contains
    # neither held-out targets nor statistics learned from held-out atoms.
    jobs = [(k, i) for k, rr in rows.items() for i in range(0, len(rr), BATCH)]
    ids = [n for n, *_ in ext]
    if ids != json.loads((extent.ROOT / "ids.json").read_text()):
        raise ValueError("External energy order changed")
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context("fork")) as pool:
        for i, _ in enumerate(pool.map(encode, jobs, buffersize=9)):
            if (i + 1) % 200 == 0:
                print(
                    "Prepared labeled spin-transfer feature blocks",
                    i + 1,
                    len(jobs),
                    flush=True,
                )
    print(
        "Spin-transfer geometry cache ready",
        {k: len(v) for k, v in rows.items()},
        flush=True,
    )


def compressed(kind, n, model):
    """Contract precomputed atomic densities with independently trained spin populations."""
    val = np.zeros((n, D))
    mols = np.zeros(n, int)
    for lo in range(0, n, BATCH):
        with np.load(ROOT / f"{kind}-{lo:06d}.npz") as f:
            t, x, ptr, st = (f[k] for k in ("t", "x", "ptr", "states"))
        for j, (q, m) in enumerate(st):
            mols[lo + j] = m
            if m == 1:
                continue
            a, b = ptr[j : j + 2]
            tt = t[a:b]
            v = x[a:b]
            pop, _ = model.population(tt, v, q, m)
            np.add.at(val[lo + j].reshape(GROUPS, DIM), tt, v * pop[:, None])
    return val, mols


def expanded(local, spin, e44, e48):
    diff = e48 - e44
    return np.column_stack((local, diff * (spin == 2), diff * (spin == 3)))


def fit():
    global rows
    rows = geometries()
    with np.load(IONS / "spin-population.npz") as f:
        basis = SpinLocalKernel.load(f)
    features = {k: compressed(k, len(rr), basis) for k, rr in rows.items()}
    external = np.vstack(
        [
            np.load(extent.ROOT / f"block-{k:06d}.npz")["rows"]
            for k in range(0, len(ext), 48)
        ]
    )
    ix = {r[0]: j for j, r in enumerate(ext)}
    settings = json.loads((WORK / "p48-public-settings.json").read_text())[
        "coefficients"
    ]
    source_eq = {}
    source_val = {}
    for t in kt.TARGETS:
        wx = np.array(settings[t])
        b = external[:, 9]
        e48 = external[:, 8] + external[:, :8] @ wx
        fx = expanded(*features["external"], b, e48)
        g = np.zeros((D + 3, D + 3))
        held_items = []
        if t == "high":
            for c in links:

                def side(ns, fx=fx, b=b):
                    ids = [ix[n] for n in ns]
                    return np.r_[
                        fx[ids].sum(axis=0), sum(refs[n] for n in ns) - b[ids].sum()
                    ]

                r, p, ts = side(c["reactant"]), side(c["product"]), side([c["ts"]])
                mid = ts - (r + p) / 2
                gap = p - r
                if c["split"] == "train":
                    g += 24 * np.outer(mid, mid) + 20 * np.outer(gap, gap)
                else:
                    held_items.append(
                        (
                            c["system"],
                            c["channel"],
                            len(c["reactant"]) + len(c["product"]),
                            gap,
                            mid,
                            ts - r,
                            ts - p,
                        )
                    )
        else:
            by = defaultdict(list)
            for n, e, m, sys, method, split in labels:
                i = ix[n]
                by[sys, method, split, m].append(np.r_[fx[i], e - b[i]])
            for (sys, method, part, m), vs in by.items():
                other = (sys, method, part, 1)
                if part != "train" or m != 3 or other not in by:
                    continue
                a = np.asarray(vs) - np.mean(by[other], axis=0)
                g += (900.0 / len(a)) * (a.T @ a)
        source_eq[t] = g
        source_val[t] = (fx, b, held_items)
    branches = {}
    scores = {}
    for t in kt.TARGETS:
        S, y, src, fold, kind, case = d = kt.equation_data(t)
        parent = ms.baseline(t, -1, "parent", PARENT)
        p48 = ms.baseline(t, -1, "parent", "p48-public")
        x = expanded(*features["public"], parent, p48)
        eq = kt.grouped_moments(d, S @ x, y - S @ parent, fold >= 0)
        G = eq["other"] + 96 * eq["mid"] + 36 * eq["gap"] + source_eq[t]
        if t == TARGET_DFT:
            bm = ms.baseline(t, -1, "modes", PARENT)
            z48 = ms.baseline(t, -1, "modes", "p48-public")
            rad = expanded(*features["radical"], bm.ravel(), z48.ravel()).reshape(
                -1, 7, D + 2
            )
            saddles = np.array([r[1][0].split("/")[-1].startswith("ts") for r in modes])
            a, ref = trial_fit.rad_rows(rad, bm, meta, np.where(saddles, 3.0, 1.0))
            G += trial_fit.reduce_gram(a, ref, np.repeat(meta[:, 0] >= 0, 3))
        # Regularize the molecular energy readout. The final two terms
        # interpolate the extensive/intensive versions for each spin, using
        # only the actual training labels for that surface.
        lhs = G[:-1, :-1].copy()
        lhs.flat[:: len(lhs) + 1] += 8.0
        coef = cho_solve(cho_factor(lhs), G[:-1, -1])
        use = fold == -1
        e = parent + x @ coef
        ff = ForceField.load(WORK / f"{PARENT}-{t}.npz", dispersion="d3")
        scaled = ForceField.load(WORK / f"p48-public-{t}.npz", dispersion="d3")
        for m in (2, 3):
            w, extra = ff.spin_readouts.get(m, (ff.weights, {}))
            donor = scaled.spin_readouts[m][0]
            c = coef[D + m - 2]
            ff.spin_readouts[m] = w + c * (donor - w), extra
            scale, dw = scaled.spin_size_readouts[m]
            ff.spin_size_readouts[m] = scale, c * dw
        local = SpinLocalKernel(
            basis.groups, basis.radii, basis.logits, basis.polarization, coef[:D]
        )
        ff.additional_bases.append(local)
        dest = WORK / f"{STAGE}-{t}.npz"
        ff.save_with_base(kt.original.BASE, dest)
        ff = ForceField.load(dest, dispersion="d3")
        for m in (1, 2, 3):
            i = next(
                i
                for i, rr in enumerate(rows["public"])
                if rr[-1] == m and 5 < len(rr[0]) < 15
            )
            z, r, q, mu = rows["public"][i]
            en, f = ff.predict(z, r, q, mu)
            np.testing.assert_allclose(en, e[i], atol=3.0e-6, rtol=0)
            dr = np.random.default_rng(i).normal(size=r.shape)
            dr /= np.linalg.norm(dr)
            fd = (
                ff.energy(z, r - 2.0e-5 * dr, q, mu)
                - ff.energy(z, r + 2.0e-5 * dr, q, mu)
            ) / 4.0e-5
            np.testing.assert_allclose(fd, np.sum(f * dr), atol=6.0e-5, rtol=0)
        np.save(WORK / f"spin49-{t}-parent--1.npy", e)
        np.savez_compressed(
            WORK / f"{STAGE}-{t}-equations.npz",
            source=src[use],
            kind=kind[use],
            case=case[use],
            error_kcal=(S @ e - y)[use] * kt.KCAL,
        )
        if t == TARGET_DFT:
            energies = bm + rad @ coef
            np.save(WORK / f"spin49-{t}-modes--1.npy", energies)
            np.save(WORK / f"{STAGE}-held-modes.npy", energies[meta[:, 0] == -1])
        fx, b, held = source_val[t]
        if held:
            grouped = defaultdict(list)
            for sys, c, n, left, mid, forward, rev in held:
                res = [
                    (v[:-1] @ coef - v[-1]) * kt.KCAL for v in (left, mid, forward, rev)
                ]
                grouped["separated fragments" if n > 2 else "connected minima"].append(
                    res
                )
            scores[t] = {
                k: {"n": len(values), "mae_kcal": np.abs(values).mean(axis=0).tolist()}
                for k, values in grouped.items()
            }
        else:
            resid = [
                (n, m, b[ix[n]] + fx[ix[n]] @ coef - e)
                for n, e, m, sys, method, part in labels
                if part == "validation"
            ]
            ref = next(v for n, m, v in resid if m == 1 and n.endswith("/0"))
            scores[t] = {
                "benzopinacol_profile_mae": np.mean(
                    [abs(v - ref) * kt.KCAL for _, _, v in resid]
                )
            }
        branches[t] = {
            "mix_doublet": coef[-2],
            "mix_triplet": coef[-1],
            "norm_local": np.linalg.norm(coef[:-2]),
        }
    kt.write_json(ROOT / "external-validation.json", scores)
    kt.write_json(
        WORK / f"{STAGE}-settings.json",
        {
            "selected": STAGE,
            "branches": dict.fromkeys(kt.TARGETS, "spin49"),
            "fit": branches,
        },
    )
    compare(STAGE)
    validate(STAGE)
    shutil.copyfile(WORK / f"{PARENT}-spectra.json", WORK / f"{STAGE}-spectra.json")
    print("Completed atom-resolved spin surface", branches, scores, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("step", choices=["prepare", "fit"])
    args = p.parse_args()
    with threadpool_limits(1):
        prepare() if args.step == "prepare" else fit()
