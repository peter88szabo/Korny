"""Transfer published spin scans and reaction networks to p42 without mixing energy zeros.

Fit one residual kernel per electronic reference, using energy differences
within composition, explicit reaction gaps and midpoint heights. Each kernel
has spin-independent and multiplicity-dependent columns. Learn how much of
each correction transfers to the original two Korny energy surfaces while
retaining the original force, equilibrium and curvature constraints. Reserve
entire scan/network systems and four BASChem19 reactions throughout.
"""

import json
import pickle
import sqlite3
from collections import defaultdict

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
from experiments.full_public.data import DB
from experiments.full_public.paper_data import OUT
from experiments.full_public.paper_features import PARENT, ROOT
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_constraints import nist_rows
from experiments.full_public.trial_curvature_fit import anchor_blocks
from experiments.full_public.trial_isomers import gram as isomer_gram
from experiments.full_public.trial_measured_curvature import load_basis
from experiments.full_public.trial_spectra import evaluate_trial
from experiments.full_public.trial_validation import run as validate
from src.forcefield import ForceField

STAGE = "p43-public"
WORK = kt.WORK


def sources():
    idx = json.loads((ROOT / "index.json").read_text())
    n = max(idx.values()) + 1
    _basis, columns = load_basis()
    raw = np.empty((n, len(columns)))
    base = np.empty(n)
    for lo in range(0, n, 32):
        with np.load(ROOT / f"block-{lo:06d}.npz") as a:
            raw[lo : lo + len(a["x"])] = a["x"]
            base[lo : lo + len(a["x"])] = a["base"]
    with sqlite3.connect(f"file:{OUT / 'electronic.sqlite'}?mode=ro", uri=True) as db:
        meta = db.execute(
            'select id,source,system,method,mult,energy_eV,split from geometries where duplicate_of is null and split in ("train","validation") order by id'
        ).fetchall()
        links = db.execute(
            "select source,system,method,channel,reactant,ts,product from channels"
        ).fetchall()
    spin = np.zeros(n, int)
    for name, _, _, _, m, _, _ in meta:
        spin[idx[name]] = m - 1
    dim = raw.shape[1]
    # Share most coefficients across spin and retain a distinct, well-conditioned
    # set of local response directions for doublet and triplet chemistry.
    x = np.column_stack((raw, raw[:, ::5] * spin[:, None]))
    return x, base, meta, idx, links, dim


def observations(x, base, meta, index, links):
    methods = sorted({r[3] for r in meta})
    gs = {m: np.zeros((x.shape[1] + 1,) * 2) for m in methods}
    groups = defaultdict(list)
    byname = {r[0]: r for r in meta}
    for name, src, system, meth, _, e, split in meta:
        if split == "train":
            groups[meth, src, system].append((index[name], e))
    for (meth, src, system), rows in groups.items():
        ids = np.array([i for i, _ in rows])
        y = np.array([e for _, e in rows]) - base[ids]
        mu = x[ids].mean(axis=0)
        y -= y.mean()
        # Cap each correlated scan's influence while retaining every frame.
        weight = min(1.0, 250 / len(ids))
        for lo in range(0, len(rows), 256):
            sl = slice(lo, lo + 256)
            a = np.column_stack((x[ids[sl]] - mu, y[sl]))
            gs[meth] += weight * (a.T @ a)
    for _, _, meth, _, r, t, p in links:
        names = [r, t] + ([p] if p else [])
        if not all(n in byname and byname[n][-1] == "train" for n in names):
            continue
        ids = [index[n] for n in names]
        a = np.column_stack(
            (x[ids], np.array([byname[n][-2] for n in names]) - base[ids])
        )
        if p:
            mid = a[1] - 0.5 * (a[0] + a[2])
            rx = a[2] - a[0]
            gs[meth] += 4 * np.outer(mid, mid) + 2 * np.outer(rx, rx)
        else:
            forward = a[1] - a[0]
            gs[meth] += 4 * np.outer(forward, forward)
    return methods, gs


def compress(full, cols, spins, common, spin):
    """Project the large cached design onto the learned methods in bounded blocks."""
    out = np.empty((len(full), common.shape[1]))
    for lo in range(0, len(full), 256):
        hg = slice(lo, min(lo + 256, len(full)))
        x = np.ascontiguousarray(full[hg][:, cols])
        out[hg] = x @ common + (x[:, ::5] @ spin) * spins[hg, None]
    return out


def original_spins():
    geom, _, _ = kt.inputs()
    with (WORK / "ids.pkl").open("rb") as f:
        _, names = pickle.load(f)
    pub = [geom[n].state[1] - 1 for n in names]
    seen = set()
    hdf = []
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as db:
        for fp, q, m in db.execute(
            'select fp,q,m from samples where split="train" order by source,name'
        ):
            if (fp, q, m) not in seen:
                seen.add((fp, q, m))
                hdf.append(m - 1)
    assert len(hdf) == 91156
    with (WORK / "radical-modes.pkl").open("rb") as f:
        records = pickle.load(f)
    radical = np.array([r[1][3][1] - 1 for r in records])
    saddle = np.array([r[1][0].split("/")[-1].startswith("ts") for r in records])
    return np.r_[pub, np.repeat(hdf, 3)], np.repeat(radical, 7), saddle


def deployment(target, b, cols, common, spin, weights, path):
    model = ForceField.load(WORK / f"{PARENT}-{target}.npz", dispersion="d3")
    k = len(model.additional_bases)
    b.weights = np.zeros(b.size)
    b.weights[cols] = common @ weights
    model.additional_bases.append(b)
    delta = spin @ weights
    for m in sorted({m for _, m in model.supported_states if m != 1}):
        coef = b.weights.copy()
        coef[cols[::5]] += (m - 1) * delta
        original, extra = model.spin_readouts.get(m, (model.weights, {}))
        model.spin_readouts[m] = original, {**extra, k: coef}
    model.save_with_base(kt.original.BASE, path)


def fit_main(target, main, rad, saddle, common, spin, methods, learned, external):
    b, cols = load_basis()
    s, y, src, fold, kind, case = d = kt.equation_data(target)
    n = s.shape[1]
    pub = main[:n]
    x = s @ pub
    base = ms.baseline(target, -1, "parent", PARENT)
    g = kt.grouped_moments(d, x, y - s @ base, fold >= 0)
    eq = g["other"] + 96 * g["mid"] + 36 * g["gap"] + isomer_gram(target, -1, pub, base)
    ef, cur = anchor_blocks(main, n, "paper43-heads")
    eq[:-1, :-1] += 0.2 * ef + 0.04 * cur
    model = ForceField.load(WORK / f"{PARENT}-{target}.npz", dispersion="d3")
    nz = nist_rows(lambda z, r: b.design(z, r, True)[1][:, cols] @ common, model)
    eq += 15 * nz.T @ nz
    # For directly compatible old and new reaction references the original
    # surface also gets extra explicit constraints. Independent-method heads
    # only transfer through correction directions learned on their own labels.
    direct = [
        i
        for i, meth in enumerate(methods)
        if (("CCSD" in meth) if target == "high" else meth == "wB97X-D3/def2-TZVP")
    ]
    design = np.zeros((len(learned) + 1, len(methods) + 1))
    design[:-1, :-1] = learned
    design[-1, -1] = 1.0
    for i in direct:
        eq += 3.0 * (design.T @ external[methods[i]] @ design)
    if target == "wb97xd3":
        meta = np.load(WORK / "rad-meta.npy")
        bm = ms.baseline(target, -1, "modes", PARENT)
        a, ref = trial_fit.rad_rows(rad, bm, meta, np.where(saddle, 3.0, 1.0))
        eq += trial_fit.reduce_gram(a, ref, np.repeat(meta[:, 0] >= 0, 3))
    lhs = eq[:-1, :-1].copy()
    lhs.flat[:: len(lhs) + 1] += 2.0
    coeff = cho_solve(cho_factor(lhs), eq[:-1, -1])
    energy = base + pub @ coeff
    key = "paper43"
    np.save(WORK / f"{key}-{target}-parent--1.npy", energy)
    path = WORK / f"{STAGE}-{target}.npz"
    deployment(target, b, cols, common, spin, coeff, path)
    ff = ForceField.load(path, dispersion="d3")
    geoms, _, _ = kt.inputs()
    with (WORK / "ids.pkl").open("rb") as f:
        _, names = pickle.load(f)
    for m in (1, 2, 3):
        i = next(
            i
            for i, n in enumerate(names)
            if geoms[n].state[1] == m and 3 < len(geoms[n].z) < 12
        )
        g = geoms[names[i]]
        z, r = g.z, g.r
        e, force = ff.predict(z, r, *g.state)
        np.testing.assert_allclose(e, energy[i], atol=3.0e-6, rtol=0)
        v = np.random.default_rng(i).normal(size=r.shape)
        v /= np.linalg.norm(v)
        fd = (
            ff.energy(z, r - 2.0e-5 * v, *g.state)
            - ff.energy(z, r + 2.0e-5 * v, *g.state)
        ) / 4.0e-5
        np.testing.assert_allclose(force.ravel() @ v.ravel(), fd, atol=5.0e-5)
    held = fold == -1
    np.savez_compressed(
        WORK / f"{STAGE}-{target}-equations.npz",
        source=src[held],
        kind=kind[held],
        case=case[held],
        error_kcal=(s @ energy - y)[held] * kt.KCAL,
    )
    if target == "wb97xd3":
        modes = bm + rad @ coeff
        np.save(WORK / f"{key}-{target}-modes--1.npy", modes)
        np.save(WORK / f"{STAGE}-held-modes.npy", modes[meta[:, 0] == -1])
    return coeff


def new_validation(x, base, meta, index, links, methods, learned, beta):
    byname = {r[0]: r for r in meta}

    def prediction(n):
        _, _, _, meth, m, ref, _ = byname[n]
        t = (
            "high"
            if "CCSD" in meth or m == 1 and meth.startswith("BASChem")
            else "wb97xd3"
        )
        i = index[n]
        return (
            np.array(
                [
                    base[i] - ref,
                    base[i] + x[i] @ learned @ beta[t] - ref,
                    base[i] + x[i] @ learned[:, methods.index(meth)] - ref,
                ]
            )
            * kt.KCAL
        )

    records = defaultdict(list)
    for src, system, meth, channel, r, t, p in links:
        names = [r, t] + ([p] if p else [])
        if not all(n in byname and byname[n][-1] == "validation" for n in names):
            continue
        dr, dt = prediction(r), prediction(t)
        records[(src, meth, "forward")].append(dt - dr)
        if p:
            dp = prediction(p)
            for key, val in [
                ("reaction", dp - dr),
                ("reverse", dt - dp),
                ("midpoint", dt - (dp + dr) / 2),
            ]:
                records[src, meth, key].append(val)
    scan = [
        r for r in meta if r[1] == "AIMNet2-NSE supplement" and r[-1] == "validation"
    ]
    for sys in sorted({r[2] for r in scan}):
        points = [r for r in scan if r[2] == sys]
        anchor = next(r[0] for r in points if r[4] == 1 and r[0].endswith("/0"))
        for r in points:
            records[(sys, r[3], "profile vs first singlet frame")].append(
                prediction(r[0]) - prediction(anchor)
            )
    return [
        {
            "source": src,
            "method": m,
            "observable": k,
            "n": len(v),
            "mae_kcal": np.mean(np.abs(v), axis=0).tolist(),
        }
        for (src, m, k), v in records.items()
    ]


def run():
    x, base, meta, index, links, dim = sources()
    methods, external = observations(x, base, meta, index, links)
    learned = []
    for method in methods:
        g = external[method]
        a = g[:-1, :-1].copy()
        a.flat[:: len(a) + 1] += 1.0
        learned.append(cho_solve(cho_factor(a), g[:-1, -1]))
    learned = np.column_stack(learned)
    np.savez(OUT / "external-linear-heads.npz", methods=methods, weights=learned)
    common, spin = learned[:dim], learned[dim:]
    _, cols = load_basis()
    ms0, rs, saddle = original_spins()
    main = compress(
        np.load(WORK / "periodic-0.85-main.npy", mmap_mode="r"), cols, ms0, common, spin
    )
    rad = compress(
        np.load(WORK / "periodic-0.85-rad.npy", mmap_mode="r"), cols, rs, common, spin
    ).reshape(-1, 7, len(methods))
    beta = {
        t: fit_main(t, main, rad, saddle, common, spin, methods, learned, external)
        for t in kt.TARGETS
    }
    summary = new_validation(x, base, meta, index, links, methods, learned, beta)
    kt.write_json(
        OUT / "p43-native-validation.json",
        {"columns": ["p42", "p43", "native auxiliary readout"], "errors": summary},
    )
    kt.write_json(
        WORK / f"{STAGE}-settings.json",
        {
            "selected": STAGE,
            "branches": dict.fromkeys(kt.TARGETS, "paper43"),
            "source_levels": methods,
            "weights": {t: b.tolist() for t, b in beta.items()},
        },
    )
    compare(STAGE)
    validate(STAGE)
    with threadpool_limits(1):
        evaluate_trial(STAGE)


if __name__ == "__main__":
    with threadpool_limits(3):
        run()
