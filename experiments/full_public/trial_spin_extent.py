"""Suppress the spurious extensive scaling of a fixed-spin electronic correction.

In the original fit a global doublet or triplet flag multiplies sums of atom
counts, pair Gaussians and local chemical environments. Large positive and
negative extensive terms can cancel on small training molecules and fail on
a larger one. Replace these three legacy spin terms by their size-normalized
values and a molecular spin offset. All geometric terms remain differentiable.
"""

import json
import multiprocessing as mp
import pickle
import shutil
import sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits

from experiments.full_public import kernel_trials as kt
from experiments.full_public import trial_aux, trial_fit
from experiments.full_public import trial_multiscale as ms
from experiments.full_public.paper_data import OUT as DATA
from experiments.full_public.paper_features import ROOT as PAPER
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_validation import run as validate
from src.forcefield import ForceField

WORK = kt.WORK
ROOT = DATA / "spin-extent48"
PARENT = "p44-public"
STAGE = "p48-public"
# Fixed before scoring: six nonhydrogen atoms give half the unrestricted
# amplitude. The scale is shared by both spins and both electronic methods.
SCALE = 6.0


class Channels:
    """Extract each current spin-gated contribution in eV using its deployed coefficients."""

    def __init__(self, target):
        self.model = ForceField.load(WORK / f"{PARENT}-{target}.npz", dispersion="d3")
        m = self.model
        nr = 1 + len(m.elements)
        common = nr + 315 + m.chem.size + m.kernel.size
        size = nr + 315 + (m.chem.size + 3) // 4 + (m.kernel.size + 3) // 4
        self.proj = np.zeros((len(m.weights), 8))
        for spin in (2, 3):
            start = common + (spin - 1) * size
            w, _ = m.spin_readouts.get(spin, (m.weights, {}))
            k = 4 * (spin - 2)
            for j, (a, b) in enumerate(((0, nr), (nr, nr + 315), (nr + 315, size))):
                ix = slice(start + a, start + b)
                self.proj[ix, k + j] = w[ix]
            self.proj[start, k + 3] = 1.0
        self.active = np.flatnonzero(np.any(self.proj, axis=1))

    def compact(self, row):
        return row[..., self.active] @ self.proj[self.active]

    def raw(self, z, r, q, m):
        if m not in (2, 3):
            return np.zeros(8)
        return self.compact(self.model.design(z, r, q, m)[0])

    def deploy(self, coef, path):
        ff = self.model
        for m in (2, 3):
            k = 4 * (m - 2)
            block = self.proj[:, k : k + 3]
            bias = self.proj[:, k + 3]
            old, extra = ff.spin_readouts.get(m, (ff.weights, {}))
            # Remove the extensive contribution for this state. Store the
            # corresponding intensive readout separately for the public API.
            ff.spin_readouts[m] = (old - block.sum(axis=1) + coef[k + 3] * bias, extra)
            ff.spin_size_readouts[m] = (SCALE, block @ coef[k : k + 3])
        ff.save_with_base(kt.original.BASE, path)


def transformed(raw, heavy):
    factor = SCALE / (SCALE + np.asarray(heavy))
    x = raw.copy()
    idx = [0, 1, 2, 4, 5, 6]
    x[..., idx] *= factor[..., None]
    return x, raw[..., idx].sum(axis=-1)


def scans():
    """p44 and p43 have identical singlet/triplet radical-surface energies."""
    lookup = json.loads((PAPER / "index.json").read_text())
    with np.load(DATA / "external-linear-heads.npz") as f:
        h = f["weights"] @ np.array(
            json.loads((WORK / "p43-public-settings.json").read_text())["weights"][
                "wb97xd3"
            ]
        )
    with sqlite3.connect(f"file:{DATA / 'electronic.sqlite'}?mode=ro", uri=True) as c:
        data = c.execute(
            "select id,z,r,charge,mult,energy_eV,method,system,split from geometries where source='AIMNet2-NSE supplement' and mult in (1,3) and duplicate_of is null and split in ('train','validation') order by id"
        ).fetchall()
    required = defaultdict(list)
    for n, *_ in data:
        required[lookup[n] // 32 * 32].append(n)
    base = {}
    for k, names in required.items():
        with np.load(PAPER / f"block-{k:06d}.npz") as d:
            for n in names:
                ix = lookup[n] % 32
                v = d["x"][ix]
                common = v @ h[: len(v)]
                base[n] = (d["base"][ix] + common, v[::5] @ h[len(v) :])
    records = []
    for n, zb, rb, q, m, e, method, system, split in data:
        val, spin = base[n]
        z = np.frombuffer(zb, np.uint8)
        r = np.frombuffer(rb, "<f8").reshape(-1, 3)
        records.append((n, z, r, q, m, val + (m - 1) * spin, TARGET_DFT))
    return records, [
        (n, e, m, system, method, split)
        for n, _, _, _, m, e, method, system, split in data
    ]


TARGET_DFT = "wb97xd3"


def network():
    """Use the authors' charges and spin for every component of separated products."""
    doc = json.loads((DATA / "fragment-channels.json").read_text())
    channels = list(doc["channels"])
    allgeo = dict(doc["geometries"])
    with sqlite3.connect(f"file:{DATA / 'electronic.sqlite'}?mode=ro", uri=True) as c:
        for method, sys, channel, r, t, p in c.execute(
            "select method,system,channel,reactant,ts,product from channels where source='AutoMeKin' and method like '%CCSD%' and product is not null"
        ):
            channels.append(
                {
                    "method": method,
                    "system": sys,
                    "channel": channel,
                    "reactant": [r],
                    "ts": t,
                    "product": [p],
                    "split": "validation" if sys == "cBD-CN" else "train",
                }
            )
        for n, zb, rb, q, m, e in c.execute(
            "select id,z,r,charge,mult,energy_eV from geometries where source='AutoMeKin' and method like '%CCSD%' and split in ('train','validation')"
        ):
            allgeo.setdefault(
                n,
                {
                    "z": np.frombuffer(zb, np.uint8),
                    "r_A": np.frombuffer(rb, "<f8").reshape(-1, 3),
                    "charge": q,
                    "mult": m,
                    "energy_eV": e,
                },
            )
    indices = {n for a in channels for n in a["reactant"] + [a["ts"]] + a["product"]}
    rows = [
        (
            n,
            np.array(a["z"], np.uint8),
            np.asarray(a["r_A"], float),
            a["charge"],
            a["mult"],
            None,
            "high",
        )
        for n, a in sorted(allgeo.items())
        if n in indices
    ]
    return rows, channels, {n: v["energy_eV"] for n, v in allgeo.items()}


def encode(lo):
    end = ROOT / f"block-{lo:06d}.npz"
    if end.exists():
        return
    out = []
    for i, (name, z, r, q, m, b, target) in enumerate(external[lo : lo + 48]):
        model = maps[target]
        if b is None:
            b = model.model.energy(z, r, q, m)
        elif (lo + i) % 1500 == 0:
            np.testing.assert_allclose(
                b, model.model.energy(z, r, q, m), atol=4.0e-6, rtol=0
            )
        raw = model.raw(z, r, q, m)
        x, lost = transformed(raw, np.count_nonzero(z > 1))
        out.append(np.r_[x, b - lost, b])
    with end.with_suffix(".pending").open("wb") as f:
        np.savez(f, rows=np.array(out))
    end.with_suffix(".pending").replace(end)


def rad_chunk(lo):
    dest = ROOT / f"rad-{lo:06d}.npz"
    if dest.exists():
        return
    out = []
    for (z, r), st in zip(radgeo[lo : lo + 64], states[lo : lo + 64], strict=True):
        x, lost = transformed(maps[TARGET_DFT].raw(z, r, *st), np.count_nonzero(z > 1))
        out.append(np.r_[x, lost])
    with dest.with_suffix(".pending").open("wb") as f:
        np.savez(f, rows=np.array(out))
    dest.with_suffix(".pending").replace(dest)


def solve():
    global maps, external, radgeo, states
    ROOT.mkdir(exist_ok=True)
    maps = {t: Channels(t) for t in kt.TARGETS}
    scan, labels = scans()
    net, links, refs = network()
    external = scan + net
    idx = {a[0]: i for i, a in enumerate(external)}
    kt.write_json(ROOT / "ids.json", list(idx))
    pts, npvt, meta = trial_aux.read_frames()
    radgeo = pts[npvt:]
    with (WORK / "radical-modes.pkl").open("rb") as f:
        radlist = pickle.load(f)
    states = np.repeat(np.array([r[1][3] for r in radlist]), 7, axis=0)
    saddle = np.array([r[1][0].split("/")[-1].startswith("ts") for r in radlist])
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context("fork")) as pool:
        for i, _ in enumerate(
            pool.map(encode, range(0, len(external), 48), buffersize=9)
        ):
            if (i + 1) % 100 == 0:
                print(
                    "Encoded spin and fragment electronic data",
                    min((i + 1) * 48, len(external)),
                    len(external),
                    flush=True,
                )
        for i, _ in enumerate(
            pool.map(rad_chunk, range(0, len(radgeo), 64), buffersize=10)
        ):
            if (i + 1) % 120 == 0:
                print(
                    "Encoded spin-dependent radical curvature rows",
                    min((i + 1) * 64, len(radgeo)),
                    len(radgeo),
                    flush=True,
                )

    def read(kind, n, step):
        blocks = []
        for i in range(0, n, step):
            with np.load(ROOT / f"{kind}-{i:06d}.npz") as a:
                blocks.append(a["rows"])
        return np.vstack(blocks)

    ext = read("block", len(external), 48)
    rraw = read("rad", len(radgeo), 64).reshape(-1, 7, 9)
    g0, _, _ = kt.inputs()
    with (WORK / "ids.pkl").open("rb") as f:
        _, names = pickle.load(f)
    heavy = np.array([np.count_nonzero(g0[n].z > 1) for n in names])
    full = np.load(WORK / "base-x.npy", mmap_mode="r")
    checks = {}
    settings = {}
    for target, model in maps.items():
        S, y, src, fold, kind, case = d = kt.equation_data(target)
        raw = np.vstack(
            [model.compact(full[k : k + 256]) for k in range(0, len(full), 256)]
        )
        x, lost = transformed(raw, heavy)
        parent = ms.baseline(target, -1, "parent", PARENT)
        base = parent - lost
        A = S @ x
        g = kt.grouped_moments(d, A, y - S @ base, fold >= 0)
        eq = g["other"] + 96 * g["mid"] + 36 * g["gap"]
        if target == "high":
            for c in links:
                if c["split"] != "train":
                    continue

                def side(ns):
                    v = ext[[idx[n] for n in ns], :9].sum(axis=0)
                    v[-1] = sum(refs[n] for n in ns) - v[-1]
                    return v

                rr, tt, pp = side(c["reactant"]), side([c["ts"]]), side(c["product"])
                mid = tt - (rr + pp) / 2
                gap = pp - rr
                eq += 32.0 * np.outer(mid, mid) + 20.0 * np.outer(gap, gap)
        else:
            by = defaultdict(list)
            for n, e, m, system, meth, split in labels:
                if split != "train":
                    continue
                v = ext[idx[n], :9].copy()
                v[-1] = e - v[-1]
                by[system, meth, m].append(v)
            for sys, meth, m in by:
                if m != 3 or (sys, meth, 1) not in by:
                    continue
                a = np.array(by[sys, meth, 3]) - np.mean(by[sys, meth, 1], axis=0)
                eq += (1800.0 / len(a)) * (a.T @ a)
            bm = ms.baseline(target, -1, "modes", PARENT)
            br = bm - rraw[:, :, 8]
            a, v = trial_fit.rad_rows(
                rraw[:, :, :8], br, meta, np.where(saddle, 3.0, 1.0)
            )
            eq += trial_fit.reduce_gram(a, v, np.repeat(meta[:, 0] >= 0, 3))
        # Ridge regularizes the small set of amplitudes. Coefficients are
        # dimensionless except for the two molecular spin offsets, in eV.
        lhs = eq[:-1, :-1].copy()
        lhs.flat[:: len(lhs) + 1] += 16.0
        coeff = cho_solve(cho_factor(lhs), eq[:-1, -1])
        e = base + x @ coeff
        key = "spin48"
        settings[target] = coeff.tolist()
        path = WORK / f"{STAGE}-{target}.npz"
        model.deploy(coeff, path)
        ff = ForceField.load(path, dispersion="d3")
        for m in (1, 2, 3):
            i = next(
                i
                for i, n in enumerate(names)
                if g0[n].state[1] == m and 4 < len(g0[n].z) < 13
            )
            obs = g0[names[i]]
            z, r = obs.z, obs.r
            en, f = ff.predict(z, r, *obs.state)
            np.testing.assert_allclose(en, e[i], atol=3e-6, rtol=0)
            v = np.random.default_rng(i).normal(size=r.shape)
            v /= np.linalg.norm(v)
            fd = (
                ff.energy(z, r - 2.0e-5 * v, *obs.state)
                - ff.energy(z, r + 2.0e-5 * v, *obs.state)
            ) / 4.0e-5
            np.testing.assert_allclose(np.sum(f * v), fd, atol=5e-5, rtol=0)
        np.save(WORK / f"{key}-{target}-parent--1.npy", e)
        use = fold == -1
        np.savez_compressed(
            WORK / f"{STAGE}-{target}-equations.npz",
            source=src[use],
            kind=kind[use],
            case=case[use],
            error_kcal=(S @ e - y)[use] * kt.KCAL,
        )
        if target == TARGET_DFT:
            modes = br + rraw[:, :, :8] @ coeff
            np.save(WORK / f"{key}-{target}-modes--1.npy", modes)
            np.save(WORK / f"{STAGE}-held-modes.npy", modes[meta[:, 0] == -1])

        def pred(n, coeff=coeff):
            v = ext[idx[n]]
            return np.array([v[9], v[8] + v[:8] @ coeff])

        if target == "high":
            errors = defaultdict(list)
            for c in links:
                if c["split"] != "validation":
                    continue

                def err(ns):
                    return (
                        sum(pred(n) for n in ns) - sum(refs[n] for n in ns)
                    ) * kt.KCAL

                a, b, h = err(c["reactant"]), err(c["product"]), err([c["ts"]])
                key = (
                    "separated fragments"
                    if len(c["reactant"]) + len(c["product"]) > 2
                    else "connected minima"
                )
                errors[key].append([b - a, h - (a + b) / 2, h - a, h - b])
            checks.update(
                {
                    k: {"n": len(v), "mae_kcal": np.abs(v).mean(axis=0).tolist()}
                    for k, v in errors.items()
                }
            )
        else:
            held = [v for v in labels if v[-1] == "validation"]
            n0, e0 = next((n, e) for n, e, m, *_ in held if m == 1 and n.endswith("/0"))
            errors = [(pred(n) - pred(n0) - (e - e0)) * kt.KCAL for n, e, *_ in held]
            checks["benzopinacol"] = {
                "n": len(errors),
                "mae_kcal": np.mean(np.abs(errors), axis=0).tolist(),
            }
            for m in (1, 3):
                n0, e0 = next(
                    (n, e) for n, e, m0, *_ in held if m0 == m and n.endswith("/0")
                )
                err = [
                    (pred(n) - pred(n0) - (e - e0)) * kt.KCAL
                    for n, e, m0, *_ in held
                    if m0 == m
                ]
                checks[f"benzopinacol within m={m}"] = {
                    "n": len(err),
                    "mae_kcal": np.mean(np.abs(err), axis=0).tolist(),
                }
    kt.write_json(
        ROOT / "external-validation.json",
        {
            "models": ["p44", "p48"],
            "reaction_observables": ["gap", "midpoint", "forward", "reverse"],
            "sets": checks,
        },
    )
    kt.write_json(
        WORK / f"{STAGE}-settings.json",
        {
            "selected": STAGE,
            "branches": dict.fromkeys(kt.TARGETS, "spin48"),
            "scale": SCALE,
            "coefficients": settings,
        },
    )
    compare(STAGE)
    validate(STAGE)
    # All four NIST geometries are singlets. Their complete p44 PES is unchanged.
    shutil.copyfile(WORK / f"{PARENT}-spectra.json", WORK / f"{STAGE}-spectra.json")
    print("Completed size-normalized spin refit", checks, flush=True)


if __name__ == "__main__":
    with threadpool_limits(1):
        solve()
