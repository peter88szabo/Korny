"""Calibrate the expanded datasets together on a single deployable Korny surface.

P49 is the common reference. On each surface the four linear coefficients
multiply the full-minus-pilot radical IRC response, the full closed-shell
AIMNet2 pair response, the AutoMeKin stationary-force/mode response and the
short-range Ions-16 redox kernel. Calibration uses training observations.
A fitted coefficient of zero discards that correction on the specified
electronic surface. All energies share a single conservative force evaluator.
"""

import argparse
import json
import multiprocessing as mp
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import lsq_linear
from threadpoolctl import threadpool_limits

from experiments.full_public import kernel_trials as kt
from experiments.full_public import trial_fit, trial_sites
from experiments.full_public import trial_multiscale as ms
from experiments.full_public import trial_network_hessians as network
from experiments.full_public import trial_spin_extent as ext
from experiments.full_public.automekin_modes import DEST as MODES
from experiments.full_public.paper_data import OUT
from experiments.full_public.rdb19_irc import OUT as IRC
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_constraints import nist_rows
from experiments.full_public.trial_curvature_fit import anchor_blocks
from experiments.full_public.trial_isomers import gram as isomers
from experiments.full_public.trial_parent_potential import CENTERS
from experiments.full_public.trial_redox_environments import ROOT as REDOX
from experiments.full_public.trial_spectra import evaluate_trial
from experiments.full_public.trial_validation import run as validate
from src.forcefield import ForceField
from src.redox_environment import RedoxEnvironmentKernel

WORK = kt.WORK
ROOT = OUT / "joint-transfer52"
PARENT = "p49-public"
STAGE = "p52-public"
LABELS = [
    "full_minus_pilot_radical_IRC",
    "full_AIMNet2_parent",
    "AutoMeKin_Hessians",
    "Ions16_redox_pairs",
]


def raw_pair(z, r, ff):
    """Match the deployed 45 group pairs and seven scaled-distance Gaussians."""
    i, j = np.triu_indices(len(z), 1)
    d = np.linalg.norm(r[i] - r[j], axis=1)
    scale = ff.radii[z[i]] + ff.radii[z[j]]
    hi = np.maximum(ff.groups[z[i]], ff.groups[z[j]])
    pair = hi * (hi + 1) // 2 + np.minimum(ff.groups[z[i]], ff.groups[z[j]])
    radial = np.exp(-0.5 * ((d[:, None] / scale[:, None] - CENTERS) / 0.22) ** 2)
    row = np.zeros((45, 7))
    np.add.at(row, pair, radial)
    return row.ravel()


def prep_one(rx):
    """Compress all three endpoint geometries without looking at split or errors."""
    dest = ROOT / "irc" / (rx + ".npz")
    if dest.exists():
        return
    with np.load(IRC / (rx + "-forw.npz")) as a, np.load(IRC / (rx + "-rev.npz")) as b:
        z = a["z"]
        xyz = [a["r_A"][-1], a["r_A"][0], b["r_A"][-1]]
        q, m = int(a["charge"]), int(a["multiplicity"])
        reference = np.array(
            [a["energy_eV"][-1], a["energy_eV"][0], b["energy_eV"][-1]]
        )
    ff = parents["wb97xd3"]
    positions = [r.astype(float) for r in xyz]
    prior = [ff.energy(z, r, q, m) for r in positions]
    pair = [raw_pair(z, r, ff) for r in positions]
    red = [redox.design(z, r)[0] for r in positions]
    with dest.with_suffix(".pending").open("wb") as f:
        np.savez_compressed(f, prior=prior, reference=reference, pair=pair, redox=red)
    dest.with_suffix(".pending").replace(dest)


def prep_irc():
    global parents, redox
    ROOT.mkdir(exist_ok=True)
    (ROOT / "irc").mkdir(exist_ok=True)
    with np.load(REDOX / "basis.npz") as f:
        redox = RedoxEnvironmentKernel.load(f)
    parents = {
        "wb97xd3": ForceField.load(WORK / "p49-public-wb97xd3.npz", dispersion="d3")
    }
    ids = [
        f.name.split("-")[0]
        for f in sorted(IRC.glob("*-forw.npz"))
        if f.with_name(f.name.replace("-forw.", "-rev.")).exists()
        and not (ROOT / "irc" / (f.name.split("-")[0] + ".npz")).is_file()
    ]
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("fork")) as pool:
        for i, _ in enumerate(pool.map(prep_one, ids, buffersize=8), 1):
            if i % 500 == 0:
                print(
                    "Encoded radical endpoint energies for joint calibration",
                    i,
                    len(ids),
                    flush=True,
                )
    print("Joint radical endpoint cache ready", len(ids), flush=True)


class Responses:
    """The four scalar responses and their exact Cartesian forces at inference."""

    def __init__(self, t):
        from experiments.full_public.trial_parent_transfer import Curves

        self.target = t
        self.parent = ForceField.load(WORK / f"{PARENT}-{t}.npz", dispersion="d3")
        pilot = ForceField.load(WORK / f"p44-public-{t}.npz")
        full = ForceField.load(WORK / f"p45-public-{t}.npz")
        self.periodic, self.selected = network.load_basis()
        self.cols = self.selected[::3]
        if t == "wb97xd3":
            self.irc_index = len(pilot.additional_bases) - 1
            self.irc_w = (
                full.spin_readouts[2][1][self.irc_index]
                - pilot.spin_readouts[2][1][self.irc_index]
            )
        else:
            self.irc_w = np.zeros(self.periodic.size)
        self.curves = Curves("full")
        self.curve_w = np.array(
            json.loads((WORK / "p47-public-settings.json").read_text())["amplitudes"][t]
        )
        self.pair_w = np.zeros(315)
        self.pair_w[self.curves.cols - 1 - len(pilot.elements)] = (
            self.curves.projection @ self.curve_w
        )
        self.modes = ForceField.load(WORK / f"p50-public-{t}.npz").additional_bases[-1]
        self.chem = ForceField.load(WORK / f"p51-public-{t}.npz").additional_bases[-1]
        keep = np.zeros_like(self.irc_w, dtype=bool)
        keep[self.cols] = True
        for w in (self.irc_w, self.modes.weights):
            np.testing.assert_array_equal(w[~keep], 0.0)

    def design(self, z, r, spin, forces=False):
        periodic, jp = self.periodic.design(z, r, forces)
        wp = self.irc_w if spin == 2 else np.zeros_like(self.irc_w)
        if forces:
            curve, jc = self.curves.design(z, r, True)
        else:
            curve = self.curves.design(z, r, False)
        chem, jr = self.chem.design(z, r, forces)
        row = np.array(
            [
                periodic @ wp,
                curve @ self.curve_w,
                periodic @ self.modes.weights,
                chem @ self.chem.weights,
            ]
        )
        der = (
            np.column_stack(
                (
                    jp @ wp,
                    jc @ self.curve_w,
                    jp @ self.modes.weights,
                    jr @ self.chem.weights,
                )
            )
            if forces
            else None
        )
        return row, der

    def deploy(self, w):
        from copy import deepcopy

        ff = deepcopy(self.parent)
        if self.target == "wb97xd3":
            old, extra = ff.spin_readouts[2]
            ff.spin_readouts[2] = (
                old,
                {**extra, self.irc_index: extra[self.irc_index] + w[0] * self.irc_w},
            )
        cols = 1 + len(ff.elements) + np.arange(315)
        ff.weights = ff.weights.copy()
        ff.weights[cols] += w[1] * self.pair_w
        for m, (base, extra) in list(ff.spin_readouts.items()):
            b = base.copy()
            b[cols] += w[1] * self.pair_w
            ff.spin_readouts[m] = b, extra
        for a, c in ((self.modes, w[2]), (self.chem, w[3])):
            b = deepcopy(a)
            b.weights = b.weights * c
            ff.additional_bases.append(b)
        ff.save_with_base(kt.original.BASE, WORK / f"{STAGE}-{self.target}.npz")


def cache_curves():
    """Supply the AIMNet2 pair response at the PBE0 energy/force anchor geometries."""
    dest = ROOT / "hdf-pair-response.npy"
    if dest.is_file():
        return
    rr = [responses[t].pair_w for t in kt.TARGETS]
    w = np.stack(rr, axis=-1)
    ff = responses["high"].parent
    data = []
    for _, z, r, dr, _, spin in trial_sites.hdf(include_state=True):
        if spin != 1:
            raise ValueError(
                "Include the doublet IRC response for a non-singlet HDF anchor"
            )
        data.extend(
            raw_pair(z, pos, ff) @ w for pos in (r, r + 0.001 * dr, r - 0.001 * dr)
        )
    with dest.with_suffix(".pending").open("wb") as f:
        np.save(f, np.asarray(data))
    dest.with_suffix(".pending").replace(dest)


def spectral(info):
    """Apply the exact p50 projected force/curvature rows to the four corrections."""
    dest = ROOT / "shape" / info["file"]
    if dest.is_file():
        return
    with np.load(MODES / info["file"]) as f:
        z, r, m = f["z"], f["r_A"], int(f["mult"])
        freq = f["freq_cm1"]
        mass = np.repeat(f["mass_u"], 3)
        mu = f["modes"].reshape(len(freq), -1)
    with np.load(network.ROOT / info["file"]) as f:
        old_rows = f["rows"].astype(float)
        old_en = f["energy"]
        old_soft = f["soft"]
        old_forces = f["forces"]
    n = len(responses["high"].cols)
    curves = responses["high"].curves
    chem = responses["high"].chem
    # The only new derivatives here are the pair and local redox functions.
    # The old periodic projections and frozen parent residuals were measured
    # with exactly the same normal displacements in p50.
    curve_w = np.column_stack([responses[t].curve_w for t in kt.TARGETS])
    chem_w = np.column_stack([responses[t].chem.weights for t in kt.TARGETS])

    def calc(pos):
        p, jp = curves.design(z, pos, True)
        c, jc = chem.design(z, pos, True)
        return np.r_[p @ curve_w, c @ chem_w], np.column_stack(
            (jp @ curve_w, jc @ chem_w)
        )

    new_rows, new_en, new_soft, new_forces = network.mode_rows(
        info["id"], r, freq, mu, mass, calc, 4
    )
    out = {}
    for j, t in enumerate(kt.TARGETS):
        model = responses[t]
        cols = model.cols
        w0 = model.irc_w[cols] * float(m == 2)
        w2 = model.modes.weights[cols]

        def combine(old, new, w0=w0, w2=w2, j=j):
            return np.stack(
                (
                    old[..., :n] @ w0,
                    new[..., j],
                    old[..., :n] @ w2,
                    new[..., 2 + j],
                    old[..., n + j],
                ),
                axis=-1,
            )

        a = combine(old_rows, new_rows)
        out[t + "_gram"] = a.T @ a
        out[t + "_energy"] = combine(old_en, new_en)
        out[t + "_soft"] = np.r_[combine(old_soft, new_soft), old_soft[-2:]]
        out[t + "_force"] = combine(old_forces, new_forces)
    with dest.with_suffix(".pending").open("wb") as f:
        np.savez_compressed(f, **out)
    dest.with_suffix(".pending").replace(dest)


def energy_groups(records, t):
    """Return actual DFT or CC reaction equations at each source's geometries."""
    import sqlite3

    from experiments.full_public.paper_data import DFT

    model = responses[t]
    joined = {}
    if t == "high":
        states, channels, refs = ext.network()
        for n, z, r, q, m, *_ in states:
            parts = n.split("/")
            fname = parts[1] + "_" + parts[-1] + ".npz"
            archive = MODES / fname
            entry = None
            if (ROOT / "shape" / fname).exists() and archive.is_file():
                with np.load(archive) as f:
                    same = (
                        int(f["mult"]) == m
                        and int(f["charge"]) == q
                        and np.array_equal(f["z"], z)
                        and np.allclose(f["r_A"], r, atol=1e-7, rtol=0)
                    )
                if same:
                    with np.load(ROOT / "shape" / fname) as f:
                        entry = f[t + "_energy"]
            if entry is None:
                entry = np.r_[model.design(z, r, m)[0], model.parent.energy(z, r, q, m)]
            entry[-1] = refs[n] - entry[-1]
            joined[n] = entry
    else:
        with sqlite3.connect(
            f"file:{OUT / 'electronic.sqlite'}?mode=ro", uri=True
        ) as db:
            refs = dict(
                db.execute(
                    "select id,energy_eV from geometries where source='AutoMeKin' and method=?",
                    (DFT,),
                )
            )
            for item in records:
                n = item["id"]
                with np.load(ROOT / "shape" / item["file"]) as f:
                    a = f[t + "_energy"]
                a[-1] = refs[n] - a[-1]
                joined[n] = a
            channels = [
                {
                    "system": sys,
                    "split": "validation" if sys == "cBD-CN" else "train",
                    "reactant": [r],
                    "ts": ts,
                    "product": [p],
                }
                for sys, r, ts, p in db.execute(
                    "select system,reactant,ts,product from channels where source='AutoMeKin' and method=? and product is not null",
                    (DFT,),
                )
            ]
    train = np.zeros((5, 5))
    held = defaultdict(list)
    for c in channels:
        if any(k not in joined for k in c["reactant"] + c["product"] + [c["ts"]]):
            continue
        a = sum(joined[n] for n in c["reactant"])
        b = sum(joined[n] for n in c["product"])
        h = joined[c["ts"]]
        gap, mid = b - a, h - (a + b) / 2
        if c["split"] == "train":
            train += 20 * np.outer(gap, gap) + 32 * np.outer(mid, mid)
        else:
            tag = (
                "separated"
                if len(c["reactant"]) + len(c["product"]) > 2
                else "connected"
            )
            held[tag].append([gap, mid, h - a, h - b])
    return train, held


def native_radicals():
    """Use only the train-partition electronic endpoints of verified full IRC pairs.

    The source Hamiltonian differs from the primary DFT surface. Allow its
    amplitudes to depart from the primary fit, eliminating the auxiliary
    coefficients analytically. Every admitted training IRC contributes its
    two directed electronic heights and its energy difference.
    """
    model = responses["wb97xd3"]
    source = json.loads((IRC.parent / "path-kernel45/manifest.json").read_text())
    G = np.zeros((5, 5))
    count = 0
    for d in source:
        if d["heldout"]:
            continue
        rx = d["rxn"]
        with np.load(IRC.parent / "path-kernel45" / (rx + ".npz")) as a:
            ends = a["ends"]
        with np.load(ROOT / "irc" / (rx + ".npz")) as f:
            np.testing.assert_allclose(ends[:, -1], f["reference"], atol=1e-8, rtol=0)
            mat = np.column_stack(
                (
                    ends[:, :-1] @ model.irc_w[model.cols],
                    f["pair"] @ model.pair_w,
                    ends[:, :-1] @ model.modes.weights[model.cols],
                    f["redox"] @ model.chem.weights,
                    f["reference"] - f["prior"],
                )
            )
        a, h, b = mat
        gap = b - a
        mid = h - (a + b) / 2
        G += np.outer(gap, gap) + 2 * np.outer(mid, mid)
        count += 1
    # Fit an independent four-amplitude source readout with a quadratic
    # penalty on departure from the primary readout. Fixed minimum shrinkage
    # prevents a source-specific coefficient from dominating unseen chemistry.
    H = G[:4, :4]
    eta = np.diag(np.maximum(np.diag(H) * 0.15, 4.0))
    shared = eta - eta @ np.linalg.solve(H + eta, eta)
    new = np.zeros_like(G)
    new[:4, :4] = (shared + shared.T) / 2
    new[:4, 4] = eta @ np.linalg.solve(H + eta, G[:4, 4])
    new[4, :4] = new[:4, 4]
    return new, count


def stage_differences(target, observable):
    """Read each response relative to the checkpoint on which it was trained."""

    def read(index):
        return ms.baseline(target, -1, observable, f"p{index}-public")

    parent = read(49)
    columns = [
        read(45) - read(44) if target == "wb97xd3" else np.zeros_like(parent),
        read(47) - read(44),
        read(50) - parent,
        read(51) - parent,
    ]
    return parent, np.stack(columns, axis=-1)


def hdf_anchors(pub, target):
    """Project the bare-backbone energy, force and curvature anchors in batches."""
    model = responses[target]
    prefix = "periodic-0.85-main.npy"
    periodic = np.load(WORK / prefix, mmap_mode="r")
    red = np.load(REDOX / "main.npy", mmap_mode="r")
    pair = np.load(ROOT / "hdf-pair-response.npy", mmap_mode="r")
    n = len(pub)
    assert len(periodic) == len(red) == n + len(pair)
    x = np.zeros((len(red), 4))
    x[:n] = pub
    x[n:, 1] = pair[:, kt.TARGETS.index(target)]
    for lo in range(n, len(x), 256):
        hi = min(lo + 256, len(x))
        x[lo:hi, 2] = periodic[lo:hi][:, model.cols] @ model.modes.weights[model.cols]
        x[lo:hi, 3] = red[lo:hi] @ model.chem.weights
    return anchor_blocks(x, n, f"joint52-{target}")


def fit():
    """Estimate four bounded amplitudes using training reactions and derivatives."""
    global responses
    responses = {t: Responses(t) for t in kt.TARGETS}
    records = json.loads((MODES / "manifest.json").read_text())["records"]
    ROOT.mkdir(exist_ok=True)
    (ROOT / "shape").mkdir(exist_ok=True)
    cache_curves()
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("fork")) as pool:
        for i, _ in enumerate(pool.map(spectral, records, buffersize=8), 1):
            if i % 500 == 0:
                print(
                    "Prepared joint stationary force and curvature rows",
                    i,
                    len(records),
                    flush=True,
                )
    meta = np.load(WORK / "rad-meta.npy")
    with (WORK / "radical-modes.pkl").open("rb") as f:
        saddle = np.array(
            [r[1][0].split("/")[-1].startswith("ts") for r in pickle.load(f)]
        )
    with (WORK / "ids.pkl").open("rb") as f:
        _, names = pickle.load(f)
    geoms, _, _ = kt.inputs()
    radical_gram, n_irc = native_radicals()
    settings, external_scores = {}, {}
    for t, model in responses.items():
        parent, pub = stage_differences(t, "parent")
        data = kt.equation_data(t)
        s, y, src, fold, kind, case = data
        groups = kt.grouped_moments(data, s @ pub, y - s @ parent, fold >= 0)
        eq = (
            groups["other"]
            + 96 * groups["mid"]
            + 36 * groups["gap"]
            + isomers(t, -1, pub, parent)
        )
        ef, curvature = hdf_anchors(pub, t)
        eq[:4, :4] += 0.12 * ef + 0.045 * curvature
        nist = nist_rows(
            lambda z, r, model=model: model.design(z, r, 1, True)[1], model.parent
        )
        eq += 20 * (nist.T @ nist)
        channels, held_channels = energy_groups(records, t)
        eq += channels
        for rec in records:
            if rec["split"] == "train":
                with np.load(ROOT / "shape" / rec["file"]) as f:
                    eq += f[t + "_gram"]
        if t == "wb97xd3":
            base_modes, rad = stage_differences(t, "modes")
            a, b = trial_fit.rad_rows(rad, base_modes, meta, np.where(saddle, 3.0, 1.0))
            eq += trial_fit.reduce_gram(a, b, np.repeat(meta[:, 0] >= 0, 3))
            eq += radical_gram
        # Weak shrinkage toward the source-trained amplitude. The box keeps
        # calibration inside the range already exercised by each source fit.
        lhs = eq[:4, :4] + 8 * np.eye(4)
        chol = cholesky(lhs, lower=True)
        rhs = solve_triangular(chol, eq[:4, 4] + 8, lower=True)
        result = lsq_linear(chol.T, rhs, bounds=(0.0, 1.0))
        if not result.success:
            raise RuntimeError(result.message)
        w = result.x
        if t == "high":
            w[0] = (
                0.0  # The radical-only response vanishes identically on this surface.
            )
        model.deploy(w)
        deployed = ForceField.load(WORK / f"{STAGE}-{t}.npz", dispersion="d3")
        e = parent + pub @ w
        for spin in (1, 2, 3):
            idx = next(
                i
                for i, k in enumerate(names)
                if geoms[k].state[1] == spin and 5 < len(geoms[k].z) < 14
            )
            g = geoms[names[idx]]
            np.testing.assert_allclose(
                model.design(g.z, g.r, spin)[0], pub[idx], rtol=0, atol=5e-6
            )
            en, force = deployed.predict(g.z, g.r, *g.state)
            np.testing.assert_allclose(en, e[idx], rtol=0, atol=5e-6)
            u = np.random.default_rng(idx).normal(size=g.r.shape)
            u /= np.linalg.norm(u)
            fd = (
                deployed.energy(g.z, g.r - 2e-5 * u, *g.state)
                - deployed.energy(g.z, g.r + 2e-5 * u, *g.state)
            ) / 4e-5
            np.testing.assert_allclose(np.sum(u * force), fd, atol=8e-5, rtol=0)
        held = fold < 0
        np.save(WORK / f"joint52-{t}-parent--1.npy", e)
        np.savez_compressed(
            WORK / f"{STAGE}-{t}-equations.npz",
            source=src[held],
            case=case[held],
            kind=kind[held],
            error_kcal=((s @ e - y) * kt.KCAL)[held],
        )
        if t == "wb97xd3":
            energies = base_modes + rad @ w
            np.save(WORK / f"joint52-{t}-modes--1.npy", energies)
            np.save(WORK / f"{STAGE}-held-modes.npy", energies[meta[:, 0] < 0])
        external_scores[t] = {}
        for k, v in held_channels.items():
            a = np.asarray(v)
            external_scores[t][k] = {
                "n": len(a),
                "gap_mid_fwd_rev_mae_kcal": (
                    np.abs(a[:, :, :4] @ w - a[:, :, 4]).mean(0) * kt.KCAL
                ).tolist(),
            }
        settings[t] = dict(zip(LABELS, w.tolist(), strict=True))
        print("Completed joint calibration", t, settings[t], flush=True)
    kt.write_json(ROOT / "external-validation.json", external_scores)
    kt.write_json(
        WORK / f"{STAGE}-settings.json",
        {
            "selected": STAGE,
            "branches": dict.fromkeys(kt.TARGETS, "joint52"),
            "amplitudes": settings,
            "training_irc_pairs": n_irc,
            "training_stationary_points": sum(v["split"] == "train" for v in records),
        },
    )
    compare(STAGE)
    validate(STAGE)
    evaluate_trial(STAGE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("prepare-irc", "fit"))
    with threadpool_limits(1):
        if parser.parse_args().step == "prepare-irc":
            prep_irc()
        else:
            fit()
