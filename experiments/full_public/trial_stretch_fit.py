"""Calibrate local X-H springs against archived force constants while protecting barriers."""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import pickle
import shutil
import numpy as np
from rdkit import Chem
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits
from experiments.full_public import (
    kernel_trials as kt,
    trial_multiscale as ms,
    trial_fit,
    trial_sites,
)
from experiments.full_public.trial_curvature_fit import anchor_blocks
from experiments.full_public.trial_stretch_labels import ROOT
from experiments.full_public.trial_constraints import nist_rows
from experiments.full_public.trial_isomers import gram as isomer_gram
from experiments.full_public.trial_spectra import evaluate_trial
from experiments.full_public.trial_validation import run as validate
from experiments.full_public.tauri_validation import compare
from src.bond_order_kernel import environments
from src.bond_environment import PAIRS
from src.four_body import RADII
from src.local_stretch import StretchKernel
from src.forcefield import ForceField

WORK = kt.WORK
STAGE = "p39-public"
PARENT = "p37-public"
WIDTHS = (0.075, 0.14)


def two_widths(z, r):
    rows = np.zeros((len(WIDTHS), basis.size))
    for k, x, gate, *_ in environments(z, r):
        if k not in slices:
            continue
        prob, r0, _ = basis.interpolation(k, x[:, 1:])
        dr = (1 + x[:, 0]) * sum(RADII[s] for s in PAIRS[k]) - r0
        for j, w in enumerate(WIDTHS):
            rows[j, slices[k]] = (
                gate * 0.5 * dr**2 * np.exp(-0.5 * (dr / w) ** 2)
            ) @ prob
    return rows


def chunk(i):
    for j in range(i, min(i + 64, len(points))):
        for a, row in zip(maps, two_widths(*points[j])):
            a[j] = row


def setup():
    global basis, slices
    with np.load(ROOT / "basis.npz") as f:
        basis = StretchKernel.load(f)
    slices = {}
    off = 0
    for k, c in basis.centers.items():
        slices[k] = slice(off, off + len(c))
        off += len(c)


def cache():
    global points, maps
    paths = [ROOT / f"main-{w}.npy" for w in WIDTHS]
    if all(p.exists() for p in paths):
        return
    gs, _, _ = kt.inputs()
    with (WORK / "ids.pkl").open("rb") as f:
        _, keys = pickle.load(f)
    points = [(gs[k].z, gs[k].r) for k in keys]
    rng = np.random.default_rng(215)
    for idx in rng.choice(len(points), 6, replace=False):
        z, r = points[idx]
        v = rng.normal(size=r.shape)
        v /= np.linalg.norm(v)
        for j, w in enumerate(WIDTHS):
            basis.width = w
            a, df = basis.design(z, r, True)
            np.testing.assert_allclose(a, two_widths(z, r)[j], atol=1.0e-10)
            fd = (
                two_widths(z, r - 1.0e-5 * v)[j] - two_widths(z, r + 1.0e-5 * v)[j]
            ) / 2.0e-5
            np.testing.assert_allclose(df.T @ v.ravel(), fd, atol=2.0e-7)
    print("Verified conservative X-H stretch energies and forces", flush=True)
    for _, z, r, v in trial_sites.hdf():
        points.extend([(z, r), (z, r + 0.001 * v), (z, r - 0.001 * v)])
    maps = [
        np.lib.format.open_memmap(
            p.with_suffix(".pending"),
            mode="w+",
            dtype=float,
            shape=(len(points), basis.size),
        )
        for p in paths
    ]
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("fork")) as pool:
        for i, _ in enumerate(
            pool.map(chunk, range(0, len(points), 64), buffersize=14)
        ):
            if (i + 1) % 1600 == 0:
                print("Encoded stretch environments", (i + 1) * 64, flush=True)
    for a, p in zip(maps, paths):
        a.flush()
        p.with_suffix(".pending").replace(p)
    print("Completed public and HDF stretch features", len(points), flush=True)


def sample(item):
    name, idx, z, r, bonds = item
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
    )
    # Prefer one bond from each represented chemical type. Uniformly sample a
    # second position when a molecule only offers C-H bonds.
    chosen = []
    for k in sorted(set(b[0] for b in bonds)):
        group = [b for b in bonds if b[0] == k]
        take = rng.choice(len(group), min(len(group), 2), replace=False)
        chosen.extend(group[i] for i in take)
    mid = two_widths(z, r)
    e0 = parent.energy(z, r)
    mass = Chem.GetPeriodicTable()
    rows = []
    label = []
    for k, env, length, spring, i, j in chosen:
        axis = (r[i] - r[j]) / length
        masses = np.array([mass.GetMostCommonIsotopeMass(int(a)) for a in (z[i], z[j])])
        v = np.zeros_like(r)
        v[i] = axis * masses[1] / sum(masses)
        v[j] = -axis * masses[0] / sum(masses)
        h = 0.002
        rp, rm = r + h * v, r - h * v
        predicted = ((parent.energy(z, rp) - e0) + (parent.energy(z, rm) - e0)) / h**2
        design = ((two_widths(z, rp) - mid) + (two_widths(z, rm) - mid)) / h**2
        rows.append(design)
        label.append([k, spring, predicted])
    formula = tuple(sorted(map(int, z)))
    fold = (
        int.from_bytes(
            hashlib.sha256(("spring39/" + str(formula)).encode()).digest()[:4], "big"
        )
        % 4
    )
    return np.asarray(rows), label, np.full(len(rows), fold)


def force_constants():
    global parent
    out = ROOT / "sampled-springs.npz"
    if out.exists():
        return
    parent = ForceField.load(WORK / f"{PARENT}-high.npz", dispersion="d3")
    with (ROOT / "measurements.pkl").open("rb") as f:
        data = pickle.load(f)
    rank = lambda a: hashlib.sha256(a[0].encode()).digest()
    # Deterministic stratification includes rarer N-H and O-H stretches.
    selected = {}
    for k in basis.centers:
        group = sorted((r for r in data if any(b[0] == k for b in r[-1])), key=rank)
        for r in group[:210]:
            selected[r[0]] = r
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("fork")) as pool:
        rows = list(pool.map(sample, selected.values(), chunksize=2, buffersize=12))
    x, y, h = (np.concatenate(a) for a in zip(*rows))
    np.savez(out, x=x, y=y, fold=h)
    print(
        "Measured parent stretch-curvature errors",
        len(selected),
        len(y),
        {
            str(PAIRS[k]): round(abs(y[y[:, 0] == k, 1] - y[y[:, 0] == k, 2]).mean(), 2)
            for k in basis.centers
        },
        flush=True,
    )


def train():
    d = kt.equation_data("high")
    s, y, src, fold, kind, case = d
    base = ms.baseline("high", -1, "parent", PARENT)
    with np.load(ROOT / "sampled-springs.npz") as f:
        raw = f["x"]
        targets = f["y"]
        bins = f["fold"]
    target = targets[:, 1] - targets[:, 2]
    n = len(base)
    train = fold >= 0
    keep = train & np.isin(src, kt.BARRIERS)
    choices = {}
    candidate = {}
    parent = ForceField.load(WORK / f"{PARENT}-high.npz", dispersion="d3")
    for j, w in enumerate(WIDTHS):
        basis.width = w
        main = np.load(ROOT / f"main-{w}.npy", mmap_mode="r")
        pub = main[:n]
        x = s @ pub
        a = kt.grouped_moments(d, x, y - s @ base, train)
        gram = (
            a["other"]
            + 48 * a["mid"]
            + 36 * a["gap"]
            + isomer_gram("high", -1, pub, base)
        )
        ef, curv = anchor_blocks(main, n, f"stretch39-{w}")
        # This independent PBE0 shape target uses the fourteen NIST training
        # minima, never the four reserved relaxation/frequency comparisons.
        nr = nist_rows(lambda z, r: basis.design(z, r, True)[1], parent)
        ng = nr.T @ nr
        rows = raw[:, j]

        def fit(sel, strength):
            # A measured spring constant has units of eV/angstrom^2. This
            # scaling puts its residual on the energy scale of a 0.10 A stretch.
            scale = np.full(sel.sum(), strength)
            for k in basis.centers:
                mask = targets[sel, 0] == k
                scale[mask] *= min(2.0, np.sqrt(len(rows) / (3 * max(1, mask.sum()))))
            weighted = rows[sel] * scale[:, None]
            mat = (
                gram[:-1, :-1]
                + 0.2 * ef
                + 0.04 * curv
                + 45 * ng[:-1, :-1]
                + weighted.T @ weighted
            )
            mat.flat[:: len(mat) + 1] += 0.012
            rhs = (
                0.28 * gram[:-1, -1]
                + 45 * ng[:-1, -1]
                + weighted.T @ (scale * target[sel])
            )
            coef = cho_solve(cho_factor(mat), rhs)
            change = np.sqrt(np.mean((x[keep] @ coef * kt.KCAL) ** 2))
            return coef * min(1.0, 0.85 / max(change, 1.0e-12))

        for strength in (0.12, 0.4, 1.2, 4.0):
            key = f"{w}-{strength}"
            loss = []
            for h in range(4):
                coef = fit(bins != h, strength)
                held = bins == h
                loss.append(float(abs(rows[held] @ coef - target[held]).mean()))
            choices[key] = loss
            candidate[key] = (w, fit(np.ones(len(rows), bool), strength))
    best = min(choices, key=lambda k: np.mean(choices[k]))
    w, coef = candidate[best]
    basis.width = w
    basis.weights = coef
    parent.additional_bases.append(basis)
    path = WORK / f"{STAGE}-high.npz"
    parent.save_with_base(kt.original.BASE, path)
    en = base + np.load(ROOT / f"main-{w}.npy", mmap_mode="r")[:n] @ coef
    np.save(WORK / "stretch39-high-parent--1.npy", en)
    restored = ForceField.load(path, dispersion="d3")
    z, r = trial_fit.sample()
    e, force = restored.predict(z, r)
    np.testing.assert_allclose(e, en[trial_fit.sample.index], rtol=0, atol=2.0e-6)
    v = np.random.default_rng(851).normal(size=r.shape)
    v /= np.linalg.norm(v)
    fd = (
        restored.energy(z, r - 2.0e-5 * v) - restored.energy(z, r + 2.0e-5 * v)
    ) / 4.0e-5
    np.testing.assert_allclose(fd, (v * force).sum(), atol=5.0e-5)
    mask = fold == -1
    np.savez_compressed(
        WORK / f"{STAGE}-high-equations.npz",
        source=src[mask],
        case=case[mask],
        kind=kind[mask],
        error_kcal=(s @ en - y)[mask] * kt.KCAL,
    )
    for suffix in ("wb97xd3.npz", "wb97xd3-equations.npz", "held-modes.npy"):
        shutil.copyfile(WORK / f"{PARENT}-{suffix}", WORK / f"{STAGE}-{suffix}")
    previous = json.loads((WORK / f"{PARENT}-settings.json").read_text())["branches"][
        "wb97xd3"
    ]
    kt.write_json(
        WORK / f"{STAGE}-settings.json",
        dict(
            selected=STAGE,
            branches={"high": "stretch39", "wb97xd3": previous},
            selected_width_strength=best,
            stretch_errors=choices,
        ),
    )
    compare(STAGE)
    validate(STAGE)
    print(
        "Selected local spring correction",
        best,
        "development force-constant MAE",
        np.mean(choices[best]),
        flush=True,
    )


if __name__ == "__main__":
    with threadpool_limits(1):
        setup()
        force_constants()
        cache()
    with threadpool_limits(3):
        train()
    with threadpool_limits(1):
        evaluate_trial(STAGE)
