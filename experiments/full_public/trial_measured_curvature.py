"""Calibrate a reaction surface on measured force changes across QM7-X conformers.

Observed PBE0 forces contain only the bare interaction. The comparison to them
therefore uses the current parent with dispersion disabled. Production predictions include D3
once, as for the other benchmarks. Corrections are constrained by all training
reaction energies and the original HDF5 directional forces and curvatures.
"""

import hashlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import pickle
import shutil
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits
from experiments.full_public import (
    kernel_trials as kt,
    trial_fit,
    trial_multiscale as ms,
)
from experiments.full_public.trial_isomers import gram as isomer_gram
from experiments.full_public.trial_spectra import evaluate_trial
from experiments.full_public.trial_validation import run as validate
from experiments.full_public.tauri_validation import compare
from experiments.reactivity.frequency_validation import TO_CM1
from src.periodic_kernel import PeriodicKernel
from src.forcefield import ForceField

WORK = kt.WORK
ROOT = WORK / "measured-curvature32"
STAGE = "p32-public"
PARENT = "p28-public"
WIDTH = 0.85


def load_basis():
    with np.load(WORK / "periodic-basis.npz") as ar:
        b = PeriodicKernel.load(ar)
    b.length = WIDTH
    b.degree = 3
    offsets = np.cumsum([0] + [len(v) for v in b.centers.values()])
    idx = np.concatenate([np.arange(a, h, 3) for a, h in zip(offsets, offsets[1:])])
    return b, idx


def record(item):
    name, z, r, fref, transform, axes, eig, split, r2, _, _ = item
    file = ROOT / (name + ".npz")
    if file.exists():
        return
    # Independently estimated halves must agree reasonably along the selected
    # reference direction. The bands include low bends and high stretches.
    freq = TO_CM1 * np.sign(eig) * np.sqrt(abs(eig))
    ids = []
    for lo, hi in ((200.0, 800.0), (800.0, 1300.0), (1300.0, 1870.0), (2400.0, 3800.0)):
        take = np.flatnonzero((freq >= lo) & (freq < hi) & (split <= 120.0))
        if len(take):
            ids.append(take[np.argmin(split[take])])
    if not ids:
        return

    def columns(pos):
        return np.column_stack(
            (basis.design(z, pos, True)[1][:, cols], model.predict(z, pos)[1].ravel())
        )

    center = columns(r)
    center[:, -1] = fref.ravel() - center[:, -1]
    blocks = [center * (2.0 / np.sqrt(r.size))]
    # Project the full Cartesian force response onto the internal axes. The
    # diagonal tests the full magnitude, off-diagonal rows constrain rotation
    # of a bending or stretching direction into neighboring modes.
    for j in ids:
        v = (transform @ axes[:, j]).reshape(r.shape)
        norm = np.linalg.norm(v)
        v /= norm
        a = (
            axes.T
            @ transform.T
            @ ((columns(r - 0.002 * v) - columns(r + 0.002 * v)) * (norm / 0.004))
        )
        a[:, -1] *= -1
        a[j, -1] += eig[j]
        scale = TO_CM1**2 / (
            2 * np.maximum(400.0, 0.5 * (np.abs(freq) + freq[j])) * 120.0
        )
        scale *= np.where(split <= 140.0, 0.2, 0.0)
        scale[j] = TO_CM1**2 / (2 * max(400.0, freq[j]) * 120.0)
        scale /= np.sqrt(len(ids))
        blocks.append(a[scale > 0] * scale[scale > 0, None])
    data = np.vstack(blocks)
    tmp = file.with_suffix(".pending")
    with tmp.open("wb") as f:
        np.savez(f, rows=data.astype(np.float32), bands=ids)
    tmp.replace(file)


def prepare():
    global model, basis, cols
    ROOT.mkdir(exist_ok=True)
    with (WORK / "qm7x-local-hessians.pkl").open("rb") as f:
        records = pickle.load(f)
    # Check only the training-fit reconstruction against independent distortions
    # of the same conformer. No held-out reaction or NIST frequency enters here.
    records = [r for r in records if r[8] > 0.80]
    model = ForceField.load(WORK / f"{PARENT}-high.npz", dispersion="off")
    basis, cols = load_basis()
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context("fork")) as pool:
        for j, _ in enumerate(pool.map(record, records, buffersize=16)):
            if (j + 1) % 64 == 0:
                print(
                    "Measured vibrational force responses",
                    j + 1,
                    "of",
                    len(records),
                    flush=True,
                )
    print(
        "Completed PBE0 force-response features",
        len(list(ROOT.glob("*.npz"))),
        flush=True,
    )


def train():
    b, idx = load_basis()
    n = len(np.load(WORK / "prior.npy", mmap_mode="r"))
    pub = np.array(
        np.load(WORK / "periodic-0.85-main.npy", mmap_mode="r")[:n, idx], order="C"
    )
    ef = np.load(WORK / "periodic-0.85-anchor-ef.npy", mmap_mode="r")[np.ix_(idx, idx)]
    curvature = np.load(WORK / "periodic-0.85-anchor-curvature.npy", mmap_mode="r")[
        np.ix_(idx, idx)
    ]
    sums = [np.zeros((len(idx) + 1,) * 2) for _ in range(4)]
    nb = 0
    for f in sorted(ROOT.glob("*.npz")):
        parent = f.stem.split("-i")[0]
        fold = int.from_bytes(hashlib.sha256(parent.encode()).digest()[:4], "big") % 4
        with np.load(f) as ar:
            x = ar["rows"].astype(float)
            nb += len(ar["bands"])
        sums[fold] += x.T @ x
    total = sum(sums)
    d = kt.equation_data("high")
    s, y, src, fold, kind, case = d
    x = s @ pub
    use = fold >= 0
    base = ms.baseline("high", -1, "parent", PARENT)
    resid = y - s @ base
    g = kt.grouped_moments(d, x, resid, use)
    equations = (
        g["other"] + 48 * g["mid"] + 36 * g["gap"] + isomer_gram("high", -1, pub, base)
    )

    # Preserve the parent on energy differences. Hyperparameters use errors on
    # entire QM7-X parents left out of the response fit and a training-reaction
    # shift budget. The final external NIST and reaction panels stay reserved.
    def weights(measured, strength):
        mat = (
            equations[:-1, :-1]
            + 0.12 * ef
            + 0.045 * curvature
            + strength * measured[:-1, :-1]
        )
        mat.flat[:: len(mat) + 1] += 0.6
        coef = cho_solve(
            cho_factor(mat, check_finite=False),
            0.30 * equations[:-1, -1] + strength * measured[:-1, -1],
            check_finite=False,
        )
        movement = x[use & np.isin(src, kt.BARRIERS)] @ coef * kt.KCAL
        size = np.sqrt(np.mean(movement**2))
        return coef * min(1.0, 1.1 / max(size, 1.0e-12))

    scores = {}
    options = (0.3, 1.5, 8.0, 35.0, 160.0)
    for st in options:
        vals = []
        for h, held in enumerate(sums):
            v = np.r_[weights(total - held, st), -1.0]
            vals.append(float(v @ held @ v / max(held[-1, -1], 1.0e-12)))
        scores[st] = vals
    best = min(scores, key=lambda k: np.mean(scores[k]))
    coef = weights(total, best)
    en = base + pub @ coef
    key = STAGE.replace("p", "measured", 1).split("-")[0]
    np.save(WORK / f"{key}-high-parent--1.npy", en)
    b.weights = np.zeros(b.size)
    b.weights[idx] = coef
    model = ForceField.load(WORK / f"{PARENT}-high.npz", dispersion="d3")
    model.additional_bases.append(b)
    path = WORK / f"{STAGE}-high.npz"
    model.save_with_base(kt.original.BASE, path)
    restored = ForceField.load(path, dispersion="d3")
    z, r = trial_fit.sample()
    energy, force = restored.predict(z, r)
    np.testing.assert_allclose(energy, en[trial_fit.sample.index], rtol=0, atol=3.0e-7)
    dr = np.random.default_rng(712).normal(size=r.shape)
    dr /= np.linalg.norm(dr)
    fd = (
        restored.energy(z, r - 2.0e-5 * dr) - restored.energy(z, r + 2.0e-5 * dr)
    ) / 4.0e-5
    np.testing.assert_allclose(np.sum(force * dr), fd, atol=4.0e-5)
    held = fold == -1
    np.savez_compressed(
        WORK / f"{STAGE}-high-equations.npz",
        source=src[held],
        case=case[held],
        kind=kind[held],
        error_kcal=(s @ en - y)[held] * kt.KCAL,
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
            branches={"high": key, "wb97xd3": previous},
            strength=best,
            shape_candidates={str(k): v for k, v in scores.items()},
            measured_modes=nb,
            force_conformers=len(list(ROOT.glob("*.npz"))),
        ),
    )
    compare(STAGE)
    validate(STAGE)
    print(
        "Calibrated on measured QM7-X force responses",
        nb,
        "directions",
        best,
        scores[best],
        flush=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orbital-parent",
        action="store_true",
        help="Calibrate the p36 bond-population potential.",
    )
    if parser.parse_args().orbital_parent:
        PARENT = "p36-public"
        STAGE = "p37-public"
        ROOT = WORK / "measured-curvature37"
    with threadpool_limits(1):
        prepare()
    with threadpool_limits(4):
        train()
    with threadpool_limits(1):
        evaluate_trial(STAGE)
