"""Fit orbital, charge and density kernels jointly with physical Hessian constraints.

Start from p22, which retained realistic equilibrium frequencies. Each inner
reference has its own exact p22 readout. Force balance and all entries of the
internal Hessian at 14 training minima constrain the new coefficients, including
off-diagonal coupling between bending and stretching directions. The other four
NIST molecules enter only the subsequent independent relaxation and evaluation.
"""

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import pickle
import numpy as np
from scipy.linalg import cho_solve, cho_factor
from threadpoolctl import threadpool_limits
from experiments.full_public import (
    kernel_trials as kt,
    trial_fit,
    trial_multiscale as ms,
)
from experiments.full_public import (
    trial_path_constraints as constraints,
    trial_parent_shapes as p12,
)
from experiments.full_public import trial_path_features as pathfeat
from experiments.full_public.trial_curvature_fit import anchor_blocks
from experiments.full_public.trial_isomers import gram as isomer_gram
from experiments.full_public.trial_constraints import OUT, POINTS, references
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_validation import run as validate
from experiments.full_public.trial_spectra import evaluate_trial
from src.atomic_states import AtomicStateKernel
from src.orbital_moments import OrbitalMomentKernel
from src.periodic_kernel import PeriodicKernel
from src.forcefield import ForceField

WORK = kt.WORK
STAGE = "p29-public"
PARENT = "p22-public"
FAMILIES = (
    ("periodic", 0.85, PeriodicKernel),
    ("orbital", 0.8, OrbitalMomentKernel),
    ("orbital", 1.5, OrbitalMomentKernel),
    ("state", 0.8, AtomicStateKernel),
    ("state", 1.5, AtomicStateKernel),
)
OPTIONS = (
    (0.012, 0.002, 30.0, 1.5),
    (0.028, 0.008, 110.0, 3.0),
    (0.07, 0.025, 330.0, 5.0),
)


def banks():
    pieces = []
    for family, width, cls in FAMILIES:
        with np.load(WORK / f"{family}-basis.npz") as f:
            b = cls.load(f)
        b.length = width
        offsets = np.cumsum([0] + [len(v) for v in b.centers.values()])
        # Retain centers in each chemical group. Alternate the centers used by
        # the two resolutions of the same descriptor.
        ids = np.concatenate(
            [
                np.arange(lo + int(width == 1.5 and hi - lo > 1), hi, 6)
                for lo, hi in zip(offsets, offsets[1:])
            ]
        )
        pieces.append((b, ids))
    return pieces


def cached():
    parts = banks()
    size = sum(len(ix) for _, ix in parts)
    for side in ("main", "rad"):
        out = WORK / f"coupled29-{side}.npy"
        if out.exists():
            continue
        inputs = [
            np.load(WORK / f"{f}-{w:g}-{side}.npy", mmap_mode="r")
            for f, w, _ in FAMILIES
        ]
        tmp = out.with_suffix(".pending")
        target = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=float, shape=(len(inputs[0]), size)
        )
        for i in range(0, len(target), 384):
            s = slice(i, i + 384)
            target[s] = np.column_stack(
                [a[s][:, ix] for a, (_, ix) in zip(inputs, parts)]
            )
        target.flush()
        del target, inputs
        tmp.replace(out)
    print("Joint energy features", size, "coefficients", flush=True)


def physical(z, r):
    """Build the six true p22 parent gradients and new conservative columns."""
    data = constraints.cartesian(z, r, path_banks)
    parents = data[:, -6:] + data[:, :-6] @ p22weights
    obs = np.column_stack([b.design(z, r, True)[1][:, ix] for b, ix in new])
    return np.column_stack((obs, parents))


def shape_case(item):
    return constraints.molecule(item, physical, new)


def shapes():
    global new, path_banks, p22weights
    p12.configure()
    new = banks()
    path_banks = pathfeat.bases()
    selected = json.loads((WORK / f"{PARENT}-settings.json").read_text())["branches"]
    p22weights = np.column_stack(
        [
            np.load(WORK / f"{selected[t]}-{t}-weights-{h}.npy")
            for t in kt.TARGETS
            for h in (0, 1, -1)
        ]
    )
    points = json.loads(POINTS.read_text())
    points.update(references())
    names = json.loads((OUT / "stationary.json").read_text())["training"]
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("fork")) as pool:
        rows = list(pool.map(shape_case, [(k, points[k]) for k in names], buffersize=8))
    for j in range(6):
        np.save(
            WORK / f"coupled29-nist-{j}.npy", np.vstack([data[j] for _, data in rows])
        )
    print(
        "Full Hessian and force targets for 14 minima in six cross-fitted readouts",
        flush=True,
    )


def fit():
    n = len(np.load(WORK / "prior.npy", mmap_mode="r"))
    main = np.load(WORK / "coupled29-main.npy", mmap_mode="r")
    pub = main[:n]
    size = pub.shape[1]
    rad = np.load(WORK / "coupled29-rad.npy", mmap_mode="r").reshape(-1, 7, size)
    ef, curv = anchor_blocks(main, n, "coupled29")
    meta = np.load(WORK / "rad-meta.npy")
    with (WORK / "radical-modes.pkl").open("rb") as f:
        saddle = np.array(
            [p[1][0].split("/")[-1].startswith("ts") for p in pickle.load(f)]
        )
    stages = {}
    scores = {}
    for ti, t in enumerate(kt.TARGETS):
        d = kt.equation_data(t)
        s, y, src, fold, kind, case = d
        primary = trial_fit.selection_mask(d, t)
        x = s @ pub
        dev = {}
        for col, h in enumerate((0, 1, -1)):
            base = ms.baseline(t, h, "parent", PARENT)
            residual = y - s @ base
            train = (fold >= 0) & (fold != h)
            block = kt.grouped_moments(d, x, residual, train)
            g = (
                block["other"]
                + 48 * block["mid"]
                + 36 * block["gap"]
                + isomer_gram(t, h, pub, base)
            )
            shape = np.load(WORK / f"coupled29-nist-{3 * ti + col}.npy").astype(float)
            ng = shape.T @ shape
            if ti:
                bm = ms.baseline(t, h, "modes", PARENT)
                z, rhs = trial_fit.rad_rows(rad, bm, meta, np.where(saddle, 3.0, 1.0))
                rg = trial_fit.reduce_gram(
                    z, rhs, np.repeat((meta[:, 0] >= 0) & (meta[:, 0] != h), 3)
                )
                del z, rhs
            chosen = None if h >= 0 else min(dev, key=lambda k: np.mean(dev[k]))
            for i, (energy, curve, nist, imag) in enumerate(OPTIONS):
                key = f"coupled29-option{i}"
                if chosen is not None and key != chosen:
                    continue
                a = g + nist * ng
                if ti:
                    a += imag * rg
                mat = a[:-1, :-1] + energy * ef + curve * curv
                mat.flat[:: size + 1] += 0.45
                coef = cho_solve(
                    cho_factor(mat, check_finite=False), a[:-1, -1], check_finite=False
                )
                en = base + pub @ coef
                err = s @ en - y
                if ti:
                    modes = bm + rad @ coef
                np.save(WORK / f"{key}-{t}-parent-{h}.npy", en)
                np.save(WORK / f"{key}-{t}-weights-{h}.npy", coef)
                if ti:
                    np.save(WORK / f"{key}-{t}-modes-{h}.npy", modes)
                if h >= 0:
                    rx, fw, bw = ms.metrics(d, err, (fold == h) & primary)
                    val = 0.4 * rx + 0.3 * (fw + bw)
                    if ti:
                        val += (
                            ms.saddle_metrics(modes, (meta[:, 0] == h) & saddle, meta)[
                                0
                            ]
                            / 350.0
                        )
                    val += np.mean(abs(shape[:, :-1] @ coef - shape[:, -1]))
                    dev.setdefault(key, []).append(float(val))
                else:
                    use = fold < 0
                    np.savez_compressed(
                        WORK / f"{STAGE}-{t}-equations.npz",
                        source=src[use],
                        case=case[use],
                        kind=kind[use],
                        error_kcal=err[use] * kt.KCAL,
                    )
                    if ti:
                        np.save(
                            WORK / f"{STAGE}-held-modes.npy", modes[meta[:, 0] == -1]
                        )
                    model = ForceField.load(WORK / f"{PARENT}-{t}.npz", dispersion="d3")
                    offset = 0
                    for b, ix in banks():
                        b.weights = np.zeros(b.size)
                        b.weights[ix] = coef[offset : offset + len(ix)]
                        model.additional_bases.append(b)
                        offset += len(ix)
                    path = WORK / f"{STAGE}-{t}.npz"
                    model.save_with_base(kt.original.BASE, path)
                    loaded = ForceField.load(path, dispersion="d3")
                    zz, rr = trial_fit.sample()
                    e, force = loaded.predict(zz, rr)
                    np.testing.assert_allclose(
                        e, en[trial_fit.sample.index], rtol=0, atol=3.0e-7
                    )
                    dr = np.random.default_rng(139).normal(size=rr.shape)
                    dr /= np.linalg.norm(dr)
                    diff = (
                        loaded.energy(zz, rr - 2.0e-5 * dr)
                        - loaded.energy(zz, rr + 2.0e-5 * dr)
                    ) / 4.0e-5
                    np.testing.assert_allclose(diff, (force * dr).sum(), atol=4.0e-5)
                    stages[t] = key
            print("Joint physical fit completed", t, h, flush=True)
        scores[t] = dev
    kt.write_json(
        WORK / f"{STAGE}-settings.json",
        dict(selected=STAGE, branches=stages, development=scores),
    )
    compare(STAGE)
    validate(STAGE)


if __name__ == "__main__":
    with threadpool_limits(1):
        cached()
        if not (WORK / "coupled29-nist-5.npy").exists():
            shapes()
    with threadpool_limits(5):
        fit()
    with threadpool_limits(1):
        evaluate_trial(STAGE)
