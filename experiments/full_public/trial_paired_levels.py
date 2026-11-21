"""Couple two electronic reference levels with matched RDB7 CC-minus-DFT labels.

A common geometry-based prior uses the singlet-trained potential for singlets
and the radical-trained potential for doublets. The reference-level outputs
have their own coefficients. Direct paired differences constrain their gap and
allow the more abundant singlet DFT structures to inform the doublet DFT fit.
"""

from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import pickle
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_limits
from experiments.full_public import (
    kernel_trials as kt,
    trial_fit,
    trial_multiscale as ms,
)
from experiments.full_public import (
    trial_coupled_shapes as joint,
    trial_path_constraints as constraints,
)
from experiments.full_public.trial_fidelity import rdb_dft
from experiments.full_public.trial_constraints import OUT, POINTS, references
from experiments.full_public.trial_isomers import gram as isomer_gram
from experiments.full_public.trial_spectra import evaluate_trial
from experiments.full_public.trial_validation import run as validate
from experiments.full_public.tauri_validation import compare
from src.forcefield import ForceField

WORK = kt.WORK
STAGE = "p31-public"
PARENT = "p28-public"
# The inner formulas choose how strongly paired electronic levels share the
# residual. Harmonic calibration uses only the fourteen NIST training minima.
OPTIONS = ((0.018, 0.005, 0.2), (0.065, 0.018, 1.0), (0.20, 0.07, 3.0))


def pieces():
    result = []
    n = 0
    for b, ids in joint.banks():
        use = (n + np.arange(len(ids))) % 2 == 0
        result.append((b, ids[use]))
        n += len(ids)
    return result


def jac(z, r):
    cols = [b.design(z, r, True)[1][:, ix] for b, ix in basis]
    return np.column_stack([*cols, minima_parent.predict(z, r)[1].ravel()])


def shape_case(item):
    return constraints.molecule(item, jac, basis)[1][0]


def shapes():
    global basis, minima_parent
    basis = pieces()
    minima_parent = ForceField.load(WORK / f"{PARENT}-high.npz", dispersion="d3")
    points = json.loads(POINTS.read_text())
    points.update(references())
    names = json.loads((OUT / "stationary.json").read_text())["training"]
    with ProcessPoolExecutor(max_workers=3, mp_context=mp.get_context("fork")) as pool:
        rows = np.vstack(
            list(pool.map(shape_case, [(n, points[n]) for n in names], buffersize=6))
        )
    np.save(WORK / "paired31-training-hessians.npy", rows)
    print(
        "Exact force and Hessian residuals at all neutral training minima",
        rows.shape,
        flush=True,
    )


def common_grams(d, x, base, h):
    block = kt.grouped_moments(d, x, d[1] - d[0] @ base, (d[3] >= 0) & (d[3] != h))
    return block["other"] + 48 * block["mid"] + 36 * block["gap"]


def routed(t, coef, path):
    """Write one portable PES, with an alternate doublet readout of shared kernels."""
    high = ForceField.load(WORK / f"{PARENT}-high.npz", dispersion="d3")
    low = ForceField.load(WORK / f"{PARENT}-wb97xd3.npz", dispersion="d3")
    from experiments.full_public.trial_energy_blend import signature

    # Collapse duplicate feature maps before overlaying the second set of weights.
    def bank(m):
        result = {}
        for b in [m.reactive_basis, *m.additional_bases]:
            if b is None:
                continue
            key = signature(b)
            if key in result:
                result[key][1][:] += b.weights
            else:
                result[key] = (b, b.weights.copy())
        return result

    first, second = bank(high), bank(low)
    high.reactive_basis = None
    high.additional_bases = []
    extra = {}
    for key in dict.fromkeys([*first, *second]):
        b = (first[key] if key in first else second[key])[0]
        b.weights = first[key][1] if key in first else np.zeros(b.size)
        alt = second[key][1] if key in second else np.zeros(b.size)
        if not np.array_equal(b.weights, alt):
            extra[len(high.additional_bases)] = alt
        high.additional_bases.append(b)
    high.spin_readouts[2] = low.weights, extra
    high.prediction_target = (low if t == "wb97xd3" else high).prediction_target
    off = 0
    for b, idx in pieces():
        b.weights = np.zeros(b.size)
        b.weights[idx] = coef[off : off + len(idx)]
        off += len(idx)
        high.additional_bases.append(b)
    assert off == len(coef)
    high.save_with_base(kt.original.BASE, path)


def solve():
    gs, _, _ = kt.inputs()
    with (WORK / "ids.pkl").open("rb") as f:
        _, names = pickle.load(f)
    is_doublet = np.array([gs[k].state[1] == 2 for k in names])
    n = len(names)
    pub = np.asarray(
        np.load(WORK / "coupled29-main.npy", mmap_mode="r")[:n, ::2], order="C"
    )
    dim = pub.shape[1]
    rad = np.asarray(
        np.load(WORK / "coupled29-rad.npy", mmap_mode="r")[:, ::2], order="C"
    ).reshape(-1, 7, dim)
    ef = np.load(WORK / "coupled29-anchor-ef.npy", mmap_mode="r")[::2, ::2]
    curv = np.load(WORK / "coupled29-anchor-curvature.npy", mmap_mode="r")[::2, ::2]
    cm = np.asarray(np.load(WORK / "paired31-training-hessians.npy"), float)
    ng = cm.T @ cm
    dh, dr = (kt.equation_data(t) for t in kt.TARGETS)
    ds = rdb_dft(dh)
    xh, xr = dh[0] @ pub, dr[0] @ pub
    use_rdb = dh[2] == "RDB7"
    xs = xh[use_rdb]
    np.testing.assert_array_equal(dh[5][use_rdb], ds[5])
    np.testing.assert_array_equal(dh[4][use_rdb], ds[4])
    gap = (ds[0], dh[1][use_rdb] - ds[1], *ds[2:])
    meta = np.load(WORK / "rad-meta.npy")
    with (WORK / "radical-modes.pkl").open("rb") as f:
        records = pickle.load(f)
        assert all(row[1][3][1] == 2 for row in records)
        saddle = np.array(
            [row[1][0].split("/")[-1].startswith("ts") for row in records]
        )
    dev = {}
    details = {}
    for h in (0, 1, -1):
        base = np.where(
            is_doublet,
            ms.baseline("wb97xd3", h, "parent", PARENT),
            ms.baseline("high", h, "parent", PARENT),
        )
        bm = ms.baseline("wb97xd3", h, "modes", PARENT)
        a = common_grams(dh, xh, base, h) + isomer_gram("high", h, pub, base)
        b = common_grams(dr, xr, base, h) + isomer_gram("wb97xd3", h, pub, base)
        # Archived RDB7 DFT labels use the same geometries as the CC labels.
        # Scale their contribution so that over 11k singlets cannot swamp the
        # ~1k radical reactions in the primary wB97X-D3 task.
        b += 0.22 * common_grams(ds, xs, base, h)
        g = common_grams(gap, xs, np.zeros(len(base)), h)
        z, y = trial_fit.rad_rows(rad, bm, meta, np.where(saddle, 4.0, 1.0))
        b += trial_fit.reduce_gram(
            z, y, np.repeat((meta[:, 0] >= 0) & (meta[:, 0] != h), 3)
        )
        del z, y
        best = None if h >= 0 else min(dev, key=lambda q: np.mean(dev[q]))
        for j, (e, shape, couple) in enumerate(OPTIONS):
            key = f"paired31-opt{j}"
            if best is not None and best != key:
                continue
            a0 = a[:-1, :-1] + e * ef + shape * curv + 90 * ng[:-1, :-1]
            b0 = b[:-1, :-1] + e * ef + shape * curv + 60 * ng[:-1, :-1]
            cross = couple * g[:-1, :-1]
            system = np.block([[a0 + cross, -cross], [-cross, b0 + cross]])
            system.flat[:: len(system) + 1] += 0.40
            rhs = np.r_[
                a[:-1, -1] + couple * g[:-1, -1], b[:-1, -1] - couple * g[:-1, -1]
            ]
            # The final parent is known exactly. Inner folds use homogeneous
            # shape protection so final-fit reaction labels cannot enter their scores.
            if h < 0:
                rhs += np.r_[90 * ng[:-1, -1], 60 * ng[:-1, -1]]
            ah, ad = np.split(
                cho_solve(
                    cho_factor(system, check_finite=False), rhs, check_finite=False
                ),
                2,
            )
            sc = []
            for t, d, v in (("high", dh, ah), ("wb97xd3", dr, ad)):
                en = base + pub @ v
                err = d[0] @ en - d[1]
                np.save(WORK / f"{key}-{t}-parent-{h}.npy", en)
                if t == "wb97xd3":
                    modes = bm + rad @ v
                    np.save(WORK / f"{key}-{t}-modes-{h}.npy", modes)
                if h >= 0:
                    ix = (d[3] == h) & trial_fit.selection_mask(d, t)
                    rx, fw, bw = ms.metrics(d, err, ix)
                    sc.append(0.4 * rx + 0.3 * (fw + bw))
                    if t == "wb97xd3":
                        sc[-1] += (
                            ms.saddle_metrics(modes, (meta[:, 0] == h) & saddle, meta)[
                                0
                            ]
                            / 400.0
                        )
                else:
                    sel = d[3] == -1
                    np.savez_compressed(
                        WORK / f"{STAGE}-{t}-equations.npz",
                        source=d[2][sel],
                        case=d[5][sel],
                        kind=d[4][sel],
                        error_kcal=err[sel] * kt.KCAL,
                    )
                    if t == "wb97xd3":
                        np.save(
                            WORK / f"{STAGE}-held-modes.npy", modes[meta[:, 0] == -1]
                        )
                    path = WORK / f"{STAGE}-{t}.npz"
                    routed(t, v, path)
                    ff = ForceField.load(path, dispersion="d3")
                    rng = np.random.default_rng(921)
                    tests = [
                        np.flatnonzero(is_doublet)[5],
                        np.flatnonzero(~is_doublet)[5],
                        next(
                            i
                            for i, k in enumerate(names)
                            if gs[k].state[0] != 0 and len(gs[k].z) > 1
                        ),
                    ]
                    for idx in tests:
                        sample = gs[names[idx]]
                        zz, rr = sample.z, sample.r
                        energy, force = ff.predict(zz, rr, *sample.state)
                        np.testing.assert_allclose(energy, en[idx], atol=2.0e-6, rtol=0)
                        direction = rng.normal(size=rr.shape)
                        direction /= np.linalg.norm(direction)
                        diff = (
                            ff.energy(zz, rr - 2.0e-5 * direction, *sample.state)
                            - ff.energy(zz, rr + 2.0e-5 * direction, *sample.state)
                        ) / 4.0e-5
                        np.testing.assert_allclose(
                            diff, (force * direction).sum(), atol=5.0e-5
                        )
            if h >= 0:
                details.setdefault(key, []).append(sc)
                dev.setdefault(key, []).append(float(np.mean(sc)))
        print("Completed paired-reference fit", h, flush=True)
    kt.write_json(
        WORK / f"{STAGE}-settings.json",
        dict(
            selected=STAGE,
            branches=dict.fromkeys(kt.TARGETS, best),
            development=details,
            paired_training_reactions=int(np.sum((ds[3] >= 0) & (ds[4] == "reaction"))),
        ),
    )
    compare(STAGE)
    validate(STAGE)


if __name__ == "__main__":
    if not (WORK / "paired31-training-hessians.npy").exists():
        with threadpool_limits(1):
            shapes()
    with threadpool_limits(4):
        solve()
    with threadpool_limits(1):
        evaluate_trial(STAGE)
