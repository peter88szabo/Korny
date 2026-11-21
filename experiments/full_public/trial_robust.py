"""Robust objectives for a scalar potential on reaction triples.

Only training reactions enter the iteratively reweighted solve. The endpoint
gap and midpoint barrier form independent coordinates of a reaction triple.
Source-normalized Huber weights prevent a small number of severe errors from
dominating either coordinate while preserving each source's total influence.
"""

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from experiments.full_public import kernel_trials as kt


def fit(d, design, target, train, pieces, anchor, tuned, threshold, strength):
    _, _, sources, _, kinds, cases = d
    scale, gap, mid = tuned
    ridge = 0.08
    combined = {
        **pieces,
        "mid": pieces["mid"] * strength,
        "gap": pieces["gap"] * strength,
    }
    coef = kt.solve(combined, anchor, gap, mid, scale, ridge)
    if threshold == 0:
        return coef
    rows = []
    for source in kt.BARRIERS:
        use = train & (sources == source)
        if not np.any(use):
            continue
        ids = [
            np.flatnonzero(use & (kinds == kind))
            for kind in ("forward", "reverse", "reaction")
        ]
        f, r, g = (ix[np.argsort(cases[ix])] for ix in ids)
        assert np.array_equal(cases[f], cases[r]) and np.array_equal(cases[f], cases[g])
        weight = strength * min(1.0, 4500 / use.sum())
        rows.append(
            (
                "mid",
                (design[f] + design[r]) * 0.5,
                (target[f] + target[r]) * 0.5,
                weight,
            )
        )
        rows.append(
            ("gap", design[g], (2 * target[g] + target[f] - target[r]) / 3, weight)
        )
    for _ in range(6):
        new = {k: np.zeros_like(pieces[k]) for k in ("gap", "mid")}
        for kind, x, y, source_weight in rows:
            err = np.abs(x @ coef - y) * kt.KCAL
            weights = np.minimum(1, threshold / np.maximum(err, 1e-12))
            weights *= source_weight / weights.mean()
            # Accumulate in small blocks so a second full weighted copy of the design is unnecessary.
            for lo in range(0, len(y), 512):
                sl = slice(lo, lo + 512)
                m = np.column_stack((x[sl], y[sl])) * np.sqrt(weights[sl, None])
                new[kind] += m.T @ m
        new["other"] = pieces["other"]
        update = kt.solve(new, anchor, gap, mid, scale, ridge)
        movement = np.max(np.abs(update - coef))
        coef = update
        if movement < 1e-5:
            break
    return coef


def robust_reactions(data, x, y, factor, least_squares, threshold_kcal):
    """Replace only the gap/midpoint squared errors by Huber losses.

    Use the Cholesky factor of the full original objective as a change of
    coordinates. Around its least-squares optimum it contributes half the
    squared displacement in those coordinates. Subtracting the quadratic
    tails of the reaction residuals leaves exactly the desired Huber objective,
    including the original PBE0, native path, force, and Hessian terms.
    """
    _, _, src, fold, kind, case = data
    blocks = []
    for source in kt.BARRIERS:
        use = (fold >= 0) & (src == source)
        if not use.any():
            continue
        weight = min(1.0, 4500 / use.sum())
        for label, a, b in kt.barrier_blocks(kind, case, use, x, y):
            blocks.append((a, a @ least_squares - b, weight * (96 if label == "mid" else 36)))
    lower, _ = factor
    threshold = threshold_kcal / kt.KCAL

    def objective(step):
        delta = solve_triangular(lower, step, lower=True, trans="T", check_finite=False)
        loss = 0.5 * (step @ step)
        grad = np.zeros_like(delta)
        for a, initial, weight in blocks:
            error = initial + a @ delta
            tail = error - np.clip(error, -threshold, threshold)
            loss -= 0.5 * weight * (tail @ tail)
            grad += weight * (a.T @ tail)
        return loss, step - solve_triangular(lower, grad, lower=True, check_finite=False)

    result = minimize(objective, np.zeros_like(least_squares), method="L-BFGS-B", jac=True,
                      options={"gtol": 1e-7, "ftol": 1e-13, "maxiter": 300})
    if not result.success:
        raise RuntimeError(f"Robust reaction optimization failed: {result.message}")
    delta = solve_triangular(lower, result.x, lower=True, trans="T", check_finite=False)
    return least_squares + delta, {
        "threshold_kcal_mol": threshold_kcal,
        "training_coordinate_rows": sum(len(a) for a, _, _ in blocks),
        "iterations": result.nit, "max_abs_preconditioned_gradient": float(np.max(np.abs(result.jac))),
    }
