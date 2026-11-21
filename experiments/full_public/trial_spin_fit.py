"""Select the midpoint/gap objective on held-out training singlets and doublets."""

import json
import numpy as np
from threadpoolctl import threadpool_limits
from experiments.full_public import kernel_trials as kt
from experiments.full_public.trial_constraints import nist_rows
from src.forcefield import ForceField

WORK, OUT = kt.WORK, kt.OUT


def selection_mask(d, target):
    _, _, src, _, kind, case = d
    gs, rows, _ = kt.inputs()
    mult = {
        (r[1], r[3]): gs[r[4][-1][0]].state[1]
        for r in rows
        if r[0] == target and r[-1] == "forward"
    }
    return np.array(
        [s != "TRIP50" and mult.get((s, n), 0) in (1, 2) for s, n in zip(src, case)]
    )


def run():
    model = ForceField.load(OUT / "joint-pbe0-reaction.npz", dispersion="d3")
    x = np.load(WORK / "base-x.npy", mmap_mode="r")
    base = np.load(WORK / "prior.npy")
    gram = np.load(OUT / "joint-gram.npy", mmap_mode="r")[:-1, :-1]
    lock = kt.original.locks(
        model, json.loads((OUT / "stationary.json").read_text())["training"]
    )
    locks = lock.T @ lock
    parent = model.weights.copy()
    nist = nist_rows(lambda z, r: model.design(z, r, 0, 1, True)[1], model)
    nist = nist.T @ nist
    opts = [
        (a, g, m)
        for a in (0.003, 0.03, 0.15)
        for g, m in ((1.5, 2.0), (1.5, 6.0), (4.5, 2.0))
    ]
    results = {}
    choices = {}
    for target in kt.TARGETS:
        d = kt.equation_data(target)
        S, ref, _, folds, _, _ = d
        design = S @ x
        y = ref - S @ base
        sel = selection_mask(d, target)
        scores = {k: [] for k in opts}
        hd = (
            np.load(OUT / "radical-derivative-gram.npy", mmap_mode="r")
            if target == "wb97xd3"
            else None
        )
        for h in (0, 1, -1):
            training = (folds >= 0) & (folds != h)
            pieces = kt.grouped_moments(d, design, y, training)
            pieces["other"][:-1, :-1] += locks
            pieces["other"][:-1, -1] -= locks @ parent
            # Include NIST geometry/frequency labels directly in the high-level
            # correction. Identical training-only rows are used for every candidate.
            pieces["other"] += 8 * nist
            if hd is not None:
                withheld = (
                    np.load(WORK / f"radical-fold-{h}.npy", mmap_mode="r")
                    if h >= 0
                    else 0.0
                )
                dh = hd - withheld
                pieces["other"][:-1, :-1] += dh[:-1, :-1]
                pieces["other"][:-1, -1] += dh[:-1, -1] - dh[:-1, :-1] @ parent
                del dh
            if h == -1:
                best = min(opts, key=lambda k: np.mean(scores[k]))
                choices[target] = dict(
                    best=best, inner=[(k, v) for k, v in scores.items()]
                )
                write_keys = [best]
            else:
                write_keys = opts
            for a, g, m in write_keys:
                coef = kt.solve(pieces, gram, g, m, a, 0.08)
                if h >= 0:
                    scores[a, g, m].append(
                        kt.score_value(
                            kt.stats(d, design @ coef - y, (folds == h) & sel)
                        )
                    )
                    np.save(WORK / f"spin-{target}-{h}-{a}-{g}-{m}.npy", coef)
                else:
                    for hh in (0, 1):
                        b = np.load(WORK / f"spin-{target}-{hh}-{a}-{g}-{m}.npy")
                        np.save(WORK / f"spin-energy-{target}-{hh}.npy", base + x @ b)
                    np.save(WORK / f"spin-energy-{target}-all.npy", base + x @ coef)
                    model.weights = parent + coef
                    model.prediction_target = ForceField.load(
                        OUT / f"joint-{target}-reaction.npz"
                    ).prediction_target
                    model.save_with_base(kt.original.BASE, WORK / f"spin-{target}.npz")
                    model.weights = parent.copy()
                    results[target] = kt.stats(d, design @ coef - y, folds == -1)
            del pieces
        print(
            "Spin-prioritized P1",
            target,
            choices[target]["best"],
            results[target],
            flush=True,
        )
    kt.write_json(WORK / "spin-choices.json", choices)
    kt.write_json(WORK / "spin.json", results)


if __name__ == "__main__":
    with threadpool_limits(limits=4, user_api="blas"):
        run()
