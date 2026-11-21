"""Compare radical-path acquisition strategies at equal reaction budgets.

Use only training formulas to build a committee of kernel energy functions.
The committee sees geometry features at both endpoints and the common saddle
when ranking additional training reactions. Evaluate once on the identical
direction-assigned, held-out electronic reactions as the nested learning curve.
"""

import hashlib
import json
from collections import Counter

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from experiments.full_public import irc_learning_curve as curve
from experiments.full_public import kernel_trials as kt


def order(record, repeat):
    salt = "fixed-irc-curve/" if repeat == 0 else f"repeat-{repeat}/fixed-irc-curve/"
    return hashlib.sha256((salt + record["rxn"]).encode()).digest()


def fit(gram, n):
    lhs = gram[:-1, :-1].copy()
    lhs.flat[:: len(lhs) + 1] += 0.015 * n
    return cho_solve(cho_factor(lhs), gram[:-1, -1])


def operators(ends, educt=0, product=2):
    a, h, b = ends[[educt, 1, product]]
    return np.stack((b - a, h - (a + b) / 2, h - a, h - b))


def run(repeat=0):
    root = curve.ROOT
    rec = json.loads((root / "learning-curve-snapshot-ids.json").read_text())
    train = sorted(
        (r for r in rec if not r["heldout"]), key=lambda row: order(row, repeat)
    )
    seed, candidates = train[:71], train[71:]
    budget = 512 - len(seed)
    assert len(candidates) >= budget
    held = []
    for r in rec:
        if r["heldout"]:
            direction = curve.direction(r["rxn"])
            if direction is not None:
                with np.load(root / (r["rxn"] + ".npz")) as f:
                    held.append(operators(f["ends"], *direction))
    held = np.stack(held)
    rng = np.random.default_rng(68347 + repeat)
    # Shared formulas appear in several paths. Sampling entire training
    # formulas builds disagreement from changes of chemical coverage.
    formulas = sorted({r["formula"] for r in seed})
    bags = [
        Counter(rng.choice(formulas, len(formulas), replace=True)) for _ in range(8)
    ]
    grams = np.zeros((len(bags) + 1, held.shape[-1], held.shape[-1]))
    n = np.zeros(len(grams), int)
    for r in seed:
        with np.load(root / (r["rxn"] + ".npz")) as f:
            g = f["gram"]
        for j, v in enumerate([1, *(b[r["formula"]] for b in bags)]):
            grams[j] += v * g
            n[j] += v
    committee = np.column_stack(
        [fit(g, size) for g, size in zip(grams[1:], n[1:], strict=True)]
    )
    score = {}
    for r in candidates:
        with np.load(root / (r["rxn"] + ".npz")) as f:
            # The final column is the archived electronic label. Only the
            # geometry columns enter acquisition and no test row is ranked.
            midgap = operators(f["ends"][:, :-1])[:2]
        score[r["rxn"]] = float(np.mean(np.var(midgap @ committee, axis=1)))
    ranked = sorted(candidates, key=lambda r: -score[r["rxn"]])
    # Control for class coverage by giving each family precisely the same
    # number of slots as it received in the random comparison.
    remaining = Counter(r["family"] for r in candidates[:budget])
    by_family = []
    for r in ranked:
        if remaining[r["family"]] > 0:
            by_family.append(r)
            remaining[r["family"]] -= 1
    strategies = {
        "random": candidates[:budget],
        "disagreement": ranked[:budget],
        "within_class_disagreement": by_family,
    }
    selections = {key: {r["rxn"] for r in rows} for key, rows in strategies.items()}
    totals = {key: grams[0].copy() for key in selections}
    total_all = grams[0].copy()
    for r in candidates:
        choices = [key for key, keep in selections.items() if r["rxn"] in keep]
        with np.load(root / (r["rxn"] + ".npz")) as f:
            g = f["gram"]
        total_all += g
        for key in choices:
            totals[key] += g
    # Half the weight stays uniform across every training path. The other half
    # upweights the committee-selected 512, preserving total weight and ridge.
    strategies.update({"all_equal": candidates, "all_weighted": candidates})
    totals["all_equal"] = total_all
    totals["all_weighted"] = 0.5 * (
        total_all + len(train) / 512 * totals["disagreement"]
    )
    results = {}
    for key, rows in strategies.items():
        size = len(seed) + len(rows)
        if key in selections:
            assert len(selections[key]) == budget
        err = held[:, :, :-1] @ fit(totals[key], size) - held[:, :, -1]
        gap, mid, forward, reverse = np.abs(err).mean(axis=0) * kt.KCAL
        results[key] = {
            "reaction": gap,
            "midpoint": mid,
            "forward": forward,
            "reverse": reverse,
            "barriers": (forward + reverse) / 2,
            "training_reactions": size,
            "training_formulas": len({r["formula"] for r in seed + rows}),
            "training_geometries": sum(r["frames"] for r in seed + rows),
            "held_reactions": len(held),
        }
    control = next(
        r
        for r in json.loads((root / "snapshot-learning-curve.json").read_text())
        if r["training_reactions"] == 512
    )
    if repeat == 0:
        for k in ("reaction", "midpoint", "forward", "reverse"):
            np.testing.assert_allclose(
                results["random"][k], control[k], atol=1e-7, rtol=0
            )
    kt.write_json(
        root
        / (
            "committee-selection.json"
            if repeat == 0
            else f"committee-selection-{repeat}.json"
        ),
        {
            "seed_ids": [r["rxn"] for r in seed],
            "acquired_ids": {k: sorted(v) for k, v in selections.items()},
            "metrics": results,
        },
    )
    print("Completed training seed", repeat, json.dumps(results), flush=True)
    return results


if __name__ == "__main__":
    runs = [run(i) for i in range(5)]
    metrics = ("reaction", "midpoint", "barriers", "forward", "reverse")
    means = {
        k: {m: float(np.mean([r[k][m] for r in runs])) for m in metrics}
        for k in runs[0]
    }
    kt.write_json(
        curve.ROOT / "committee-five-seeds.json", {"means": means, "seeds": runs}
    )
