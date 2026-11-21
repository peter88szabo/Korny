"""Grouped reaction fits on the saved public split, starting with a fixed basis.

The midpoint barrier and endpoint gap are two independent differences of one
potential. Searching their weights never introduces separate barrier predictors.
The bare PBE0 checkpoint remains the common reference for both reaction branches.
"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import pickle

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from experiments.full_public import fit_reactions as original
from experiments.full_public.data import COMMON_REACTIONS, OUT, fingerprint, load_reaction_data
from experiments.reactivity.thermochemistry import EV_TO_KCAL as KCAL
from src.forcefield import ForceField

WORK = OUT / "kernel-trials"
TARGETS = ("high", "wb97xd3")
BARRIERS = tuple(COMMON_REACTIONS)


def write_json(file, data):
    temp = file.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(file)


def inputs():
    return load_reaction_data(preserve_geometry_index=True)


def prepare():
    """Cache exactly the same feature rows, geometry representatives and D3 as the deployed fit."""
    WORK.mkdir(exist_ok=True)
    geoms, _, _ = inputs()
    ids, lookup, names = {}, {}, []
    for k, g in geoms.items():
        signature = fingerprint(g.z, g.r), g.state
        if signature not in lookup:
            lookup[signature] = len(names)
            names.append(k)
        ids[k] = lookup[signature]
    original.surface = ForceField.load(OUT / "joint-pbe0-reaction.npz")
    original.unique, original.geoms = names, geoms
    shape = (len(names), len(original.surface.weights))
    x = np.lib.format.open_memmap(
        WORK / "base-x.npy", mode="w+", dtype=float, shape=shape
    )
    base = np.empty(len(names))
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context("fork")) as pool:
        for i, feat, e, d in pool.map(
            original.precompute, range(len(names)), chunksize=32, buffersize=14
        ):
            x[i], base[i] = feat, e + d
    # Include this fixed PBE0 readout and D3 exactly once in each fit's prior.
    base += x @ original.surface.weights
    x.flush()
    np.save(WORK / "prior.npy", base)
    with (WORK / "ids.pkl").open("wb") as f:
        pickle.dump((ids, names), f)
    print("Prepared fixed PBE0+D3 energies and design", shape, flush=True)


def equation_data(target):
    geoms, rows, _ = inputs()
    with (WORK / "ids.pkl").open("rb") as f:
        ids, names = pickle.load(f)
    selected = [r for r in rows if r[0] == target]
    nn, jj, values = [0], [], []
    sources, splits, kinds, refs, cases = [], [], [], [], []
    ts = {(r[1], r[3]): geoms[r[4][-1][0]].z for r in selected if r[-1] == "forward"}
    for _, src, split, name, terms, ref, kind in selected:
        if (src, name) in ts:
            comp = tuple(sorted(map(int, ts[src, name])))
        else:
            z = geoms[max(terms, key=lambda t: len(geoms[t[0]].z))[0]].z
            comp = tuple(sorted(int(i) for i in z if i != 1))
        # The salt and buckets are fixed independently of references and predictions.
        h = (
            int.from_bytes(
                hashlib.sha256(("kt1/" + str(comp)).encode()).digest()[:4], "big"
            )
            % 8
        )
        splits.append(-1 if split == "validation" else h)
        for k, c in terms:
            jj.append(ids[k])
            values.append(c)
        nn.append(len(jj))
        sources.append(src)
        kinds.append(kind)
        refs.append(ref)
        cases.append(name)
    S = csr_matrix((values, jj, nn), shape=(len(selected), len(names)))
    return (
        S,
        np.array(refs),
        np.array(sources),
        np.array(splits),
        np.array(kinds),
        np.array(cases),
    )


def stats(d, resid, mask):
    """Report each original directional error as well as their pooled MAE."""
    _, _, src, _, kind, _ = d
    out = {}
    for s in np.unique(src[mask]):
        out[s] = {}
        for k in sorted(set(kind[mask & (src == s)])):
            v = resid[mask & (src == s) & (kind == k)] * KCAL
            out[s][k] = [len(v), float(np.abs(v).mean())]
        if "forward" in out[s]:
            a, b = out[s]["forward"], out[s]["reverse"]
            out[s]["barriers"] = [
                a[0] + b[0],
                (a[1] * a[0] + b[1] * b[0]) / (a[0] + b[0]),
            ]
    return out


def barrier_blocks(kind, case, use, features, y, row_weights=None):
    """Yield aligned midpoint and endpoint-gap rows for one training reaction source."""
    labels = ("forward", "reverse", "reaction")
    ids = [np.flatnonzero(use & (kind == k)) for k in labels]
    f, r, g = (ix[np.argsort(case[ix])] for ix in ids)
    if not (np.array_equal(case[f], case[r]) and np.array_equal(case[f], case[g])):
        raise ValueError("Forward, reverse and reaction labels must share the same IDs")
    for lo in range(0, len(f), 384):
        ff, rr, gg = (ix[lo : lo + 384] for ix in (f, r, g))
        mid, mid_y = (features[ff] + features[rr]) * 0.5, (y[ff] + y[rr]) * 0.5
        # The reference gap is sometimes rounded independently of the barriers.
        gap, gap_y = features[gg], (2 * y[gg] + y[ff] - y[rr]) / 3
        if row_weights is not None:
            root = np.sqrt(row_weights[gg])
            mid, mid_y = mid * root[:, None], mid_y * root
            gap, gap_y = gap * root[:, None], gap_y * root
        yield "mid", mid, mid_y
        yield "gap", gap, gap_y


def grouped_moments(d, features, y, mask, row_weights=None):
    """Return exact sufficient statistics for independent midpoint, gap and other rows."""
    _, _, src, _, kind, case = d
    dim = features.shape[1]
    pieces = {k: np.zeros((dim + 1, dim + 1)) for k in ("mid", "gap", "other")}
    for s in np.unique(src[mask]):
        use = mask & (src == s)
        # Source balancing matches the old full-data objective on refit.
        w = min(1.0, 4500 / use.sum())

        def add(which, mat, ys):
            block = np.column_stack((mat, ys))
            pieces[which] += w * (block.T @ block)

        if s not in BARRIERS:
            ix = np.flatnonzero(use)
            for a in range(0, len(ix), 384):
                sel = ix[a : a + 384]
                add("other", features[sel], y[sel])
            continue
        for label, matrix, target in barrier_blocks(kind, case, use, features, y, row_weights):
            add(label, matrix, target)
    return pieces


def solve(pieces, anchor, gap, mid, scale, ridge):
    a = pieces["other"] + gap * pieces["gap"] + mid * pieces["mid"]
    a[:-1, :-1] += scale * anchor
    a.flat[:: a.shape[0] + 1] += ridge
    return cho_solve(
        cho_factor(a[:-1, :-1], check_finite=False), a[:-1, -1], check_finite=False
    )


def score_value(table):
    # Macro-average reaction and directional barrier errors across development
    # sources, so the large number of RDB7 channels cannot hide BH9 failures.
    return float(
        np.mean(
            [
                (v["reaction"][1] + v["barriers"][1]) * 0.5
                for name, v in table.items()
                if name in BARRIERS
            ]
        )
    )


def first():
    if not (WORK / "ids.pkl").exists():
        prepare()
    model = ForceField.load(OUT / "joint-pbe0-reaction.npz")
    x = np.load(WORK / "base-x.npy", mmap_mode="r")
    base = np.load(WORK / "prior.npy")
    gram = np.load(OUT / "joint-gram.npy", mmap_mode="r")[:-1, :-1]
    lock = original.locks(
        model, json.loads((OUT / "stationary.json").read_text())["training"]
    )
    locks = lock.T @ lock
    parent = model.weights.copy()
    options = [
        (a, g, m)
        for a in (0.003, 0.03, 0.15)
        for g, m in ((1.5, 2.0), (1.5, 6.0), (4.5, 2.0))
    ]
    results = {}
    for target in TARGETS:
        d = equation_data(target)
        S, ref, src, folds, _, _ = d
        design = S @ x
        y = ref - S @ base
        # Choose settings on two disjoint grouped inner holdouts of the high-level
        # training set. Transfer the chosen settings to the radical branch, whose
        # saved derivative Gram includes all Grambow training structures.
        if target == "high":
            trials = defaultdict(list)
            for h in (0, 1):
                mask = (folds >= 0) & (folds != h)
                pieces = grouped_moments(d, design, y, mask)
                pieces["other"][:-1, :-1] += locks
                pieces["other"][:-1, -1] -= locks @ parent
                for a, g, m in options:
                    coef = solve(pieces, gram, g, m, a, 0.08)
                    pred = design @ coef - y
                    t = stats(d, pred, folds == h)
                    trials[a, g, m].append(score_value(t))
                    np.save(WORK / f"inner-{h}-{a}-{g}-{m}.npy", coef)
                print("Priority 1 inner fold", h, "complete", flush=True)
                del pieces
            best = min(options, key=lambda k: np.mean(trials[k]))
            write_json(
                WORK / "p1-search.json",
                dict(best=best, trials=[(k, v) for k, v in trials.items()]),
            )
        a, g, m = best
        pieces = grouped_moments(d, design, y, folds >= 0)
        pieces["other"][:-1, :-1] += locks
        pieces["other"][:-1, -1] -= locks @ parent
        if target == "wb97xd3":
            hd = np.load(OUT / "radical-derivative-gram.npy", mmap_mode="r")
            pieces["other"][:-1, :-1] += hd[:-1, :-1]
            pieces["other"][:-1, -1] += hd[:-1, -1] - hd[:-1, :-1] @ parent
        control = solve(pieces, gram, 1.5, 2.0, 0.15, 0.08)
        previous = ForceField.load(OUT / f"joint-{target}-reaction.npz")
        np.testing.assert_allclose(
            design @ control, design @ (previous.weights - parent), atol=2e-5, rtol=1e-7
        )
        coef = solve(pieces, gram, g, m, a, 0.08)
        model.weights = parent + coef
        model.prediction_target = previous.prediction_target
        model.save_with_base(original.BASE, WORK / f"p1-{target}.npz")
        results[target] = stats(d, design @ coef - y, folds == -1)
        np.save(WORK / f"p1-energies-{target}.npy", base + x @ coef)
        print("Priority 1 full validation", target, results[target], flush=True)
        del pieces, design
    write_json(WORK / "p1.json", results)


if __name__ == "__main__":
    with threadpool_limits(limits=4, user_api="blas"):
        first()
