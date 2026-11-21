"""Measure where the RDB7 coupled-cluster error enters a two-level potential.

Fit CCSD(T)-F12a minus archived DFT at individual stationary geometries and,
as a control, on energy differences using the same local kernel columns.
The quantum-baseline rows are retrospective. The predicted-baseline rows
use only geometry and the previously trained Korny DFT surface at evaluation.
"""

import csv
import json
import pickle
import shutil
from collections import defaultdict

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from experiments.full_public import kernel_trials as kt
from experiments.full_public.tauri_validation import compare
from experiments.full_public.trial_measured_curvature import load_basis
from experiments.full_public.trial_multiscale import baseline
from experiments.full_public.trial_spectra import evaluate_trial
from experiments.full_public.trial_validation import run as validate
from src.forcefield import ForceField

WORK = kt.WORK
RESULTS = kt.OUT / "rdb7-baseline-audit"
KINDS = ("reaction", "forward", "reverse")
RIDGES = (1e-6, 1e-4, 1e-2)
STAGE = "p53-public"


def read_pickle(name):
    with (WORK / name).open("rb") as stream:
        return pickle.load(stream)


def fit_ridges(features, labels, counts=None):
    """Remove train-only elemental offsets before regularizing kernel weights.

    A stoichiometrically balanced equation cancels all elemental offsets.
    Projecting both design and labels removes their unpenalized contribution
    exactly, even when some formula counts are linearly dependent.
    """
    if counts is not None:
        design = np.column_stack((features, labels))
        design -= counts @ np.linalg.lstsq(counts, design, rcond=None)[0]
        features, labels = design[:, :-1], design[:, -1]
    gram = features.T @ features
    rhs = features.T @ labels
    unit = gram.trace() / len(gram)
    return {
        ridge: cho_solve(cho_factor(gram + np.eye(len(gram)) * ridge * unit), rhs)
        for ridge in RIDGES
    }


def metrics(errors, kinds):
    values = {k: float(np.mean(np.abs(errors[kinds == k])) * kt.KCAL) for k in KINDS}
    values["barriers"] = (values["forward"] + values["reverse"]) / 2
    return values


def paired_intervals(stage, parent="p49-public"):
    """Resample entire TS formulas while preserving the active source mixture."""
    geoms = {}
    # Read coordinates and formulas from the same ledger as the scored equations.
    all_geom, rows, _ = kt.inputs()
    for _, src, split, case, terms, _, kind in rows:
        if split == "validation" and kind == "forward":
            geoms[src, case] = tuple(sorted(map(int, all_geom[terms[-1][0]].z)))

    def read(st):
        path = kt.OUT / f"so3lr-tauri-matched/{st}-comparison.csv"
        result = {}
        with path.open() as f:
            for row in csv.DictReader(f):
                if row["source"] not in kt.BARRIERS:
                    continue
                values = [
                    row[k + " error_kcal"]
                    for k in ("Korny", "Original SO3LR", "Tauri-M-2606", "Tauri-L-2606")
                ]
                if all(values) and np.isfinite(np.asarray(values, float)).all():
                    result[row["source"], row["id"], row["observable"]] = abs(
                        float(values[0])
                    )
        return result

    old, new = read(parent), read(stage)
    n = sum(kt.COMMON_REACTIONS.values())
    assert new.keys() == old.keys() and len(new) == n * 3
    grouped = defaultdict(lambda: defaultdict(list))
    for src, case, kind in old:
        if kind == "reaction":
            delta = np.array([new[src, case, k] - old[src, case, k] for k in KINDS])
            grouped[src][geoms[src, case]].append(delta)
    rng = np.random.default_rng(5746)
    reps = np.zeros((4000, 3))
    for groups in grouped.values():
        sizes = np.array([len(v) for v in groups.values()])
        sums = np.stack([np.sum(v, axis=0) for v in groups.values()])
        draws = rng.integers(len(sizes), size=(len(reps), len(sizes)))
        reps += sums[draws].sum(1) / sizes[draws].sum(1)[:, None] * (sizes.sum() / n)
    out = {
        k: np.quantile(reps[:, j], [0.025, 0.975]).tolist() for j, k in enumerate(KINDS)
    }
    out["barriers"] = np.quantile(reps[:, 1:].mean(1), [0.025, 0.975]).tolist()
    return out


def fit_geometry_baseline(
    bank,
    columns,
    gi,
    dft_values,
    composition,
    cc_weights,
    parent,
    matrix,
    cc,
    dft,
    kind,
    panel,
):
    """Fit a separate geometry-only DFT readout around p49's high-level PES.

    Alternate kernel centers between the DFT base and the small CC correction.
    With identical columns and regularization, the two successive linear fits
    would collapse algebraically to one ordinary fit directly on CC energies.
    The gauge-dependent elemental terms cancel in every balanced reaction.
    Both scalar changes differentiate through the existing periodic kernel.
    """
    basis, full = load_basis()
    base_cols = full[1::2]
    x_base = np.ascontiguousarray(bank[: len(parent), :][:, base_cols], dtype=float)
    # Fixed before scoring the outer panel. Match the relative ridge strength
    # used by the inner-fold stationary-geometry CC correction.
    base_weight = fit_ridges(x_base[gi], dft_values - parent[gi], composition)[1e-2]
    estimated_dft = parent + x_base @ base_weight
    del x_base
    basis.weights = np.zeros(basis.size)
    basis.weights[base_cols] = base_weight
    basis.weights[columns] = cc_weights
    new = estimated_dft + bank[: len(parent), :][:, columns] @ cc_weights
    stage = STAGE
    ff = ForceField.load(WORK / "p49-public-high.npz", dispersion="d3")
    ff.additional_bases.append(basis)
    path = WORK / f"{stage}-high.npz"
    ff.save_with_base(kt.original.BASE, path)
    deployed = ForceField.load(path, dispersion="d3")
    geoms, _, _ = kt.inputs()
    _, names = read_pickle("ids.pkl")
    for mult in (1, 2, 3):
        i = next(
            i
            for i, name in enumerate(names)
            if geoms[name].state[1] == mult and 5 < len(geoms[name].z) < 14
        )
        g = geoms[names[i]]
        energy, force = deployed.predict(g.z, g.r, *g.state)
        np.testing.assert_allclose(energy, new[i], rtol=0, atol=5e-6)
        u = np.random.default_rng(i).normal(size=g.r.shape)
        u /= np.linalg.norm(u)
        fd = (
            deployed.energy(g.z, g.r - 2e-5 * u, *g.state)
            - deployed.energy(g.z, g.r + 2e-5 * u, *g.state)
        ) / 4e-5
        np.testing.assert_allclose(np.sum(u * force), fd, atol=8e-5, rtol=0)
    s, y, source, fold, types, ids = kt.equation_data("high")
    held = fold < 0
    np.save(WORK / "paired53-high-parent--1.npy", new)
    np.savez_compressed(
        WORK / f"{stage}-high-equations.npz",
        source=source[held],
        kind=types[held],
        case=ids[held],
        error_kcal=((s @ new - y) * kt.KCAL)[held],
    )
    for suffix in ("wb97xd3.npz", "wb97xd3-equations.npz", "held-modes.npy"):
        shutil.copyfile(WORK / ("p49-public-" + suffix), WORK / (stage + "-" + suffix))
    kt.write_json(
        WORK / f"{stage}-settings.json",
        {
            "selected": stage,
            "branches": {"high": "paired53", "wb97xd3": "spin49"},
            "dft_centers": len(base_cols),
            "cc_centers": len(columns),
            "parent": "p49-public",
        },
    )
    compare(stage)
    validate(stage)
    kt.write_json(
        RESULTS / f"{stage}-minus-p49-intervals.json", paired_intervals(stage)
    )
    return {
        "geometry_dft_vs_dft": metrics(
            (matrix @ estimated_dft - dft)[panel], kind[panel]
        ),
        "geometry_dft_vs_cc": metrics(
            (matrix @ estimated_dft - cc)[panel], kind[panel]
        ),
        "geometry_dft_plus_delta_vs_cc": metrics(
            (matrix @ new - cc)[panel], kind[panel]
        ),
    }


def run():
    ids, names = read_pickle("ids.pkl")
    archived = read_pickle("rdb-dft-energies.pkl")
    observations = read_pickle("isomer-observations.pkl")["RDB7"]
    dft_copies = defaultdict(list)
    for key, energy in archived.items():
        dft_copies[ids[key]].append(energy)
    geometry_ids, delta, composition, membership = [], [], [], []
    for (elements, _), records in observations.items():
        formula = [elements.count(z) for z in (1, 6, 7, 8)]
        if sum(formula) != len(elements):
            raise ValueError(f"Unexpected RDB7 elements {elements}")
        for idx, cc, folds in records:
            copies = dft_copies[idx]
            if copies and np.ptp(copies) <= 0.001:
                geometry_ids.append(idx)
                delta.append(cc - np.mean(copies))
                composition.append(formula)
                membership.append(folds)
    gi, delta, composition = map(np.asarray, (geometry_ids, delta, composition))
    assert len(set(gi)) == len(gi) and all(-1 not in fs for fs in membership)

    # Keep names from the original archive for DFT reaction sums. The same
    # geometry can be repeated in different reaction folders with rounded labels.
    all_data = kt.equation_data("high")
    rdb = all_data[2] == "RDB7"
    matrix, cc, _, fold, kind, case = (a[rdb] for a in all_data)
    _, rows, _ = kt.inputs()
    refs = {
        (row[3], row[-1]): sum(s * archived[n] for n, s in row[4])
        for row in rows
        if row[0] == "high" and row[1] == "RDB7"
    }
    dft = np.array([refs[n, k] for n, k in zip(case, kind, strict=True)])
    residual = cc - dft
    _, columns = load_basis()
    columns = columns[::2]
    bank = np.load(WORK / "periodic-0.85-main.npy", mmap_mode="r")
    pub = np.ascontiguousarray(bank[: len(names), :][:, columns], dtype=float)
    x = matrix @ pub
    with (kt.OUT / "so3lr-tauri-matched/p49-public-comparison.csv").open() as stream:
        common = {
            (r["id"], r["observable"])
            for r in csv.DictReader(stream)
            if r["source"] == "RDB7"
        }
    panel = np.array([(n, k) in common for n, k in zip(case, kind, strict=True)])
    assert panel.sum() == len(common) == 3 * 259 and (fold[panel] == -1).all()
    assert set(kind) == set(KINDS)

    held_ids = set(all_data[0][all_data[3] < 0].indices.tolist())
    dft_data = kt.equation_data("wb97xd3")
    held_ids.update(dft_data[0][dft_data[3] < 0].indices.tolist())
    assert held_ids.isdisjoint(gi)
    tune = {"stationary": defaultdict(list), "differences": defaultdict(list)}
    counts = {}
    for h in (0, 1, -1):
        blocked = held_ids | (set(matrix[fold == h].indices) if h >= 0 else set())
        valid_geometry = np.ones(len(names), dtype=bool)
        valid_geometry[list(blocked)] = False
        shape_train = valid_geometry[gi] & np.array([h not in fs for fs in membership])
        # The incidence matrix can contain negative stoichiometries. Absolute
        # coefficients detect every blocked participant without cancellation.
        eq_train = (
            (fold >= 0)
            & (fold != h)
            & ((abs(matrix) @ (~valid_geometry).astype(float)) == 0)
        )
        weights = {
            "stationary": fit_ridges(
                pub[gi[shape_train]], delta[shape_train], composition[shape_train]
            ),
            "differences": fit_ridges(x[eq_train], residual[eq_train]),
        }
        if h >= 0:
            dev = fold == h
            for method, fits in weights.items():
                for ridge, w in fits.items():
                    err = metrics(x[dev] @ w - residual[dev], kind[dev])
                    tune[method][ridge].append((err["reaction"] + err["barriers"]) / 2)
            continue
        counts = {
            "stationary_geometries": int(shape_train.sum()),
            "equation_rows": int(eq_train.sum()),
            "equation_reactions": len(set(case[eq_train])),
            "kernel_columns": len(columns),
            "held_reactions": 259,
        }
    # Both methods and all baseline substitutions are scored on the same IDs.
    # Selecting regularization uses only inner folds of the training archive.
    selected = {
        m: min(scores, key=lambda r: np.mean(scores[r])) for m, scores in tune.items()
    }
    dft_ml = (matrix @ baseline("wb97xd3", -1, "parent", "p49-public"))[panel]
    p49 = (matrix @ baseline("high", -1, "parent", "p49-public"))[panel]
    pred = {"archived_dft": dft[panel], "korny_dft": dft_ml, "korny_high": p49}
    for method, ridge in selected.items():
        change = x[panel] @ weights[method][ridge]
        pred["archived_dft_" + method] = dft[panel] + change
        pred["korny_dft_" + method] = dft_ml + change
    scores = {m: metrics(v - cc[panel], kind[panel]) for m, v in pred.items()}
    scores_dft = metrics(dft_ml - dft[panel], kind[panel])
    summary = json.loads(
        (kt.OUT / "so3lr-tauri-matched/p49-public-summary.json").read_text()
    )
    for k, val in scores["korny_high"].items():
        np.testing.assert_allclose(
            val, summary["common"]["RDB7"]["Korny"][k], atol=1e-7
        )
    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "predictions.csv").open("w", newline="") as stream:
        out = csv.writer(stream)
        out.writerow(["case", "observable", "cc_kcal", *(m + "_kcal" for m in pred)])
        for row in zip(
            case[panel], kind[panel], cc[panel], *pred.values(), strict=True
        ):
            out.writerow([*row[:2], *(v * kt.KCAL for v in row[2:])])
    result = {
        "counts": counts,
        "selected": selected,
        "metrics_vs_cc": scores,
        "korny_dft_vs_archived_dft": scores_dft,
    }
    kt.write_json(RESULTS / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    parent = baseline("high", -1, "parent", "p49-public")
    use = gi[shape_train]
    result["geometry_only_experiment"] = fit_geometry_baseline(
        bank,
        columns,
        use,
        np.array([np.mean(dft_copies[idx]) for idx in use]),
        composition[shape_train],
        weights["stationary"][selected["stationary"]],
        parent,
        matrix,
        cc,
        dft,
        kind,
        panel,
    )
    kt.write_json(RESULTS / "summary.json", result)
    print("Geometry-only fit complete", result["geometry_only_experiment"], flush=True)
    evaluate_trial(STAGE)


if __name__ == "__main__":
    run()
