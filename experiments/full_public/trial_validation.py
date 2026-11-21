"""Compare the exact public reaction IDs and probe the radical saddle curvature.

Models see all their own supported elements. Matched SO3LR results include only
IDs with a saved prediction. The bootstrap resamples whole transition-state
formulas within each source while preserving the original source proportions.
"""

import csv
import json
import pickle
from collections import defaultdict

import numpy as np

from experiments.full_public import kernel_trials as kt
from experiments.reactivity.frequency_validation import TO_CM1
from src.forcefield import ForceField


PANELS = kt.BARRIERS
KINDS = ("reaction", "forward", "reverse")


def errors(key):
    result = {}
    for target in kt.TARGETS:
        with np.load(kt.WORK / f"{key}-{target}-equations.npz") as f:
            for src, name, kind, e in zip(
                f["source"], f["case"], f["kind"], f["error_kcal"]
            ):
                if src not in PANELS:
                    continue
                idx = (str(src), str(name), str(kind))
                assert idx not in result
                result[idx] = float(e)
    return result


def ablation(formulas):
    template = "base-w1.65-a0.65-n128-four1"
    public, both = errors(template + "-public"), errors(template)
    assert public.keys() == both.keys()
    rng = np.random.default_rng(731)
    bins = defaultdict(lambda: defaultdict(list))
    for src, name, kind in public:
        if kind != "reaction":
            continue
        delta = np.array(
            [abs(both[src, name, k]) - abs(public[src, name, k]) for k in KINDS]
        )
        bins[src][formulas[src, name]].append(delta)
    pooled = np.zeros((4000, 3))
    n = sum(len(v) for groups in bins.values() for v in groups.values())
    for src, groups in bins.items():
        sums = np.array([np.sum(group, axis=0) for group in groups.values()])
        counts = np.array([len(group) for group in groups.values()])
        picks = rng.integers(0, len(sums), size=(4000, len(sums)))
        pooled += (
            sums[picks].sum(axis=1)
            / counts[picks].sum(axis=1)[:, None]
            * counts.sum()
            / n
        )
    return dict(
        formulas={s: len(v) for s, v in bins.items()},
        percentiles={
            "reaction": np.percentile(pooled[:, 0], [2.5, 97.5]).tolist(),
            "barriers": np.percentile(pooled[:, 1:].mean(axis=1), [2.5, 97.5]).tolist(),
        },
    )


def radical_energies(stage):
    saved = kt.WORK / f"{stage}-held-modes.npy"
    if saved.exists():
        return np.load(saved)
    cfg = json.loads((kt.WORK / f"{stage}-settings.json").read_text())
    if "components" in cfg:
        return sum(
            w * radical_energies(k) for k, w in zip(cfg["components"], cfg["weights"])
        )
    model = ForceField.load(kt.WORK / f"{stage}-wb97xd3.npz")
    b = model.reactive_basis
    assert b.spec["nr"] == 2
    width = len(b.centers[1])
    cols = np.concatenate([np.arange(256 * i, 256 * i + width) for i in range(4)])
    ready = json.loads((kt.WORK / "base-aux-ready.json").read_text())
    start = ready["private"]
    meta = np.load(kt.WORK / "rad-meta.npy")
    held = np.flatnonzero(meta[:, 0] == -1)
    indices = (start + 7 * held[:, None] + np.arange(7)).ravel()
    path = kt.WORK / f"base-w{b.length:g}-a{b.angular:g}-aux-bank2-features.npy"
    parts = [np.load(path, mmap_mode="r")[np.ix_(indices, cols)]]
    if b.four:
        parts.append(np.load(kt.WORK / "four-aux.npy", mmap_mode="r")[indices])
    if b.bond:
        suffix = "-angle" if b.bond.angular else ""
        # This stage uses a single-scale (length 1) bond kernel.
        parts.append(
            np.load(kt.WORK / f"bond-{b.bond.width}-1{suffix}-aux.npy", mmap_mode="r")[
                indices
            ]
        )
    e = (np.column_stack(parts) @ b.weights).reshape(-1, 7) + np.load(
        kt.WORK / "rad-prior.npy"
    )[held, :, 2]
    return e


def radical_modes(stage):
    meta = np.load(kt.WORK / "rad-meta.npy")
    held = np.flatnonzero(meta[:, 0] == -1)
    e = radical_energies(stage)
    curv = np.column_stack(
        [(e[:, i] + e[:, i + 1] - 2 * e[:, 0]) / 0.004**2 for i in (3, 5)]
    )
    ref, mass = meta[held, 2:4], meta[held, 4:6]

    def freq(x):
        return TO_CM1 * np.sign(x) * np.sqrt(np.abs(x) / mass)

    err = np.abs(freq(curv) - freq(ref))
    with (kt.WORK / "radical-modes.pkl").open("rb") as f:
        records = pickle.load(f)
    saddle = np.array([records[i][1][0].split("/")[-1].startswith("ts") for i in held])
    assert saddle.any() and not saddle.all()
    return {
        s: dict(
            n=int(mask.sum()),
            mae_cm1=err[mask].mean(axis=0).tolist(),
            sign_agreement=(np.sign(curv[mask]) == np.sign(ref[mask]))
            .mean(axis=0)
            .tolist(),
        )
        for s, mask in (("endpoint", ~saddle), ("saddle", saddle))
    }


def run(stage):
    candidate = json.loads((kt.WORK / f"{stage}-settings.json").read_text())["selected"]
    pred = errors(candidate)
    gs, rows, _ = kt.inputs()
    formulas = {
        (src, n): tuple(sorted(gs[terms[-1][0]].z))
        for _, src, role, n, terms, _, k in rows
        if k == "forward" and role == "validation"
    }
    summary = {}
    totals = []
    with (kt.WORK / f"{stage}-matched-comparison.csv").open("w") as out:
        w = csv.writer(out)
        w.writerow(
            (
                "source",
                "id",
                "observable",
                "reference_kcal",
                "korny_kcal",
                "so3lr_kcal",
                "korny_error_kcal",
                "so3lr_error_kcal",
            )
        )
        seen = set()
        for s in PANELS:
            bins = defaultdict(list)
            with (kt.OUT / "so3lr-original-matched" / f"{s}-equations.csv").open() as f:
                for row in csv.DictReader(f):
                    key = s, row["id"], row["observable"]
                    e = pred[key]
                    assert key not in seen
                    seen.add(key)
                    ref = float(row["reference_kcal"])
                    w.writerow(
                        (
                            s,
                            row["id"],
                            key[2],
                            ref,
                            ref + e,
                            row["so3lr_kcal"],
                            e,
                            row["so3lr_error_kcal"],
                        )
                    )
                    if not row["so3lr_error_kcal"]:
                        continue
                    bins[key[2]].append([abs(e), abs(float(row["so3lr_error_kcal"]))])
            bins["barriers"] = bins["forward"] + bins["reverse"]
            summary[s] = {k: np.mean(v, axis=0).tolist() for k, v in bins.items()}
            totals.append(bins)
        assert seen == pred.keys()
    summary["pooled"] = {
        k: np.mean(np.vstack([t[k] for t in totals]), axis=0).tolist()
        for k in (*KINDS, "barriers")
    }
    kt.write_json(
        kt.WORK / f"{stage}-diagnostics.json",
        dict(
            matched=summary,
            isoprene_ablation=ablation(formulas),
            radical=radical_modes(stage),
        ),
    )


if __name__ == "__main__":
    import sys

    run(sys.argv[1])
