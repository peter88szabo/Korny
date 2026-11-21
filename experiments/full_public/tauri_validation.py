"""Evaluate both published SO3LR Tauri checkpoints on the exact Korny reaction IDs.

Keep per-geometry total charge and multiplicity, use the same 12 Å cutoff as the
original SO3LR comparison, and reconstruct endpoints with their full stoichiometry.
Older predictions are eligible only for the same named state and coordinates.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from experiments.full_public import kernel_trials as kt
from experiments.full_public import so3lr_validation as original

MODELS = ("so3lr-tauri-m-2606", "so3lr-tauri-l-2606")
ROOT = kt.OUT / "so3lr-tauri-matched"


def evaluate(model, panel, rows, geoms, meta, exe):
    work = ROOT / model
    work.mkdir(parents=True, exist_ok=True)
    old = original.read_results(
        original.OLD / (model + ".extxyz"), meta["keys"], meta["geometries"]
    )
    keys = sorted({k for row in rows for k, _ in row[4]})
    values = {}
    fresh = []
    records = {}
    for k in keys:
        g = geoms[k]
        q, m = g.state
        prior = meta["geometries"].get(k)
        if (
            prior
            and (q, m) == (prior["q"], prior["m"])
            and np.array_equal(g.z, prior["z"])
            and np.allclose(g.r, prior["r"], rtol=0, atol=1e-10)
        ):
            values[k] = old[k]
        else:
            fresh.append(k)
            records[k] = dict(z=g.z, r=g.r, q=q, m=m)
    if fresh:
        input_file = work / f"{panel}-input.extxyz"
        original.write_inputs(input_file, fresh, geoms)
        signature = dict(
            ids=fresh,
            sha256=hashlib.sha256(input_file.read_bytes()).hexdigest(),
            model=model,
            cutoff_A=12,
            precision="float64",
        )
        manifest = work / f"{panel}-manifest.json"
        output = work / f"{panel}-output.extxyz"
        if output.exists() and (
            not manifest.exists() or json.loads(manifest.read_text()) != signature
        ):
            raise ValueError(f"Stale model or structure manifest for {panel}")
        kt.write_json(manifest, signature)
        if not output.exists():
            temp = work / f"{panel}-pending.extxyz"
            with (work / f"{panel}.log").open("w") as log:
                subprocess.run(
                    [
                        str(exe),
                        "eval",
                        "--model",
                        model,
                        "--targets",
                        "energy",
                        "--batch-size",
                        "8",
                        "--precision",
                        "float64",
                        "--lr-cutoff",
                        "12",
                        "--datafile",
                        str(input_file),
                        "--save-to",
                        str(temp),
                        "--overwrite",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            values.update(original.read_results(temp, fresh, records))
            temp.replace(output)
        else:
            values.update(original.read_results(output, fresh, records))
    errors = {k: [] for k in ("reaction", "forward", "reverse")}
    with (work / f"{panel}-equations.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(
            ("id", "observable", "reference_kcal", "prediction_kcal", "error_kcal")
        )
        for _, _, _, name, terms, ref, k in rows:
            pred = sum(c * values[key] for key, c in terms) * kt.KCAL
            error = pred - ref * kt.KCAL
            writer.writerow((name, k, ref * kt.KCAL, pred, error))
            errors[k].append(error)
    errors["barriers"] = errors["forward"] + errors["reverse"]
    result = dict(
        n=len(errors["reaction"]),
        cached=len(keys) - len(fresh),
        fresh=len(fresh),
        mae={k: float(np.abs(x).mean()) for k, x in errors.items()},
    )
    kt.write_json(work / f"{panel}.json", result)
    print(model, panel, result, flush=True)


def run(executable):
    geoms, rows, _ = kt.inputs()
    meta = json.loads((original.OLD / "metadata.json").read_text())
    for p in original.PANELS:
        panelrows = [r for r in rows if r[1] == p and r[2] == "validation"]
        for m in MODELS:
            evaluate(m, p, panelrows, geoms, meta, executable)


def compare(stage):
    """Join models by source, reaction ID and observable, preserving unsupported original rows."""
    from experiments.full_public.trial_validation import errors

    selected = json.loads((kt.WORK / f"{stage}-settings.json").read_text())["selected"]
    korny = errors(selected)
    entries = []
    for panel in original.PANELS:

        def index(path):
            with path.open() as f:
                return {(r["id"], r["observable"]): r for r in csv.DictReader(f)}

        archive = index(original.WORK / f"{panel}-equations.csv")
        tauri = [index(ROOT / m / f"{panel}-equations.csv") for m in MODELS]
        assert all(d.keys() == archive.keys() for d in tauri)
        for key, ref in archive.items():
            expected = float(ref["reference_kcal"])
            np.testing.assert_allclose(
                [float(d[key]["reference_kcal"]) for d in tauri],
                expected,
                rtol=0,
                atol=1e-7,
            )
            val = ref["so3lr_error_kcal"]
            entries.append(
                (
                    panel,
                    *key,
                    expected,
                    korny[(panel,) + key],
                    float(val) if val else None,
                    *[float(d[key]["error_kcal"]) for d in tauri],
                )
            )
    assert len(entries) == len(korny)
    labels = ("Korny", "Original SO3LR", "Tauri-M-2606", "Tauri-L-2606")
    with (ROOT / f"{stage}-comparison.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "source",
                "id",
                "observable",
                "reference_kcal",
                *[k + " error_kcal" for k in labels],
            ]
        )
        w.writerows(entries)
    result = {}
    for scope in ("common", "native"):
        result[scope] = {}
        for panel in (*original.PANELS, "priority", "all"):
            out = {}
            for idx, name in enumerate(labels, 4):
                rows = [
                    r
                    for r in entries
                    if (
                        panel in ("all", "priority")
                        or r[0] == panel
                    )
                    and (scope == "native" or r[5] is not None)
                    and r[idx] is not None
                ]
                stats = {
                    k: [abs(r[idx]) for r in rows if r[2] == k]
                    for k in ("reaction", "forward", "reverse")
                }
                stats["barriers"] = stats["forward"] + stats["reverse"]
                out[name] = dict(
                    n=len(stats["reaction"]),
                    **{k: float(np.mean(v)) for k, v in stats.items()},
                )
            result[scope][panel] = out
    kt.write_json(ROOT / f"{stage}-summary.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--executable", default="so3lr", type=Path)
    p.add_argument(
        "--compare",
        metavar="STAGE",
        help="Score saved predictions with this Korny checkpoint",
    )
    args = p.parse_args()
    if args.compare:
        compare(args.compare)
    else:
        run(args.executable)
