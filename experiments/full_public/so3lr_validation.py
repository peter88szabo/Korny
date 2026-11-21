"""Evaluate the original SO3LR checkpoint on Korny's saved reaction IDs.

The old comparison used a different split. Reuse a cached energy only when
the geometry ID, atoms, coordinates, charge and multiplicity match the current
ledger. Reconstruct energies of detached fragments with the saved stoichiometry.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shlex
import subprocess

import numpy as np
from rdkit import Chem

from experiments.full_public.data import COMMON_REACTIONS, OUT, fingerprint, load_reaction_data
from experiments.reactivity.thermochemistry import EV_TO_KCAL
from src.forcefield import ForceField

PANELS = tuple(COMMON_REACTIONS)
WORK = OUT / "so3lr-original-matched"
OLD = OUT.parent / "comparison"
# Element domain of the original (pre-Tauri) SO3LR checkpoint.
SUPPORTED = {1, 6, 7, 8, 9, 15, 16, 17}
PT = Chem.GetPeriodicTable()


def read_results(path, keys, geometries):
    """Verify frame identity on readback, including the input manifest's state."""
    values = {}
    with path.open() as stream:
        for key in keys:
            count = int(next(stream))
            info = dict(w.split("=", 1) for w in shlex.split(next(stream)) if "=" in w)
            atoms = [next(stream).split() for _ in range(count)]
            entry = geometries[key]
            zs = [PT.GetAtomicNumber(a[0]) for a in atoms]
            if zs != list(entry["z"]) or int(info["charge"]) != entry["q"]:
                raise ValueError(f"Element or charge mismatch for {key}")
            if "multiplicity" in info and int(info["multiplicity"]) != entry["m"]:
                raise ValueError(f"Multiplicity mismatch for {key}")
            np.testing.assert_allclose([list(map(float, a[1:4])) for a in atoms],
                                       entry["r"], rtol=0, atol=1e-7)
            value = float(info["energy_so3lr"])
            if not np.isfinite(value):
                raise ValueError(f"Nonfinite energy for {key}")
            values[key] = value
        if stream.read().strip():
            raise ValueError("Extra frames in " + str(path))
    return values


def write_inputs(path, keys, geoms):
    """Each detached fragment retains its own charge, multiplicity and reference coordinates."""
    with path.open("w") as f:
        for k in keys:
            g = geoms[k]
            f.write(f"{len(g.z)}\nProperties=species:S:1:pos:R:3 charge={g.state[0]} multiplicity={g.state[1]}\n")
            for z, pos in zip(g.z, g.r, strict=True):
                f.write(PT.GetElementSymbol(int(z)) + " " + " ".join(f"{x:.12f}" for x in pos) + "\n")


def run(executable):
    WORK.mkdir(exist_ok=True)
    geoms, all_rows, _ = load_reaction_data(preserve_geometry_index=True)
    original = json.loads((OUT / "joint-reaction-validation.json").read_text())["validation"]
    meta = json.loads((OLD / "metadata.json").read_text())
    old = read_results(OLD / "so3lr.extxyz", meta["keys"], meta["geometries"])
    # Korny's fit scored once per fingerprint. Keep its representative even when
    # a differently named fragment in another reaction has the same fingerprint.
    representatives, by_key = {}, {}
    for k, g in geoms.items():
        by_key[k] = representatives.setdefault((fingerprint(g.z, g.r), g.state), k)
    kn = {t: ForceField.load(OUT / f"joint-{t}-reaction.npz", dispersion="d3")
          for t in ("high", "wb97xd3")}
    korny_energies = {}
    summary = {}
    for panel in PANELS:
        rows = [r for r in all_rows if r[1] == panel and r[2] == "validation"]
        keys = sorted({k for row in rows for k, _ in row[4]})
        fresh, values, reasons = [], {}, {}
        for k in keys:
            g = geoms[k]
            unknown = sorted(set(g.z) - SUPPORTED)
            if unknown:
                reasons[k] = "unsupported " + ", ".join(PT.GetElementSymbol(int(a)) for a in unknown)
                continue
            prior = meta["geometries"].get(k)
            if (prior and (prior["q"], prior["m"]) == g.state and np.array_equal(prior["z"], g.z)
                    and np.allclose(prior["r"], g.r, atol=1e-10, rtol=0)):
                values[k] = old[k]
            else:
                fresh.append(k)
        cached = len(values)
        print(f"{panel}: {len(keys)} geometry IDs, {cached} cached, {len(fresh)} fresh, {len(reasons)} unsupported", flush=True)
        datafile = WORK / f"{panel}-input.extxyz"
        outfile = WORK / f"{panel}-output.extxyz"
        manifest = WORK / f"{panel}-input-ids.json"
        encoded = json.dumps(fresh, indent=2) + "\n"
        if fresh:
            if outfile.exists() and (not manifest.exists() or manifest.read_text() != encoded):
                raise ValueError("Cached SO3LR output has a different input manifest")
            manifest.write_text(encoded)
            write_inputs(datafile, fresh, geoms)
            # Match the original cache's checkpoint, units, cutoff, and precision.
            if not outfile.exists():
                tmp = WORK / f"{panel}-pending.extxyz"
                with (WORK / f"{panel}.log").open("w") as log:
                    subprocess.run([executable,"eval","--model","so3lr","--targets","energy",
                                    "--batch-size","8","--precision","float64","--lr-cutoff","12",
                                    "--datafile",str(datafile),"--save-to",str(tmp),"--overwrite"],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                tmp.replace(outfile)
            records = {k: dict(z=geoms[k].z, r=geoms[k].r, q=geoms[k].state[0], m=geoms[k].state[1]) for k in fresh}
            values.update(read_results(outfile, fresh, records))
        full, paired = defaultdict(list), defaultdict(list)
        with (WORK / f"{panel}-equations.csv").open("w") as f:
            w = csv.writer(f)
            w.writerow(["id","observable","reference_kcal","korny_kcal","so3lr_kcal","korny_error_kcal","so3lr_error_kcal","missing"])
            for target, _, _, name, terms, ref, kind in rows:
                for k, c in terms:
                    ident = (target, by_key[k])
                    if ident not in korny_energies:
                        g = geoms[by_key[k]]
                        korny_energies[ident] = kn[target].energy(g.z, g.r, *g.state)
                ke = sum(c*korny_energies[target,by_key[k]] for k,c in terms)*EV_TO_KCAL
                ref *= EV_TO_KCAL
                miss = {k: reasons[k] for k, c in terms if k not in values}
                se = None if miss else sum(c*values[k] for k,c in terms)*EV_TO_KCAL
                full[kind].append(ke-ref)
                if se is not None:
                    paired[kind].append([ke-ref,se-ref])
                w.writerow([name,kind,ref,ke,se,ke-ref,None if se is None else se-ref,json.dumps(miss) if miss else ""])
        for kind in ("forward","reverse","reaction"):
            expected = original[rows[0][0]][panel + "/" + kind]
            assert len(full[kind]) == expected["n"]
            np.testing.assert_allclose(np.abs(full[kind]).mean(), expected["mae"], rtol=0, atol=1e-6)
        for values_by_type in (full,paired):
            values_by_type["both_barriers"] = values_by_type["forward"] + values_by_type["reverse"]
        metrics = {}
        for kind, all_errors in full.items():
            pair = np.array(paired[kind])
            metrics[kind] = dict(total=len(all_errors), scored=len(pair),
                                 korny_full=float(np.abs(all_errors).mean()),
                                 korny_matched=float(np.abs(pair[:,0]).mean()) if len(pair) else None,
                                 so3lr_matched=float(np.abs(pair[:,1]).mean()) if len(pair) else None,
                                 so3lr_rmse=float(np.sqrt(np.mean(pair[:,1]**2))) if len(pair) else None)
        summary[panel] = dict(cached=cached, fresh=len(fresh), unsupported=reasons, metrics=metrics)
        (WORK / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(f"{panel} done: {metrics}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", default="so3lr", help="SO3LR CLI in its own environment")
    run(parser.parse_args().executable)
