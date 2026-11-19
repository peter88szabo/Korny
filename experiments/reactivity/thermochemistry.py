"""Score atom-balanced reactions and genuine stationary-point barriers at fixed geometries.

The independent GMTKN55 electronic reference values (kcal/mol) and stoichiometry
are read from Behemoth's existing benchmark manifests. This module does no QC.
The small closed-shell panel is fixed by identifiers before comparing variants.
"""
import argparse
from collections import Counter
import json
import re
from pathlib import Path

import numpy as np

from experiments.des15k.benchmark import ELEMENTS
from experiments.des15k.run import OUT
from src.forcefield import ForceField

BENCH = Path("/home/peter/DevCode/Behemoth/validation/ynk_kernel_benchmark")
# First-row reaction panel plus small sulfur hydrides and SO2. The energy
# difference for HCN -> HNC follows from the two published BH76 barriers.
REACTIONS = (1, 4, 7, 9)
BARRIERS = ("bh76_hcn", "bh76_hnc", "inv24_h2o", "inv24_h2s", "inv24_so2")
EV_TO_KCAL = 23.060548867


def score(errors):
    """Use the reaction (or directional barrier) as the unit of observation."""
    a = np.asarray(errors)
    return dict(n=len(a), mae_kcal_mol=float(np.abs(a).mean()),
                rmsd_kcal_mol=float(np.sqrt(np.mean(a*a))))


def panel(root=BENCH):
    """Return (name, stoichiometric terms, reference, category) for the fixed panel."""
    out = []
    def entries(folder):
        return json.loads((root/folder/"manifest.json").read_text())["reactions"]
    for i in REACTIONS:
        r = entries("gmtkn55_reactions")[i]
        terms = [(c, root/"pbe0_reference/tz"/("g2rc_"+n+".xyz"))
                 for c,n in zip(r["coeffs"],r["names"], strict=True)]
        out.append((f"G2RC subset index {i}", terms, r["ref"], "reaction"))
    bh = {}
    for r in entries("gmtkn55_barriers"):
        if r["stems"][0] not in BARRIERS:
            continue
        paths = [root/"gmtkn55_barriers/tz"/(n+".xyz") for n in r["stems"]]
        out.append((r["stems"][0], list(zip(r["coeffs"],paths,strict=True)), r["ref"], "barrier"))
        bh[r["stems"][0]] = (paths[0],r["ref"])
    hcn, bf = bh["bh76_hcn"]
    hnc, br = bh["bh76_hnc"]
    out.append(("HCN -> HNC", [(-1,hcn),(1,hnc)], bf-br, "reaction"))
    return out


def archived_pbe0(terms):
    """Read an existing Behemoth SCF result when every reaction term has one.

    The local G2RC archive includes cc-pVTZ PBE0 values followed by correlated
    energies. Use the SCF line explicitly. Unsupported terms have no entry.
    """
    total = 0.
    for coefficient, geometry in terms:
        logfile = geometry.with_name(geometry.stem+"_pbe0.out")
        if not logfile.exists():
            return None
        match = re.search(r"Total energy \(all terms\):\s+([-+\d.]+) Hartree", logfile.read_text())
        if not match:
            raise ValueError(f"Missing PBE0 energy in {logfile}")
        total += coefficient*float(match[1])
    from src.units import HARTREE_EV
    return total*HARTREE_EV*EV_TO_KCAL


def assess(model, root=BENCH):
    """Evaluate each molecule once; verify nuclear balance before subtracting energies."""
    energies, formulas, rows = {}, {}, []
    for name, terms, reference, category in panel(root):
        total, balance = 0., Counter()
        for coefficient, path in terms:
            if path not in energies:
                xyz = [s.split() for s in path.read_text().splitlines()[2:] if s.strip()]
                z = np.array([ELEMENTS[s[0]] for s in xyz])
                r = np.array([s[1:4] for s in xyz], dtype=float)
                energies[path] = model.predict(z,r,0)[0]
                formulas[path] = Counter(z)
            total += coefficient*energies[path]
            for z, count in formulas[path].items():
                balance[z] += coefficient*count
        if any(balance.values()):
            raise ValueError(f"Unbalanced reaction {name}")
        predicted = total*EV_TO_KCAL
        row = dict(name=name, category=category, reference=reference, prediction=predicted,
                   error=predicted-reference)
        same_functional = archived_pbe0(terms)
        if same_functional is not None:
            row["archived_PBE0_cc_pVTZ"] = same_functional
            row["error_vs_archived_PBE0"] = predicted-same_functional
        rows.append(row)
    return {**{k:score([r["error"] for r in rows if r["category"] == k])
               for k in ("reaction","barrier")}, "rows":rows}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--benchmark-root", type=Path, default=BENCH)
    a = p.parse_args()
    checkpoint = a.model or OUT/json.loads((OUT/"results.json").read_text())["checkpoint"]
    volumes = OUT/json.loads((OUT/"volumes.json").read_text())["checkpoint"]
    results = {s:assess(ForceField.load(checkpoint, dispersion=s, volumes=volumes),a.benchmark_root)
               for s in ("off","ts","d3")}
    a.out.write_text(json.dumps(results, indent=2)+"\n")
    for s, v in results.items():
        print(s, v["reaction"], v["barrier"])
