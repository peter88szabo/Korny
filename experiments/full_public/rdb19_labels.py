"""Join gas-phase IRC trajectories to the RDB19-Rad reaction/class metadata.

Retain Gibbs labels as distinct observables and keep both path directions in
one fold. The forward/reverse convention from the Gaussian logfile is local
to that calculation. A product-side label requires an endpoint assignment.
"""

import csv
import json
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.full_public.rdb19_irc import OUT, ROOT


def load():
    tables = {}
    names = {"reaction_types", "all_rxn_water_forward_noresonance"}
    with tarfile.open(ROOT / "Datasets.tar.bz2", "r|bz2") as t:
        for m in t:
            name = Path(m.name).stem
            if name not in names:
                continue
            with t.extractfile(m) as f:
                tables[name] = {
                    int(row["rxn_idx"]): row
                    for row in csv.DictReader(line.decode("utf8") for line in f)
                }
            if len(tables) == len(names):
                break
    classes = tables["reaction_types"]
    return {
        i: {
            "reaction_type": classes[i]["reaction_type"],
            "cluster": int(v["Kmeans_split_cluster_idx"]),
            "smiles": v["rxn_smi"],
            "gas_barrier_G_kcal": float(v["DeltaG_ET_gas_molar_kcal_mol"]),
            "gas_reaction_G_kcal": float(v["DeltaG_EP_gas_molar_kcal_mol"]),
        }
        for i, v in tables["all_rxn_water_forward_noresonance"].items()
    }


def report():
    metadata = load()
    OUT.mkdir(exist_ok=True)
    temp = OUT / "reaction-identities.pending"
    temp.write_text(json.dumps(metadata, indent=1))
    temp.replace(OUT / "reaction-identities.json")
    classes = Counter()
    sources = {}
    errors = []
    spins = Counter()
    seen = set()
    for path in sorted(OUT.glob("*.npz")):
        with np.load(path) as a:
            i = int(a["rxn_id"])
            m = int(a["multiplicity"])
            label = metadata[i]
            if i not in seen:
                classes[label["reaction_type"]] += 1
                seen.add(i)
            z = a["z"]
            r = a["r_A"].astype(float)
            f = a["force_eV_A"].astype(float)
            e = a["energy_eV"]
            spins[m] += len(e)
            # Independent numerical sign/unit test using the trapezoidal work
            # along consecutive accepted geometries in the same Cartesian frame.
            de = np.diff(e)
            work = -np.einsum("tij,tij->t", 0.5 * (f[1:] + f[:-1]), np.diff(r, axis=0))
            errors.extend(abs(de - work))
            sources.setdefault(i, {})[str(a["direction"])] = (r[0], e[0], z)
    matches = []
    for i, paths in sources.items():
        if set(paths) != {"forw", "rev"}:
            continue
        a, b = paths["forw"], paths["rev"]
        if not np.array_equal(a[2], b[2]):
            continue
        # Gaussian input coordinates use a fixed frame in this deposit.
        matches.append([i, float(np.max(abs(a[0] - b[0]))), abs(a[1] - b[1])])
    result = {
        "published_reactions": len(metadata),
        "published_classes": len({v["reaction_type"] for v in metadata.values()}),
        "imported_reactions": len(seen),
        "imported_logs": sum(map(len, sources.values())),
        "labels_by_multiplicity": spins,
        "classes": classes,
        "work_abs_error_eV_median_p95": np.quantile(errors, [0.5, 0.95]).tolist()
        if errors
        else [],
        "same_starting_TS_pairs": int(
            sum(v[1] < 1.0e-4 and v[2] < 1.0e-4 for v in matches)
        ),
        "pairs_checked": len(matches),
    }
    temp = OUT / "path-inventory.pending"
    temp.write_text(json.dumps(result, indent=2))
    temp.replace(OUT / "path-inventory.json")
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    report()
