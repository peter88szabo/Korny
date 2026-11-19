"""Validate torsional energy ranges against all 500 downloaded PBE0 torsion scans.

Each fragment has 24 constrained geometries. For each method, calculate its
maximum minus its minimum across that same grid. These barriers measure a
constrained torsion profile; chemical transition states have a separate panel.
"""
import argparse
import csv
from pathlib import Path
import json
import h5py
import numpy as np

from experiments.des15k.run import DATA
from .thermochemistry import EV_TO_KCAL, score
from src.forcefield import ForceField


def evaluate(checkpoint, out):
    model = ForceField.load(checkpoint)
    profiles = {}
    with h5py.File(DATA/"TorsionNet500.hdf5") as f:
        for name,g in f.items():
            fragment, index = name.rsplit("_conf_",1)
            data = g[name]
            predicted = model.energy(data["atNUM"][:],data["atXYZ"][:],round(data["hCHG"][:].sum()))
            profiles.setdefault(fragment,[]).append((int(index),float(data["ePBE0"][0]),predicted))
    if len(profiles) != 500 or any(len(p)!=24 for p in profiles.values()):
        raise ValueError("Incomplete TorsionNet500 release")
    rows = []
    for name,p in sorted(profiles.items()):
        _,ref,pred = np.array(sorted(p)).T
        rows.append((name,np.ptp(ref)*EV_TO_KCAL,np.ptp(pred)*EV_TO_KCAL))
    with out.with_suffix(".csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["fragment","reference_range_kcal_mol","predicted_range_kcal_mol"])
        writer.writerows(rows)
    stats = dict(checkpoint=checkpoint.name,reference="PBE0", **score([p-r for _,r,p in rows]))
    out.write_text(json.dumps(stats,indent=2)+"\n")
    print(stats)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint",type=Path)
    p.add_argument("out",type=Path)
    a = p.parse_args()
    evaluate(a.checkpoint,a.out)
