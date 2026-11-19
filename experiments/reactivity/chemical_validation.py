"""Paired MD-related validation of functional groups at fixed and relaxed geometries.

Use the existing peroxy H-shift reference files for both checkpoints. No
quantum jobs are run. The seven PBE0 harmonic references are compared after
independent relaxation from identical starting geometries. A failed relaxation
is recorded and excluded from numerical frequency averages.
"""
import argparse
import json
from pathlib import Path

from experiments.reactivity.larger_vibrations import evaluate, references
from experiments.reactivity.matern_fit import OUT
from experiments.reactivity.radical_validation import validate
from src.forcefield import ForceField


def run(report, control_report):
    fitted = json.loads(report.read_text())
    ctrl = json.loads(control_report.read_text())
    fit_ratio = fitted["curve"][fitted["selected_trial"]]["total_RDB7_to_BH9_weight"]
    control_ratio = ctrl["curve"][ctrl["selected_trial"]]["total_RDB7_to_BH9_weight"]
    folders = json.loads((OUT/"matern_spin_validation.json").read_text())["paths"]
    points = references()
    output = {"dispersion":"d3", "control_loss_weight":control_ratio,
              "chemical_loss_weight":fit_ratio,"models":{}}
    for name, row in (("control",ctrl),("chemical",fitted)):
        model = ForceField.load(OUT/row["checkpoint"],dispersion="d3")
        paths = {}
        for tag,ref in folders.items():
            path = paths[tag] = validate(model,Path(ref["folder"]))
            print(name, tag, "reaction / forward / reverse errors (kcal/mol)",
                  *(round(path["error_kcal_mol"][k],2) for k in
                    ("reaction","forward_barrier","reverse_barrier")), flush=True)
        print(name, "relaxed molecule tests",flush=True)
        output["models"][name] = dict(checkpoint=row["checkpoint"], paths=paths, minima=evaluate(model,points))
    return output


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,default=OUT/"rdb7_balanced_centers128_chemical.json")
    parser.add_argument("--control",type=Path,default=OUT/"rdb7_balanced_centers128.json")
    parser.add_argument("--out",type=Path,default=OUT/"chemical_validation.json")
    args=parser.parse_args()
    result=run(args.report,args.control)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
