"""Score one saved checkpoint by TS spin and on archived peroxy H-shift saddles."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from experiments.reactivity.matern_fit import OUT,partition,reactions
from experiments.reactivity.radical_validation import validate
from experiments.reactivity.thermochemistry import EV_TO_KCAL,score
from src.forcefield import ForceField


def evaluate(model_path):
    model=ForceField.load(model_path,dispersion="d3")
    gs,rs,_=reactions(model.charge_offsets)
    en={};errors=defaultdict(list)
    for rec in rs:
        if partition(gs[rec.ts].z)!="test":
            continue
        for k in (rec.ts,*rec.reactants,*rec.products):
            if k not in en:
                g=gs[k]
                en[k]=model.energy(g.z,g.r,*g.state)
        prediction=[sum(en[k]*c for k,c in eq)*EV_TO_KCAL for eq in rec.equations()]
        delta=np.array(prediction)-rec.reference
        errors[str(gs[rec.ts].state[1])].append(delta)
        errors["all"].append(delta)
    spin={}
    for key,v in errors.items():
        arr=np.array(v)
        spin[key]=dict(reaction=score(arr[:,2]),barrier=score(arr[:,:2].ravel()))
    folders=json.loads((OUT/"matern_spin_validation.json").read_text())["paths"]
    paths={k:validate(model,Path(v["folder"])) for k,v in folders.items()}
    return dict(checkpoint=model_path.name,dispersion="d3",BH9_TRIP50=spin,paths=paths)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit",type=Path,required=True,help="Report containing the checkpoint to evaluate")
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    model_path=args.fit.parent/json.loads(args.fit.read_text())["checkpoint"]
    report=evaluate(model_path)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(report["checkpoint"],report["BH9_TRIP50"]["all"],flush=True)
