"""Fit an oxygen-group correction on archived C5H9O5 doublet relative energies.

Use only completed, internally consistent Molpro UCCSD(T)-F12b/cc-pVDZ-F12
single points. Select ridge on entire held-out chemical species within the
training composition. The three C5H9O3 peroxy H shifts use another composition
and a different electronic reference level. They are external transfer tests.
"""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re

import numpy as np

from experiments.reactivity.matern_fit import OUT
from experiments.reactivity.radical_validation import validate
from experiments.reactivity.spin_data import geometry
from experiments.reactivity.thermochemistry import EV_TO_KCAL
from src.forcefield import ForceField
from src.chemical import ChemicalEnvironment, TRANSFER_PATTERNS
from src.units import BOHR_A, HARTREE_EV

ROOT = Path("/home/peter/Dropbox/Research_Leuven/LIM1_reevol/Ready_calc/Saving_from_Dirac")
TARGET = sorted([6]*5+[1]*9+[8]*5)


def read_archive():
    """Verify energy, input wavefunction, composition, geometry units and duplicates."""
    records, rejected, seen = [], Counter(), set()
    for file in sorted(ROOT.rglob("*.out")):
        log = file.read_text(errors="replace")
        family = re.search(r"/Zhen_([^/]+?)_UCCSDT_",str(file))
        output = re.findall(r"^\s*!RHF-UCCSD\(T\)-F12b energy\s+([-\d.]+)\s*$",log,re.M)
        table = re.findall(r"UCCSD\(T\)-F12B/cc-pVDZ-F12 energy=\s*([-\d.]+)",log)
        if (not family or len(output)!=1 or len(table)!=1 or abs(float(output[0])-float(table[0]))>1e-8
                or "Molpro calculation terminated" not in log
                or re.search(r"No convergence|failed to converge",log,re.I)):
            rejected["method_or_convergence_or_family"] += 1
            continue
        geom = re.search(r"geometry\s*=\s*\{([^}]+)\}",log,re.I)
        wf = re.search(r"\{\s*uccsd\(t\)-f12b\s*;\s*wf\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)",log,re.I)
        if not geom or not wf:
            rejected["wavefunction_or_geometry"] += 1
            continue
        lines = re.findall(r"^\s*[A-Z][a-z]?\s+[-.\d]+\s+[-.\d]+\s+[-.\d]+\s*$",geom[1],re.M)
        zsum = sum({"H":1,"C":6,"O":8}[line.split()[0]] for line in lines)
        g = geometry(lines,zsum-int(wf[1]),int(wf[3])+1)
        if sorted(g.z)!=TARGET or g.state!=(0,2):
            rejected["reserved_composition_or_spin"] += 1
            continue
        # Molpro prints atomic coordinates in bohr independently of the XYZ
        # input. Check pair distances to allow a translated or rotated origin.
        atom_table = log.split("NR  ATOM    CHARGE",1)[1].split("Bond lengths",1)[0]
        printed = re.findall(r"^\s*\d+\s+[A-Z][a-z]?\s+[.\d]+\s+([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)\s*$",
                             atom_table,re.M)
        def distances(r):
            return np.linalg.norm(r[:,None,:]-r[None,:,:],axis=2)
        d = distances(g.r)
        np.testing.assert_allclose(d,distances(np.array(printed,float)*BOHR_A),atol=2e-5)
        fingerprint = tuple(np.sort(d[np.triu_indices(len(d),1)]).round(4))
        if fingerprint in seen:
            rejected["duplicate_geometry"] += 1
            continue
        seen.add(fingerprint)
        records.append((family[1],g,float(output[0])*HARTREE_EV))
    return records,rejected


def run(*, bridges=False):
    parent = json.loads((OUT/"rdb7_chemical_residual.json").read_text())["checkpoint"]
    model = ForceField.load(OUT/parent,dispersion="d3")
    if bridges:
        old = model.chemical
        expanded = ChemicalEnvironment(schema="smooth-oxygen-groups/2")
        expanded.weights = {state:np.pad(w,(0,expanded.size-old.size)) for state,w in old.weights.items()}
        model.chemical = expanded
    records,excluded = read_archive()
    families = np.array([label for label,_,_ in records])
    features, target = [], []
    for _,g,ref in records:
        features.append(model.chemical.design(g.z,g.r)[0])
        target.append(ref-model.energy(g.z,g.r,*g.state))
    x,y = np.array(features),np.array(target)
    groups = np.unique(families)

    def fit(mask,ridge):
        # Balance species so repeated conformers do not dominate, then eliminate
        # the arbitrary absolute energy zero. Only energy differences can
        # constrain the new MD potential on this composition.
        freq = Counter(families[mask])
        weights = np.array([1/freq[key] for key in families[mask]])
        a,b = x[mask],y[mask]
        av = np.average(a,axis=0,weights=weights)
        offset = np.average(b,weights=weights)
        a,b = a-av,b-offset
        gram = a.T@(weights[:,None]*a)
        gram.flat[::len(gram)+1] += ridge
        w = np.linalg.solve(gram,a.T@(weights*b))
        return w,offset-av@w

    trials = []
    for ridge in (.03,.3,3.,30.,300.):
        errors = []
        for group in groups:
            valid = families==group
            w,b = fit(~valid,ridge)
            errors.append(float(np.abs(y[valid]-x[valid]@w-b).mean()*EV_TO_KCAL))
        trials.append(dict(ridge=ridge,mean_held_species_MAE_kcal_mol=float(np.mean(errors))))
    selected = min(trials,key=lambda t:t["mean_held_species_MAE_kcal_mol"])
    w,_ = fit(np.ones(len(x),bool),selected["ridge"])
    # Only update neutral doublets. This parameter table follows electronic
    # state across geometries and does not use the molecular formula at runtime.
    model.chemical.weights[(0,2)] += w
    model.electronic.target += "; neutral doublet functional groups additionally calibrated on C5H9O5 UCCSD(T)-F12b relative energies"
    buf=BytesIO()
    model.save(buf)
    data=buf.getvalue()
    name="peroxy-"+sha256(data).hexdigest()[:16]+".npz"
    (OUT/name).write_bytes(data)
    deployed=ForceField.load(OUT/name,dispersion="d3")
    reference=json.loads((OUT/"matern_spin_validation.json").read_text())["paths"]
    result={}
    for tag,ref in reference.items():
        path=result[tag]=validate(deployed,Path(ref["folder"]))
        print(tag,path["error_kcal_mol"],"negative modes",len(path["model_negative_modes_cm1"]),flush=True)
    report=dict(checkpoint=name,parent=parent,training_formula="C5H9O5",training_state=[0,2],
                excluded=excluded,reference="RHF-UCCSD(T)-F12b/cc-pVDZ-F12",n_training=len(x),
                families=Counter(families),regularization_trials=trials,selected=selected,
                test_formula="C5H9O3",paths=result)
    if bridges:
        report["shared_H_training_counts"] = {
            label:int(np.count_nonzero(x[:,model.chemical.slices[label].start]>=.15))
            for label,_,_ in TRANSFER_PATTERNS}
    filename="peroxy_transfer_bridges.json" if bridges else "peroxy_transfer.json"
    (OUT/filename).write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("training",len(x),len(groups),"selected",selected,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridges",action="store_true",help="Fit shared-H features to the same coupled-cluster labels")
    run(bridges=parser.parse_args().bridges)
