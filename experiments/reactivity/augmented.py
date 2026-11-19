"""Use the additional bare PBE0 labels already on disk to improve the local energy model.

QM7-X molecule IDs are partitioned before choosing any conformers or fitting
any centers. Isomerization examples compare different held-out IDs of identical
formula. The frozen GMTKN transition-state panel and TorsionNet profiles never
participate in this regression.
"""
from collections import defaultdict
import csv
import json
from itertools import combinations
from pathlib import Path

import h5py
import numpy as np

from experiments.des15k.run import ATOMIC, DATA, OUT as DES_OUT, ROOT, SEED, Sample, electronic_labels, fit, read_des, split
from experiments.reactivity.thermochemistry import EV_TO_KCAL, assess, score
from src.forcefield import ForceField

OUT = Path(__file__).resolve().parent
QM = ROOT/"dataset_for_training/QM7_X/8000_starting_set/8000.hdf5"


def sample(group, name, parent, *, weight=1.):
    """Convert both collections to the common atom-referenced bare-PBE0 convention."""
    z = group["atNUM"][:]
    e = float(group["ePBE0"][0])-sum(ATOMIC[i] for i in z)
    charge, multiplicity = electronic_labels(group)
    return Sample(name, str(parent), z, group["atXYZ"][:], charge, e, group["pbe0FOR"][:],
                  weight, multiplicity=multiplicity)


def qm_partition():
    """Sample each training conformer's optimized geometry and twelve distortions.

    Hold out at least two molecule IDs in each eligible formula group to permit
    balanced isomerization reactions. Singletons cannot contribute such pairs.
    """
    rng = np.random.default_rng(8103)
    train, test = [], defaultdict(list)
    with h5py.File(QM) as f:
        by_formula = defaultdict(list)
        for mol in sorted(f):
            z = f[mol][next(iter(f[mol]))]["atNUM"][:]
            by_formula[tuple(sorted(z))].append(mol)
        heldout = set()
        for members in by_formula.values():
            if len(members) >= 3:
                heldout.update(rng.choice(members, max(2,round(.25*len(members))),replace=False))
        for mol in sorted(f):
            families = defaultdict(list)
            for key in sorted(f[mol]):
                root, suffix = key.rsplit("-",1)
                families[root].append(suffix)
            for root,suffixes in families.items():
                opt = root+"-opt"
                eq = sample(f[mol][opt],opt,mol,weight=4.)
                if mol in heldout:
                    test[mol].append(eq)
                    continue
                train.append(eq)
                distorted = [s for s in suffixes if s != "opt"]
                for s in rng.choice(distorted,min(12,len(distorted)),replace=False):
                    key = root+"-"+s
                    train.append(sample(f[mol][key],key,mol))
    return train,test


def isomer_errors(test, model):
    """Compare ensemble minima across distinct molecules, with no adjustable reference offset."""
    means = defaultdict(list)
    for mol, structures in test.items():
        formula = tuple(sorted(structures[0].z))
        pred = min(model.predict(s.z,s.r,s.charge)[0] for s in structures)
        ref = min(s.energy for s in structures)
        means[formula].append((mol,pred,ref))
    errors = []
    for members in means.values():
        for (m,a,ea),(n,b,eb) in combinations(members,2):
            errors.append((m,n,(eb-ea)*EV_TO_KCAL,(b-a)*EV_TO_KCAL))
    return errors


def run():
    """Fit once on DES training dimers plus the two independent monomer training sources."""
    rng = np.random.default_rng(SEED)
    des,_ = split(read_des(DATA/"DES15k.hdf5"),rng)
    qm,test = qm_partition()
    with h5py.File(DATA/"SPICE_dipeptides.hdf5") as h:
        # One small deterministic sample broadens intramolecular coverage at low cost.
        names = rng.choice(sorted(h),640,replace=False)
        spice = [sample(h[n][n],n,n) for n in names]
    train = des+qm+spice
    rng.shuffle(train)
    print("Training geometries",len(des),len(qm),len(spice),"held-out QM IDs",len(test),flush=True)
    name, seconds, dimension = fit(train,rng,OUT,center_count=80)
    base = DES_OUT/json.loads((DES_OUT/"results.json").read_text())["checkpoint"]
    results = dict(checkpoint=name, fit_seconds=seconds, coefficients=dimension,
                   training=dict(DES=len(des),QM7X=len(qm),SPICE=len(spice)),
                   heldout_QM7X_ids=sorted(test), isomerizations={}, gmtkn={})
    rows = []
    for label, path in (("DES",base),("augmented",OUT/name)):
        m = ForceField.load(path)
        pairs = isomer_errors(test,m)
        results["isomerizations"][label] = score([p[3]-p[2] for p in pairs])
        results["gmtkn"][label] = assess(m)
        rows.extend((label,*p) for p in pairs)
    with (OUT/"qm7x_isomers.csv").open("w") as f:
        writer = csv.writer(f)
        writer.writerow(["model","from_id","to_id","reference_kcal_mol","predicted_kcal_mol"])
        writer.writerows(rows)
    (OUT/"results.json").write_text(json.dumps(results,indent=2)+"\n")
    print("Held-out isomerizations",results["isomerizations"],flush=True)


if __name__ == "__main__":
    run()
