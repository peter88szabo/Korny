"""Fit smooth shared-hydrogen terms on frequency-confirmed archived radical reactions.

The train/development/test assignment follows the whole transition-state formula.
The additional readout acts only on neutral doublets. Energy differences, force
stationarity and the curvature along the unstable Cartesian direction all
differentiate the same energy. Preserve both the original chemical coefficients
and all the fitted hydrocarbon terms.
"""
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import tarfile

import numpy as np

from experiments.reactivity.matern_fit import OUT, partition
from experiments.reactivity.radical_curvatures import ARCHIVE, final_hessian
from experiments.reactivity.radical_validation import validate
from experiments.reactivity.spin_data import geometry
from experiments.reactivity.thermochemistry import EV_TO_KCAL, score
from src.chemical import ChemicalEnvironment, TRANSFER_PATTERNS
from src.forcefield import ForceField
from src.units import HARTREE_EV


def read_reactions(chem):
    """Require a balanced doublet channel with one TS mode and two stable minima."""
    candidates = defaultdict(dict)
    rejected = Counter()
    with tarfile.open(ARCHIVE,"r|gz") as archive:
        for info in archive:
            match = re.fullmatch(r"wb97xd3/(rxn\d+)/(ts|r|p)\d+\.log",info.name)
            if not match:
                continue
            group,role = match.groups()
            log = archive.extractfile(info).read().decode(errors="replace")
            job = log.rsplit("Standard Nuclear Orientation (Angstroms)",1)[-1]
            if ("Thank you very much for using Q-Chem" not in job or "wB97X-D3" not in job
                    or "Grimme D3" not in job):
                rejected["incomplete_or_other_method"] += 1
                continue
            state = re.search(r"\$molecule\s+(-?\d+)\s+(\d+)",log,re.I)
            if not state or tuple(map(int,state.groups()))!=(0,2):
                continue
            atoms = re.findall(r"^\s*\d+\s+([A-Z][a-z]?\s+[-.\d]+\s+[-.\d]+\s+[-.\d]+)\s*$",
                               job.split("Nuclear Repulsion Energy",1)[0],re.M)
            g = geometry(atoms,0,2)
            if g is None or 8 not in g.z:
                continue
            frequencies = [float(v) for line in re.findall(r"^\s*Frequency:([^\n]+)",job,re.M) for v in line.split()]
            totals = re.findall(r"Total energy in the final basis set\s*=\s*([-\d.]+)",job)
            if (len(totals)!=1 or len(frequencies)!=3*len(g.z)-6
                    or sum(v < -1. for v in frequencies)!=int(role=="ts")):
                rejected["unverified_energy_or_stationary_point"] += 1
                continue
            # Keep only the TS Hessian, and only for plausible shared-H
            # geometries. Most archived radical reactions change other bonds.
            h = None
            if role=="ts":
                x = chem.design(g.z,g.r)[0]
                occupancy = max(x[chem.slices[k].start] for k,_,_ in TRANSFER_PATTERNS)
                if occupancy < .15:
                    continue
                read = final_hessian(log)
                if read is None:
                    continue
                z,r,_,h = read
                np.testing.assert_array_equal(z,g.z)
                np.testing.assert_allclose(r,g.r,atol=1e-8)
            candidates[group][role]=(g,float(totals[0])*HARTREE_EV,h)
    complete = {}
    for group,items in candidates.items():
        if set(items)!={"r","ts","p"}:
            continue
        if not all(sorted(items[k][0].z)==sorted(items["ts"][0].z) for k in ("r","p")):
            raise ValueError("Unbalanced radical reaction "+group)
        complete[group]=items
    return complete,rejected


def main():
    parent = json.loads((OUT/"peroxy_transfer.json").read_text())["checkpoint"]
    model = ForceField.load(OUT/parent,dispersion="d3")
    old = model.chemical
    rich = ChemicalEnvironment(schema="smooth-oxygen-groups/2")
    rich.weights = {s:np.pad(w,(0,rich.size-old.size)) for s,w in old.weights.items()}
    model.chemical=rich
    start=old.size
    reactions,rejected=read_reactions(rich)
    a,b=[],[]
    data=defaultdict(list)
    n_peroxy=Counter()
    for group,items in sorted(reactions.items()):
        g,_,h = items["ts"]
        subset=partition(g.z)
        features,prior,target=[],[],[]
        for key in ("r","ts","p"):
            geo,ref,_ = items[key]
            x=rich.design(geo.z,geo.r)[0]
            features.append(x[start:])
            prior.append(model.energy(geo.z,geo.r,*geo.state))
            target.append(ref)
        stoich=np.array([[-1,1,0],[0,1,-1],[-1,0,1]])
        x=stoich@features
        residue=stoich@(np.array(target)-prior)
        data[subset].append((group,x,residue))
        peroxy = [rich.slices[k].start-start for k in ("carbon_H_peroxy","oxygen_H_peroxy")]
        n_peroxy[subset]+=int(np.max(features[1][peroxy])>=.15)
        if subset!="train":
            continue
        a.extend(x); b.extend(residue)
        # The Q-Chem saddle is stationary. The most negative eigenvector of
        # its Cartesian Hessian probes the energy-lowering displacement.
        _,vectors=np.linalg.eigh(h)
        u=vectors[:,0].reshape(g.r.shape)
        step=.001
        jac=rich.design(g.z,g.r,gradient=True)[1][:,start:]
        force=model.predict(g.z,g.r,*g.state)[1]
        scale=.5/np.sqrt(g.r.size)
        a.extend(scale*jac); b.extend(-scale*force.ravel())
        curvature=[]
        for sign in (-1,1):
            r=g.r+sign*step*u
            f=model.predict(g.z,r,*g.state)[1]
            j=rich.design(g.z,r,gradient=True)[1][:,start:]
            curvature.append((u.ravel()@j,float(np.sum(u*f))))
        delta=(curvature[0][0]-curvature[1][0])/(2*step)
        present=(curvature[0][1]-curvature[1][1])/(2*step)
        a.append(.04*delta);b.append(.04*(u.ravel()@h@u.ravel()-present))
    print("verified shared-H saddle reactions", {k:len(v) for k,v in data.items()},
          "of which peroxy",dict(n_peroxy),flush=True)
    if any(not data[k] for k in ("train","development","test")):
        raise ValueError("Missing formula-disjoint stationary-point reactions in a split")
    a,b=np.array(a),np.array(b)
    def metrics(subset,w):
        errors=np.array([(x@w-y)*EV_TO_KCAL for _,x,y in data[subset]])
        return dict(reaction=score(errors[:,2]),barrier=score(errors[:,:2].ravel()))
    grid=[]; solutions=[]
    # Treat retaining the previous surface as a genuine development option.
    for ridge in (None,.03,.3,3.,30.,300.):
        w=np.zeros(a.shape[1])
        if ridge is not None:
            gram=a.T@a
            gram.flat[::len(gram)+1]+=ridge
            w=np.linalg.solve(gram,a.T@b)
        dev=metrics("development",w)
        grid.append(dict(ridge=ridge,development=dev))
        solutions.append(w)
    selected=min(range(len(grid)),key=lambda i:sum(v["mae_kcal_mol"] for v in grid[i]["development"].values()))
    w=solutions[selected]
    rich.weights[(0,2)][start:]=w
    model.electronic.target += "; neutral doublet shared-H terms calibrated to Grambow wB97X-D3/def2-TZVP stationary points"
    buf=BytesIO();model.save(buf);payload=buf.getvalue()
    name="peroxy-"+sha256(payload).hexdigest()[:16]+".npz"
    (OUT/name).write_bytes(payload)
    deployed=ForceField.load(OUT/name,dispersion="d3")
    test_paths=json.loads((OUT/"matern_spin_validation.json").read_text())["paths"]
    paths={k:validate(deployed,Path(v["folder"])) for k,v in test_paths.items()}
    for k,path in paths.items():
        print(k,path["error_kcal_mol"],flush=True)
    # Verify restored predictions using full energies from the forward provider.
    for group,x,y in data["test"]:
        rows=reactions[group]
        diffs=[]
        for role in ("r","ts","p"):
            g,en,_=rows[role]
            diffs.append(deployed.energy(g.z,g.r,*g.state)-en)
        np.testing.assert_allclose(stoich@diffs,x@w-y,atol=3e-8)
    report=dict(checkpoint=name,parent=parent,reference="wB97X-D3/def2-TZVP electronic energy",
                source="10.5281/zenodo.3731554",rejected_logs=rejected,peroxy_saddles=n_peroxy,
                groups={k:[g for g,_,_ in v] for k,v in data.items()},development_trials=grid,
                selected=selected,control=metrics("test",solutions[0]),test=metrics("test",w),paths=paths)
    (OUT/"hydrogen_transfer.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print("test",report["test"],"control",report["control"],flush=True)


if __name__=="__main__":
    main()
