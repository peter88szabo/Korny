"""A controlled learning curve for bare PBE0 and CCSD(T)/CBS doublet thermochemistry.

The publisher supplies spin-labelled geometries, bare PBE0 absolute energies
and W1-F12 atomization and reaction energies. Each quantum method has its own
atom-referenced zero. Relative energies always use the same method on both
sides. OSOS has equilibrium thermochemistry and does not provide saddle points.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from io import BytesIO, StringIO
import json
from pathlib import Path
import re
from time import perf_counter
from zipfile import ZipFile

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from experiments.des15k.benchmark import ELEMENTS
from experiments.reactivity.matern_fit import basis
from experiments.reactivity.spin_data import DATA, Geometry, geometry, reactions
from experiments.reactivity.spin_fit import BACKBONE, OUT, partition
from experiments.reactivity.thermochemistry import EV_TO_KCAL, panel, score
from src.dispersion.d3 import d3_energy_forces
from src.chemical import ChemicalEnvironment
from src.electronic import ElectronicCorrection, ElectronicFeatures
from src.forcefield import ForceField
from src.units import HARTREE_EV

FILES = {"atomization":"TAE_3360", "H_abstraction":"H-ABST_3360",
         "H_shift":"H-SHIFT_2292","isomerization":"ISOMER_922"}


def group_hash(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],"big")


def group_split(z):
    """Group by heavy-element composition, shared by related parents and radicals."""
    key = ",".join(str(a) for a in sorted(z[z!=1]))
    f = group_hash(key) % 10
    return "test" if f<2 else "development" if f<4 else "train"


def load():
    geometries, equations = {},defaultdict(list)
    with ZipFile(DATA/"OSOS/jp6c03915_si_002.zip") as archive:
        def csvfile(key):
            return StringIO(archive.read(key).decode("utf-8-sig"))
        pbe = {k:float(e) for k,e in csv.reader(csvfile("SI_Absolute_Energies/PBE0.csv"))}
        for entry in archive.namelist():
            if entry.startswith("SI_REF_DATA/XYZ/") and entry.endswith(".xyz"):
                lines = archive.read(entry).decode().splitlines()
                q,m = map(int,lines[1].split()[-2:])
                g = geometry(lines[2:],q,m)
                key = Path(entry).stem
                if g is not None and key in pbe:
                    geometries[key] = g
        for kind,fn in FILES.items():
            for row in csv.DictReader(csvfile("SI_REF_DATA/REF_"+fn+".csv")):
                equations[kind].append((row["Reactants"].split(";"),row["Products"].split(";"),
                                        float(row["Ref_Energy_kcal_mol"])))
    # A parent's TAE equals its radical's TAE plus the hydrogen dissociation
    # energy. Multiple labelled radical sites provide a check on this cycle.
    cc = {a[0]:-e for a,_,e in equations["atomization"]}
    parent_values=defaultdict(list)
    for left,right,ref in equations["H_abstraction"]:
        radical, = (k for k in right if k!="H_atom")
        parent_values[left[0]].append(cc[radical]-ref)
    inconsistent=set()
    for key,values in parent_values.items():
        if np.ptp(values)>.031:
            inconsistent.add(key)
            continue
        cc[key] = np.mean(values)
    # Quarantine all molecules and equations of an inconsistent parent. Keep an
    # audit of which names failed instead of choosing one of its conflicting TAEs.
    def usable(k):
        return k not in inconsistent and not any(k.startswith(bad+"_") for bad in inconsistent)
    geometries={k:g for k,g in geometries.items() if usable(k)}
    equations={kind:[(l,r,e) for l,r,e in rows if all(usable(k) for k in l+r)]
               for kind,rows in equations.items()}
    pbe_atoms = {z:pbe[k] for k,z in ELEMENTS.items() if k in pbe}
    targets = {k:((pbe[k]-sum(pbe_atoms[a] for a in g.z))*HARTREE_EV,cc[k]/EV_TO_KCAL)
               for k,g in geometries.items() if k in cc}
    geometries={k:g for k,g in geometries.items() if k in targets}
    geometries["H_atom"]=Geometry(np.array([1]),np.zeros((1,3)),(0,2))
    targets["H_atom"]=(0.,0.)
    # Verify H-shifts/isomerizations independently against the listed reaction energies.
    for kind in ("H_abstraction","H_shift","isomerization"):
        for left,right,e in equations[kind]:
            calc=sum(targets[k][1] for k in right)-sum(targets[k][1] for k in left)
            if abs(calc*EV_TO_KCAL-e)>.031:
                raise ValueError(f"Inconsistent published {kind} energy")
    return geometries,targets,equations,sorted(inconsistent)


def fit(*, chemical=False):
    start=perf_counter()
    geoms,targets,eqs,inconsistent=load()
    model=ForceField.load(OUT/BACKBONE)
    species=model.pair.species
    states=((0,1),(0,2))
    electronic=ElectronicFeatures(species,radial_width=.25)
    model.electronic=ElectronicCorrection(species,states,np.zeros((2,electronic.size)),"PBE0")
    splits={k:group_split(g.z) for k,g in geoms.items()}
    # Keep the entire known fixed panel, and formulas of the seven NIST test
    # molecules, out of the new corpus. BH9/TRIP50 heldout complete TS formulas
    # are also reserved. Shared H is an explicitly anchored free-atom reference.
    protected=set()
    for _,terms,_,_ in panel():
        for _,p in terms:
            protected.add(tuple(sorted(ELEMENTS[r.split()[0]] for r in p.read_text().splitlines()[2:] if r.strip())))
    from experiments.reactivity.larger_vibrations import references
    for pt in references().values():
        protected.add(tuple(sorted(ELEMENTS[a] for a in pt["atoms"])))
    sg,recs,_=reactions(model.charge_offsets)
    for row in recs:
        if partition(sg[row.ts].z)!="train":
            for k in [row.ts,*row.reactants,*row.products]:
                protected.add(tuple(sorted(sg[k].z)))
    for k,g in geoms.items():
        if tuple(sorted(g.z)) in protected and k!="H_atom":
            splits[k]="excluded"
    def parent(k):
        return re.match(r"dsgdb9nsd_\d+",k)[0]
    train_keys=sorted(k for k,s in splits.items() if s=="train" and k!="H_atom")
    parents=sorted({parent(k) for k in train_keys},key=group_hash)
    def subset(f):
        families=set(parents[:round(len(parents)*f)])
        return {k for k in train_keys if parent(k) in families}|{"H_atom"}
    kernel=basis((geoms[k] for k in sorted(subset(.25))),species,max_centers=144)
    kernel.family="double_matern52"
    chem = ChemicalEnvironment() if chemical else None
    rows,baseline,ys,keys=[],[],[],[]
    for k,g in geoms.items():
        if splits[k]=="excluded":
            continue
        e=electronic.design(g.z,g.r)[0]
        v=kernel.design(kernel.project(g.z,g.r,gradient=False))[0]
        state_weight=.45 if g.state==(0,2) else 0.
        row = np.r_[e,state_weight*e,v,state_weight*v]
        if chemical:
            group = chem.design(g.z,g.r)[0]
            row = np.r_[row,group,state_weight*group]
        rows.append(row)
        base=model.energy(g.z,g.r,*g.state)
        disp=d3_energy_forces(g.z,g.r)[0]
        baseline.append([base,base+disp])
        ys.append(targets[k])
        keys.append(k)
    keys_to_i={k:i for i,k in enumerate(keys)}
    x=np.array(rows);prior=np.array(baseline);target=np.array(ys);y=target-prior
    eq=[]
    for kind in ("H_abstraction","H_shift","isomerization"):
        for left,right,cc_ref in eqs[kind]:
            participants=[k for k in left+right if k!="H_atom"]
            if not all(k in keys_to_i for k in participants):
                continue
            split=splits[participants[0]]
            if any(splits[k]!=split for k in participants):
                raise ValueError("A reaction crosses heavy-composition partitions")
            idx=np.array([keys_to_i[k] for k in left+right])
            nu=np.array([-1]*len(left)+[1]*len(right))
            eq.append((kind,split,idx,nu))
    print("OSOS geometries",Counter(splits.values()),"equations",Counter(r[1] for r in eq),
          "basis",x.shape,flush=True)
    active=np.any(x[[keys_to_i[k] for k in subset(1.)]]!=0.,axis=0)
    x=x[:,active]
    # The same kernel, centers, regularization, data splits, feature scaling and
    # test labels are used at all three sizes. Only the number of labelled parents varies.
    curve=[]
    # Quantify the reference-level discrepancy separately from the ML error.
    # The number for each reaction compares two electronic-structure labels
    # on identical geometries and the same held-out chemical compositions.
    reference_gap = defaultdict(list)
    for kind,split,idx,nu in eq:
        if split=="test":
            reference_gap[kind].append(float(nu@(target[idx,0]-target[idx,1])*EV_TO_KCAL))
    finalweights=None
    for fraction in (.25,.5,1.):
        used=subset(fraction)
        mask=[keys_to_i[k] for k in sorted(used)]
        a=[.65*x[mask]];b=[.65*y[mask]]
        for _,split,idx,nu in eq:
            if split=="train" and all(keys[i] in used for i in idx):
                a.append((nu@x[idx])[None,:]);b.append((nu@y[idx])[None,:])
        # The free H reference controls the zero for dissociation into H + radical.
        a.append(15*x[keys_to_i["H_atom"]][None,:]);b.append(15*y[keys_to_i["H_atom"]][None,:])
        a,b=np.vstack(a),np.vstack(b)
        gram=a.T@a;gram.flat[::len(gram)+1]+=.03
        weights=cho_solve(cho_factor(gram),a.T@b)
        predictions=prior+x@weights
        stats={}
        for split in ("development","test"):
            errors=defaultdict(list)
            for kind,s,idx,nu in eq:
                if split==s:
                    errors[kind].append(nu@(predictions[idx]-target[idx])*EV_TO_KCAL)
            stats[split]={kind:{label:score(np.array(errs)[:,j]) for j,label in enumerate(("PBE0","CCSD(T)/CBS"))}
                          for kind,errs in errors.items()}
        curve.append(dict(fraction=fraction,train_parents=round(len(parents)*fraction),
                          train_geometries=len(used)-1,metrics=stats))
        print("fraction",fraction, "test CC", {k:round(v["CCSD(T)/CBS"]["mae_kcal_mol"],2)
              for k,v in stats["test"].items()},flush=True)
        finalweights=weights
    names={}
    for col,label in enumerate(("PBE0","OSOS W1-F12 CCSD(T)/CBS atom-referenced electronic energies, calibrated with PBE0-D3(BJ)")):
        w=np.zeros(len(active));w[active]=finalweights[:,col]
        n=electronic.size; d=kernel.size
        model.electronic=ElectronicCorrection(species,states,[w[:n],w[:n]+.45*w[n:2*n]],label)
        w=w[2*n:]
        kernel.weights={(0,1):w[:d],(0,2):w[:d]+.45*w[d:2*d]}
        model.matern=kernel
        if chemical:
            common,spin = w[2*d:].reshape(2,chem.size)
            model.chemical = ChemicalEnvironment({(0,1):common,(0,2):common+.45*spin})
        buf=BytesIO();model.save(buf);data=buf.getvalue()
        name=("pbe0-osos-" if col==0 else "cc-osos-")+hashlib.sha256(data).hexdigest()[:16]+".npz"
        (OUT/name).write_bytes(data)
        names["PBE0" if col==0 else "CCSD(T)/CBS"]=name
        check=ForceField.load(OUT/name,dispersion="off" if col==0 else "d3")
        for i in np.linspace(0,len(keys)-1,16,dtype=int):
            g=geoms[keys[i]]
            np.testing.assert_allclose(check.energy(g.z,g.r,*g.state),predictions[i,col],atol=2e-7)
    report=dict(checkpoints=names,source="10.1021/acs.jpca.6c03915",states=states,splits=Counter(splits.values()),
                split_method="complete heavy-element composition; nested parent subsets; common isolated H",
                inconsistent_parents_excluded=inconsistent,reg=.03,targets=["bare PBE0","W1-F12 CCSD(T)/CBS"],kernel=kernel.family,curve=curve,
                seconds=perf_counter()-start,reference_PBE0_minus_W1_test={k:score(v) for k,v in reference_gap.items()})
    filename = "osos_chemical.json" if chemical else "osos_learning_curve.json"
    if chemical:
        report["functional_group_schema"] = chem.schema
    (OUT/filename).write_text(json.dumps(report,indent=2)+"\n")
    print(names,report["seconds"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chemical",action="store_true",help="Include functional groups in the singlet and radical fits")
    fit(chemical=parser.parse_args().chemical)
