"""Join published self-consistent bare PBE0 energies to GMTKN55 spin-labelled structures.

The Yonsei D2C-DFT table distinguishes PBE0, HF-PBE0 and dispersion-corrected
variants. Only the exact PBE0 calculation type supplies this fit. Each join also
checks the independent benchmark reference to detect indexing or version errors.
"""
import argparse
import ast
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from experiments.reactivity.high_level_fit import DATA as STRUCTURES, fingerprint
from experiments.reactivity.spin_data import DATA, geometry, reactions
from experiments.reactivity.spin_fit import BACKBONE, OUT, partition, shape_constraints
from experiments.reactivity.thermochemistry import EV_TO_KCAL, assess, panel, score
from src.electronic import ElectronicCorrection, ElectronicFeatures
from src.forcefield import ForceField

SOURCE = "https://tccl.yonsei.ac.kr/data/database"
FAMILIES = {"W4-11","BH76","BH76RC","BHROT27","BHPERI","BHDIV10","WCPT18","G21EA","G21IP",
            "FH51","ISO34","ISOL24","TAUT15","RSE43","DARC","BSR36","PX13","CDIE20","DC13",
            "PA26","PArel","DIPCS10","SIE4x4","RC21","AHB21"}
DEVELOPMENT = {"ISO34","BHDIV10"}
TEST = {"BH76","BH76RC"}


def dataset(charges):
    """Keep benchmark-family splits and protect the other experiments' validation structures."""
    local = defaultdict(list)
    with (STRUCTURES/"PBEh-3c_reactions.csv").open() as f:
        for row in csv.DictReader(f):
            local[row["Subset"]].append(row)
    fixed = set()
    for _, terms, _, _ in panel():
        for _, p in terms:
            g = geometry(p.read_text().splitlines()[2:], 0, 1)
            fixed.add((tuple(sorted(g.z)), g.state))
    geoms, records = reactions(charges)[:2]
    protected = defaultdict(list)
    for rec in records:
        if partition(geoms[rec.ts].z) != "train":
            for key in [rec.ts,*rec.reactants,*rec.products]:
                g = geoms[key]
                formula, ds = fingerprint(g.z, g.r)
                protected[formula,g.state].append(ds)
    @lru_cache(None)
    def load(key):
        path = STRUCTURES/key
        def flag(name):
            p = path/name
            return int(p.read_text()) if p.exists() else 0
        q, m = flag(".CHRG"), 1+flag(".UHF")
        if q not in charges or m > 3:
            return None
        return geometry((path/"struc.xyz").read_text().splitlines()[2:], q, m)
    rows, skipped, gs = [], Counter(), {}
    with (DATA/"GMTKN55_PBE0/d2cdft.csv").open() as f:
        for row in csv.DictReader(f):
            family = row["setname"]
            if row["calctype"] != "PBE0" or family not in FAMILIES:
                continue
            idx = int(row["rxnidx"])
            localrow = local[family][idx-1]
            if abs(float(localrow["ReferenceValue"])-float(row["Eref"])) > .051:
                raise ValueError(f"Different GMTKN reference for {family}/{idx}")
            names = ast.literal_eval(localrow["Reaction"])
            keys = [str(Path("BH76" if family=="BH76RC" else family)/n) for n in names]
            gs.update({k:load(k) for k in keys})
            if any(gs[k] is None for k in keys):
                skipped["size_element_or_state"] += 1
                continue
            if any((tuple(sorted(gs[k].z)),gs[k].state) in fixed for k in keys):
                skipped["protected_singlet_formula"] += 1
                continue
            nu = ast.literal_eval(localrow["Stochiometry"])
            balance = Counter()
            for k, c in zip(keys,nu,strict=True):
                for z in gs[k].z:
                    balance[z] += c
            # Isolated-proton conventions omit one hydrogen nucleus in the file.
            # Korny uses atom-referenced energies, so these unbalanced rows need a
            # separate energy-zero conversion. Keep only complete nuclear equations.
            if any(balance.values()) and (family=="PA26" or (family,idx)==("G21IP",1)):
                skipped["omitted_proton"] += 1
                continue
            if any(balance.values()):
                skipped["other_omitted_nuclei"] += 1
                continue
            split = "test" if family in TEST else "development" if family in DEVELOPMENT else "train"
            if split != "train":
                for k in keys:
                    g=gs[k]
                    formula, ds=fingerprint(g.z,g.r)
                    protected[formula,g.state].append(ds)
            rows.append((family,idx,split, list(zip(keys,nu,strict=True)),float(row["Ecalc"])))
    def overlaps(key):
        g = gs[key]
        formula, ds = fingerprint(g.z, g.r)
        return any(np.all(abs(d-ds)<.04) for d in protected[formula,g.state])
    out=[]
    for family,idx,split,eq,e in rows:
        # Isolated atoms are shared references. Report that separately from transfer
        # to an unseen polyatomic molecule, and retain atomic IPs and affinities.
        if split=="train" and any(len(gs[k].z)>1 and overlaps(k) for k,_ in eq):
            skipped["shared_validation_geometry"] += 1
            continue
        out.append((family,idx,split,eq,e))
    return gs,out,skipped


def pbe0_force_rows(model, phi, states):
    """Add bare PBE0 energy and Cartesian force residuals from backbone training IDs.

    Force differences at distorted molecules anchor the molecular curvature.
    Train-only DES parents cover both ionic charges. A block QR within each
    state preserves the least-squares problem while bounding peak memory.
    """
    from experiments.des15k.run import DATA as DES, SEED, read_des, split
    from experiments.reactivity.augmented import qm_partition
    from experiments.reactivity.cluster_forces import projected_rows
    from experiments.reactivity.stationary_fit import POINTS, curvature_rows
    rng = np.random.default_rng(SEED)
    des, _ = split(read_des(DES/"DES15k.hdf5"), rng)
    qm, _ = qm_partition()
    selected = []
    for q in (-1, 0, 1):
        group = [r for r in des if r.charge == q]
        selected.extend(group[i] for i in rng.choice(len(group), 128, replace=False))
    selected.extend(qm[i] for i in rng.choice(len(qm), 192, replace=False))
    chunks = defaultdict(list)
    for row in selected:
        state = row.charge,row.multiplicity
        energy, force = model.predict(row.z, row.r, *state)
        e, f = phi.design(row.z, row.r)
        # Both labels use the original atom-referenced PBE0 energy convention.
        chunks[state].append(.50*np.r_[e,row.energy-energy][None, :])
        chunks[state].append((2./np.sqrt(row.r.size))*np.column_stack((f,(row.force-force).ravel())))
    cluster, _, _ = projected_rows(model, phi, POINTS.parents[1]/"QCell/qcell_ions_water.h5")
    secants, _ = curvature_rows(model,phi)
    for weight, data in ((.45, cluster["train"]), (.15, secants)):
        # Both readers express residuals as base_force minus target_force.
        data[:,-1] *= -1
        chunks[0,1].append(weight*data)
    xs, ys = [], []
    for state, parts in chunks.items():
        a = np.vstack(parts)
        active = np.any(a!=0, axis=0)
        small = np.linalg.qr(a[:,active], mode="r")
        block = np.zeros((len(small), phi.size+1))
        block[:,active] = small
        design = np.zeros((len(block),len(states)+1,phi.size))
        design[:,0] = block[:,:-1]
        design[:,1+states.index(state)] = block[:,:-1]
        xs.append(design.reshape(len(block),-1))
        ys.append(block[:,-1])
    return np.vstack(xs), np.concatenate(ys), dict(DES=384,QM7X=192,
                 QCell_force_projections=len(cluster["train"]),QM7X_force_secants=len(secants))


def fit(*, forces=False):
    from hashlib import sha256
    from io import BytesIO
    from time import perf_counter
    from scipy.linalg import eigh
    start=perf_counter()
    model=ForceField.load(OUT/BACKBONE)
    geoms, rows, skipped=dataset(model.charge_offsets)
    keys={k for _,_,s,eq,_ in rows if s=="train" for k,_ in eq}
    states=sorted({geoms[k].state for k in keys})
    phi=ElectronicFeatures(model.pair.species, environment=True)
    model.electronic=ElectronicCorrection(model.pair.species,states,np.zeros((len(states),phi.size)),
                                          "PBE0",environment=True)
    @lru_cache(None)
    def design(k):
        g=geoms[k]
        vec=phi.design(g.z,g.r)[0]
        blocks=np.zeros((len(states)+1,phi.size))
        blocks[0]=vec
        blocks[1+states.index(g.state)]=vec
        return model.energy(g.z,g.r,*g.state),blocks.ravel()
    x,y,meta=[],[],[]
    for family,idx,split,eq,e in rows:
        if any(geoms[k].state not in states for k,_ in eq):
            skipped["absent_from_train"]+=1
            continue
        prior=sum(c*design(k)[0] for k,c in eq)
        x.append(sum(c*design(k)[1] for k,c in eq))
        y.append(e/EV_TO_KCAL-prior)
        meta.append(dict(family=family,idx=idx,split=split,reference_pbe0=e,
                         prior_kcal_mol=prior*EV_TO_KCAL))
    x,y=np.array(x),np.array(y)
    train=np.array([r["split"]=="train" for r in meta])
    dev=np.array([r["split"]=="development" for r in meta])
    # Preserve fitted equilibrium force and curvature at the existing NIST training
    # points. Other GMTKN labels are energies, including vertical electronic states,
    # so their geometries need not be stationary on the bare PBE0 surface.
    lock,b,_=shape_constraints(model,phi,{},[],states)
    force_sources = {}
    if forces:
        xf,yf,force_sources = pbe0_force_rows(model,phi,states)
        lock,b = np.vstack((lock,xf)),np.r_[b,yf]
    choices=[]
    for scale in (.25,.75):
        penalty=np.r_[np.ones(phi.size),np.full(len(states)*phi.size,scale)]
        a=np.vstack((x[train],lock))*penalty
        active=np.any(a!=0,axis=0)
        aa=a[:,active]
        # The selected subspace is enough to represent the complete ridge solution.
        vals,vecs=eigh(aa @ aa.T)
        rhs=vecs.T @ np.r_[y[train],b]
        proj=aa.T @ vecs
        for ridge in (.003,.03,.3,3.):
            w=np.zeros(len(penalty))
            w[active]=penalty[active]*(proj @ (rhs/(np.maximum(vals,0)+ridge)))
            loss=np.mean(np.abs(x[dev]@w-y[dev]))
            choices.append((loss,ridge,scale,w))
    _,ridge,scale,w=min(choices,key=lambda v:v[0])
    ws=w.reshape(1+len(states),phi.size)
    model.electronic=ElectronicCorrection(model.pair.species,states,ws[1:]+ws[0],"PBE0",environment=True)
    buf=BytesIO()
    model.save(buf)
    raw=buf.getvalue()
    name="pbe0-electronic-"+sha256(raw).hexdigest()[:16]+".npz"
    (OUT/name).write_bytes(raw)
    errors=(x@w-y)*EV_TO_KCAL
    for r,err in zip(meta,errors,strict=True):
        r["error_kcal_mol"]=float(err)
    metrics={family:score([r["error_kcal_mol"] for r in meta if r["family"]==family])
             for family in sorted({r["family"] for r in meta})}
    report=dict(checkpoint=name,backbone=BACKBONE,source=SOURCE,calctype="PBE0",
                basis="def2-QZVPPD",states=states,counts=Counter(r["split"] for r in meta),
                excluded=skipped,shape_constraints=len(lock),force_sources=force_sources,metrics=metrics,
                ridge=ridge,state_feature_scale=scale,fit_seconds=perf_counter()-start,
                fixed_panel_high_level=assess(model),rows=meta)
    (OUT/("gmtkn_pbe0_state_forces.json" if forces else "gmtkn_pbe0_states.json")).write_text(json.dumps(report,indent=2)+"\n")
    print(name,report["counts"],"seconds",report["fit_seconds"],flush=True)
    print({k:v for k,v in metrics.items() if k in TEST|DEVELOPMENT},flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-forces",action="store_true")
    fit(forces=parser.parse_args().joint_forces)
