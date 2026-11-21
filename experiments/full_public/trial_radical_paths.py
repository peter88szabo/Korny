"""Recover energies and forces along the published Grambow optimization paths.

The final electronic energy and atom ordering must match the stationary-point
ledger. Each path inherits its reaction fold. Shared endpoint geometries use
the union of their fold exclusions. Intermediate derivatives remain in the
printed Cartesian frame so the force directions rotate with their structures.
"""
from collections import Counter, defaultdict
import pickle
import re
import sqlite3
import tarfile

import numpy as np

from experiments.full_public import kernel_trials as kt
from experiments.full_public.references import RAD
from experiments.full_public.trial_rdb_dft import ENERGY, MARK, geometry, force
from src.units import HARTREE_EV


def parse():
    geoms, _, _ = kt.inputs()
    matrix, _, _, fold, _, _ = kt.equation_data("wb97xd3")
    with (kt.WORK / "ids.pkl").open("rb") as handle:
        ids, _ = pickle.load(handle)
    roles = defaultdict(set)
    for i, h in enumerate(fold):
        for j in matrix.indices[matrix.indptr[i]:matrix.indptr[i+1]]:
            roles[j].add(int(h))
    archive_path = kt.WORK / "grambow-steps.sqlite"
    temp = archive_path.with_suffix(".pending")
    if temp.exists():
        temp.unlink()
    counts = Counter()
    with sqlite3.connect(temp) as db, tarfile.open(RAD, "r|gz") as arc:
        db.execute("create table paths(name text primary key, coord_id integer, folds text, z blob, q integer, m integer, energy real)")
        db.execute("create table steps(name text, step integer, r blob, de real, force blob, primary key(name,step))")
        for member in arc:
            match = re.search(r"/rxn(\d+)/(ts|p|r)\d+\.log$", member.name)
            if not match or not member.isfile():
                continue
            key = f"Grambow/{match[1]}/{match[2]}"
            if key not in ids:
                continue
            g = geoms[key]
            idx = ids[key]
            if -1 in roles[idx]:
                counts["paths reserved for outer validation"] += 1
                continue
            text = arc.extractfile(member).read().decode(errors="replace")
            sections = text.split(MARK)[1:]
            last = geometry(sections[-1])
            energies = ENERGY.findall(sections[-1])
            if last is None or not energies or not np.array_equal(g.z,last[0]):
                raise ValueError(f"Atom ordering or final energy missing in {key}")
            def distances(r):
                return np.linalg.norm(r[:,None]-r[None,:],axis=-1)
            if np.max(abs(distances(g.r)-distances(last[1])))>3.e-4:
                counts["final geometry disagreement"] += 1
                continue
            state = re.search(r"\$molecule\s+(-?\d+)\s+(\d+)\s",text,re.I)
            if tuple(map(int,state.groups()))!=g.state:
                raise ValueError(f"Electronic state differs for {key}")
            e = float(energies[-1])*HARTREE_EV
            db.execute("insert into paths values(?,?,?,?,?,?,?)",
                       (key,idx,",".join(map(str,sorted(roles[idx]))),last[0].tobytes(),*g.state,e))
            counts["training paths"] += 1
            seen = set()
            for i, block in enumerate(sections[:-1]):
                val = ENERGY.findall(block)
                geo = geometry(block)
                if len(val)!=1 or geo is None or not np.array_equal(geo[0],g.z):
                    continue
                pos = geo[1].tobytes()
                if pos in seen:
                    continue
                try:
                    f = force(block,len(g.z))
                except (ValueError,StopIteration):
                    f = None
                seen.add(pos)
                db.execute("insert into steps values(?,?,?,?,?)",(key,i,pos,float(val[0])*HARTREE_EV-e,None if f is None else f.tobytes()))
                counts["distinct steps within paths"] += 1
                counts["force arrays"] += int(f is not None)
            if counts["training paths"]%500==0:
                db.commit()
                print(dict(counts),flush=True)
    temp.replace(archive_path)
    kt.write_json(kt.WORK/"grambow-steps.json",dict(counts))
    print("Completed Grambow archive",dict(counts),flush=True)


if __name__=="__main__":
    parse()


def audit():
    """Check the path energy zero and apply state/geometry exclusions to every frame."""
    from experiments.full_public.data import fingerprint
    from experiments.full_public.trial_path_features import TAG
    geoms,rows,_=kt.inputs()
    roles=defaultdict(set)
    for target in kt.TARGETS:
        d=kt.equation_data(target)
        partition={(src,case,kind):int(h) for src,case,kind,h in zip(d[2],d[5],d[4],d[3])}
        for method,src,_,name,terms,_,kind in rows:
            if method==target:
                for key,_ in terms:
                    roles[key].add(partition[src,name,kind])
    bins=defaultdict(set)
    for k,hs in roles.items():
        g=geoms[k]
        bins[(fingerprint(g.z,g.r),*g.state)].update(hs)
    with sqlite3.connect(f"file:{kt.WORK/'grambow-steps.sqlite'}?mode=ro",uri=True) as db:
        energies=dict(db.execute("select name,energy from paths"))
        errors=[]
        for _,src,_,_,terms,e,_ in rows:
            if src=="Grambow" and all(k in energies for k,_ in terms):
                errors.append(abs(sum(c*energies[k] for k,c in terms)-e))
        if not errors or max(errors)>1.e-7:
            raise ValueError(f"Grambow stationary and trajectory reference energy zeros disagree: {max(errors,default=None)} eV")
        data=list(db.execute("select p.name,p.folds,p.z,p.q,p.m,s.r from paths p join steps s using(name) order by p.name,s.step"))
    signatures=[]
    for _,fold,z,q,m,r in data:
        sig=fingerprint(np.frombuffer(z,np.uint8),np.frombuffer(r,float).reshape(-1,3)),q,m
        bins[sig].update(map(int,fold.split(",")))
        signatures.append(sig)
    masks=np.array([[(-1 not in bins[sig]) and (h not in bins[sig]) for h in (0,1,-1)] for sig in signatures])
    np.save(kt.WORK/f"{TAG}-eligible.npy",masks)
    result=dict(checked_reaction_equations=len(errors),max_energy_disagreement_eV=max(errors),eligible_by_fold=masks.sum(axis=0).tolist())
    kt.write_json(kt.WORK/"grambow-path-audit.json",result)
    print("Audited archive zero, coordinates, state and folds",result,flush=True)
