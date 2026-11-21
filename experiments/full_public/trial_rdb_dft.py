"""Match published RDB7 wB97X-D3 jobs to the CC stationary-point geometry ledger.

Store electronic energies on the common state/geometry. For derivative work keep
the actual DFT Cartesian frame, its printed Hessian, and the optimization history.
Each frame inherits its parent reaction's split. Duplicate stationary fragments
are screened again by geometry when generating training constraints.
"""
import json
import pickle
import re
import sqlite3
import tarfile
from collections import Counter
import numpy as np
from experiments.full_public import kernel_trials as kt
from experiments.reactivity.radical_curvatures import final_hessian
from experiments.reactivity.rdb7 import NAME
from src.units import BOHR_A, HARTREE_EV

OUT = kt.WORK
MARK = "Standard Nuclear Orientation (Angstroms)"
ENERGY = re.compile(r"Total energy in the final basis set\s*=\s*([-\d.]+)",re.I)
ATOMS = re.compile(r"^\s*\d+\s+([A-Z][a-z]?)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$",re.M)
ELEMENT = {"H":1,"C":6,"N":7,"O":8}


def geometry(block):
    coords = ATOMS.findall(block.split("Nuclear Repulsion Energy",1)[0])
    if not coords:
        return None
    return np.array([ELEMENT[a] for a,*_ in coords],np.uint8), np.array([v for a,*v in coords],float)


def force(block, n):
    """Read the Q-Chem three-by-atoms SCF gradient and convert to forces in eV/Å."""
    if "Gradient of SCF Energy" not in block:
        return None
    it = iter(block.split("Gradient of SCF Energy",1)[1].strip().splitlines())
    array = np.empty((n,3))
    for start in range(0,n,6):
        end = min(start+6,n)
        if list(map(int,next(it).split())) != list(range(start+1,end+1)):
            raise ValueError("Gradient slice disagreement")
        for axis in range(3):
            tokens = next(it).split()
            if len(tokens)!=1+end-start or int(tokens[0])!=axis+1:
                raise ValueError("Incomplete gradient")
            array[start:end,axis] = np.asarray(tokens[1:],float)
    return -array*(HARTREE_EV/BOHR_A)


def parse():
    gs, rows, _ = kt.inputs()
    split = {s[3]:s[2] for s in rows if s[1]=="RDB7" and s[-1]=="forward"}
    anchor = {}
    with (OUT/"ids.pkl").open("rb") as f:
        ids,_=pickle.load(f)
    dbpath = OUT/"rdb-dft-frames.sqlite"
    tmp = dbpath.with_suffix(".pending")
    if tmp.exists():
        tmp.unlink()
    counts = Counter()
    with sqlite3.connect(tmp) as db, tarfile.open(kt.OUT.parent/"RDB7/wb97xd3.tar.gz","r|gz") as archive:
        db.execute("create table stationary(name text primary key, reaction text, split text, coord_id integer, z blob, r blob, q integer, m integer, energy real, hessian blob, converged integer)")
        db.execute("create table steps(name text, step integer, r blob, energy real, force blob, primary key(name,step))")
        for entry in archive:
            match=NAME.match(entry.name)
            if not entry.isfile() or not match:
                continue
            case,label,_ = match.groups()
            key = f"RDB7/{case}/{label}"
            if key not in gs:
                continue
            role = split.get("RDB7/"+case,split.get(case))
            if role is None:
                raise ValueError(f"Missing RDB7 role for {case}")
            txt=archive.extractfile(entry).read().decode(errors="replace")
            sections = txt.split(MARK)[1:]
            if not sections or "Thank you very much for using Q-Chem" not in sections[-1]:
                counts["incomplete"]+=1
                continue
            found=ENERGY.findall(sections[-1])
            geom=geometry(sections[-1])
            if len(found)!=1 or geom is None:
                counts["missing final energy or coordinates"]+=1
                continue
            z,r=geom
            st = tuple(map(int,re.search(r"\$molecule\s+(-?\d+)\s+(\d+)\s",txt,re.I).groups()))
            g=gs[key]
            if not np.array_equal(z,g.z) or g.state!=st:
                counts["state or atom order mismatch"]+=1
                continue
            distances=lambda x:np.linalg.norm(x[:,None]-x[None,:],axis=-1)
            if np.max(abs(distances(r)-distances(g.r)))>2e-5:
                counts["different geometry"]+=1
                continue
            e=float(found[0])*HARTREE_EV
            anchor[key]=e
            try:
                h=final_hessian(txt)
            except (ValueError,IndexError,StopIteration):
                h=None
            db.execute("insert into stationary values(?,?,?,?,?,?,?,?,?,?,?)",
                (key,case,role,ids[key],z.tobytes(),r.tobytes(),*st,e,
                 None if h is None else h[3].tobytes(),int("OPTIMIZATION CONVERGED" in txt)))
            counts[f"{role} matched final"]+=1
            counts[f"{role} Hessian"]+=int(h is not None)
            for i,part in enumerate(sections[:-1]):
                energies=ENERGY.findall(part)
                if len(energies)!=1:
                    continue
                geo=geometry(part)
                if geo is None or not np.array_equal(geo[0],z):
                    continue
                try:
                    f=force(part,len(z))
                except (ValueError,StopIteration):
                    f=None
                db.execute("insert into steps values(?,?,?,?,?)",(key,i,geo[1].tobytes(),float(energies[0])*HARTREE_EV,None if f is None else f.tobytes()))
                counts[f"{role} optimization frames"]+=1
                counts[f"{role} force arrays"]+=int(f is not None)
            if len(anchor)%5000==0:
                print("Matched stationary records",len(anchor),flush=True)
                db.commit()
        db.commit()
    tmp.replace(dbpath)
    with (OUT/"rdb-dft-energies.pkl").open("wb") as f:
        pickle.dump(anchor,f)
    kt.write_json(OUT/"rdb-dft-coverage.json",dict(counts))
    print("Completed RDB7 DFT inventory",dict(counts),flush=True)


if __name__=="__main__":
    parse()
