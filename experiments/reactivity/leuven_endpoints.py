"""Audit IRC endpoint labels against ORCA's actual path and independent minima.

Graph isomorphism here establishes single-bond connectivity only. Align atoms
using element-preserving graph mappings and compare Cartesian coordinates.
A geometrical match identifies a candidate conformer; it cannot prove that a
separate optimization descended from this particular endpoint.
"""
import json
from pathlib import Path
import re
import sqlite3

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import rdMolAlign

from experiments.reactivity.isoprene_irc_fit import ROOT as IRC
from experiments.reactivity.leuven_hshift import CACHE, DIFF, EV_TO_KCAL, ForceField, references

RADII={1:.31,6:.76,7:.71,8:.66,9:.57,15:1.07,16:1.05,17:1.02}


def connectivity(z,r):
    mol=Chem.RWMol()
    for a in z:
        atom=Chem.Atom(int(a));atom.SetNoImplicit(True);mol.AddAtom(atom)
    for i in range(len(z)):
        for j in range(i):
            if np.linalg.norm(r[i]-r[j])<1.2*(RADII[z[i]]+RADII[z[j]]):
                mol.AddBond(i,j,Chem.BondType.SINGLE)
    geom=Chem.Conformer(len(z))
    for i,(x,y,zz) in enumerate(r):
        geom.SetAtomPosition(i,Point3D(x,y,zz))
    mol.AddConformer(geom)
    return mol


def key(mol):
    return Chem.MolToSmiles(mol,canonical=True,allHsExplicit=True)


def run():
    rows,_=references()
    fit=json.loads((CACHE/'hshift_cc.json').read_text())
    model=ForceField.load(CACHE/fit['checkpoint'],dispersion='d3')
    db=sqlite3.connect(CACHE/'labels.sqlite')
    db.execute('ATTACH ? AS f',(str(CACHE/'files.sqlite'),))
    minima={}
    for h,zb,rb,fr in db.execute("SELECT hash,z,r,freq FROM labels WHERE q=0 AND m=2 AND n=17 AND kind='optimized' AND level LIKE '%m062x%gas' AND freq IS NOT NULL"):
        z=np.frombuffer(zb,np.uint8)
        if sorted(z)!=[1]*9+[6]*5+[8]*3 or min(json.loads(fr)[6:]) < -1:
            continue
        mol=connectivity(z,np.frombuffer(rb,'<f8').reshape(-1,3))
        minima.setdefault(key(mol),[]).append((h,mol))
    report={}
    for tag,role,pts,cc in rows:
        trace=next((IRC/'IRC_from_TS-Opt_only_realTSs_finestep_morestep'/tag).glob('iso*.out')).read_text()
        section=trace.rsplit('IRC PATH SUMMARY',1)[1]
        steps=re.findall(r'^\s*(\d+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$',section,re.M)
        if not steps:
            raise ValueError('Missing IRC gradients at '+tag)
        # Match end-of-path electronic energies to the frequency job, avoiding
        # an assumed orientation of ORCA's forward/backward directions.
        result=[]
        for p in (pts[0],pts[-1]):
            from src.units import HARTREE_EV
            end=min([steps[0],steps[-1]],key=lambda row:abs(float(row[1])-p.energy/HARTREE_EV))
            if abs(float(end[1])-p.energy/HARTREE_EV)>2e-6:
                raise ValueError('IRC and endpoint energies disagree for '+tag)
            mol=connectivity(p.z,p.r); candidates=minima.get(key(mol),[])
            nearest=None
            if candidates:
                rmsd,h=min((rdMolAlign.GetBestRMS(Chem.Mol(mol),m,maxMatches=10000),h) for h,m in candidates)
                source=db.execute('SELECT name FROM f.files WHERE hash=? ORDER BY length(name) LIMIT 1',(h,)).fetchone()[0]
                nearest={'all_atom_rmsd_A':rmsd,'source':source}
            result.append(dict(step=int(end[0]),max_gradient_Eh_bohr=float(end[3]),rms_gradient_Eh_bohr=float(end[4]),negative_internal_modes=p.negatives,
                               optimized_same_connectivity=len(candidates),nearest=nearest))
        pred=[model.energy(p.z,p.r,0,2) for p in pts]
        report[tag]=dict(split=role,irc_endpoints=result,
                         cc_saddle_minus_start_end_and_end_minus_start=(DIFF@cc*EV_TO_KCAL).tolist(),
                         model_saddle_minus_start_end_and_end_minus_start=(DIFF@pred*EV_TO_KCAL).tolist())
    (CACHE/'irc_endpoint_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['conformer_15'],indent=2),flush=True)


if __name__=='__main__':
    run()
