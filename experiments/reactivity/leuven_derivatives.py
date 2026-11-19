"""Match archived ORCA derivatives, optimized XYZs and Hessians to the actual job stage.

Exact filenames locate candidate files, including ORCA Compound basenames.
Coordinates, atom identities and (for gradients) electronic energies must
agree before a derivative is attached. Keep provenance in the private cache.
"""
import json
from pathlib import Path
import sqlite3

import numpy as np

from experiments.reactivity.leuven_files import CACHE
from experiments.reactivity.leuven_labels import ELEMENT
from src.units import BOHR_A,HARTREE_EV


def gradient(path,z,r,e):
    lines=[s.strip() for s in Path(path).read_text().splitlines() if s.strip() and not s.startswith('#')]
    n=len(z)
    xyz=[s.split() for s in lines[2+3*n:]]
    if int(lines[0])!=n or len(xyz)!=n or abs(float(lines[1])-e)>2e-6:
        return
    if bytes(int(x[0]) for x in xyz)!=z or not np.allclose(np.array([x[1:] for x in xyz],float)*BOHR_A,r,atol=2e-5,rtol=0):
        return
    f=-np.array(lines[2:2+3*n],dtype='<f8')*HARTREE_EV/BOHR_A
    if np.isfinite(f).all():
        return f.tobytes()


def match_atoms(path,z,r,hessian):
    text=Path(path).read_text(errors='replace')
    if hessian:
        if '$atoms\n' not in text or '$hessian\n' not in text:
            return False
        section=text.split('$atoms\n',1)[1].strip().splitlines()
        start,col,unit=1,2,BOHR_A
        if int(text.split('$hessian\n',1)[1].split()[0])!=3*len(z):
            return False
    else:
        section=text.splitlines()
        start,col,unit=2,1,1.
    if int(section[0])!=len(z):
        return False
    a=[s.split() for s in section[start:start+len(z)]]
    return (bytes(ELEMENT[x[0]] for x in a)==z and
            np.allclose(unit*np.array([x[col:col+3] for x in a],float),r,atol=2e-5,rtol=0))


def run():
    with sqlite3.connect(CACHE/'labels.sqlite') as db:
        db.execute('ATTACH ? AS idx',(str(CACHE/'files.sqlite'),))
        db.execute('CREATE TABLE IF NOT EXISTS derivatives (hash TEXT,stage INTEGER,xyz TEXT,hess TEXT,engrad TEXT,force BLOB,PRIMARY KEY(hash,stage))')
        rows=list(db.execute('SELECT hash,stage,stem,z,r,energy FROM labels'))
        for i,(key,stage,stem,z,raw,e) in enumerate(rows):
            r=np.frombuffer(raw,'<f8').reshape(-1,3)
            files={}
            for name, in db.execute('SELECT name FROM idx.files WHERE hash=?',(key,)):
                # A sibling of an archived job must be inside the same archive.
                parent=name.removesuffix(name.split('::')[-1].split('/')[-1])
                for ext in ('xyz','hess','engrad'):
                    candidate=parent+stem+'.'+ext
                    f=db.execute('SELECT f.hash,o.local FROM idx.files f JOIN idx.objects o USING(hash) WHERE f.name=?',(candidate,)).fetchone()
                    if f and ext not in files:
                        files[ext]=f
            matched={};force=None
            for ext,(h,p) in files.items():
                try:
                    if ext=='engrad':
                        force=gradient(p,z,r,e)
                        if force: matched[ext]=h
                    elif match_atoms(p,z,r,ext=='hess'):
                        matched[ext]=h
                except (IndexError,KeyError,ValueError):
                    continue
            db.execute('INSERT OR REPLACE INTO derivatives VALUES(?,?,?,?,?,?)',(key,stage,matched.get('xyz'),matched.get('hess'),matched.get('engrad'),force))
            if i%1000==0:
                db.commit()
        db.commit()
        counts=db.execute('SELECT count(xyz),count(hess),count(force) FROM derivatives').fetchone()
        print('Matched optimized/final coordinates XYZ, Cartesian Hessian, matched gradient:',counts,flush=True)


if __name__=='__main__':
    run()
