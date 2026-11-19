"""Fit high-level reaction differences on independent subsets of GMTKN55.

A short-range pair-kernel correction sits on top of the geometry/frequency
checkpoint with PBE0-D3(BJ). Its target is ReferenceValue (kcal/mol), the
benchmark electronic energy difference. The PBEh-3c MethodValue is unused.
Regularization uses independent development subsets and penalizes changes in
forces and curvature at the NIST *training* geometries.
"""
import ast
from collections import Counter, defaultdict
import csv
from functools import lru_cache
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np

from experiments.des15k.benchmark import ELEMENTS
from experiments.reactivity.augmented import OUT
from experiments.reactivity.stationary_fit import POINTS
from experiments.reactivity.thermochemistry import EV_TO_KCAL, assess, panel, score
from src.forcefield import ForceField
from src.pair import PairBaseline

DATA = Path('/home/peter/DevCode/Behemoth/validation/GMTKN55')
TRAIN = {'W4-11','BHPERI','BHROT27','WCPT18','PX13','BSR36','DARC','FH51','ISOL24','TAUT15','CDIE20','DC13'}
DEV = {'BHDIV10','ISO34'}


@lru_cache(None)
def geometry(path):
    """Read closed-shell neutral molecular states and the W4 free-atom energy references."""
    if not path.exists():
        return None
    rows = [line.split() for line in path.read_text().splitlines()[2:] if line.strip()]
    if any(row[0] not in ELEMENTS for row in rows) or len(rows)>36:
        return None
    z = np.array([ELEMENTS[s[0]] for s in rows])
    for filename in ('.CHRG','.UHF'):
        indicator = path.parent/filename
        if indicator.exists() and int(indicator.read_text().strip()) != 0:
            # Free atoms define the zero in a balanced W4 atomization equation.
            # No open-shell molecular reaction or barrier is described by this model.
            if filename=='.UHF' and path.parent.parent.name=='W4-11' and len(z)==1:
                continue
            return None
    return z,np.array([row[1:4] for row in rows],float)


def fingerprint(z,r):
    """Distinguish different geometries of one formula independent of atom ordering."""
    i,j = np.triu_indices(len(z),1)
    ds = np.linalg.norm(r[i]-r[j],axis=1)
    order = np.lexsort((ds,np.maximum(z[i],z[j]),np.minimum(z[i],z[j])))
    return tuple(sorted(z)),ds[order]


def dataset():
    """Protect whole fixed-panel formulas and geometrically equivalent development endpoints."""
    protected = {tuple(sorted(geometry(path)[0])) for _,terms,_,_ in panel() for _,path in terms}
    records = []
    for row in csv.DictReader((DATA/'PBEh-3c_reactions.csv').open()):
        family = row['Subset']
        if family not in TRAIN|DEV:
            continue
        names = ast.literal_eval(row['Reaction'])
        nu = ast.literal_eval(row['Stochiometry'])
        paths = [DATA/family/n/'struc.xyz' for n in names]
        states = [geometry(p) for p in paths]
        if any(g is None or tuple(sorted(g[0])) in protected for g in states):
            continue
        balance = Counter()
        for (z,_),c in zip(states,nu,strict=True):
            for a in z:
                balance[a] += c
        if any(balance.values()):
            raise ValueError(f'Unbalanced {family} {names}')
        records.append((family,list(zip(paths,nu,strict=True)),float(row['ReferenceValue'])))
    dev_coords = defaultdict(list)
    for family,terms,_ in records:
        if family in DEV:
            for p,_ in terms:
                key,ds = fingerprint(*geometry(p))
                dev_coords[key].append(ds)
    out,seen = [],set()
    for family,terms,target in records:
        duplicate = False
        equation = Counter()
        for p,c in terms:
            formula,ds = fingerprint(*geometry(p))
            if family in TRAIN and any(np.all(np.abs(ds-d)<.06) for d in dev_coords[formula]):
                duplicate = True
            equation[(formula,tuple(ds.round(3)))] += c
        eq = tuple(sorted((key,v) for key,v in equation.items() if v))
        inv = tuple((key,-v) for key,v in eq)
        if not duplicate and eq and min(eq,inv) not in seen:
            seen.add(min(eq,inv))
            out.append((family,terms,target))
    return out


class Calibrated:
    """Conservative experimental surface. Both components return eV and eV/Å."""
    def __init__(self,base,pair):
        self.base,self.pair=base,pair

    def predict(self,z,r,charge=0):
        e,f=self.base.predict(z,r,charge)
        de,df=self.pair.predict(z,r)
        return e+de,f+df


def main():
    name = json.loads((OUT/'repulsion_audit.json').read_text())['checkpoint']
    base = ForceField.load(OUT/name,dispersion='d3')
    records = dataset()
    pair = PairBaseline(base.pair.species, np.array([.70,.92,1.14,1.36,1.58,1.8,2.1,2.5]),.26,2.2,3.1)
    @lru_cache(None)
    def state(path):
        z,r=geometry(path)
        return base.energy(z,r),pair.design(z,r)[0]
    rows,delta = [],[]
    for family,terms,target in records:
        value=sum(c*state(p)[0] for p,c in terms)
        rows.append(sum(c*state(p)[1] for p,c in terms))
        delta.append(target/EV_TO_KCAL-value)
    x,y=np.array(rows),np.array(delta)
    mask=np.array([family in TRAIN for family,_,_ in records])
    # Preserve forces and Hessian-vector products at independent training
    # points. Atomic composition/intercept offsets cancel in balanced equations.
    locks=[]
    rng=np.random.default_rng(72)
    for p in json.loads(POINTS.read_text()).values():
        if p['split']!='train':
            continue
        z=np.array([ELEMENTS[s] for s in p['atoms']])
        r=np.array(p['positions_A'])
        locks.extend(.4*pair.design(z,r)[1])
        for u in rng.normal(size=(3,*r.shape)):
            u/=np.linalg.norm(u)
            h=(pair.design(z,r+.001*u)[1]-pair.design(z,r-.001*u)[1])/.002
            locks.extend(.05*h)
    locks=np.asarray(locks)
    candidates=[]
    for ridge in (.001,.01,.1,1.,10.):
        weights=np.linalg.lstsq(np.vstack((x[mask],locks,np.sqrt(ridge)*np.eye(x.shape[1]))),
                              np.r_[y[mask],np.zeros(len(locks)+x.shape[1])],rcond=None)[0]
        # Equal weight to the held-out isomerization and barrier families.
        losses=[np.mean(abs((x@weights-y)[[r[0]==fam for r in records]])) for fam in sorted(DEV)]
        candidates.append((np.mean(losses),ridge,weights))
    _,ridge,pair.weights=min(candidates,key=lambda trial:trial[0])
    stream=BytesIO()
    pair.save(stream)
    content=stream.getvalue()
    correction='gmtkn-d3-delta-'+hashlib.sha256(content).hexdigest()[:16]+'.npz'
    (OUT/correction).write_bytes(content)
    report=dict(backbone=name,dispersion='d3',correction=correction,ridge=ridge,n_train=int(mask.sum()),
                n_development=int((~mask).sum()),fixed_panel=assess(Calibrated(base,pair)),families={},
                records=[])
    errors=(x@pair.weights-y)*EV_TO_KCAL
    for fam in sorted({r[0] for r in records}):
        idx=np.array([a==fam for a,_,_ in records])
        report['families'][fam]=dict(split='train' if fam in TRAIN else 'development',
                before=score(y[idx]*EV_TO_KCAL),after=score(errors[idx]))
    for (fam,terms,target),err in zip(records,errors,strict=True):
        report['records'].append(dict(family=fam,terms=[[p.parent.name,c] for p,c in terms],
                                      reference_kcal_mol=target,error_kcal_mol=err))
    (OUT/'high_level_fit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('training/dev',report['n_train'],report['n_development'],'ridge',ridge,flush=True)
    print('fixed',*(report['fixed_panel'][k] for k in ('reaction','barrier')),flush=True)


if __name__=='__main__':
    main()
