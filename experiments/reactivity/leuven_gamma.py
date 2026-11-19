"""Matched stationary-point comparison for explicitly named private Gamma channels.

Use the archived Case II R/S labels exactly as written. Select CC single points
with a converged, frequency-characterized ORCA geometry from the same compound
job, and require agreement of every atom and coordinate. Compare the lowest
archived electronic CC conformer of each stationary species. Missing fragments,
stereo mismatches and unconfirmed source pathways are never guessed.
"""
from collections import Counter,defaultdict
import json
import re
import sqlite3
import numpy as np
from experiments.reactivity import compare_potentials as cp
from experiments.reactivity.leuven_hshift import CACHE,HARTREE_EV,EV_TO_KCAL
from experiments.reactivity.spin_data import Geometry,Reaction
from src.forcefield import ForceField


def export():
    work=cp.WORK/'gamma';work.mkdir(exist_ok=True)
    d=sqlite3.connect(CACHE/'labels.sqlite')
    d.execute('ATTACH ? AS idx',(str(CACHE/'files.sqlite'),))
    prefix='Saving_VSC/Case2/CCSDT_Gamma-Channel/'
    matched=defaultdict(list);seen=set()
    sql="""SELECT f.name, c.z, c.r, c.energy, c.q, c.m, o.r, o.freq FROM idx.files f
           JOIN labels c ON c.hash=f.hash JOIN labels o ON o.hash=c.hash AND o.stage<c.stage
           WHERE f.name LIKE ? AND f.name NOT LIKE '%::%' AND c.level='ORCA6.1.0/ccsd(t)-f12/ri/cc-pvdz-f12/gas'
           AND o.kind='optimized' AND o.level='ORCA6.1.0/m062x/aug-cc-pvtz/d3zero/gas'
           AND o.freq IS NOT NULL AND o.z=c.z AND o.q=c.q AND o.m=c.m"""
    for name,z,r,e,q,m,ro,freq in d.execute(sql,(prefix+'%',)):
        path=name[len(prefix):].split('/')[0]
        if not re.fullmatch(r'(?:TS_)?Gamma-\d+(?:-\d+)?_ZZ_[RS]',path):continue
        if not np.allclose(np.frombuffer(r,'<f8'),np.frombuffer(ro,'<f8'),atol=2e-5,rtol=0):continue
        v=np.sort(json.loads(freq)[6:])
        if (v< -1.).sum() != int(path.startswith('TS_')) or (path.startswith('TS_') and v[0]>-20.):continue
        if (path,z,r) in seen:continue
        seen.add((path,z,r));matched[path].append((e,name,Geometry(np.frombuffer(z,np.uint8).astype(int),np.frombuffer(r,'<f8').reshape(-1,3),(q,m))))
    panels=[];geometries={};used={};skipped={}
    for name,group in sorted(matched.items()):
        pattern=re.fullmatch(r'TS_Gamma-(\d+)-(\d+)_ZZ_([RS])',name)
        if not pattern:continue
        start='Gamma-'+pattern[1]+'_ZZ_'+pattern[3];end='Gamma-'+pattern[2]+'_ZZ_'+pattern[3]
        if not (start in matched and end in matched):
            skipped[name]='missing exactly labelled endpoint';continue
        endpoints=[min(matched[k],key=lambda row:row[0]) for k in (start,name,end)]
        if len({(tuple(sorted(v[2].z)),v[2].state) for v in endpoints})>1:
            skipped[name]='unbalanced standalone species or state';continue
        a,s,b=[v[0]*HARTREE_EV*EV_TO_KCAL for v in endpoints]
        panels.append(Reaction(name,'source-labelled Gamma',[start],[end],name,(s-a,s-b,b-a)))
        for k,v in zip((start,name,end),endpoints):
            geometries[k]=v[2];used[k]=dict(source=v[1],conformers=len(matched[k]))
    if not panels:raise RuntimeError('No source-labelled, balanced channels passed stationary-point checks')
    info=json.loads((cp.OUT/'reaction_centers_shape.json').read_text());model=ForceField.load(cp.OUT/info['checkpoint'],dispersion='d3')
    records={}
    with (work/'geometries.extxyz').open('w') as f:
        for k,g in sorted(geometries.items()):
            q,m=g.state;f.write(f'{len(g.z)}\nProperties=species:S:1:pos:R:3 charge={q} multiplicity={m}\n')
            for z,(x,y,zz) in zip(g.z,g.r):f.write(f'{cp.SYMBOLS[z]} {x:.11f} {y:.11f} {zz:.11f}\n')
            records[k]=dict(z=g.z.tolist(),r=g.r.tolist(),q=q,m=m,base=model.energy(g.z,g.r,*g.state))
    (work/'metadata.json').write_text(json.dumps(dict(parent=info['checkpoint'],keys=sorted(records),geometries=records,panels={'Gamma_CaseII':[vars(r) for r in panels]},provenance=used,exclusions=skipped),indent=2))
    print('Verified named channels',[(r.name,np.round(r.reference,3).tolist()) for r in panels],flush=True)


if __name__=='__main__':
    import sys
    if sys.argv[-1]=='export':export()
    else:
        cp.WORK=cp.WORK/'gamma'
        cp.assess(sys.argv[-1])
