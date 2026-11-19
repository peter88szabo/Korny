"""Score transferable potentials on identical reactions and electronic references.

Export each molecular geometry separately with its explicit charge and spin,
then sum the energies of detached fragments with the published stoichiometry.
The SO3LR evaluator runs in its own environment. Verify coordinates on readback
to catch any changes in output order before calculating any differences.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shlex

import numpy as np

from experiments.reactivity.leuven_hshift import DIFF, references
from experiments.reactivity.matern_fit import OUT
from experiments.reactivity.rdb7 import dataset
from experiments.reactivity.spin_data import DATA, Geometry, Reaction, reactions
from experiments.reactivity.spin_fit import partition
from experiments.reactivity.thermochemistry import EV_TO_KCAL, score
from src.forcefield import ForceField

WORK=DATA/'comparison'
SYMBOLS={1:'H',6:'C',7:'N',8:'O',9:'F',15:'P',16:'S',17:'Cl'}


def export(full_private=False):
    WORK.mkdir(exist_ok=True)
    info=json.loads((OUT/'rdb7_chemical_residual_v4.json').read_text())
    model=ForceField.load(OUT/info['checkpoint'],dispersion='d3')
    if full_private:
        gs,panels={},{}
    else:
        gs,rows,_=dataset()
        selected=set(info['test_ids'])
        panels={'RDB7':[row for row in rows if row.name in selected]}
        bg,br,_=reactions(model.charge_offsets)
        gs.update(bg)
        panels['BH9_TRIP50']=[row for row in br if partition(gs[row.ts].z)=='test']
        assert len(panels['RDB7'])==400 and len(panels['BH9_TRIP50'])==65
    hrows,_=references()
    panels['Leuven_IRC']=[]
    for name,role,points,cc in hrows:
        if not full_private and role!='test':
            continue
        keys=['Leuven/'+name+'/'+s for s in ('start','saddle','end')]
        for k,p in zip(keys,points):
            gs[k]=Geometry(p.z,p.r,(0,2))
        panels['Leuven_IRC'].append(Reaction(name,'intra_H_migration',[keys[0]],[keys[2]],keys[1],list(DIFF@cc*EV_TO_KCAL)))
    keys=sorted({k for rows in panels.values() for r in rows for k in (r.ts,*r.reactants,*r.products)})
    records={}
    with (WORK/'geometries.extxyz').open('w') as f:
        for k in keys:
            g=gs[k];q,m=g.state
            f.write(f'{len(g.z)}\nProperties=species:S:1:pos:R:3 charge={q} multiplicity={m}\n')
            for z,(x,y,zz) in zip(g.z,g.r):
                f.write(f'{SYMBOLS[z]} {x:.11f} {y:.11f} {zz:.11f}\n')
            records[k]=dict(z=g.z.tolist(),r=g.r.tolist(),q=q,m=m,base=model.energy(g.z,g.r,*g.state))
    data=dict(parent=info['checkpoint'],keys=keys,geometries=records,
              panels={key:[vars(row) for row in rows] for key,rows in panels.items()})
    (WORK/'metadata.json').write_text(json.dumps(data))
    print('Exported',len(keys),{k:len(v) for k,v in panels.items()},flush=True)


def assess(model):
    data=json.loads((WORK/'metadata.json').read_text())
    vals={}
    if model=='korny':
        vals={k:g['base'] for k,g in data['geometries'].items()}
    elif model in ('korny-centers','korny-centers-shape'):
        fname='reaction_centers.json' if model=='korny-centers' else 'reaction_centers_shape.json'
        latest=ForceField.load(OUT/json.loads((OUT/fname).read_text())['checkpoint'],dispersion='d3')
        vals={k:latest.energy(g['z'],g['r'],g['q'],g['m']) for k,g in data['geometries'].items()}
    else:
        with (WORK/(model+'.extxyz')).open() as f:
            for k in data['keys']:
                n=int(next(f));fields=dict(x.split('=',1) for x in shlex.split(next(f)) if '=' in x)
                g=data['geometries'][k];atoms=[next(f).split() for _ in range(n)]
                if [v[0] for v in atoms]!=[SYMBOLS[z] for z in g['z']]:
                    raise ValueError('Unexpected atom order in '+k)
                np.testing.assert_allclose(np.array([a[1:4] for a in atoms],float),g['r'],atol=1e-6,rtol=0)
                vals[k]=float(fields['energy_so3lr'])
            if f.read().strip():
                raise ValueError('Unexpected output structures')
    result={}
    for panel,rows in data['panels'].items():
        errors=defaultdict(list)
        for raw in rows:
            row=Reaction(**raw)
            pred=np.array([sum(vals[k]*c for k,c in eq) for eq in row.equations()])*EV_TO_KCAL
            error=pred-row.reference
            if not np.isfinite(error).all():
                raise ValueError('Nonfinite reaction '+row.name)
            labels=['all', 'TS_multiplicity_'+str(data['geometries'][row.ts]['m'])]
            if all(data['geometries'][k]['q']==0 and data['geometries'][k]['m']==1 for k in [row.ts,*row.reactants,*row.products]):
                labels.append('neutral_singlets_only')
            for label in labels:
                errors[label].append(error)
        result[panel]={}
        for label,v in errors.items():
            a=np.array(v)
            result[panel][label]=dict(reaction=score(a[:,2]),barrier=score(a[:,:2].ravel()),forward=score(a[:,0]),reverse=score(a[:,1]))
            print(model,panel,label,len(a),*(round(result[panel][label][p]['mae_kcal_mol'],3) for p in ('reaction','barrier')),flush=True)
    (WORK/(model+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['export','korny','korny-centers','korny-centers-shape','so3lr','so3lr-tauri-m-2606','so3lr-tauri-l-2606'])
    parser.add_argument('--all-private',action='store_true')
    args=parser.parse_args()
    if args.all_private:WORK=WORK/'all_irc'
    export(args.all_private) if args.mode=='export' else assess(args.mode)
