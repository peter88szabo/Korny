"""Train and test isoprene 1,6-H shifts on the author's paired electronic CC tables.

Read all three matching ORCA coordinate files for each IRC, verify the saddle's
single imaginary mode, and use only the ECC_VDZF12 column for all three roles.
Select the ridge on development paths. Both directions of a path and nearby
conformers have the same role. Absolute total-energy zeros cancel exactly.
"""
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from experiments.reactivity.isoprene_irc_fit import path_data, OUT
from experiments.reactivity.leuven_files import ROOT, CACHE
from experiments.reactivity.thermochemistry import EV_TO_KCAL, score
from src.forcefield import ForceField
from src.units import HARTREE_EV


def references():
    folder=ROOT/'Saving_from_Dirac'
    tables=[]
    for name in ('IRC_start_16HShift','TS-16HShift','IRC_final_16HShift'):
        path=folder/(name+'_Case-2_ccsdt-vdz-f12_final_energies.dat')
        lines=path.read_text().splitlines()
        assert lines[0].split()[-1]=='ECC_VDZF12'
        values={}
        for line in lines[1:]:
            fields=line.split()
            if len(fields)==6 and fields[0].isdigit():
                values['conformer_'+fields[0]]=float(fields[-1])*HARTREE_EV
        tables.append(values)
    paths,_,clusters,roles=path_data()
    common=sorted(set(paths).intersection(*tables))
    return [(key,roles[key],paths[key],np.array([t[key] for t in tables])) for key in common],clusters


# rows are the forward and reverse barriers, and the reaction energy.
DIFF=np.array([[-1.,1.,0.],[0.,1.,-1.],[-1.,0.,1.]])


def calculate(model,records):
    a=[];y=[];geom=[];force=[]
    for _,_,points,cc in records:
        hs=[]; prior=[]; grad=[]; residual=[]
        for p in points:
            h,g=model.chemical.design(p.z,p.r,gradient=True)
            val,f=model.predict(p.z,p.r,0,2)
            hs.append(h); prior.append(val)
            if p.force is not None:
                grad.extend(g/np.sqrt(p.r.size))
                residual.extend((p.force.ravel()-f.ravel())/np.sqrt(p.r.size))
        a.append(DIFF@hs)
        y.append(DIFF@(cc-prior))
        geom.append(np.array(grad).reshape(-1,model.chemical.size))
        force.append(np.array(residual))
    return np.array(a),np.array(y),geom,force


def run(private_parent=False):
    rows,clusters=references()
    source=json.loads((OUT/'rdb7_chemical_residual_v4.json').read_text())['checkpoint']
    if private_parent:
        source=json.loads((CACHE/'global_fit.json').read_text())['fits']['CC']['checkpoint']
    parent=ForceField.load((CACHE if private_parent else OUT)/source,dispersion='d3')
    a,y,geo,force=calculate(parent,rows)
    masks={s:np.array([r[1]==s for r in rows]) for s in ('train','development','test','overlap_with_NEB')}
    train=masks['train']
    # Equal total weight per structural cluster, even for near-identical conformers.
    weights=np.array([1/np.sqrt(sum(x[0] in group and x[1]==r[1] for x in rows))
                      for r in rows for group in clusters if r[0] in group])
    x=(a[train]*weights[train,None,None]).reshape(-1,a.shape[-1])
    target=(y[train]*weights[train,None]).ravel()
    for i in np.flatnonzero(train):
        x=np.vstack((x,weights[i]*geo[i]))
        target=np.r_[target,weights[i]*force[i]]
    active=np.any(x!=0,axis=0)
    gram=x[:,active].T@x[:,active];rhs=x[:,active].T@target
    def metrics(w,split):
        err=(a[masks[split]]@w-y[masks[split]])*EV_TO_KCAL
        return dict(saddle_minus_start=score(err[:,0]),saddle_minus_end=score(err[:,1]),saddle_to_endpoints=score(err[:,:2].ravel()),endpoint_gap=score(err[:,2]))
    candidates=[np.zeros(a.shape[-1])]
    for ridge in (.001,.01,.1,1.,10.,100.):
        w=candidates[0].copy()
        w[active]=cho_solve(cho_factor(gram+ridge*np.eye(len(gram))),rhs)
        candidates.append(w)
    dev=[metrics(w,'development') for w in candidates] if masks['development'].any() else []
    best=min(range(len(dev)),key=lambda i:dev[i]['endpoint_gap']['mae_kcal_mol']+dev[i]['saddle_to_endpoints']['mae_kcal_mol']) if dev else 4  # fixed unit ridge when the CC development column is missing
    w=candidates[best]
    parent.chemical.weights[0,2]+=w
    parent.electronic.target+='; neutral doublet specialized on private Case-II 1,6-H-shift CC electronic differences'
    CACHE.mkdir(exist_ok=True)
    data=BytesIO();parent.save(data); blob=data.getvalue()
    model_path=CACHE/('hshift-'+sha256(blob).hexdigest()[:12]+'.npz')
    model_path.write_bytes(blob)
    restored=ForceField.load(model_path,dispersion='d3')
    for i,(_,_,pts,cc) in enumerate(rows):
        actual=DIFF@(np.array([restored.energy(p.z,p.r,0,2) for p in pts])-cc)
        np.testing.assert_allclose(actual,a[i]@w-y[i],atol=1e-8)
    report=dict(checkpoint=model_path.name,parent=source,method='ECC_VDZF12 (Hartree), electronic energy',reaction_class='intra_H_migration',
                ridge=[None,.001,.01,.1,1.,10.,100.][best],paths={r[0]:r[1] for r in rows},clusters=[sorted(c) for c in clusters],counts=dict(Counter(r[1] for r in rows)),
                distinct_test_clusters=sum(any(r[0] in c and r[1]=='test' for r in rows) for c in clusters),
                regularization_selection='development' if dev else 'fixed ridge 1.0, no complete CC development paths',overlap_diagnostic=metrics(w,'overlap_with_NEB'),before=metrics(candidates[0],'test'),after=metrics(w,'test'),dev=dev,train_gradient_components=sum(len(force[i]) for i in np.flatnonzero(train)))
    (CACHE/('hshift_cc_transfer.json' if private_parent else 'hshift_cc.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('counts','distinct_test_clusters','ridge','before','after')},indent=2),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-parent',action='store_true')
    run(parser.parse_args().private_parent)
