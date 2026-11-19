"""Local sGDML comparison for one fixed 17-atom, neutral-doublet H-shift.

Use the official sGDML kernel assembler, symmetry representation and predictor.
Only three of nine training geometries have a matching Cartesian force vector.
Solve the observed submatrix for the mixed-label experiment. Also run the
unmodified sGDML training routine on the three fully labelled saddle points.
Both use real archived forces. Test energies never enter either training fit.
"""
import json
import numpy as np
from scipy.linalg import solve
from rdkit import Chem
# The local upstream revision references torch.backends during import even if
# optional PyTorch is absent. In that case copy its CPU implementation to a
# temporary directory and bind the missing name only. No solver math changes.
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile

if importlib.util.find_spec('torch') is None:
    spec=importlib.util.find_spec('sgdml')
    if spec is None:
        raise ImportError('Add the local sGDML checkout to PYTHONPATH')
    root=Path(spec.origin).parent
    pattern='except ImportError:\n    _has_torch = False'
    if any(pattern in (root/name).read_text() for name in ('train.py','predict.py')):
        cpu_copy=tempfile.TemporaryDirectory()
        pkg=Path(cpu_copy.name)/'sgdml'
        shutil.copytree(root,pkg,ignore=shutil.ignore_patterns('__pycache__'))
        for name in ('train.py','predict.py'):
            f=pkg/name
            f.write_text(f.read_text().replace(pattern,'except ImportError:\n    torch = None\n    _has_torch = False'))
        sys.path.insert(0,cpu_copy.name)
from sgdml.train import GDMLTrain
from sgdml.predict import GDMLPredict
from sgdml.utils.desc import Desc
from experiments.reactivity.leuven_endpoints import connectivity
from experiments.reactivity.leuven_hshift import references, DIFF, EV_TO_KCAL, CACHE, score

def graph(points):
    """Mark the donor/acceptor using connectivity at both ends; preserve the atom map."""
    first=connectivity(points[0].z,points[0].r)
    last=connectivity(points[-1].z,points[-1].r)
    for bond in list(first.GetBonds()):
        a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        if last.GetBondBetweenAtoms(a,b) is None:
            bond.SetBondType(Chem.BondType.DOUBLE)
    for bond in last.GetBonds():
        a,b=bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
        if first.GetBondBetweenAtoms(a,b) is None:
            first.AddBond(a,b,Chem.BondType.TRIPLE)
    return first


def mapping(template,points):
    """Choose one permutation for the whole path from the start geometry alone."""
    perms=graph(points).GetSubstructMatches(graph(template),uniquify=False,maxMatches=10000)
    if not perms:
        raise ValueError('Different reaction graphs')
    a=template[0].r-template[0].r.mean(axis=0)
    def loss(order):
        b=points[0].r[list(order)]-points[0].r.mean(axis=0)
        u,_,vt=np.linalg.svd(b.T@a)
        u[:,-1]*=np.linalg.det(u@vt)
        return np.linalg.norm(b@u@vt-a)
    return np.array(min(perms,key=loss))


def run(outer=None):
    rows,clusters=references()
    clusters=[c for c in clusters if any(r[0] in c for r in rows)]
    train=[r for r in rows if (r[1]=='train' if outer is None else r[0] not in clusters[outer])]
    test=[r for r in rows if (r[1]=='test' if outer is None else r[0] in clusters[outer])]
    template=train[0][2]
    perms=np.array(graph(template).GetSubstructMatches(graph(template),uniquify=False,maxMatches=10000))
    r,e,f,observed=[],[],[],[]
    for _,_,pts,cc in train:
        order=mapping(template,pts)
        for p,v in zip(pts,cc):
            r.append(p.r[order]);e.append(v*EV_TO_KCAL)
            observed.append(p.force is not None)
            f.append((p.force.reshape(-1,3)[order]*EV_TO_KCAL) if p.force is not None else np.zeros_like(p.r))
    f,r,e,observed=map(np.array,(f,r,e,observed))
    trainer=GDMLTrain(max_memory=1,max_processes=1,use_torch=False)
    def make_task(r,e,f):
        data=dict(R=r,F=f,E=e,z=template[0].z,name=np.array('Leuven_1_6_H_shift'),theory=np.array('ECC_VDZF12 with M06-2X-D3 forces'))
        return trainer.create_task(data,len(r),data,0,sig=1.,lam=1e-8,use_E=True,use_E_cstr=True,perms=perms)
    def evaluate(model):
        predictor=GDMLPredict(model,max_processes=1,use_torch=False)
        err=[]
        for _,_,pts,cc in test:
            order=mapping(template,pts)
            energies,_=predictor.predict(np.stack([p.r[order].ravel() for p in pts]))
            err.append(DIFF@(energies.ravel()-cc*EV_TO_KCAL))
        a=np.array(err)
        return dict(endpoint_gap=score(a[:,2]),saddle_to_ends=score(a[:,:2].ravel()),saddle_minus_start=score(a[:,0]),saddle_minus_end=score(a[:,1]))
    stock=trainer.train(make_task(r[observed],e[observed],f[observed]))
    result=dict(stock_saddle_only=evaluate(stock))
    task=make_task(r,e,f)
    ids=task['idxs_train']
    geom=r[ids]
    n,nat=geom.shape[:2]; dim=3*nat
    desc=Desc(nat,max_processes=1)
    x,dx=desc.from_R(geom.reshape(n,-1))
    ps=(np.array([Desc.perm(p) for p in perms])+np.arange(len(perms))[:,None]*desc.dim).flatten('F')
    kernel=-trainer._assemble_kernel_mat(x,dx,ps,task['sig'],desc,use_E_cstr=True)
    mask=np.r_[np.repeat(observed[ids],dim),np.ones(n,bool)]
    mean=e.mean()
    y=np.r_[f[ids].ravel(),mean-e[ids]]
    # Drop every unavailable Cartesian component before solving. Zero entries
    # stored for array bookkeeping therefore contribute no observations.
    k=kernel[np.ix_(mask,mask)]
    k.flat[::len(k)+1]+=task['lam']
    alpha=np.zeros(len(mask));alpha[mask]=-solve(k,y[mask],assume_a='pos')
    model=trainer.create_model(task,'analytic',x,dx,ps,1.,alpha[:-n],alphas_E=alpha[-n:])
    model['c']=mean
    predictor=GDMLPredict(model,max_processes=1,use_torch=False)
    et,ft=predictor.predict(geom.reshape(n,-1))
    result['mixed_training_energy_mae_kcal_mol']=float(np.mean(abs(et.ravel()-e[ids])))
    result['mixed_training_force_mae_kcal_mol_A']=float(np.mean(abs(ft.reshape(n,nat,3)[observed[ids]]-f[ids][observed[ids]])))
    result.update(masked_observations=evaluate(model),fixed_sigma=1.,lambda_=1e-8,n_permutations=len(perms),CC_train_energies=n,DFT_train_force_vectors=int(observed.sum()),test_paths=len(test))
    (CACHE/('sgdml_hshift.json' if outer is None else f'sgdml_hshift_fold_{outer}.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fold',type=int,choices=range(4))
    run(parser.parse_args().fold)
