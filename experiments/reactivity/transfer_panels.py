"""Test the fixed QM7-X-only base on AQM gas and recomputed SO3LR MD22.

Compare electronic energies within a molecule, using one geometry as the
energy origin and excluding that anchor from each energy MAE. Force errors
use every selected Cartesian component. No target participates in selection.
"""
from collections import defaultdict
from hashlib import sha256
import codecs
import json
from pathlib import Path
import tarfile

import h5py
import numpy as np
from rdkit import Chem
from experiments.des15k.benchmark import frames
from experiments.reactivity.spin_data import DATA
from experiments.reactivity.thermochemistry import EV_TO_KCAL
from src.forcefield import ForceField

QM=Path(__file__).resolve().parents[1]/'qm7x_pilot';CHECK=QM/'korny_qm7x_pbe0.npz'


def aqm():
    smiles={line.split()[0]:line.split()[1] for line in (DATA/'Aquamarine/info-final-AQM.csv').read_text().splitlines() if line.strip()}
    bins=defaultdict(list)
    with h5py.File(DATA/'Aquamarine/AQM-gas.hdf5') as data:
        for group in data.values():
            if len(group)<2:continue
            names=sorted(group);first=group[names[0]];mol_id=names[0].rsplit('-c',1)[0]
            z=first['atNUM'][:].ravel()
            if not set(z)<={1,6,7,8}:continue
            mol=Chem.MolFromSmiles(smiles[mol_id])
            if mol is None or Chem.GetFormalCharge(mol)!=0 or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):continue
            if sorted(a.GetAtomicNum() for a in Chem.AddHs(mol).GetAtoms())!=sorted(z):continue
            size='<=20' if len(z)<=20 else '21-40' if len(z)<=40 else '>40'
            bins[size].append((mol_id,group.name))
        print('AQM eligible parents',{k:len(v) for k,v in bins.items()},flush=True)
        for size,parents in bins.items():
            for mol_id,path in sorted(parents,key=lambda v:sha256(v[0].encode()).digest())[:16]:
                group=data[path];names=sorted(group,key=lambda s:sha256(s.encode()).digest())[:6]
                yield size,mol_id,[(g['atNUM'][:].ravel(),g['atXYZ'][:],g['pbe0FOR'][:],float(g['ePBE0'][0]),0,1) for g in (group[k] for k in names)]


def md22():
    with tarfile.open(DATA/'DES15K/so3lr_test.tar.gz','r|gz') as arch:
        for member in arch:
            if not member.isfile() or '/md22_' not in member.name or not member.name.endswith('.extxyz'):continue
            name=Path(member.name).stem
            with codecs.getreader('utf-8')(arch.extractfile(member)) as stream:
                # Deterministic reservoir keeps the lowest hashes and spans the
                # full available trajectory without retaining it in memory.
                selected=[]
                for i,(z,r,f,e,q,_,m) in enumerate(frames(stream)):
                    selected.append((sha256(f'{name}-{i}'.encode()).digest(),(z,r,f,e,q,m)))
                    if len(selected)>40:
                        selected.sort(key=lambda v:v[0]);selected.pop()
            yield name,name,[v for _,v in sorted(selected,key=lambda item:item[0])]


def run(groups,models):
    results=defaultdict(lambda:defaultdict(lambda:dict(e=[],f=[],parents=0,n=0,atoms=[])))
    for group,parent,records in groups:
        for tag,model in models.items():
            out=results[group][tag];out['parents']+=1;out['n']+=len(records);base=None
            for z,r,f,e,q,m in records:
                en,force=model.predict(z,r,q,m);err=en-e
                if base is None:base=err
                else:out['e'].append((err-base)*EV_TO_KCAL)
                out['f'].extend(((force-f)*EV_TO_KCAL).ravel().tolist());out['atoms'].append(len(z))
    final={}
    for group,items in results.items():
        final[group]={}
        for tag,d in items.items():
            ee,ff=np.array(d['e']),np.array(d['f'])
            final[group][tag]=dict(parents=d['parents'],geometries=d['n'],n_relative_energies=len(ee),n_force_components=len(ff),atom_range=[min(d['atoms']),max(d['atoms'])],
                                   relative_energy_mae_kcal_mol=np.abs(ee).mean(),relative_energy_rmse_kcal_mol=np.sqrt(np.mean(ee**2)),force_mae_kcal_mol_A=np.abs(ff).mean(),force_rmse_kcal_mol_A=np.sqrt(np.mean(ff**2)))
            print(group,tag,d['n'],round(float(np.abs(ee).mean()),3),round(float(np.abs(ff).mean()),3),flush=True)
    return final


if __name__=='__main__':
    off=ForceField.load(CHECK)
    out=DATA/'comparison'
    a=run(aqm(),{'off':off});(out/'aqm_transfer.json').write_text(json.dumps(a,indent=2,allow_nan=False))
    md=run(md22(),{'off':off,'ts':ForceField.load(CHECK,dispersion='ts',volumes=QM/'volumes.npz'),'d3':ForceField.load(CHECK,dispersion='d3')})
    (out/'md22_transfer.json').write_text(json.dumps(md,indent=2,allow_nan=False))
