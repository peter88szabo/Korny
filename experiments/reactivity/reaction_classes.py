"""Assign training-only barrier families from publisher labels and mapped bond edits.

Names prefixed KinBot follow kinbot/reaction_finder*.py, revision 684e513c.
Retain the publisher's more general labels when they do not imply a particular
KinBot mechanism. A geometry needs no reaction-class input during dynamics.
"""
from collections import Counter
import csv
from pathlib import Path
from zipfile import ZipFile
from rdkit import Chem
from experiments.reactivity.spin_data import DATA

RMG={'ketoenol':'ketoenol','1,3_Insertion_ROR':'r13_insertion_ROR','Retroene':'Retro_Ene',
     '2+2_cycloaddition':'r22_cycloaddition','Intra_2+2_cycloaddition_Cd':'r22_cycloaddition',
     '1+2_Cycloaddition':'r12_cycloaddition','1,3_Insertion_CO2':'r13_insertion_CO2',
     'Diels_alder_addition':'Diels_alder_addition'}


def hydrogen_migration(row):
    """Identify a single intramolecular H move to a KinBot-eligible acceptor."""
    args=Chem.SmilesParserParams();args.removeHs=False;args.sanitize=False
    left,right=[Chem.MolFromSmiles(row[k],args) for k in ('rsmi','psmi')]
    if any(g is None or len(Chem.GetMolFrags(g))!=1 for g in (left,right)):
        return False
    def graph(m):
        atoms={a.GetAtomMapNum():a for a in m.GetAtoms()}
        bonds={frozenset((b.GetBeginAtom().GetAtomMapNum(),b.GetEndAtom().GetAtomMapNum())) for b in m.GetBonds()}
        return atoms,bonds
    a,l=graph(left);b,r=graph(right)
    if {i:x.GetAtomicNum() for i,x in a.items()}!={i:x.GetAtomicNum() for i,x in b.items()} or len(l-r)!=1 or len(r-l)!=1:
        return False
    broken=next(iter(l-r));formed=next(iter(r-l));other=broken&formed
    if len(other)!=1 or a[next(iter(other))].GetAtomicNum()!=1:
        return False
    for atoms,site in ((a,next(iter(formed-other))),(b,next(iter(broken-other)))):
        x=atoms[site]
        if x.GetAtomicNum() in (7,8,16) or x.GetNumRadicalElectrons()>0 or any(v.GetBondTypeAsDouble()>1 for v in x.GetBonds()):
            return True
    return False


def lookup(*, include_trip50=False):
    labels={}
    with (DATA/'RDB7/ccsdtf12_dz.csv').open() as f:
        for row in csv.DictReader(f):
            key='RDB7/rxn'+row['idx'].zfill(6)
            name=row['rmg_family']
            if name in RMG:
                family='KinBot/'+RMG[name]
            elif name:
                family='RMG/'+name
            else:
                family='KinBot/intra_H_migration' if hydrogen_migration(row) else 'unassigned'
            labels[key]=family
    with ZipFile(DATA/'BH9/ct2c00362_si_001.zip') as f:
        for line in f.read('BH9_SI/Reference.org').decode().splitlines()[1:]:
            if line.startswith('|'):
                _,_,sub,tag,*_=[s.strip() for s in line.strip('|').split('|')]
                labels['BH9/'+tag]='KinBot/Diels_alder_addition' if sub=='Diels-Alder' else 'BH9/'+sub
    if include_trip50:
        with (DATA/'TRIP50/ct6c00144_si_001.csv').open(encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                if row['Theory']=='Reference':
                    labels['TRIP50/'+row['Rxn Name']]='TRIP50/'+row['Rxn Type']
    return labels


def family(row,geoms,labels):
    value=labels[row.name]
    if value in ('BH9/Hydrogen atom transfer','TRIP50/HAT'):
        if len(row.reactants)==2 and any(geoms[k].state[1]>1 for k in row.reactants):
            return 'KinBot/abstraction'
    return value
