"""Read published CCSD(T)-F12a/cc-pVDZ-F12 electronic stationary-point energies.

The release CSV also contains thermal quantities. The present application needs
the electronic potential, so read the explicitly labelled F12a total from each
Molpro output and sum detached reactants and products separately.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
import tarfile

import numpy as np

from experiments.reactivity.spin_data import DATA, Geometry, Reaction, geometry
from experiments.reactivity.thermochemistry import EV_TO_KCAL
from src.units import HARTREE_EV

ARCHIVE = DATA/"RDB7/ccsdtf12_dz.tar.gz"
NAME = re.compile(r"^qm_logs/(rxn\d+)/((r|p|ts)\d+(?:_\d+)?)\.log$")


def complete_f12a_table(log, energy):
    """Verify the printed final electronic energy even if the timing footer is truncated."""
    table=log.split('New reference energy')[-1].split('F12b corrections for ansatz')[0]
    reference=re.match(r'\s*([-\d.]+)',table)
    pair=re.search(r'CCSD-F12a correlation energy\s+([-\d.]+)',table)
    triples=re.search(r'Triples \(T\) contribution\s+([-\d.]+)',table)
    if not(reference and pair and triples and '!CCSD(T)-F12b total energy' in log):
        return False
    return abs(sum(float(x[1]) for x in (reference,pair,triples))-float(energy))<1e-8


def dataset(*, accept_verified_table=False):
    """Skip incomplete channels and record exclusions; check nuclear and charge balance."""
    gs, energies, rejected = {}, {}, Counter()
    roles = defaultdict(lambda: defaultdict(list))
    invalid = set()
    with tarfile.open(ARCHIVE, "r|gz") as stream:
        for member in stream:
            match = NAME.match(member.name)
            if not match or not member.isfile():
                continue
            case, label, role = match.groups()
            log = stream.extractfile(member).read().decode(errors="replace")
            coord = re.search(r"geometry\s*=\s*\{\s*angstrom\s*;([^}]+)\}", log, re.I)
            state = re.search(r"wf\s*,\s*spin\s*=\s*(\d+)\s*,\s*charge\s*=\s*(-?\d+)", log, re.I)
            en = re.findall(r"^\s*!(?:U|R)?CCSD\(T\)-F12a total energy\s+([-\d.Ee+]+)", log, re.M)
            finished="Molpro calculation terminated" in log
            if accept_verified_table and len(en)==1 and not finished:
                finished=complete_f12a_table(log,en[0])
            if not (coord and state and len(en)==1 and finished):
                invalid.add(case)
                rejected["unverified_log"]+=1
                continue
            unpaired,q = map(int,state.groups())
            g = geometry(coord[1].replace(";","\n").splitlines(),q,unpaired+1)
            if g is None:
                invalid.add(case)
                rejected["element_or_size"]+=1
                continue
            k="RDB7/"+case+"/"+label
            gs[k],energies[k] = g,float(en[0])
            roles[case][role].append(k)
    records=[]
    for case,d in roles.items():
        if case in invalid or len(d["ts"])!=1 or not d["r"] or not d["p"]:
            rejected["incomplete_reaction"]+=1
            continue
        ts=d["ts"][0]
        z=Counter(gs[ts].z)
        def balanced(side):
            return Counter(a for k in side for a in gs[k].z)==z and sum(gs[k].state[0] for k in side)==gs[ts].state[0]
        if not all(balanced(d[s]) for s in ("r","p")):
            rejected["unbalanced_reaction"]+=1
            continue
        r,p=(sum(energies[k] for k in d[s]) for s in ("r","p"))
        e=energies[ts]
        ref=tuple(v*HARTREE_EV*EV_TO_KCAL for v in (e-r,e-p,p-r))
        records.append(Reaction("RDB7/"+case,"RDB7",d["r"],d["p"],ts,ref))
    return gs, records, rejected


if __name__=="__main__":
    gs, rs, excluded=dataset()
    print(len(gs), len(rs), excluded)
    print("TS populations",Counter(gs[r.ts].state for r in rs))
