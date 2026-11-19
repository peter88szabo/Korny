"""Project archived Grambow radical Hessians onto reaction and vibrational directions.

The native Q-Chem files contain final electronic Hessians at wB97X-D3/def2-TZVP.
Their curvatures are optional auxiliary observations in a high-level fit. They
never enter the bare PBE0 energy/force regression. Choose files by ID before
looking at their energies or mode errors and exclude validation TS formulas.
"""
import hashlib
from pathlib import Path
import re
import tarfile

import numpy as np

from experiments.des15k.benchmark import ELEMENTS
from experiments.reactivity.spin_data import DATA
from src.electronic import validate_electrons
from src.units import BOHR_A, HARTREE_EV

ARCHIVE = DATA / "Grambow_radicals/wb97xd3_rad.tar.gz"


def final_hessian(log):
    """Read one complete final-frequency job in its printed Cartesian atom order."""
    # An optimization log can include several Hessians at different geometries.
    job = log.rsplit("Standard Nuclear Orientation (Angstroms)", 1)[-1]
    if "Hessian of the SCF Energy" not in job or "Thank you very much for using Q-Chem" not in job:
        return None
    xyz = []
    for atom, x,y,z in re.findall(r"^\s*\d+\s+([A-Z][a-z]?)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$",
                                  job.split("Nuclear Repulsion Energy")[0], re.M):
        xyz.append((ELEMENTS[atom],x,y,z))
    q, m = map(int, re.search(r"\$molecule\s+(-?\d+)\s+(\d+)\s", log, re.I).groups())
    z = np.array([row[0] for row in xyz], int)
    r = np.array([row[1:] for row in xyz], float)
    validate_electrons(z, q, m)
    h = np.full((r.size, r.size), np.nan)
    rows = iter(job.split("Hessian of the SCF Energy", 1)[1].strip().splitlines())
    for start in range(0, r.size, 6):
        ids = np.array(next(rows).split(), int)-1
        if not np.array_equal(ids, np.arange(start, min(start+6,r.size))):
            raise ValueError("Hessian column numbers disagree with atom count")
        for i in range(r.size):
            a=next(rows).split()
            if int(a[0])!=i+1:
                raise ValueError("Missing Hessian row")
            h[i,ids]=np.array(a[1:],float)
    if not np.allclose(h,h.T,atol=2e-6):
        raise ValueError("Invalid Cartesian Hessian")
    return z,r,(q,m),h*(HARTREE_EV/BOHR_A**2)


def curvature_samples(held_formulas, limit=64):
    """Deterministically select complete, disjoint neutral-doublet inputs."""
    seen=0
    with tarfile.open(ARCHIVE, mode="r:gz") as archive:
        members=[f for f in archive if f.isfile() and f.name.endswith(".log")]
        members.sort(key=lambda f:hashlib.sha256(f.name.encode()).digest())
        # gzip seeks are costly. Read the selected names in a second streaming pass.
        wanted={f.name for f in members[:8*limit]}
    with tarfile.open(ARCHIVE, mode="r|gz") as archive:
        for f in archive:
            if f.name not in wanted:
                continue
            log=archive.extractfile(f).read().decode(errors="replace")
            if "OPTIMIZATION CONVERGED" not in log:
                continue
            sample=final_hessian(log)
            if sample is None:
                continue
            z,r,state,h=sample
            if state!=(0,2) or tuple(sorted(z)) in held_formulas:
                continue
            yield f.name,z,r,state,h
            seen+=1
            if seen==limit:
                return
