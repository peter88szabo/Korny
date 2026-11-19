"""Verify the installed Smite PES route on the archived 17-atom peroxy radical.

Run in a temporary work directory. Coordinates and force units cross Smite's
atomic-unit boundary here. The small NVE trajectory checks integration and
energy consistency on one fixed doublet surface.
"""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from experiments.reactivity.matern_fit import OUT
from experiments.reactivity.radical_validation import frames
from src.forcefield import ForceField
from src.units import BOHR_A,HARTREE_EV


def run(checkpoint):
    from core.molecule import Molecule
    from qchem_interfaces.pesrun import PES_Energy,PES_Force,PES_Hessian
    from qchem_interfaces.qchem_validation import validate_qchem_input
    from utils.atomic_masses import get_mass_vector

    model=ForceField.load(checkpoint,dispersion="d3")
    folder=Path(next(iter(json.loads((OUT/"matern_spin_validation.json").read_text())["paths"].values()))["folder"])
    z,r,_=next(frames(next(folder.glob("*_MEP_trj.xyz"))))
    atoms=[{1:"H",6:"C",8:"O"}[a] for a in z]
    cfg=validate_qchem_input(dict(qchem="PES",pes_name="Korny",korny_model=str(checkpoint.resolve()),
                                  korny_dispersion="d3",charge=0,multiplicity=2))
    origin=Path.cwd()
    with TemporaryDirectory() as tmp:
        try:
            os.chdir(tmp)
            x=r.ravel()/BOHR_A
            en,f=model.predict(z,r,0,2)
            np.testing.assert_allclose(PES_Energy(None,x,atoms,cfg)*HARTREE_EV,en,atol=1e-9,rtol=0.)
            force=PES_Force(x,atoms,cfg).reshape(r.shape)*HARTREE_EV/BOHR_A
            np.testing.assert_allclose(force,f,atol=1e-9,rtol=0.)
            dx=2e-5
            fd=np.array([(model.energy(z,r-u*dx,0,2)-model.energy(z,r+u*dx,0,2))/(2*dx)
                         for u in np.eye(r.size).reshape(-1,*r.shape)]).reshape(r.shape)
            np.testing.assert_allclose(fd,force,atol=2e-5,rtol=1e-6)
            h=PES_Hessian(x,atoms,cfg)
            if h.shape!=(r.size,r.size) or not np.isfinite(h).all():
                raise ValueError("Invalid Smite Hessian")
            mol=Molecule(atoms,get_mass_vector(atoms),x,np.zeros(r.size))
            mol.qchem=cfg;mol.fname="peroxy"
            mol.p[0]=.1;mol.p[-3]=-.1
            initial=mol.get_energy()[-1]
            mol.run_trajectory(integrator="verlet",timestep=.05,maxstep=60,iprint=60,Rstop=20.)
            if mol.termination_reason!="maxstep" or mol.termination_step!=60:
                raise ValueError("Smite trajectory ended early")
            drift=float((mol.get_energy()[-1]-initial)*HARTREE_EV*23.060548867)
        finally:
            os.chdir(origin)
    return dict(checkpoint=checkpoint.name,charge=0,multiplicity=2,n_atoms=len(z),dt_fs=.05,steps=60,
                maximum_FD_force_error_eV_A=float(np.max(np.abs(fd-force))),
                hessian_asymmetry_Eh_bohr2=float(np.max(np.abs(h-h.T))),
                final_minus_initial_kcal_mol=drift)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    checkpoint=args.fit.parent/json.loads(args.fit.read_text())["checkpoint"]
    report=run(checkpoint)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(report)
