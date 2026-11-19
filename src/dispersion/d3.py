"""Grimme D3 with Becke–Johnson damping, via simple-dftd3."""
import numpy as np


def d3_energy_gradient_bohr(Z, R, method, atm):
    """Return Hartree energy and Hartree/bohr gradient; coordinates are in bohr."""
    from dftd3.interface import DispersionModel, RationalDampingParam

    model = DispersionModel(np.asarray(Z), np.asarray(R))
    params = RationalDampingParam(method=method, atm=atm)
    result = model.get_dispersion(params, grad=True)
    return float(result["energy"]), np.asarray(result["gradient"])


def d3_energy_forces(z, r):
    """PBE0-D3(BJ) with the ATM term. Return eV and eV/Å for positions in Å."""
    from ..units import BOHR_A, HARTREE_EV
    e, grad = d3_energy_gradient_bohr(z, np.asarray(r)/BOHR_A, "pbe0", True)
    return e*HARTREE_EV, -grad*(HARTREE_EV/BOHR_A)
