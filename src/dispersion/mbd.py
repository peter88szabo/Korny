"""Molecular MBD@rsSCS dispersion via the libMBD Fortran implementation."""
import numpy as np


def mbd_energy_bohr(Z, R, log_volumes, beta=0.85):
    """Energy in Hartree for coordinates in bohr; beta defaults to PBE0."""
    from pymbd.fortran import MBDGeom

    symbols = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S", 17: "Cl"}
    coords = np.asarray(R, dtype=float)
    logs = np.asarray(log_volumes)
    if coords.shape != (len(Z), 3) or logs.shape not in ((len(Z),), (len(Z), 1)):
        raise ValueError("Expected one log volume and one coordinate vector per atom")
    volumes = np.exp(logs.reshape(len(Z)))
    geom = MBDGeom(coords)
    return float(geom.mbd_energy_species([symbols[z] for z in Z], volumes, beta=beta))
