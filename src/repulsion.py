"""Screened nuclear repulsion at distances shorter than ordinary chemical bonds.

Use the published element-pair NLH screening coefficients (Nordlund, Lehtola
and Hobler, DOI 10.1103/PhysRevA.111.032818). The polynomial switch is specific
to Korny. It prevents adding a second copy of equilibrium short-range forces to
a backbone already trained on full bare-PBE0 energies and gradients.
"""
from itertools import combinations_with_replacement

import numpy as np


class NuclearRepulsion:
    """Evaluate each unordered pair once, with distances in Å and energy in eV."""
    coulomb = 14.39964548  # eV Å
    radii_A = {1:.31, 6:.76, 7:.71, 8:.66, 9:.57, 15:1.07, 16:1.05, 17:1.02}

    def __init__(self, species, coefficients, radii=None):
        pairs = list(combinations_with_replacement(sorted(species), 2))
        self.coefficients = np.array(coefficients, dtype=float)
        if self.coefficients.shape != (len(pairs), 6) or not np.isfinite(self.coefficients).all():
            raise ValueError("Expected three finite NLH amplitude/exponent pairs per element pair")
        self.table = np.zeros((max(species)+1,max(species)+1,6))
        self.radii = np.zeros(max(species)+1)
        radii = self.radii_A if radii is None else radii
        for (a,b), row in zip(pairs,self.coefficients,strict=True):
            self.table[a,b] = self.table[b,a] = row
            self.radii[a],self.radii[b] = radii[a],radii[b]

    def predict(self, z, r, *, include_atoms=None):
        """Return E and −dE/dR, including the derivative of the joining function."""
        i,j = np.triu_indices(len(z),1)
        if include_atoms is not None:
            # An extended surface can add only pairs outside the parent core.
            active = include_atoms[i] | include_atoms[j]
            i, j = i[active], j[active]
        v = r[j]-r[i]
        length = np.linalg.norm(v,axis=1)
        scale = self.radii[z[i]]+self.radii[z[j]]
        # Full core below 60% of the covalent-radius sum, zero above 82%.
        # This engineering choice is verified separately on compressed pairs
        # and on relaxed molecules. It is not a fit to molecular frequencies.
        active = length < .82*scale
        force = np.zeros_like(r)
        if not np.any(active):
            return 0.,force
        i,j,v,length,scale = (a[active] for a in (i,j,v,length,scale))
        if np.any(length < 1e-10):
            raise ValueError("Coincident nuclei")
        t = np.clip((length/scale-.60)/.22,0.,1.)
        switch = 1+t**4*(-35+t*(84+t*(-70+20*t)))
        slope = -140*t**3*(1-t)**3/(.22*scale)
        coeff = self.table[z[i],z[j]]
        decay = coeff[:,1::2]
        terms = coeff[:,::2]*np.exp(-decay*length[:,None])
        factor = self.coulomb*z[i]*z[j]/length
        potential = factor*terms.sum(axis=1)
        derivative = -potential/length-factor*(decay*terms).sum(axis=1)
        outward = -(slope*potential+switch*derivative)
        fij = (outward/length)[:,None]*v
        np.add.at(force,j,fij)
        np.add.at(force,i,-fij)
        return float(switch@potential),force
