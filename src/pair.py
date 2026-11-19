"""Smooth element-pair radial expansion in eV and angstroms."""
from itertools import combinations_with_replacement

import numpy as np


class PairBaseline:
    """Smooth pair potential: one linear radial expansion per element pair."""
    def __init__(self, species, centers, width, ron, rc):
        self.pairs = list(combinations_with_replacement(sorted(species), 2))
        self.centers = np.array(centers)
        self.width, self.ron, self.rc = width, ron, rc
        self.species = tuple(species)
        self.group = np.full((max(species)+1, max(species)+1), -1, dtype=int)
        for k, (a, b) in enumerate(self.pairs):
            self.group[a, b] = self.group[b, a] = k

    def _radial(self, z, r):
        i, j = np.triu_indices(len(z), 1)
        displacement = r[i]-r[j]
        dist = np.linalg.norm(displacement, axis=1)
        inside = dist < self.rc
        i, j, displacement, dist = (a[inside] for a in (i, j, displacement, dist))
        if np.any(dist < 1e-10):
            raise ValueError("Coincident nuclei")
        groups = self.group[z[i], z[j]]
        delta = (dist[:, None]-self.centers)/self.width
        radial = np.exp(-0.5*delta**2)
        t = np.clip((dist-self.ron)/(self.rc-self.ron), 0., 1.)
        cutoff = 1. + t**4*(-35. + t*(84. + t*(-70. + 20.*t)))
        slope = -140.*t**3*(1.-t)**3/(self.rc-self.ron)
        basis = radial*cutoff[:, None]
        derivative = radial*slope[:, None] - basis*delta/self.width
        return i, j, groups, displacement/dist[:, None], basis, derivative

    def design(self, z, r):
        """Return pair energy features and their negative Cartesian derivatives.

        Each element pair has its own radial expansion. Accumulating the two
        opposite force contributions per pair preserves total linear momentum.
        """
        i, j, groups, direction, basis, derivative = self._radial(z, r)
        energy = np.zeros((len(self.pairs), len(self.centers)))
        force = np.zeros((len(z), len(self.pairs), 3, len(self.centers)))
        np.add.at(energy, groups, basis)
        fij = -direction[:, :, None]*derivative[:, None, :]
        np.add.at(force, (i, groups), fij)
        np.add.at(force, (j, groups), -fij)
        return energy.ravel(), force.transpose(0, 2, 1, 3).reshape(r.size, -1)

    def fit(self, records, ridge, force_weight=4.):
        design, target = [], []
        for z, r, e, f in records:
            row, forces = self.design(z, r)
            scale = force_weight/np.sqrt(r.size)
            design.append(np.vstack((row, scale*forces)))
            target.append(np.r_[e, scale*f.ravel()])
        n = design[0].shape[1]
        self.weights = np.linalg.lstsq(np.vstack([*design, np.sqrt(ridge)*np.eye(n)]),
                                       np.concatenate([*target, np.zeros(n)]), rcond=None)[0]

    def predict(self, z, r):
        """Contract only the active pair weights, avoiding a full design matrix."""
        i, j, groups, direction, basis, derivative = self._radial(z, r)
        w = self.weights.reshape(len(self.pairs), -1)[groups]
        force = np.zeros_like(r, dtype=float)
        fij = -np.einsum("ij,ij->i", derivative, w)[:, None]*direction
        np.add.at(force, i, fij)
        np.add.at(force, j, -fij)
        return np.einsum("ij,ij->", basis, w), force

    def save(self, file):
        np.savez(file, species=self.species, centers=self.centers, width=self.width,
                 ron=self.ron, rc=self.rc, weights=self.weights)

    @classmethod
    def load(cls, file):
        with np.load(file, allow_pickle=False) as a:
            obj = cls(a["species"].tolist(), a["centers"], float(a["width"]),
                      float(a["ron"]), float(a["rc"]))
            obj.weights = a["weights"]
        return obj
