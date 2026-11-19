"""Conservative corrections conditioned on total charge and spin multiplicity.

Each observed (charge, 2S+1) has its own coefficients multiplying the same
distance and coordination features. Both the energy and its Cartesian
derivative therefore depend on the electronic state. The labels describe the
whole input molecule or complex, including when computing a dissociation.
"""
from operator import index

import numpy as np

from .pair import PairBaseline


def state_key(charge, multiplicity):
    """Normalize discrete quantum numbers without silently rounding user input."""
    values = []
    for name, value in (("charge", charge), ("multiplicity", multiplicity)):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be an integer")
        try:
            values.append(index(value))
        except TypeError as exc:
            raise ValueError(f"{name} must be an integer") from exc
    if values[1] < 1:
        raise ValueError("Multiplicity must be at least one")
    return tuple(values)


def validate_electrons(numbers, charge, multiplicity):
    """Check that 2S unpaired electrons can occur for this electron count."""
    electrons = int(np.sum(numbers)) - charge
    unpaired = multiplicity - 1
    if electrons < unpaired or (electrons - unpaired) % 2:
        raise ValueError(f"{electrons} electrons are incompatible with multiplicity {multiplicity}")


class ElectronicFeatures:
    """Invariant radial and local coordination features for electronic-state regression.

    A short-range radial expansion distinguishes bond lengths. Gaussian functions
    of the smooth coordination at each atom distinguish atoms with different
    numbers of bonded neighbors. Both representations stay smooth as a bond
    forms or breaks. The first terms allow different atomic and molecular offsets.
    """
    radii = {1: .31, 6: .76, 7: .71, 8: .66, 9: .57, 15: 1.07, 16: 1.05, 17: 1.02}

    def __init__(self, species, *, environment=False, radial_width=.25):
        if not np.isfinite(radial_width) or radial_width <= 0:
            raise ValueError("Radial width must be positive and finite")
        self.environment = environment
        self.species = tuple(species)
        self.radial = PairBaseline(species, [.7, .94, 1.16, 1.38, 1.6, 1.84, 2.12, 2.48],
                                   radial_width, 2.5, 3.3)
        self.coord_centers = np.arange(6)
        self.size = (1 + len(species) * (1 + len(self.coord_centers))
                     + len(self.radial.pairs) * len(self.radial.centers))

        if environment:
            # Element-resolved neighbor pairs distinguish e.g. O-C-H and O-C-O.
            self.neighbor_pairs = np.triu_indices(len(species))
            self.size += len(species) * len(self.neighbor_pairs[0])

    def _neighbor_products(self, count, species, i, j, bond, force_dir):
        """Quadratic products of element-resolved smooth coordination at each center."""
        n = len(species)
        nc = len(self.species)
        neighbors = np.zeros((n, nc))
        np.add.at(neighbors, (i, species[j]), bond)
        np.add.at(neighbors, (j, species[i]), bond)
        a, b = self.neighbor_pairs
        values = np.zeros((nc, len(a)))
        np.add.at(values, species, neighbors[:, a] * neighbors[:, b])
        jac = np.zeros((n, 3, nc, len(a)))
        for center, other in ((i, j), (j, i)):
            t = species[other, None]
            partial = (t == a) * neighbors[center][:, b] + (t == b) * neighbors[center][:, a]
            df = force_dir[:, :, None] * partial[:, None]
            np.add.at(jac, (i, slice(None), species[center]), df)
            np.add.at(jac, (j, slice(None), species[center]), -df)
        return values.ravel(), jac.reshape(count, -1)

    def design(self, z, r):
        """Compute a feature vector and its negative Jacobian in inverse angstroms."""
        radial, dradial = self.radial.design(z, r)
        species_id = np.array([self.species.index(a) for a in z])
        counts = np.bincount(species_id, minlength=len(self.species))
        i, j = np.triu_indices(len(z), 1)
        delta = r[i] - r[j]
        dist = np.linalg.norm(delta, axis=1)
        scale = np.array([self.radii[a] for a in z])
        radii = scale[i] + scale[j]
        # A logistic bond count gives a differentiable coordination measure.
        bond = .5 * (1 - np.tanh(6 * (dist / radii - 1.20)))
        coord = np.bincount(i, bond, len(z)) + np.bincount(j, bond, len(z))
        x = (coord[:, None] - self.coord_centers) / .55
        site = np.exp(-.5 * x**2)
        feat = np.zeros((len(self.species), len(self.coord_centers)))
        np.add.at(feat, species_id, site)
        slope = -x * site / .55
        jac = np.zeros((len(z), 3, *feat.shape))
        # The bond-count derivative enters the environments at both endpoints.
        force_dir = (12 * bond * (1 - bond) / (radii * dist))[:, None] * delta
        for endpoint in (i, j):
            df = force_dir[:, :, None] * slope[endpoint, None, :]
            np.add.at(jac, (i, slice(None), species_id[endpoint]), df)
            np.add.at(jac, (j, slice(None), species_id[endpoint]), -df)
        values = np.r_[1., counts, radial, feat.ravel()]
        forces = np.column_stack((np.zeros((r.size, 1 + len(counts))), dradial,
                                  jac.reshape(r.size, -1)))
        if self.environment:
            context, df = self._neighbor_products(r.size, species_id, i, j, bond, force_dir)
            values, forces = np.r_[values, context], np.column_stack((forces, df))
        return values, forces


class ElectronicCorrection:
    """A serializable table of energy coefficients indexed by exact state."""
    def __init__(self, species, states, weights, target, *, environment=False, radial_width=.25):
        self.features = ElectronicFeatures(species, environment=environment, radial_width=radial_width)
        keys = [state_key(q, m) for q, m in states]
        weights = np.asarray(weights, dtype=float)
        if (not keys or len(set(keys)) != len(keys) or not np.isfinite(weights).all()
                or weights.shape != (len(keys), self.features.size)):
            raise ValueError("Invalid electronic-state checkpoint")
        self.weights = dict(zip(keys, weights, strict=True))
        self.target = str(target)

    def predict(self, z, r, state):
        values, force = self.features.design(z, r)
        weights = self.weights[state]
        return float(values @ weights), (force @ weights).reshape(r.shape)
