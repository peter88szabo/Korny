"""Smooth, fragment-local charge transfer as a molecular energy correction.

A bond conductance tends continuously to zero on separation. Charges induced
by electronegativity differences minimize a quadratic energy with atomic
hardness, a softened Coulomb interaction, and a penalty for bond charge flow.
The scalar energy features describe responses to element and coordination
potentials. Their fitted mixture is an energy correction, not a population
analysis. The parent calculator handles the supplied net charge and spin.
"""

import numpy as np

from .four_body import switch


class SplitChargeResponse:
    def __init__(self, groups, radii, weights):
        self.groups = np.asarray(groups, int)
        self.radii = np.asarray(radii, float)
        self.weights = np.asarray(weights, float)
        # The first eight sources cover H and valence groups 1 through 7.
        # Six more sources couple coordination changes to the indicated group.
        self.coordination_groups = (0, 3, 4, 5, 6, 7)
        self.i, self.j = np.triu_indices(8 + len(self.coordination_groups))
        self.size = len(self.i)
        if self.weights.shape != (self.size,):
            raise ValueError("Wrong number of split-charge response coefficients")

    def design(self, z, r, forces=False):
        z, r = np.asarray(z, int), np.asarray(r, float)
        n = len(z)
        a, b = np.triu_indices(n, 1)
        dr = r[b] - r[a]
        dist = np.linalg.norm(dr, axis=1)
        if np.any(dist < 1e-8):
            raise ValueError("Coincident nuclei in charge response")
        scale = self.radii[z[a]] + self.radii[z[b]]
        x = dist / scale
        contact, dc = switch(x, 1.05, 1.85)
        g, dg = contact / 4.0, dc / (4.0 * scale)
        # L maps atom potentials to locally conserved charge flows. Each
        # disconnected covalent component has zero net induced charge.
        lap = np.zeros((n, n))
        lap[a, b] = lap[b, a] = -g
        diag = np.diag_indices(n)
        lap[diag] = -lap.sum(1)
        typ = self.groups[z]
        one_hot = (typ[:, None] == np.arange(8)).astype(float)
        cn_channels = (typ[:, None] == self.coordination_groups).astype(float)
        occ, do = switch(x, 1.00, 1.55)
        cn = np.bincount(a, occ, minlength=n) + np.bincount(b, occ, minlength=n)
        # Eight eV sets the scale of the source potentials. Referencing each
        # channel to its usual valence handles saturated and deficient sites.
        valence = np.array([1, 1, 2, 3, 4, 3, 2, 1, 0])[typ]
        mismatch = cn - valence
        env = mismatch / np.sqrt(1 + (mismatch / 2) ** 2)
        chi = 8.0 * np.column_stack([one_hot, env[:, None] * cn_channels])

        # J is positive definite. Solving for effective potentials with
        # multiple right-hand sides gives all response features in one solve.
        invr = 1 / np.sqrt(dist**2 + 1.2**2)
        interaction = np.zeros_like(lap)
        interaction[a, b] = interaction[b, a] = 14.4 * invr
        interaction[diag] = 7.0 + 14.4 / 1.2
        u = np.linalg.solve(np.eye(n) + interaction @ lap, chi)
        q = -lap @ u
        i, j = self.i, self.j
        multiplicity = np.where(i == j, 1.0, 2.0)
        response = 0.5 * (chi.T @ q)[i, j] * multiplicity
        if not forces:
            return response, None

        # At the variational minimum the envelope theorem eliminates dq/dr.
        # Include the explicit geometry dependence of chi, J and L separately.
        dchi = np.zeros_like(chi)
        dchi[:, 8:] = 8.0 * cn_channels / (1 + (mismatch[:, None] / 2) ** 2) ** 1.5

        def endpoint(atom):
            return dchi[atom][:, i] * q[atom][:, j] + dchi[atom][:, j] * q[atom][:, i]

        de = (do / scale)[:, None] * (endpoint(a) + endpoint(b))
        dj = -14.4 * dist * invr**3
        de += dj[:, None] * (q[a][:, i] * q[b][:, j] + q[b][:, i] * q[a][:, j])
        du = u[a] - u[b]
        de -= dg[:, None] * du[:, i] * du[:, j]
        pair_force = (dr / dist[:, None])[:, :, None] * (0.5 * de * multiplicity)[
            :, None, :
        ]
        jac = np.zeros((n, 3, self.size))
        np.add.at(jac, a, pair_force)
        np.add.at(jac, b, -pair_force)
        return response, jac.reshape(r.size, self.size)

    def fields(self):
        return dict(
            split_charge_weights=self.weights,
            split_charge_groups=self.groups,
            split_charge_radii=self.radii,
        )

    @classmethod
    def load(cls, data):
        return cls(
            data["split_charge_groups"],
            data["split_charge_radii"],
            data["split_charge_weights"],
        )
