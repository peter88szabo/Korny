"""Short-range donor and acceptor environment features for a scalar energy.

Two local chemical scores are learned from molecular ionization energies
and electron affinities. Their variation within each element group modulates
distance features for atom pairs. Scores describe fitted environments and
have no unique interpretation as isolated-atom ionization energies. A smooth
cutoff gives additive fragment limits when every interfragment neighborhood
lies outside the cutoff.
"""

import numpy as np

from .four_body import switch
from .spin_local import DIM, GROUPS, local_sites

RADII = np.array([1.0, 1.42, 2.0])
WIDTHS = np.array([0.17, 0.24, 0.34])
PAIRS = GROUPS * (GROUPS + 1) // 2
DIMENSION = PAIRS * len(RADII) * 3


class RedoxEnvironmentKernel:
    def __init__(self, groups, radii, scores, bias, weights):
        self.groups = np.asarray(groups, int)
        self.radii = np.asarray(radii, float)
        self.scores = np.asarray(scores, float)
        self.bias = np.asarray(bias, float)
        self.weights = np.asarray(weights, float)
        self.size = DIMENSION
        if (
            self.scores.shape != (GROUPS, DIM, 2)
            or self.bias.shape != (GROUPS, 2)
            or self.weights.shape != (self.size,)
        ):
            raise ValueError("Invalid redox-environment readout")

    def design(self, z, r, forces=False):
        # Reuse exactly the geometry map used for molecular redox supervision.
        types, x, jac = local_sites(self.groups, self.radii, z, r, forces)
        scores = np.tanh(
            np.einsum("nd,nda->na", x, self.scores[types]) + self.bias[types]
        )
        i, j = np.triu_indices(len(z), 1)
        vec = r[i] - r[j]
        d = np.linalg.norm(vec, axis=1)
        scale = self.radii[z[i]] + self.radii[z[j]]
        dist = d / scale
        active = dist < 3.3
        i, j, vec, d, scale, dist = (v[active] for v in (i, j, vec, d, scale, dist))
        top = np.maximum(types[i], types[j])
        pair = top * (top + 1) // 2 + np.minimum(types[i], types[j])
        delta = (dist[:, None] - RADII) / WIDTHS
        g = np.exp(-0.5 * delta**2)
        fc, dfc = switch(dist, 2.6, 3.3)
        shell = g * fc[:, None]
        a, b = scores[i], scores[j]
        # The first two moments distinguish donor-rich and acceptor-rich
        # bonds. The third couples a donor environment across a bond to an
        # acceptor environment. Exchanging the two atoms leaves all invariant.
        h = np.column_stack(
            (
                a[:, 0] + b[:, 0],
                a[:, 1] + b[:, 1],
                a[:, 0] * b[:, 1] + b[:, 0] * a[:, 1],
            )
        )
        value = np.zeros((PAIRS, len(RADII), 3))
        np.add.at(value, pair, shell[:, :, None] * h[:, None, :])
        if not forces:
            return value.ravel(), None
        ds = (
            np.einsum("nda,ndb->nab", self.scores[types], jac)
            * (1 - scores**2)[:, :, None]
        )
        da, db = ds[i], ds[j]
        dh = np.stack(
            (
                da[:, 0] + db[:, 0],
                da[:, 1] + db[:, 1],
                da[:, 0] * b[:, 1, None]
                + db[:, 1] * a[:, 0, None]
                + db[:, 0] * a[:, 1, None]
                + da[:, 1] * b[:, 0, None],
            ),
            axis=1,
        )
        # Accumulate by pair type before allocating a Cartesian derivative.
        # This keeps memory proportional to the local pair count times N for
        # the two site scores, avoiding an extra dimension for all shells.
        der = np.zeros((*value.shape, r.size))
        radial = (dfc[:, None] * g - shell * delta / WIDTHS) / (scale * d)[:, None]
        for k in range(len(RADII)):
            for c in range(3):
                part = der[:, k, c]
                np.add.at(part, pair, dh[:, c] * shell[:, k, None])
                term = (radial[:, k] * h[:, c])[:, None] * vec
                for xyz in range(3):
                    np.add.at(part, (pair, 3 * i + xyz), term[:, xyz])
                    np.add.at(part, (pair, 3 * j + xyz), -term[:, xyz])
        return value.ravel(), -der.reshape(self.size, r.size).T

    def fields(self):
        return {
            "redox_groups": self.groups,
            "redox_radii": self.radii,
            "redox_scores": self.scores,
            "redox_bias": self.bias,
            "redox_weights": self.weights,
        }

    @classmethod
    def load(cls, d):
        return cls(
            *(d["redox_" + s] for s in ("groups", "radii", "scores", "bias", "weights"))
        )
