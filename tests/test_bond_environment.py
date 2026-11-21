"""Guard force propagation, extensivity, and homonuclear symmetry for reactive features."""

import unittest

import numpy as np

from src.bond_environment import BondBasis
from src.four_body import four_body


class BondEnvironmentTests(unittest.TestCase):
    def test_bridge_forces_permutation_and_separated_fragments(self):
        # A shared hydrogen and a skew O-C-C arm stress the switch derivatives,
        # homonuclear symmetrization, and four-body torsion contractions.
        z = np.array([8, 1, 8, 6, 6, 1, 1, 7])
        r = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.04, 0.13, 0.02],
                [2.31, 0.10, 0.04],
                [-0.65, 1.16, 0.28],
                [-1.91, 1.8, 0.16],
                [-0.37, 1.61, 1.23],
                [2.56, -0.62, 0.60],
                [-2.68, 1.72, 1.35],
            ]
        )
        rng = np.random.default_rng(81)
        rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        order = rng.permutation(len(z))
        direction = rng.normal(size=r.shape)
        for design in (
            BondBasis.generate(12, 0.8).design,
            BondBasis.generate(12, 0.8, True).design,
            four_body,
        ):
            with self.subTest(design=design):
                v, force = design(z, r, True)
                v2, f2 = design(z[order], r[order] @ rotation + 3, True)
                np.testing.assert_allclose(v, v2, atol=1e-11)
                transformed = np.einsum(
                    "iac,ab->ibc", force.reshape(len(z), 3, -1)[order], rotation
                )
                np.testing.assert_allclose(
                    f2.reshape(transformed.shape), transformed, atol=1e-10
                )
                fd = (
                    design(z, r - 1e-5 * direction)[0]
                    - design(z, r + 1e-5 * direction)[0]
                ) / 2e-5
                np.testing.assert_allclose(fd, direction.ravel() @ force, atol=1e-6)
                distant = np.vstack((r, r + 30))
                both, fb = design(np.tile(z, 2), distant, True)
                np.testing.assert_allclose(both, 2 * v, atol=1e-11)
                np.testing.assert_allclose(fb, np.tile(force, (2, 1)), atol=1e-10)
