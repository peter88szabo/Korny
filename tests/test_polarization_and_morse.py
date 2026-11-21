"""The optional physical energy terms must retain symmetry and conservative forces."""

import unittest

import numpy as np

from src.morse_bends import MorseBendBasis
from src.split_charge import SplitChargeResponse


class PhysicalBasisTests(unittest.TestCase):
    def test_energy_derivatives_and_dissociation(self):
        groups, radii = np.zeros(9, int), np.ones(9)
        groups[[1, 6, 7, 8]] = [0, 4, 5, 6]
        radii[[1, 6, 7, 8]] = [0.31, 0.76, 0.71, 0.66]
        z = np.array([1, 6, 7, 8])
        r = np.array(
            [
                [0.84, 0.15, 0.06],
                [0.0, 0.0, 0.0],
                [-0.7, 1.05, 0.07],
                [-0.8, -0.97, 0.2],
            ]
        )
        v = np.random.default_rng(721).normal(size=r.shape)
        for model in (
            MorseBendBasis(groups, radii, np.zeros(342)),
            SplitChargeResponse(groups, radii, np.zeros(105)),
        ):
            with self.subTest(model=type(model).__name__):
                x, j = model.design(z, r, True)
                step = 1e-6
                fd = (
                    model.design(z, r + step * v)[0] - model.design(z, r - step * v)[0]
                ) / (2 * step)
                np.testing.assert_allclose(fd, -v.ravel() @ j, atol=2e-6, rtol=1e-5)
                np.testing.assert_allclose(
                    model.design(z[::-1], r[::-1])[0], x, atol=1e-10
                )
                np.testing.assert_allclose(
                    j.reshape(len(z), 3, -1).sum(0), 0, atol=1e-10
                )
                isolated = model.design(
                    z, np.arange(4)[:, None] * np.array([[100.0, 0.0, 0.0]]), True
                )
                for field in isolated:
                    np.testing.assert_allclose(field, 0, atol=1e-12)
