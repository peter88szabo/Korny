"""Regress two-contact symmetries, extraction order and analytic forces."""

import unittest
import numpy as np

from src.hydrogen_transfer import HydrogenTransferKernel, POWERS


class HydrogenTransferTests(unittest.TestCase):
    def test_stretched_h_cartesian_force_and_atom_permutation(self):
        radii = np.full(32, 0.75)
        radii[[1, 6, 8]] = [0.31, 0.76, 0.66]
        z = np.array([6, 1, 8, 6])
        r = np.array(
            [[0.0, 0.0, 0.0], [1.16, 0.05, 0.0], [1.92, 0.99, 0.0], [1.35, -1.42, 0.18]]
        )
        rng = np.random.default_rng(86)
        v = rng.normal(size=r.shape)
        perm = np.array([3, 1, 0, 2])
        # Equal-element endpoints and the assignment to unlike ends must both
        # survive permutations and round-trip checkpoint loading.
        for family in ("chebyshev", "krawtchouk"):
            if family == "chebyshev":
                coeff = np.eye(5)
            else:
                # The normalized binomial expectation of a balanced
                # Krawtchouk mode is x**n, expressed here in Chebyshev modes.
                coeff = np.zeros((5, 5))
                for n in range(5):
                    c = np.polynomial.chebyshev.poly2cheb(np.eye(5)[n])
                    coeff[n, : len(c)] = c
            b = HydrogenTransferKernel(
                radii, family, coeff, rng.normal(size=21 * len(POWERS))
            )
            b = HydrogenTransferKernel.load(b.fields())
            x, j = b.design(z, r, True)
            xp, jp = b.design(z[perm], r[perm], True)
            np.testing.assert_allclose(x, xp, atol=1e-13)
            np.testing.assert_allclose(
                j.reshape(4, 3, -1)[perm], jp.reshape(4, 3, -1), atol=1e-12
            )
            h = 1e-5
            numerical = (b.design(z, r + h * v)[0] - b.design(z, r - h * v)[0]) / (
                2 * h
            )
            np.testing.assert_allclose(-v.ravel() @ j, numerical, atol=7e-8)
            np.testing.assert_allclose(j.reshape(4, 3, -1).sum(0), 0, atol=2e-12)
            np.testing.assert_array_equal(b.design(z, r * 5)[0], 0)

    def test_angle_window_force_and_inactive_bend(self):
        radii = np.full(16, 0.75)
        radii[1] = 0.31
        window = np.cos(np.deg2rad([160.0, 110.0]))
        basis = HydrogenTransferKernel(
            radii, "chebyshev", np.eye(5), np.ones(21 * len(POWERS)), window
        )
        z = np.array([6, 1, 8])

        def xyz(deg):
            angle = np.deg2rad(deg)
            return np.array(
                [
                    [1.32, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [1.25 * np.cos(angle), 1.25 * np.sin(angle), 0.0],
                ]
            )

        scale = 1e-4
        x, j = basis.design(z, xyz(136.0), True)
        grad = (
            basis.design(z, xyz(136.0 + scale))[0]
            - basis.design(z, xyz(136.0 - scale))[0]
        ) / (2 * scale)
        tangent = (xyz(136.0 + scale) - xyz(136.0 - scale)) / (2 * scale)
        np.testing.assert_allclose(-tangent.ravel() @ j, grad, atol=2e-9)
        zero, forces = basis.design(z, xyz(100.0), True)
        np.testing.assert_allclose(zero, 0, atol=1e-14)
        np.testing.assert_allclose(forces, 0, atol=1e-14)
        reloaded = HydrogenTransferKernel.load(basis.fields())
        np.testing.assert_allclose(reloaded.design(z, xyz(136.0))[0], x)
