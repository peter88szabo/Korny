"""Finite-difference regression for mixed organic/heavy-element spin features."""

from types import SimpleNamespace
import unittest

import numpy as np

from src.reaction_surface import ReactionSurface


class ZeroLocalKernel:
    """Isolate the chemical and radial parts, whose Cartesian indexing differs."""

    species = (1, 6, 7, 8, 9, 15, 16, 17)
    size = 0

    def project(self, z, r, gradient):
        return None

    def design(self, sites, ncart=0):
        return np.zeros(0), np.zeros((ncart, 0))


class ReactionSurfaceTest(unittest.TestCase):
    def test_heavy_atom_forces_and_spin_follow_atom_order(self):
        z = np.array([6, 35, 8, 1])
        r = np.array(
            [[0.0, 0.0, 0.0], [-1.87, 0.1, 0.2], [1.36, 0.06, 0.0], [1.87, 0.8, 0.04]]
        )
        radii = np.ones(36)
        radii[[1, 6, 8, 35]] = [0.31, 0.76, 0.66, 1.20]
        groups = np.zeros(36, dtype=int)
        groups[[1, 6, 8, 35]] = [1, 4, 6, 7]
        surface = ReactionSurface(
            SimpleNamespace(pair=None, local=None),
            ZeroLocalKernel(),
            sorted(z),
            radii,
            groups,
            [(1, 2)],
            [],
            "test",
        )
        state = (1, 2)
        v, j = surface.design(z, r, *state, True)
        rng = np.random.default_rng(931)
        direction = rng.normal(size=r.shape)
        weights = rng.normal(size=len(v))
        h = 1e-6
        plus = surface.design(z, r + h * direction, *state)[0]
        minus = surface.design(z, r - h * direction, *state)[0]
        np.testing.assert_allclose(
            direction.ravel() @ j @ weights,
            (minus - plus) @ weights / (2 * h),
            atol=1e-7,
        )
        order = [3, 1, 0, 2]
        vv, jj = surface.design(z[order], r[order], *state, True)
        np.testing.assert_allclose(v, vv, atol=1e-12)
        np.testing.assert_allclose(
            j.reshape(len(z), 3, -1)[order], jj.reshape(len(z), 3, -1), atol=1e-11
        )
