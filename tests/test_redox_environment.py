"""Check the local redox kernel's forces, symmetries and dissociation limit."""

import unittest

import numpy as np

from src.redox_environment import DIMENSION, RedoxEnvironmentKernel
from src.spin_local import DIM, GROUPS


class RedoxKernelTests(unittest.TestCase):
    def test_conservative_permutation_invariant_and_additive(self):
        rand = np.random.default_rng(812)
        elements = np.arange(119) % GROUPS
        model = RedoxEnvironmentKernel(
            elements,
            np.full(119, 0.68),
            rand.normal(size=(GROUPS, DIM, 2)),
            rand.normal(size=(GROUPS, 2)),
            rand.normal(size=DIMENSION),
        )
        z = np.array([8, 6, 1, 7, 1, 1])
        r = np.array(
            [
                [0.04, 0.1, -0.12],
                [1.34, 0.2, 0.11],
                [-0.3, 0.96, 0.0],
                [2.4, 1.19, -0.18],
                [1.28, -0.8, 0.4],
                [3.04, 0.63, 0.61],
            ]
        )
        x, j = model.design(z, r, True)
        v = rand.normal(size=r.shape)
        v /= np.linalg.norm(v)
        d = 1e-5
        fd = (model.design(z, r - d * v)[0] - model.design(z, r + d * v)[0]) / (2 * d)
        np.testing.assert_allclose(v.ravel() @ j, fd, atol=2e-8)
        ix = rand.permutation(len(z))
        rot, _ = np.linalg.qr(rand.normal(size=(3, 3)))
        y, jr = model.design(z[ix], r[ix] @ rot + 1.7, True)
        np.testing.assert_allclose(x, y, atol=2e-14)
        f = (j @ model.weights).reshape(-1, 3)
        np.testing.assert_allclose(
            (jr @ model.weights).reshape(-1, 3), f[ix] @ rot, atol=2e-12
        )
        separate = r.copy()
        separate[3:] += 30.0
        x1 = model.design(z[:3], r[:3])[0]
        x2 = model.design(z[3:], r[3:])[0]
        np.testing.assert_allclose(model.design(z, separate)[0], x1 + x2, atol=1e-13)
        loaded = RedoxEnvironmentKernel.load(model.fields())
        np.testing.assert_array_equal(loaded.design(z, r)[0], x)
