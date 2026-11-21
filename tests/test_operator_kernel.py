"""Check that the finite operator bases give conservative deployable energies."""

import unittest

import numpy as np

from experiments.full_public.trial_operator_kernels import FAMILIES, build
from src.local import descriptor_layout
from src.operator_kernel import OperatorKernel
from src.reactive_basis import ReactiveBasis


class OperatorKernelTests(unittest.TestCase):
    def test_cartesian_derivatives_and_reload_for_all_families(self):
        rng = np.random.default_rng(731)
        dim = sum(descriptor_layout(1, 2, 2)[0])
        base = ReactiveBasis(
            (1,),
            (2, 2, 1.8, 3.2),
            {1: rng.normal(size=(dim, 18))},
            {1: rng.normal(size=(95, 18))},
            {1: 2},
            1,
            1,
            np.zeros(95),
        )
        z = np.array([1, 1, 1])
        r = np.array([[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [-0.25, 0.91, 0.12]])
        v = rng.normal(size=r.shape)
        for name in (*FAMILIES, "tensor_legendre"):
            with self.subTest(family=name):
                basis = build(name, 0.8, base)
                basis = OperatorKernel.load(basis.fields())
                energy, jac = basis.design(z, r, True)
                np.testing.assert_allclose(energy, basis.design(z, r)[0], atol=1e-12)
                np.testing.assert_allclose(basis.design(z[:1], r[:1])[0], 0, atol=1e-12)
                h = 1e-5
                fd = (basis.design(z, r + h * v)[0] - basis.design(z, r - h * v)[0]) / (
                    2 * h
                )
                np.testing.assert_allclose(fd, -v.ravel() @ jac, rtol=3e-5, atol=2e-6)
                np.testing.assert_allclose(jac.reshape(3, 3, -1).sum(0), 0, atol=2e-12)
