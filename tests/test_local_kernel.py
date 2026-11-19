"""Check the energy derivatives and invariances used by reaction MD and frequencies."""
from io import BytesIO
from pathlib import Path
import unittest

import numpy as np

from src.forcefield import ForceField


class KernelTests(unittest.TestCase):
    def test_saved_kernels_conserve_forces_for_each_family(self):
        model_path = (Path(__file__).resolve().parents[1] / "experiments/reactivity"
                      / "matern-15d11dacaf49a610.npz")
        model = ForceField.load(model_path)
        # A distorted planar doublet avoids symmetry-induced zero derivatives.
        z = np.array([8, 1, 1, 1])
        r = np.array([[.01, .02, .03], [.96, .02, .12], [-.31, .90, -.03], [-.57, -.77, .08]])
        state = (0, 2)
        permutation = [1, 3, 0, 2]
        rotation = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
        for family in ("gaussian", "matern52", "double_matern52"):
            with self.subTest(family=family):
                model.matern.family = family
                buf = BytesIO()
                model.save(buf)
                buf.seek(0)
                saved = ForceField.load(buf)
                self.assertEqual(saved.matern.family, family)
                energy, force = model.predict(z, r, *state)
                self.assertAlmostEqual(saved.energy(z, r, *state), energy, places=9)
                shifted = (r @ rotation + [3., 1., -2.])[permutation]
                transformed, f = saved.predict(z[permutation], shifted, *state)
                self.assertAlmostEqual(energy, transformed, places=9)
                np.testing.assert_allclose(f, (force@rotation)[permutation], atol=1e-8)
                np.testing.assert_allclose(force.sum(axis=0), 0., atol=1e-9)
                h = 1e-5
                fd = [(saved.energy(z, r-h*u, *state) - saved.energy(z, r+h*u, *state))/(2*h)
                      for u in np.eye(r.size).reshape(-1, *r.shape)]
                np.testing.assert_allclose(force.ravel(), fd, atol=5e-6, rtol=1e-6)
                # At a center the distance is exactly zero. The directional
                # center basis still has a finite, generally nonzero gradient.
                kernel = saved.matern
                site = kernel.centers[8][:1]
                v = kernel.tangents[8][:1]
                _, analytic = kernel._kernel(8, site, True)
                upper, _ = kernel._kernel(8, site+h*v, False)
                lower, _ = kernel._kernel(8, site-h*v, False)
                np.testing.assert_allclose(analytic[0]@v[0], ((upper-lower)/(2*h))[0],
                                           atol=1e-6, rtol=1e-6)
