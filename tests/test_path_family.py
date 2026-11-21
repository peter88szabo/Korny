"""Check that the local class probabilities contribute to the Cartesian force."""

import numpy as np

from src.local import descriptor_layout
from src.path_family import PathFamilyKernel
from src.periodic_kernel import PeriodicKernel


def test_family_mixture_coordinate_response():
    rng = np.random.default_rng(988)
    groups = np.zeros(10, int)
    groups[6] = 4
    radii = np.ones(10)
    radii[[1, 6]] = [0.31, 0.76]
    shells = np.array([[0.95, 1.5], [0.20, 0.35]])
    dim = sum(descriptor_layout(9, 2, 2)[0])
    proj = {k: rng.normal(0, 0.12, (dim, 4)) for k in (0, 4)}
    centers = {k: rng.normal(0, 0.3, (5, 4)) for k in proj}
    banks = [
        PeriodicKernel(
            groups, radii, proj, centers, shells, (2.8, 3.8), length, deg, np.zeros(10)
        )
        for length, deg in ((0.85, 3), (1.35, 6))
    ]
    native = {k: rng.normal(0, 0.8, (10, 3)) for k in proj}
    logits = {k: rng.normal(0, 0.8, (11, 3)) for k in proj}
    model = PathFamilyKernel.load(
        PathFamilyKernel(banks, native, logits, np.zeros(5)).fields()
    )
    z = np.array([6, 6, 1, 1])
    r = np.array(
        [[0.0, 0.0, 0.0], [1.41, 0.07, 0.0], [-0.52, 0.88, 0.11], [1.92, 0.89, -0.09]]
    )
    e, force = model.design_state(z, r, 0, 2, True)
    v = rng.normal(size=r.shape)
    eps = 1e-6
    m, _ = model.design_state(z, r - eps * v, 0, 2)
    p, _ = model.design_state(z, r + eps * v, 0, 2)
    np.testing.assert_allclose(force.T @ v.ravel(), (m - p) / (2 * eps), atol=1e-8)
    perm = [3, 1, 0, 2]
    np.testing.assert_allclose(
        model.design_state(z[perm], r[perm], 0, 2)[0], e, atol=1e-12
    )
    pair, _ = model.design_state(np.tile(z, 2), np.vstack((r, r + 20.0)), 0, 2)
    np.testing.assert_allclose(pair, 2 * e, atol=1e-12)
    assert not np.any(model.design_state(z, r, 0, 1)[0])
