"""The new local derivative kernel must supply a smooth conservative energy."""

import numpy as np

from src.local import descriptor_layout
from src.path_hermite import PathHermiteKernel


def test_path_features_are_invariant_and_have_cartesian_forces():
    rng = np.random.default_rng(107)
    species = (1, 6, 8)
    spec = (2, 1, 1.8, 3.3)
    dim = sum(descriptor_layout(len(species), *spec[:2])[0])
    projections = {s: rng.normal(0, 0.12, (dim, 5)) for s in species}
    centers = {}
    for s in species:
        c, t = rng.normal(0, 0.4, (2, 3, 5))
        t /= np.linalg.norm(t, axis=1)[:, None]
        centers[s] = np.column_stack((c, t, [0.4, 0.7, 1.1], [0, 1, 0]))
    model = PathHermiteKernel(
        species,
        spec,
        projections,
        centers,
        dict.fromkeys(species, 2),
        1.0,
        1.0,
        np.zeros(9),
    )
    model = PathHermiteKernel.load(model.fields())
    z = np.array([6, 8, 1, 1])
    r = np.array(
        [[0.0, 0.0, 0.0], [1.28, 0.12, 0.0], [-0.51, 0.9, 0.07], [-0.46, -0.83, 0.21]]
    )
    x, jac = model.design(z, r, True)
    v = rng.normal(size=r.shape)
    eps = 2e-6
    minus, _ = model.design(z, r - eps * v)
    plus, _ = model.design(z, r + eps * v)
    np.testing.assert_allclose(jac.T @ v.ravel(), (minus - plus) / (2 * eps), atol=1e-8)
    perm = [2, 0, 3, 1]
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    np.testing.assert_allclose(
        model.design(z[perm], r[perm] @ rotation)[0], x, atol=1e-12
    )
    doubled, _ = model.design(np.tile(z, 2), np.vstack((r, r + 25.0)))
    np.testing.assert_allclose(doubled, 2 * x, atol=1e-12)
    np.testing.assert_allclose(model.design(z[:1], r[:1])[0], 0.0, atol=1e-12)
