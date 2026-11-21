"""Check conservation and the Cartesian force of a doublet spin-transfer edge."""

import numpy as np

from src.spin_exchange import PAIRS, RADII, SpinExchangeKernel, population_design
from src.spin_local import DIM, GROUPS


def test_edge_transfer_and_conservative_energy():
    rng = np.random.default_rng(1291)
    groups = np.zeros(10, int)
    groups[[6, 8]] = [4, 6]
    radii = np.ones(10)
    radii[[1, 6, 8]] = [0.31, 0.76, 0.66]
    logits, polar = rng.normal(0, 0.3, (2, 3, GROUPS, DIM))
    exchange = rng.normal(0, 0.2, (3, PAIRS, len(RADII)))
    basis = SpinExchangeKernel(
        groups, radii, logits, polar, np.zeros(GROUPS * DIM), exchange
    )
    basis = SpinExchangeKernel.load(basis.fields())
    z = np.array([6, 8, 1, 1])
    r = np.array([[0.0, 0.0, 0.0], [1.4, 0.1, 0.0], [-0.3, 1.0, 0.1], [20.0, 0.0, 0.0]])
    types, sites, dx = basis.sites(z, r, True)
    for q in (-1, 0, 1):
        spin, ds = basis.population(types, sites, q, 2, dx)
        flow, jac = basis.redistribution(z, r, types, spin, q, ds)
        design = population_design(types, z, r, radii, spin)
        np.testing.assert_allclose(
            design @ exchange[basis.charge_channel(q)].ravel(), flow, atol=1e-12
        )
        np.testing.assert_allclose(jac.sum(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose((spin + flow).sum(), 1.0, atol=1e-12)
        assert flow[-1] == 0.0
        x, force = basis.design_state(z, r, q, 2, True)
        v = rng.normal(size=r.shape)
        eps = 2e-6
        minus, _ = basis.design_state(z, r - eps * v, q, 2)
        plus, _ = basis.design_state(z, r + eps * v, q, 2)
        np.testing.assert_allclose(
            force.T @ v.ravel(), (minus - plus) / (2 * eps), atol=1e-8
        )
        perm = [3, 2, 0, 1]
        np.testing.assert_allclose(
            basis.design_state(z[perm], r[perm], q, 2)[0], x, atol=1e-12
        )
    for m in (1, 3):
        x, force = basis.design_state(z, r, 0, m, True)
        assert not np.any(x) and not np.any(force)
