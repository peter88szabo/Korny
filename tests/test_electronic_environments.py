"""Check invariants, charge conservation and analytic forces through electronic features."""

import numpy as np
import pytest
from src.atomic_states import StateFeatures, AtomicStateKernel
from src.orbital_moments import OrbitalMomentKernel, WIDTH
from src.periodic_kernel import PeriodicKernel
from src.local import descriptor_layout


def make_kernel(kind):
    rng = np.random.default_rng(814)
    groups = np.zeros(10, dtype=int)
    groups[6:9] = [4, 5, 6]
    radii = np.ones(10)
    radii[[1, 6, 7, 8]] = [0.31, 0.76, 0.71, 0.66]
    keys = (0, 4, 6)
    if kind == "orbital":
        proj = {g: rng.normal(0, 0.12, (WIDTH, 4)) for g in keys}
        c = {g: rng.normal(0, 0.4, (4, 4)) for g in keys}
        return OrbitalMomentKernel.load(
            OrbitalMomentKernel(
                groups, radii, c, proj, 0.92, rng.normal(size=12)
            ).fields()
        )
    shells = np.array([[1.0, 1.5], [0.25, 0.4]])
    width = sum(descriptor_layout(9, 2, 2)[0])
    proj = {g: rng.normal(0, 0.15, (width, 4)) for g in keys}
    centers = {g: rng.normal(0, 0.25, (3, 4)) for g in keys}
    density = PeriodicKernel(
        groups, radii, proj, centers, shells, (2.8, 3.8), 0.9, 3, np.zeros(9)
    )
    teacher = StateFeatures(density, {g: rng.normal(0, 0.2, (7, 5)) for g in keys})
    env = {g: rng.normal(0, 0.13, (20, 6)) for g in keys}
    c = {g: rng.normal(0, 0.28, (4, 10)) for g in keys}
    return AtomicStateKernel.load(
        AtomicStateKernel(teacher, c, env, 0.92, rng.normal(size=12)).fields()
    )


@pytest.mark.parametrize("kind", ["state", "orbital"])
def test_electronic_kernel_cartesian_response(kind):
    model = make_kernel(kind)
    z = np.array([6, 6, 8, 1, 1])
    r = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.36, 0.3, 0.07],
            [1.2, 1.68, -0.15],
            [-0.68, -0.8, 0.06],
            [0.18, 1.74, 0.41],
        ]
    )
    rng = np.random.default_rng(27)
    v = rng.normal(size=r.shape)
    if kind == "state":
        for q, m in ((0, 1), (1, 2), (-1, 2)):
            u, _ = model.teacher.projected(z, r, False)
            fields = model.teacher.context(z, r, q, m, u)[2]
            np.testing.assert_allclose(fields[:, :2].sum(axis=0), q, atol=1.0e-12)

        def call(z, r, force=False):
            return model.design_state(z, r, 1, 2, force)

        assert np.linalg.norm(call(z, r)[0] - model.design(z, r)[0]) > 0.001
    else:
        call = model.design
    a, j = call(z, r, True)
    step = 2.0e-6
    fd = (call(z, r - step * v)[0] - call(z, r + step * v)[0]) / (2 * step)
    np.testing.assert_allclose(j.T @ v.ravel(), fd, atol=1.0e-7)
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    perm = rng.permutation(len(z))
    b, k = call(z[perm], r[perm] @ rotation - 1.9, True)
    np.testing.assert_allclose(a, b, atol=1.0e-12)
    np.testing.assert_allclose(
        k.reshape(len(z), 3, -1),
        np.einsum("nik,ij->njk", j.reshape(len(z), 3, -1)[perm], rotation),
        atol=1.0e-10,
    )
    if kind == "orbital":
        pair, _ = call(np.tile(z, 2), np.vstack((r, r + 30)))
        np.testing.assert_allclose(pair, 2 * a, atol=1.0e-12)


def test_competing_bonds_are_smooth_through_symmetric_transfer():
    from src.valence_competition import CompetitionKernel, events, WIDTH as width

    rng = np.random.default_rng(712)
    z = np.array([8, 1, 8, 6])
    r = np.array(
        [[-1.23, 0.03, 0.0], [0.01, 0.03, 0.09], [1.27, 0.01, 0.06], [2.35, 1.05, 0.03]]
    )
    groups = np.zeros(10, int)
    groups[6] = 4
    groups[8] = 6
    radii = np.ones(10)
    radii[[1, 6, 8]] = [0.31, 0.76, 0.66]
    keys = np.unique(events(z, r, groups, radii)[0])
    proj = {k: rng.normal(0, 0.2, (width, 6)) for k in keys}
    centers = {k: rng.normal(0, 0.3, (4, 6)) for k in keys}
    model = CompetitionKernel.load(
        CompetitionKernel(
            groups, radii, centers, proj, 0.9, rng.normal(size=4 * len(keys))
        ).fields()
    )
    a, j = model.design(z, r, True)
    direction = rng.normal(size=r.shape)
    fd = (
        model.design(z, r - 1.0e-6 * direction)[0]
        - model.design(z, r + 1.0e-6 * direction)[0]
    ) / 2.0e-6
    np.testing.assert_allclose(j.T @ direction.ravel(), fd, atol=2.0e-7)
    perm = [2, 1, 0, 3]
    np.testing.assert_allclose(model.design(z[perm], r[perm])[0], a, atol=1.0e-12)


@pytest.mark.parametrize("q,m", [(0, 1), (0, 3), (1, 2)])
def test_spin_response_preserves_occupations_and_rotation(q, m):
    from src.occupied_orbitals import environments

    z = np.array([6, 8, 1, 1])
    r = np.array(
        [
            [0.07, 0.09, 0.0],
            [1.31, -0.01, 0.05],
            [-0.45, 0.91, 0.02],
            [-0.54, -0.93, 0.05],
        ]
    )
    groups = np.zeros(10, int)
    groups[6] = 4
    groups[8] = 6
    radii = np.ones(10)
    radii[[1, 6, 8]] = [0.31, 0.76, 0.66]
    x, j = environments(z, r, q, m, groups, radii, True)
    np.testing.assert_allclose(x[:, [0, 8]].sum(), 12 - q, atol=2.0e-12)
    np.testing.assert_allclose(x[:, [1, 9]].sum(), m - 1, atol=2.0e-12)
    rng = np.random.default_rng(312)
    dr = rng.normal(size=r.shape)
    fd = (
        environments(z, r + 1.0e-6 * dr, q, m, groups, radii)[0]
        - environments(z, r - 1.0e-6 * dr, q, m, groups, radii)[0]
    ) / 2.0e-6
    np.testing.assert_allclose(j @ dr.ravel(), fd, atol=2.0e-7)
    rot = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    np.testing.assert_allclose(
        environments(z, r @ rot, q, m, groups, radii)[0], x, atol=2.0e-12
    )


@pytest.mark.parametrize("q,m", [(0, 1), (0, 3), (1, 2)])
def test_directional_bond_population_forces_rotate_with_molecule(q, m):
    """Detect index transpositions in polar and spin-resolved off-diagonal blocks."""
    from src.orbital_bonds import environments

    z = np.array([6, 8, 1, 1])
    r = np.array(
        [[0, 0, 0], [1.23, 0.15, 0.01], [-0.5, 0.86, 0.1], [-0.56, -0.82, 0.08]]
    )
    groups = np.zeros(10, int)
    groups[[6, 8]] = [4, 6]
    radii = np.ones(10)
    radii[[1, 6, 8]] = [0.31, 0.76, 0.66]

    def env(z, r, forces=False):
        return environments(z, r, q, m, groups, radii, forces)

    values, jac = env(z, r, True)
    rng = np.random.default_rng(219)
    dr = rng.normal(size=r.shape)
    fd = (env(z, r + 1.0e-6 * dr)[0] - env(z, r - 1.0e-6 * dr)[0]) / 2.0e-6
    np.testing.assert_allclose(jac @ dr.ravel(), fd, atol=3.0e-7)
    perm = [2, 1, 3, 0]
    rot = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    np.testing.assert_allclose(
        env(z[perm], r[perm] @ rot + 1.3)[0], values[perm], atol=1.0e-11
    )


def test_compact_kernel_individual_support_and_atomic_reference():
    """A search-tree bin can contain inside, outside and exactly-on-cutoff centers."""
    from src.compact_kernel import CompactKernel

    spec = dict(nr=1, lmax=0, ron=2.0, rc=3.0, inverse_radius=True, species=(1,))
    model = CompactKernel(
        (1,),
        spec,
        {1: np.eye(1)},
        {1: np.array([[0.2], [1.0], [0.5]])},
        {1: np.array([0.6, 0.5, 0.4])},
        np.array([1.0, 2.0, 3.0]),
    )
    model = CompactKernel.load(model.fields())
    x = np.array([[0.0], [0.5], [0.7], [0.9]])
    owner = np.array([0, 1, 1, 2])
    phi = lambda u: (
        np.maximum(0.0, 1.0 - ((u - model.centers[1].T) / model.radii[1]) ** 2) ** 4
    )
    expected = np.zeros((3, 3))
    np.add.at(expected, owner, phi(x) - phi(0.0))
    np.testing.assert_allclose(
        model.sparse(1, x, owner, 3).toarray(), expected, atol=2.0e-15
    )


def test_stretch_equilibrium_and_environmental_force():
    """An equilibrium spring changes curvature while exerting zero first-order force."""
    from src.bond_environment import PAIRS
    from src.bond_order_kernel import environments
    from src.local_stretch import StretchKernel

    z = np.array([6, 1, 1, 8])
    r = np.array(
        [[0.0, 0.0, 0.0], [1.1, 0.1, 0.04], [-0.55, 0.90, -0.15], [-0.4, -1.25, 0.2]]
    )
    k = PAIRS.index((1, 6))
    context = next(x[:, 1:] for key, x, *other in environments(z, r) if key == k)
    model = StretchKernel(
        {k: context.copy()}, {k: np.array([1.08, 1.12])}, 0.12, np.ones(2)
    )
    model = StretchKernel.load(model.fields())
    v = np.random.default_rng(712).normal(size=r.shape)
    e, j = model.design(z, r, True)
    fd = (
        model.design(z, r - 1.0e-6 * v)[0] - model.design(z, r + 1.0e-6 * v)[0]
    ) / 2.0e-6
    np.testing.assert_allclose(j.T @ v.ravel(), fd, atol=1.0e-8)
    assert (e > 0).all()
    z = np.array([1, 6])
    r = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])
    iso = StretchKernel({k: np.zeros((1, 24))}, {k: np.array([1.1])}, 0.12, np.ones(1))
    val, force = iso.design(z, r, True)
    np.testing.assert_allclose(val, 0.0, atol=1.0e-20)
    np.testing.assert_allclose(force, 0.0, atol=1.0e-12)


def test_reactive_hessian_response_uses_signed_curvature_and_transverse_forces():
    """A constant parent and conservative harmonic correction exactly recover the target."""
    from experiments.full_public.trial_stationary_response import stationary_rows

    z = np.array([6, 8, 1, 1])
    r = np.array(
        [[0.0, 0.0, 0.0], [1.3, 0.1, 0.05], [-0.4, 0.9, 0.03], [-0.6, -0.8, -0.05]]
    )
    rng = np.random.default_rng(246)
    h = rng.normal(size=(r.size, r.size))
    h = (h + h.T) * 0.5
    f0 = rng.normal(size=r.size)

    def columns(x):
        # The parent supplies one quarter of the Hessian and has nonzero force
        # at the archived geometry. The trainable column supplies its complement.
        delta = (x - r).ravel()
        return np.column_stack((-0.75 * (h @ delta) - f0, -0.25 * (h @ delta) + f0))

    a = stationary_rows(z, r, h, columns)
    np.testing.assert_allclose(a[:, 0], a[:, -1], atol=1.0e-9)
    assert len(a) > r.size


@pytest.mark.parametrize("charge,multiplicity", [(0, 1), (1, 2), (-1, 2), (0, 3)])
def test_local_spin_conservation_and_conservative_force(charge, multiplicity):
    """Signed local spin and its Cartesian response conserve the molecular total."""
    from src.spin_local import DIM, GROUPS, SpinLocalKernel

    rng = np.random.default_rng(142)
    z = np.array([6, 8, 1, 1])
    r = np.array([[0., 0., 0.], [1.32, 0.11, 0.], [-.4, .88, .03], [-.5, -.84, -.05]])
    groups = np.arange(9)
    radii = np.array([0., .31, .28, 1.28, .96, .84, .76, .71, .66])
    theta = rng.normal(size=(3, GROUPS, DIM))
    pol = rng.normal(size=theta.shape)
    model = SpinLocalKernel(groups, radii, theta, pol, rng.normal(size=GROUPS*DIM))
    model = SpinLocalKernel.load(model.fields())
    types, x, jac = model.sites(z, r, True)
    spin, dspin = model.population(types, x, charge, multiplicity, jac)
    np.testing.assert_allclose(spin.sum(), multiplicity-1, atol=2.e-14)
    np.testing.assert_allclose(dspin.sum(axis=0), 0., atol=3.e-14)
    values, df = model.design_state(z, r, charge, multiplicity, True)
    direction = rng.normal(size=r.shape)
    fd = (model.design_state(z, r-1.e-6*direction, charge, multiplicity)[0]
          - model.design_state(z, r+1.e-6*direction, charge, multiplicity)[0])/2.e-6
    np.testing.assert_allclose(df.T @ direction.ravel(), fd, atol=1.e-8)
    perm = [3, 0, 2, 1]
    np.testing.assert_allclose(values, model.design_state(z[perm], r[perm]+1.5, charge, multiplicity)[0], atol=1.e-12)
