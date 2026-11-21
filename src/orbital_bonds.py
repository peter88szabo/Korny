"""Element-resolved covalency features from a geometry and spin-dependent density matrix.

Atomic populations alone omit how an electron is shared across a bond. Pair
blocks of the same occupied-orbital matrix resolve s/p and sigma components,
spin-dependent sharing and bond energy. The representation sums these smooth
quantities into the local environments at both ends of each bond.
"""

import numpy as np
from .occupied_orbitals import matrix, population
from .four_body import switch

CHANNELS = 9
WIDTH = 9 * CHANNELS + 4


def bond_properties(b, s, e, u, change=None):
    """Return rotational scalars of a bond block or their exact directional derivatives."""
    ss = b[:, 0, 0]
    sp = b[:, 0, 1:]
    ps = b[:, 1:, 0]
    pp = b[:, 1:, 1:]
    along = np.einsum("ai,aij,aj->a", u, pp, u)
    squares = np.einsum("aij,aij->a", b, b)
    dot = lambda a, c: np.sum(a * c, axis=tuple(range(1, a.ndim)))
    if change is None:
        return np.column_stack(
            (
                ss**2,
                dot(sp, sp),
                dot(ps, ps),
                along**2,
                dot(pp, pp) - along**2,
                squares,
                squares * squares,
                dot(b, s),
                dot(b, e),
            )
        )
    db, ds, de, du = change
    dsigma = (
        np.einsum("ai,aij,aj->a", du, pp, u)
        + np.einsum("ai,aij,aj->a", u, db[:, 1:, 1:], u)
        + np.einsum("ai,aij,aj->a", u, pp, du)
    )
    dq = 2 * dot(b, db)
    return np.column_stack(
        (
            2 * ss * db[:, 0, 0],
            2 * dot(sp, db[:, 0, 1:]),
            2 * dot(ps, db[:, 1:, 0]),
            2 * along * dsigma,
            2 * dot(pp, db[:, 1:, 1:]) - 2 * along * dsigma,
            dq,
            2 * squares * dq,
            dot(db, s) + dot(b, ds),
            dot(db, e) + dot(b, de),
        )
    )


def environments(z, r, q, m, groups, radii, derivatives=False):
    n = len(z)
    typ = groups[z]
    h, active, (a, b, grad) = matrix(z, r, groups, radii, derivatives)
    energy, orbit = np.linalg.eigh(h)
    count = int(np.sum(np.where(z == 1, 1, np.where(z == 2, 2, typ)))) - int(q)
    if (count + m - 1) % 2:
        raise ValueError("Molecular charge and spin disagree")
    up = (count + m - 1) // 2
    alpha, sa = population(energy, up)
    beta, sb = population(energy, count - up)
    tot = alpha + beta
    e = energy + 1.5
    f = np.stack([tot, alpha - beta, e * tot])
    # Embed the H-only s basis back into uniform 4x4 pair blocks.
    full_orbit = np.zeros((4 * n, len(energy)))
    full_orbit[active] = orbit

    def pair_blocks(diagonal):
        full = ((full_orbit * diagonal) @ full_orbit.T).reshape(n, 4, n, 4)
        atoms = np.arange(n)
        return full[a, :, b, :], full[atoms, :, atoms, :].diagonal(axis1=1, axis2=2)

    items = [pair_blocks(v) for v in f]
    density, spin, band = [p[0] for p in items]
    delta = r[b] - r[a]
    d = np.linalg.norm(delta, axis=1)
    u = delta / d[:, None]
    scale = radii[z[a]] + radii[z[b]]
    gate, dgate = switch(d / scale, 1.48, 2.35)
    directed = np.r_[a, b]
    other = typ[np.r_[b, a]]
    local = bond_properties(density, spin, band, u)
    swap = [0, 2, 1, 3, 4, 5, 6, 7, 8]

    def aggregate(props, atomic):
        val = np.zeros((n, CHANNELS, 9))
        np.add.at(val, (directed, other), np.vstack((props, props[:, swap])))
        return np.column_stack(
            (
                val.reshape(n, -1),
                atomic[0][:, 0],
                atomic[0][:, 1:].sum(axis=1),
                atomic[1][:, 0],
                atomic[1][:, 1:].sum(axis=1),
            )
        )

    x = aggregate(local * gate[:, None], [p[1] for p in items])
    if not derivatives:
        return x, None
    gap = energy[:, None] - energy
    close = abs(gap) < 1.0e-7
    qa = np.stack((sa, sa, e * sa))
    qb = np.stack((sb, -sb, e * sb))
    fp = -qa - qb
    fp[2] += tot
    divided = np.divide(
        f[:, :, None] - f[:, None, :],
        gap,
        out=np.zeros((3, len(energy), len(energy))),
        where=~close,
    )
    divided = np.where(close[None], 0.5 * (fp[:, :, None] + fp[:, None, :]), divided)
    eye = np.eye(3)
    index = np.arange(n)
    jac = np.zeros((n, WIDTH, r.size))
    full = np.zeros((4 * n, 4 * n, 3))
    for site in range(n):
        full.fill(0.0)
        blocks = full.reshape(n, 4, n, 4, 3)
        sign = (b == site).astype(float) - (a == site)
        for ix, sig in (
            (np.flatnonzero(b == site), 1.0),
            (np.flatnonzero(a == site), -1.0),
        ):
            blocks[a[ix], :, b[ix], :, :] = sig * grad[ix]
            blocks[b[ix], :, a[ix], :, :] = sig * grad[ix].transpose(0, 2, 1, 3)
        transformed = orbit.T @ full[active][:, active].transpose(2, 0, 1) @ orbit
        du = (
            sign[:, None, None]
            * (eye - u[:, :, None] * u[:, None, :])
            / d[:, None, None]
        )
        for axis, dh in enumerate(transformed):
            deriv = divided * dh
            dmua = dh.diagonal() @ sa / max(sa.sum(), 1.0e-14)
            dmub = dh.diagonal() @ sb / max(sb.sum(), 1.0e-14)
            ii = np.arange(len(energy))
            deriv[:, ii, ii] += qa * dmua + qb * dmub
            transform = (full_orbit @ deriv @ full_orbit.T).reshape(3, n, 4, n, 4)
            pairs = [v[a, :, b, :] for v in transform]
            single = [
                v[index, :, index, :].diagonal(axis1=1, axis2=2) for v in transform[:2]
            ]
            dp = bond_properties(density, spin, band, u, (*pairs, du[:, :, axis]))
            dg = dgate * sign * u[:, axis] / scale
            jac[:, :, 3 * site + axis] = aggregate(
                dp * gate[:, None] + local * dg[:, None], single
            )
    return x, jac
