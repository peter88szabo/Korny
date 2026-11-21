"""Conservative local energy kernels informed by partial charge and radical character.

A first kernel estimates atomic electronic properties from geometry. Its partial
charges come from antisymmetric charge transfers between neighbors, preserving
the molecular input charge exactly. Formal charge and radical character are
separate auxiliary targets learned from mapped Lewis structures. Spatial sums
of these properties describe the environment of each atom for a second energy
kernel. Every derivative, including the response of the charge model, enters
the Cartesian force. Distances are in angstrom and partial charges are in e.
"""

import numpy as np
from .four_body import switch
from .local import descriptor_array
from .periodic_kernel import PeriodicKernel


def neighbors(z, r, radii):
    """Compute smooth bond-transfer, coordination, and screened-field weights."""
    a, b = np.triu_indices(len(z), 1)
    dr = r[b] - r[a]
    d = np.linalg.norm(dr, axis=1)
    if np.any(d < 1.0e-10):
        raise ValueError("Coincident nuclei")
    ix = d < 8.0
    a, b, d, dr = a[ix], b[ix], d[ix], dr[ix]
    scale = radii[z[a]] + radii[z[b]]
    x = d / scale
    bond, db = switch(x, 1.03, 1.86)
    outer, do = switch(d, 2.3, 5.3)
    long, dl = switch(d, 5.0, 8.0)
    denom = np.sqrt(d * d + 1.0)
    w = np.column_stack((bond, outer, long / denom))
    dw = np.column_stack((db / scale, do, dl / denom - long * d / denom**3))
    return a, b, dr / d[:, None], w, dw


def laplacian(n, a, b, bond):
    """Map atomic scalar potentials to charge transfers with vanishing column sums."""
    out = np.zeros((n, n))
    np.add.at(out, (a, a), bond)
    np.add.at(out, (b, b), bond)
    out[a, b] = out[b, a] = -bond
    return out


class StateFeatures:
    def __init__(self, periodic, coefficients):
        self.periodic = periodic
        self.coefficients = coefficients

    def interpolation(self, g, u, jac=None):
        c = self.periodic.centers[g]
        delta = u[:, None, :] - c
        sq = np.einsum("abd,abd->ab", delta, delta)
        cols = [np.ones((len(u), 1))]
        derivatives = (
            [np.zeros((len(u), 1, jac.shape[-1]))] if jac is not None else None
        )
        for width in (0.60, 1.45):
            k = np.exp(-0.5 * sq / width**2)
            cols.append(k)
            if jac is not None:
                ds = (
                    -np.einsum("abd,adi->abi", delta, jac, optimize=True)
                    * k[:, :, None]
                    / width**2
                )
                derivatives.append(ds)
        return np.column_stack(cols), (
            np.concatenate(derivatives, axis=1) if derivatives is not None else None
        )

    def projected(self, z, r, forces):
        """Build element-group density coordinates and optional positive Jacobians."""
        p = self.periodic
        tp = p.groups[z]
        proj = {}
        der = {}
        raw = []
        jac = []
        for i in range(len(z)):
            a, b = descriptor_array(
                r,
                tp,
                i,
                nr=p.shells.shape[1],
                lmax=2,
                ron=p.cutoff[0],
                rc=p.cutoff[1],
                species=tuple(range(9)),
                gaussian_shells=p.shells,
                atom_radii=p.radii[z],
                gradient=forces,
            )
            raw.append(a)
            if forces:
                jac.append(b)
        for g, m in p.projections.items():
            ix = np.flatnonzero(tp == g)
            if len(ix):
                proj[g] = np.asarray(raw)[ix] @ m
                if forces:
                    der[g] = m.T @ np.asarray(jac)[ix].reshape(len(ix), len(m), r.size)
        return proj, der if forces else None

    def context(self, z, r, q, m, u, du=None):
        """Return bounded electronic features and their full local spatial response."""
        p = self.periodic
        n = len(z)
        tp = p.groups[z]
        fields = np.zeros((n, 4))
        df = np.zeros((n, 4, r.size)) if du is not None else None
        for g, proj in u.items():
            ix = np.flatnonzero(tp == g)
            phi, dphi = self.interpolation(g, proj, None if du is None else du[g])
            # Ordinary and open-shell endpoint labels use separately regularized heads.
            coeff = self.coefficients[g][:, [0, 1, 2, 3 if m == 1 else 4]]
            fields[ix] = phi @ coeff
            if du is not None:
                df[ix] = np.einsum("abc,bd->adc", dphi, coeff)
        a, b, unit, w, dw = neighbors(z, r, p.radii)
        L = laplacian(n, a, b, w[:, 0])
        initial = fields[:, :2].copy()
        shares = p.radii[z] ** 2
        fields[:, :2] = L @ initial + q * (shares / shares.sum())[:, None]
        if du is not None:
            df[:, :2] = np.einsum("ab,bkd->akd", L, df[:, :2])
            for src, delta in (
                (a, initial[a] - initial[b]),
                (b, initial[b] - initial[a]),
            ):
                direct = delta[:, :, None] * (dw[:, 0, None] * unit)[:, None, :]
                view = df[:, :2].reshape(n, 2, n, 3).transpose(0, 2, 1, 3)
                np.add.at(view, (src, b), direct)
                np.add.at(view, (src, a), -direct)
        # Bounded local variables avoid amplifying extrapolation in charged
        # molecules or unusual coordination environments.
        scales = np.array([0.4, 0.65, 1.4, 0.75])
        val = np.tanh(fields / scales)
        dval = (
            None
            if du is None
            else (1 - val * val)[:, :, None] * df / scales[None, :, None]
        )
        moments = []
        derivatives = []
        for k in range(3):
            weight = w[:, k]
            avg = np.zeros_like(val)
            np.add.at(avg, a, weight[:, None] * val[b])
            np.add.at(avg, b, weight[:, None] * val[a])
            moments.append(avg)
            if du is not None:
                dadj = np.zeros_like(df)
                for i, j in ((a, b), (b, a)):
                    np.add.at(dadj, i, weight[:, None, None] * dval[j])
                    pair = (dw[:, k, None] * unit)[:, None, :] * val[j, :, None]
                    view = dadj.reshape(n, 4, n, 3).transpose(0, 2, 1, 3)
                    np.add.at(view, (i, b), pair)
                    np.add.at(view, (i, a), -pair)
                derivatives.append(dadj)
        env = np.column_stack([val, *moments, val * moments[-1]])
        denv = None
        if du is not None:
            interaction = (
                dval * moments[-1][:, :, None] + val[:, :, None] * derivatives[-1]
            )
            denv = np.concatenate([dval, *derivatives, interaction], axis=1)
        return env, denv, fields


class AtomicStateKernel:
    """Predict a scalar energy from charge, spin character and their neighborhoods."""

    def __init__(self, teacher, centers, projections, length, weights):
        self.teacher = teacher
        self.centers = centers
        self.projections = projections
        self.length = float(length)
        self.weights = np.asarray(weights)
        self.size = sum(len(c) for c in centers.values())
        if self.weights.shape != (self.size,) or self.length <= 0:
            raise ValueError("Invalid atomic-state kernel")

    def design_state(self, z, r, charge, multiplicity, forces=False):
        u, du = self.teacher.projected(z, r, forces)
        env, de, _ = self.teacher.context(z, r, charge, multiplicity, u, du)
        types = self.teacher.periodic.groups[z]
        vals = np.zeros(self.size)
        jac = np.zeros((r.size, self.size)) if forces else None
        offset = 0
        for g, centers in self.centers.items():
            sl = slice(offset, offset + len(centers))
            offset = sl.stop
            if g not in u:
                continue
            ix = np.flatnonzero(types == g)
            proj = self.projections[g]
            sites = np.column_stack((0.65 * u[g], env[ix] @ proj))
            delta = sites[:, None, :] - centers
            kernel = np.exp(
                -0.5 * np.einsum("acd,acd->ac", delta, delta) / self.length**2
            )
            vals[sl] = kernel.sum(axis=0)
            if forces:
                dsites = np.concatenate((0.65 * du[g], proj.T @ de[ix]), axis=1)
                jac[:, sl] = (
                    np.einsum("ac,acd,adi->ic", kernel, delta, dsites, optimize=True)
                    / self.length**2
                )
        return vals, jac

    def design(self, z, r, forces=False):
        """Use the neutral singlet state for the equilibrium training Hessian."""
        return self.design_state(z, r, 0, 1, forces)

    def fields(self):
        d = {"state_" + k: v for k, v in self.teacher.periodic.fields().items()}
        d.update(
            state_length=self.length,
            state_keys=list(self.centers),
            state_weights=self.weights,
        )
        for g in self.centers:
            d[f"state_coef_{g}"] = self.teacher.coefficients[g]
            d[f"state_centers_{g}"] = self.centers[g]
            d[f"state_proj_{g}"] = self.projections[g]
        return d

    @classmethod
    def load(cls, d):
        keys = list(map(int, d["state_keys"]))
        periodic = PeriodicKernel.load(
            {k[6:]: v for k, v in d.items() if k.startswith("state_periodic_")}
        )
        teacher = StateFeatures(periodic, {g: d[f"state_coef_{g}"] for g in keys})
        return cls(
            teacher,
            {g: d[f"state_centers_{g}"] for g in keys},
            {g: d[f"state_proj_{g}"] for g in keys},
            d["state_length"],
            d["state_weights"],
        )
