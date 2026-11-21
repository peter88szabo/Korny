"""Smooth additional atom-centered energy bases for reaction-calibrated surfaces.

The extension references the same element ordering as the backbone. Its own
stored descriptor and projection allow nonlinear geometry scales to change
without altering either the frozen backbone or its Cartesian derivatives.
"""

import numpy as np
from .local import descriptor_array


class ReactiveBasis:
    def __init__(
        self,
        species,
        spec,
        projections,
        centers,
        radial_dims,
        length,
        angular,
        weights,
        four=False,
        bond=None,
        gaussian_shells=None,
        profile=None,
    ):
        self.species = tuple(map(int, species))
        self.spec = dict(
            nr=int(spec[0]),
            lmax=int(spec[1]),
            ron=float(spec[2]),
            rc=float(spec[3]),
            inverse_radius=True,
            species=self.species,
        )
        if gaussian_shells is not None:
            self.spec["gaussian_shells"] = np.asarray(gaussian_shells, dtype=float)
        self.projections, self.centers, self.radial_dims = (
            projections,
            centers,
            radial_dims,
        )
        self.widths = {s: len(c) for s,c in centers.items()}
        self.length, self.angular = float(length), float(angular)
        self.profile = profile
        if profile is not None and profile not in ("gaussian", "matern52", "rational_quadratic", "exponential"):
            raise ValueError(f"Unknown radial kernel {profile}")
        self.four = bool(four)
        self.bond = bond
        self.weights = np.asarray(weights)
        self.size = (
            sum(map(len, centers.values()))
            + (12 if self.four else 0)
            + (bond.size if bond else 0)
        )
        if self.weights.shape != (self.size,) or min(self.length, self.angular) <= 0:
            raise ValueError("Invalid reactive basis weights or kernel scale")

    def values(self, s, sites, grad=False):
        """Double Matérn 5/2, with the energy of each isolated atom subtracted."""
        c = self.centers[s]
        w = np.ones(c.shape[1])
        w[self.radial_dims[s] :] = self.angular
        delta = (sites[:, None, :] - c[None, :, :]) * w
        d2 = np.einsum("ijd,ijd->ij", delta, delta)
        d0 = np.square(c * w).sum(axis=1)
        if self.profile is not None:
            value, slope = radial_profile(self.profile, d2, self.length, grad)
            zero, _ = radial_profile(self.profile, d0, self.length)
            return value - zero, (slope[:, :, None] * delta * w if grad else None)
        v = np.zeros_like(d2)
        b = np.zeros_like(v)
        for width in (0.5 * self.length, 2 * self.length):
            a = 5 / width**2
            t = np.sqrt(a * d2)
            t0 = np.sqrt(a * d0)
            v += 0.5 * (
                (1 + t + t * t / 3) * np.exp(-t) - (1 + t0 + t0 * t0 / 3) * np.exp(-t0)
            )
            if grad:
                b -= a / 6 * (1 + t) * np.exp(-t)
        return v, (b[:, :, None] * delta * w) if grad else None

    def design(self, z, r, forces=False):
        """Sum atomic kernels, propagating analytic negative gradients to all atoms."""
        result = np.zeros(self.size)
        force = np.zeros((len(z), 3, self.size)) if forces else None
        keep = np.isin(z, self.species)
        zz, rr = z[keep], r[keep]
        start = 0
        for s in self.centers:
            stop = start + self.widths[s]
            for i in np.flatnonzero(zz == s):
                raw, jac = descriptor_array(rr, zz, i, **self.spec, gradient=forces)
                p = self.projections[s]
                val, der = self.values(s, (raw @ p)[None, :], forces)
                result[start:stop] += val[0]
                if forces:
                    local = (der[0] @ (p.T @ jac.reshape(len(raw), -1))).T
                    force[keep, :, start:stop] -= local.reshape(len(zz), 3, -1)
            start = stop
        if self.four:
            from .four_body import four_body

            result[start : start + 12], grad = four_body(z, r, forces)
            if forces:
                force[:, :, start : start + 12] = grad.reshape(len(z), 3, -1)
            start += 12
        if self.bond is not None:
            result[start:], grad = self.bond.design(z, r, forces)
            if forces:
                force[:, :, start:] = grad.reshape(len(z), 3, -1)
        return result, force.reshape(r.size, -1) if forces else None

    def fields(self):
        names = ("nr", "lmax", "ron", "rc")
        d = dict(
            rb_spec=[self.spec[n] for n in names],
            rb_species=self.species,
            rb_active=list(self.centers),
            rb_length=self.length,
            rb_angular=self.angular,
            rb_weights=self.weights,
            rb_four=self.four,
        )
        if "gaussian_shells" in self.spec:
            d["rb_shells"] = self.spec["gaussian_shells"]
        if self.profile is not None:
            d["rb_profile"] = self.profile
        if self.bond is not None:
            d.update(
                rb_bond_frequencies=self.bond.frequencies,
                rb_bond_phases=self.bond.phases,
            )
        for s in self.centers:
            d[f"rb_proj_{s}"] = self.projections[s]
            d[f"rb_cent_{s}"] = self.centers[s]
            d[f"rb_rad_{s}"] = self.radial_dims[s]
        return d

    @classmethod
    def load(cls, d):
        active = list(map(int, d["rb_active"]))

        def unpack(key):
            return {s: d[f"rb_{key}_{s}"] for s in active}

        from .bond_environment import BondBasis

        bond = (
            BondBasis(d["rb_bond_frequencies"], d["rb_bond_phases"])
            if "rb_bond_phases" in d
            else None
        )
        return cls(
            d["rb_species"],
            d["rb_spec"],
            unpack("proj"),
            unpack("cent"),
            unpack("rad"),
            d["rb_length"],
            d["rb_angular"],
            d["rb_weights"],
            d["rb_four"],
            bond,
            d.get("rb_shells"),
            str(d["rb_profile"]) if "rb_profile" in d else None,
        )


def radial_profile(name, squared_distance, half_width, derivative=False):
    """Return k and, optionally, twice dk/d(r²) for an analytic Cartesian pullback.

    All profiles have height one at zero and half height at the specified
    distance. The exponential is available for energy comparisons only because
    its radial cusp has no Cartesian gradient at a coincident environment.
    """
    u = squared_distance / half_width**2
    if name == "gaussian":
        k = np.exp(-np.log(2) * u)
        slope = -2 * np.log(2) * k if derivative else None
    elif name == "rational_quadratic":
        k = 1 / (1 + u)
        slope = -2 * k**2 if derivative else None
    elif name == "matern52":
        # Positive root of (1+t+t²/3) exp(-t) = 1/2.
        a = 2.330256192156008
        t = a * np.sqrt(u)
        decay = np.exp(-t)
        k = (1 + t + t*t/3) * decay
        slope = -(a*a/3) * (1+t) * decay if derivative else None
    elif name == "exponential":
        if derivative:
            raise ValueError("The exponential kernel is restricted to energy screening")
        k = np.exp(-np.log(2) * np.sqrt(u))
        slope = None
    else:
        raise ValueError(f"Unknown radial kernel {name}")
    return k, (slope / half_width**2 if derivative else None)
