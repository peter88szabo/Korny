"""Conservative local-kernel correction for reaction and spin-state energetics.

The checkpoint retains the fitted PBE0 backbone. Extra element-specific and
periodic-group descriptors cover atoms present only in reaction benchmarks.
Spin and charge change both the energy and its analytic Cartesian derivative.
"""

import numpy as np

from .chemical import ChemicalEnvironment
from .electronic import state_key, validate_electrons
from .matern import LocalMatern
from .repulsion import NuclearRepulsion


class ReactionSurface:
    def __init__(self, base, kernel, elements, radii, groups, states, weights, target):
        self.base, self.kernel = base, kernel
        self.elements = tuple(map(int, elements))
        self.radii = np.asarray(radii)
        self.groups = np.asarray(groups, dtype=int)
        self.supported_states = tuple(map(tuple, states))
        self.weights = np.asarray(weights)
        self.prediction_target = target
        self.chem = ChemicalEnvironment(schema="smooth-organic-groups/1")
        self.correction = None
        self.extension_core = None
        self.pair = base.pair
        self.local = base.local

    def prior(self, z, r, q, m, forces=False):
        """Evaluate the trained organic contribution on its supported atoms.

        New spins start from the same-charge singlet prior. State-dependent
        fitted residuals supply the open-shell correction. Extension elements
        enter the new local environment features and their Cartesian forces.
        """
        mask = np.isin(z, self.base.local.species)
        force = np.zeros_like(r) if forces else None
        b = self.base
        extra_energy = 0.0
        if self.extension_core is not None and not mask.all():
            extra_energy, extra_force = self.extension_core.predict(
                z, r, include_atoms=~mask
            )
            if forces:
                force += extra_force
        if not mask.any():
            return extra_energy, force
        zz, rr = z[mask], r[mask]
        st = (q, m) if (q, m) in b.supported_states else (q, 1)
        if st not in b.supported_states:
            st = (0, 1)
        e = sum(
            np.count_nonzero(zz == s) * a
            for s, a in zip(b.local.species, b.offsets, strict=True)
        ) + b.charge_offsets.get(q, 0.0)
        ep, fp = b.pair.predict(zz, rr)
        if forces:
            el, fl, _ = b.local.observables(zz, rr)
            force[mask] += fp + fl
        else:
            el = b.local.energy(zz, rr)
        e += ep + el
        for head in (b.valence, b.repulsion):
            if head is not None:
                eh, fh = head.predict(zz, rr)
                e += eh
                if forces:
                    force[mask] += fh
        if b.electronic is not None:
            ee, ef = b.electronic.predict(zz, rr, st)
            e += ee
            if forces:
                force[mask] += ef
        return float(e + extra_energy), force

    def design(self, z, r, q, m, forces=False):
        """Build invariant energy features, including smooth pair and site kernels."""
        n = len(z)
        a, b = np.triu_indices(n, 1)
        vec = r[a] - r[b]
        d = np.linalg.norm(vec, axis=1)
        if np.any(d < 1e-9):
            raise ValueError("Coincident atoms")
        scale = self.radii[z[a]] + self.radii[z[b]]
        x = d / scale
        # Smoothly localized Gaussians on scaled interatomic distances; grouping
        # by valence electrons shares radial information across periodic rows.
        centers = np.array([0.75, 1.0, 1.25, 1.55, 2.05, 2.75, 3.5])
        t = (x[:, None] - centers) / 0.22
        radial = np.exp(-0.5 * t * t)
        g1, g2 = (
            np.minimum(self.groups[z[a]], self.groups[z[b]]),
            np.maximum(self.groups[z[a]], self.groups[z[b]]),
        )
        idx = g2 * (g2 + 1) // 2 + g1
        ng = 45
        pairs = np.zeros((ng, len(centers)))
        np.add.at(pairs, idx, radial)
        count = np.array([np.count_nonzero(z == s) for s in self.elements])
        mask = np.isin(z, self.kernel.species)
        c = np.zeros(self.chem.size)
        dc = np.zeros((r.size, self.chem.size)) if forces else None
        if np.any(mask):
            zz, rr = z[mask], r[mask]
            c, jchem = self.chem.design(zz, rr, gradient=forces)
            if forces:
                dc.reshape(n, 3, -1)[mask] = jchem.reshape(len(zz), 3, -1)
            v, df = self.kernel.design(
                self.kernel.project(zz, rr, gradient=forces), rr.size if forces else 0
            )
        else:
            v = np.zeros(self.kernel.size)
            df = None
        common = np.r_[1.0, count, pairs.ravel(), c, v]
        # Deviations use the same chemical groups and a subset of the local
        # centers. Their reduced basis regularizes the less abundant spin data.
        extra = np.r_[common[: 1 + len(count)], pairs.ravel(), c[::4], v[::4]]
        gates = (float(q), float(m == 2), float(m == 3), float(max(0, m - 3)))
        row = np.concatenate([common, *[g * extra for g in gates]])
        if not forces:
            return row, None
        dp = np.zeros((n, 3, ng, len(centers)))
        neg = (t * radial) / (0.22 * scale[:, None])
        delta = vec[:, :, None] / d[:, None, None] * neg[:, None, :]
        np.add.at(dp, (a, slice(None), idx), delta)
        np.add.at(dp, (b, slice(None), idx), -delta)
        dp = dp.reshape(r.size, -1)
        dv = np.zeros((n, 3, len(v)))
        if df is not None:
            dv[mask] = df.reshape(np.count_nonzero(mask), 3, len(v))
        dv = dv.reshape(r.size, -1)
        const = np.zeros((r.size, 1 + len(count)))
        full = np.column_stack((const, dp, dc, dv))
        ext = np.column_stack((const, dp, dc[:, ::4], dv[:, ::4]))
        return row, np.column_stack((full, *[g * ext for g in gates]))

    def validate_state(self, charge, multiplicity):
        """Expose the same charge/multiplicity contract used by the Smite adapter."""
        state = state_key(charge, multiplicity)
        if state not in self.supported_states:
            raise ValueError(f"Unsupported charge/spin {state}")
        return state

    def _predict(self, z, r, q, m, forces):
        z = np.asarray(z)
        r = np.asarray(r, dtype=float)
        q, m = self.validate_state(q, m)
        if (
            r.shape != (len(z), 3)
            or not np.isfinite(r).all()
            or not np.isin(z, self.elements).all()
        ):
            raise ValueError("Unsupported elements or geometry")
        validate_electrons(z, q, m)
        e, f = self.prior(z, r, q, m, forces)
        v, j = self.design(z, r, q, m, forces)
        e += v @ self.weights
        if forces:
            f += (j @ self.weights).reshape(r.shape)
        if self.correction is not None:
            de, df = self.correction(z, r)
            e += de
            if forces:
                f += df
        return float(e), f

    def predict(self, numbers, positions, charge=0, multiplicity=1):
        return self._predict(numbers, positions, charge, multiplicity, True)

    def energy(self, numbers, positions, charge=0, multiplicity=1):
        return self._predict(numbers, positions, charge, multiplicity, False)[0]

    def save_with_base(self, base_path, path):
        with np.load(base_path, allow_pickle=False) as original:
            fields = dict(original)
        fields.update({"rx_" + k: v for k, v in self.kernel.fields().items()})
        fields.update(
            rx_elements=self.elements,
            rx_radii=self.radii,
            rx_groups=self.groups,
            rx_states=self.supported_states,
            rx_weights=self.weights,
            rx_target=self.prediction_target,
        )
        if self.extension_core is not None:
            fields["rx_extension_core"] = self.extension_core.coefficients
        np.savez_compressed(path, **fields)

    @classmethod
    def restore(cls, base, d):
        kernel = LocalMatern.load(dict(d, species=base.local.species))
        model = cls(
            base,
            kernel,
            d["elements"],
            d["radii"],
            d["groups"],
            d["states"],
            d["weights"],
            str(d["target"]),
        )
        if "extension_core" in d:
            model.extension_core = NuclearRepulsion(
                model.elements, d["extension_core"], model.radii
            )
        return model
