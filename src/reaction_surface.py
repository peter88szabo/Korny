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
        self.reactive_basis = None
        self.additional_bases = []
        # A spin-specific readout shares the same descriptors and PBE0 prior.
        self.spin_readouts = {}
        # A fixed number of unpaired electrons can occupy a molecule of any
        # size. These optional coefficients scale an extensive local feature
        # sum by a heavy-atom count. The factor is constant under Cartesian
        # displacements, so the same coefficients give the analytic forces.
        self.spin_size_readouts = {}
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
        weights, extra = self.spin_readouts.get(m, (self.weights, {}))
        if m in self.spin_size_readouts:
            scale, correction = self.spin_size_readouts[m]
            weights = weights + correction * (scale / (scale + np.count_nonzero(z > 1)))
        e += v @ weights
        if forces:
            f += (j @ weights).reshape(r.shape)
        if self.reactive_basis is not None:
            extension, jac = self.reactive_basis.design(z, r, forces)
            e += extension @ self.reactive_basis.weights
            if forces:
                f += (jac @ self.reactive_basis.weights).reshape(r.shape)
        for index, basis in enumerate(self.additional_bases):
            weights = extra.get(index, basis.weights)
            if not np.any(weights):
                continue
            # Each resolution describes the same Cartesian potential. Taking
            # the derivative of its energy readout preserves conservative MD.
            if hasattr(basis, "design_state"):
                extension, jac = basis.design_state(z, r, q, m, forces)
            else:
                extension, jac = basis.design(z, r, forces)
            e += extension @ weights
            if forces:
                f += (jac @ weights).reshape(r.shape)
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
        if self.reactive_basis is not None:
            fields.update({"rx_" + k: v for k, v in self.reactive_basis.fields().items()})
        if self.additional_bases:
            fields["rx_additional_count"] = len(self.additional_bases)
            for i, basis in enumerate(self.additional_bases):
                fields.update({f"rx_additional_{i}_" + k: v for k, v in basis.fields().items()})
        if self.spin_readouts:
            fields["rx_readout_spins"] = sorted(self.spin_readouts)
            for m, (weights, extra) in self.spin_readouts.items():
                fields[f"rx_readout_{m}_weights"] = weights
                fields[f"rx_readout_{m}_indices"] = sorted(extra)
                for index, coef in extra.items():
                    fields[f"rx_readout_{m}_extra_{index}"] = coef
        if self.spin_size_readouts:
            fields["rx_size_spins"] = sorted(self.spin_size_readouts)
            for m, (scale, coef) in self.spin_size_readouts.items():
                fields[f"rx_size_{m}_scale"] = scale
                fields[f"rx_size_{m}_weights"] = coef
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
        if "rb_spec" in d:
            from .reactive_basis import ReactiveBasis
            model.reactive_basis = ReactiveBasis.load(d)
        for i in range(int(d.get("additional_count", 0))):
            from .reactive_basis import ReactiveBasis
            prefix = f"additional_{i}_"
            fields = {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}
            if "path_family_weights" in fields:
                from .path_family import PathFamilyKernel
                model.additional_bases.append(PathFamilyKernel.load(fields))
            elif "path_hermite" in fields:
                from .path_hermite import PathHermiteKernel
                model.additional_bases.append(PathHermiteKernel.load(fields))
            elif "htransfer_family" in fields:
                from .hydrogen_transfer import HydrogenTransferKernel
                model.additional_bases.append(HydrogenTransferKernel.load(fields))
            elif "operator_family" in fields:
                from .operator_kernel import OperatorKernel
                model.additional_bases.append(OperatorKernel.load(fields))
            elif "morse_weights" in fields:
                from .morse_bends import MorseBendBasis
                model.additional_bases.append(MorseBendBasis.load(fields))
            elif "split_charge_weights" in fields:
                from .split_charge import SplitChargeResponse
                model.additional_bases.append(SplitChargeResponse.load(fields))
            elif "spinexchange_weights" in fields:
                from .spin_exchange import SpinExchangeKernel
                model.additional_bases.append(SpinExchangeKernel.load(fields))
            elif "redox_scores" in fields:
                from .redox_environment import RedoxEnvironmentKernel
                model.additional_bases.append(RedoxEnvironmentKernel.load(fields))
            elif "spinlocal_logits" in fields:
                from .spin_local import SpinLocalKernel
                model.additional_bases.append(SpinLocalKernel.load(fields))
            elif "stretch_keys" in fields:
                from .local_stretch import StretchKernel
                model.additional_bases.append(StretchKernel.load(fields))
            elif "compact_keys" in fields:
                from .compact_kernel import CompactKernel
                model.additional_bases.append(CompactKernel.load(fields))
            elif "spectral_keys" in fields:
                from .adaptive_spectrum import SpectralKernel
                model.additional_bases.append(SpectralKernel.load(fields))
            elif "occupied_groups" in fields:
                from .occupied_orbitals import OccupationKernel
                model.additional_bases.append(OccupationKernel.load(fields))
            elif "competition_groups" in fields:
                from .valence_competition import CompetitionKernel
                model.additional_bases.append(CompetitionKernel.load(fields))
            elif "state_periodic_groups" in fields:
                from .atomic_states import AtomicStateKernel
                model.additional_bases.append(AtomicStateKernel.load(fields))
            elif "orbital_groups" in fields:
                from .orbital_moments import OrbitalMomentKernel
                model.additional_bases.append(OrbitalMomentKernel.load(fields))
            elif "periodic_groups" in fields:
                from .periodic_kernel import PeriodicKernel
                model.additional_bases.append(PeriodicKernel.load(fields))
            elif "message_seeds" in fields:
                from .message_kernel import MessageKernel
                model.additional_bases.append(MessageKernel.load(fields))
            elif "order_centers" in fields:
                from .bond_order_kernel import BondOrderKernel
                model.additional_bases.append(BondOrderKernel.load(fields))
            else:
                model.additional_bases.append(ReactiveBasis.load(fields))
        for spin in d.get("readout_spins", []):
            m = int(spin)
            weights = d[f"readout_{m}_weights"]
            extra = {int(i): d[f"readout_{m}_extra_{i}"] for i in d[f"readout_{m}_indices"]}
            if weights.shape != model.weights.shape or any(
                i < 0 or i >= len(model.additional_bases) or v.shape != model.additional_bases[i].weights.shape
                for i, v in extra.items()
            ):
                raise ValueError("Invalid spin-specific energy coefficients")
            model.spin_readouts[m] = weights, extra
        for value in d.get("size_spins", []):
            m = int(value)
            scale = float(d[f"size_{m}_scale"])
            coef = d[f"size_{m}_weights"]
            if (coef.shape != model.weights.shape or not np.isfinite(coef).all()
                    or not np.isfinite(scale) or scale <= 0):
                raise ValueError("Invalid size-dependent spin coefficients")
            model.spin_size_readouts[m] = scale, coef
        return model
