"""Portable joint pair and local energy model. Coordinates in Å, energies in eV."""
from io import BytesIO

import numpy as np

from .dispersion import correction_for
from .chemical import ChemicalEnvironment
from .electronic import ElectronicCorrection, state_key, validate_electrons
from .local import Molecules
from .matern import LocalMatern
from .pair import PairBaseline
from .repulsion import NuclearRepulsion
from .valence import Valence


class ForceField:
    """Compose one bare PBE0 checkpoint with an optional independent dispersion term.

    Atomic offsets use the dataset's energy zero. Discrete charge intercepts
    shift energies between fitted charge states. Neither term has a Cartesian
    derivative. A fitted electronic-state head changes both energy and force
    with charge and multiplicity. prediction_target describes the full fit.
    Energies are in eV, input coordinates in Å, and forces in eV/Å.
    """

    def __init__(self, pair, local, offsets, charge_offsets, target):
        if tuple(pair.species) != local.species:
            raise ValueError("Inconsistent species ordering in checkpoint")
        self.pair, self.local = pair, local
        self.offsets = np.asarray(offsets, dtype=float)
        self.charge_offsets = dict(charge_offsets)
        if target not in ("PBE0", "PBE0 (-eAT)"):
            raise ValueError("A Korny base checkpoint must target bare PBE0")
        self.target = target
        self.correction = None
        self.valence = None
        self.repulsion = None
        self.electronic = None
        self.matern = None
        self.chemical = None

    @property
    def supported_states(self):
        """Only advertise charge and multiplicity pairs represented in this checkpoint."""
        if self.electronic is not None:
            return tuple(self.electronic.weights)
        return tuple((q, 1) for q in self.charge_offsets)

    @property
    def prediction_target(self):
        return self.electronic.target if self.electronic is not None else self.target

    def validate_state(self, charge, multiplicity):
        state = state_key(charge, multiplicity)
        if state not in self.supported_states:
            raise ValueError(f"Unfitted electronic state {state}. Supported (charge, multiplicity): "
                             f"{self.supported_states}")
        return state

    def _geometry_and_offset(self, numbers, positions, charge, multiplicity):
        """Validate elements and total charge, then compute the zero of energy."""
        z, r = self.local.geometry(numbers, positions)
        state = self.validate_state(charge, multiplicity)
        validate_electrons(z, *state)
        z = z.astype(int)
        counts = [np.count_nonzero(z == s) for s in self.local.species]
        return z, r, float(counts @ self.offsets + self.charge_offsets[charge]), state

    def energy(self, numbers, positions, charge=0, multiplicity=1):
        """Fast scalar evaluation for reaction energies and fixed-geometry scans."""
        z, r, offset, state = self._geometry_and_offset(numbers, positions, charge, multiplicity)
        energy = offset + self.pair.predict(z, r)[0] + self.local.energy(z, r)
        if self.valence is not None:
            energy += self.valence.predict(z, r)[0]
        if self.repulsion is not None:
            energy += self.repulsion.predict(z, r)[0]
        if self.electronic is not None:
            energy += self.electronic.predict(z, r, state)[0]
        if self.matern is not None:
            energy += self.matern.evaluate(z, r, state, forces=False)[0]
        if self.chemical is not None:
            energy += self.chemical.evaluate(z, r, state, forces=False)[0]
        if self.correction is not None:
            energy += self.correction(z, r)[0]
        return float(energy)

    def predict(self, numbers, positions, charge=0, multiplicity=1):
        """Return a conservative energy/force pair, adding the chosen correction once."""
        z, r, offset, state = self._geometry_and_offset(numbers, positions, charge, multiplicity)
        ep, fp = self.pair.predict(z, r)
        el, fl, _ = self.local.observables(z, r)
        energy = offset + ep + el
        force = fp + fl
        if self.valence is not None:
            de, df = self.valence.predict(z, r)
            energy += de
            force += df
        if self.repulsion is not None:
            de, df = self.repulsion.predict(z, r)
            energy += de
            force += df
        if self.electronic is not None:
            de, df = self.electronic.predict(z, r, state)
            energy += de
            force += df
        if self.matern is not None:
            de, df = self.matern.evaluate(z, r, state, forces=True)
            energy += de
            force += df
        if self.chemical is not None:
            de, df = self.chemical.evaluate(z, r, state, forces=True)
            energy += de
            force += df
        if self.correction is not None:
            de, df = self.correction(z, r)
            energy += de
            force += df
        return energy, force

    def save(self, path):
        """Bundle the base pair and local coefficients; dispersion is a run-time option."""
        fields = {}
        for save in (self.local.save_local, self.pair.save):
            stream = BytesIO()
            save(stream)
            stream.seek(0)
            with np.load(stream, allow_pickle=False) as part:
                fields.update({k: part[k] for k in part.files})
        version = "korny/1"
        if self.valence is not None:
            version = "korny/2"
            fields["valence_weights"] = self.valence.weights
        if self.repulsion is not None:
            version = "korny/3"
            fields["nlh_coefficients"] = self.repulsion.coefficients
        if self.electronic is not None:
            version = "korny/4"
            fields["electronic_states"] = list(self.electronic.weights)
            fields["electronic_weights"] = list(self.electronic.weights.values())
            fields["electronic_target"] = self.electronic.target
            fields["electronic_environment"] = self.electronic.features.environment
            if self.electronic.features.radial.width != .25:
                version = "korny/5"
                fields["electronic_radial_width"] = self.electronic.features.radial.width
        if self.matern is not None:
            if self.electronic is None or set(self.matern.weights) != set(self.supported_states):
                raise ValueError("Local kernel and electronic states disagree")
            version = "korny/5"
            fields.update(self.matern.fields())
        if self.chemical is not None:
            if self.electronic is None or set(self.chemical.weights) != set(self.supported_states):
                raise ValueError("Functional-group and electronic states disagree")
            version = "korny/6"
            fields.update(chemical_schema=self.chemical.schema,
                          chemical_weights=[self.chemical.weights[s] for s in self.supported_states])
        np.savez_compressed(path, **fields, bundle=version, offsets=self.offsets,
                            charges=list(self.charge_offsets),
                            charge_offsets=list(self.charge_offsets.values()), target=self.target)

    @classmethod
    def load(cls, path, *, dispersion="off", volumes=None):
        with np.load(path, allow_pickle=False) as d:
            reaction_fields = {k[3:]: d[k] for k in d.files if k.startswith("rx_")}
            version = str(d["bundle"])
            if version not in ("korny/1", "korny/2", "korny/3", "korny/4", "korny/5", "korny/6"):
                raise ValueError("Unknown Korny checkpoint format")
            valence = Valence(d["valence_weights"]) if version == "korny/2" or "valence_weights" in d else None
            core = NuclearRepulsion(d["species"],d["nlh_coefficients"]) if "nlh_coefficients" in d else None
            electronic = (ElectronicCorrection(d["species"], d["electronic_states"],
                                               d["electronic_weights"], str(d["electronic_target"]),
                                               environment=bool(d["electronic_environment"])
                                               if "electronic_environment" in d else False,
                                               radial_width=float(d["electronic_radial_width"])
                                               if "electronic_radial_width" in d else .25)
                          if version in ("korny/4", "korny/5", "korny/6") else None)
            chemical = None
            if version == "korny/6":
                chemical = ChemicalEnvironment(dict(zip(electronic.weights, d["chemical_weights"],strict=True)),
                                               schema=str(d["chemical_schema"]))
            matern = LocalMatern.load(d) if "matern_scales" in d else None
            if matern and set(matern.weights) != set(electronic.weights):
                raise ValueError("Inconsistent checkpoint states")
            offsets, target = d["offsets"], str(d["target"])
            charges = dict(zip(d["charges"].tolist(), d["charge_offsets"].tolist(), strict=True))
        model = cls(PairBaseline.load(path), Molecules.load_local(path), offsets, charges, target)
        model.valence = valence
        model.repulsion = core
        model.electronic = electronic
        model.matern = matern
        model.chemical = chemical
        if reaction_fields:
            from .reaction_surface import ReactionSurface
            model = ReactionSurface.restore(model, reaction_fields)
        model.correction = correction_for(dispersion, volumes)
        return model
