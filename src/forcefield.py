"""Portable joint pair and local energy model. Coordinates in Å, energies in eV."""
from io import BytesIO

import numpy as np

from .dispersion import correction_for
from .local import Molecules
from .pair import PairBaseline
from .valence import Valence


class ForceField:
    """Compose one bare PBE0 checkpoint with an optional independent dispersion term.

    Atomic offsets use the dataset's energy zero. Discrete charge intercepts
    shift energies between fitted charge states. Neither term has a Cartesian
    derivative. Energies are in eV, input coordinates in Å, and forces in eV/Å.
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

    def _geometry_and_offset(self, numbers, positions, charge):
        """Validate elements and total charge, then compute the zero of energy."""
        z, r = self.local.geometry(numbers, positions)
        if charge not in self.charge_offsets:
            raise ValueError(f"Charge {charge} outside fitted states {list(self.charge_offsets)}")
        z = z.astype(int)
        counts = [np.count_nonzero(z == s) for s in self.local.species]
        return z, r, float(counts @ self.offsets + self.charge_offsets[charge])

    def energy(self, numbers, positions, charge=0):
        """Fast scalar evaluation for reaction energies and fixed-geometry scans."""
        z, r, offset = self._geometry_and_offset(numbers, positions, charge)
        energy = offset + self.pair.predict(z, r)[0] + self.local.energy(z, r)
        if self.valence is not None:
            energy += self.valence.predict(z, r)[0]
        if self.correction is not None:
            energy += self.correction(z, r)[0]
        return float(energy)

    def predict(self, numbers, positions, charge=0):
        """Return a conservative energy/force pair, adding the chosen correction once."""
        z, r, offset = self._geometry_and_offset(numbers, positions, charge)
        ep, fp = self.pair.predict(z, r)
        el, fl, _ = self.local.observables(z, r)
        energy = offset + ep + el
        force = fp + fl
        if self.valence is not None:
            de, df = self.valence.predict(z, r)
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
        np.savez_compressed(path, **fields, bundle=version, offsets=self.offsets,
                            charges=list(self.charge_offsets),
                            charge_offsets=list(self.charge_offsets.values()), target=self.target)

    @classmethod
    def load(cls, path, *, dispersion="off", volumes=None):
        with np.load(path, allow_pickle=False) as d:
            version = str(d["bundle"])
            if version not in ("korny/1", "korny/2"):
                raise ValueError("Unknown Korny checkpoint format")
            valence = Valence(d["valence_weights"]) if version == "korny/2" else None
            offsets, target = d["offsets"], str(d["target"])
            charges = dict(zip(d["charges"].tolist(), d["charge_offsets"].tolist(), strict=True))
        model = cls(PairBaseline.load(path), Molecules.load_local(path), offsets, charges, target)
        model.valence = valence
        model.correction = correction_for(dispersion, volumes)
        return model
