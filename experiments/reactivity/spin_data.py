"""Read published, spin-labelled stationary points without running quantum chemistry.

Each row is a nuclear- and charge-balanced energy difference. Forward and reverse
barriers and their reaction energy are kept in the same split. The formula of
the full transition state groups related reactions across both collections.
"""
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from experiments.des15k.benchmark import ELEMENTS
from experiments.des15k.run import ROOT
from src.electronic import validate_electrons

DATA = ROOT / "dataset_for_training"


@dataclass
class Geometry:
    z: np.ndarray
    r: np.ndarray
    state: tuple


@dataclass
class Reaction:
    name: str
    family: str
    reactants: list
    products: list
    ts: str
    # Forward barrier, reverse barrier, and reaction energy, in kcal/mol.
    reference: tuple

    def equations(self):
        left = [(a, -1) for a in self.reactants]
        right = [(a, -1) for a in self.products]
        return [left + [(self.ts, 1)], right + [(self.ts, 1)],
                left + [(a, 1) for a in self.products]]


def geometry(lines, q, m, *, max_atoms=36):
    """Return None for species outside the element set or small-molecule scope."""
    atoms = [line.split() for line in lines if line.strip()]
    if (max_atoms is not None and len(atoms) > max_atoms) or any(row[0] not in ELEMENTS for row in atoms):
        return None
    z = np.array([ELEMENTS[row[0]] for row in atoms])
    validate_electrons(z, q, m)
    return Geometry(z, np.array([row[1:4] for row in atoms], float), (q, m))


def bh9(*, max_atoms=36):
    geoms, rows = {}, []
    with ZipFile(DATA / "BH9/ct2c00362_si_001.zip") as archive:
        for p in archive.namelist():
            if "/XYZ_files/" in p and p.endswith(".xyz"):
                lines = archive.read(p).decode().splitlines()
                q, m = map(int, lines[1].split())
                geoms["BH9/" + Path(p).stem] = geometry(lines[2:], q, m, max_atoms=max_atoms)
        for line in archive.read("BH9_SI/Reference.org").decode().splitlines()[1:]:
            if not line.startswith("|"):
                continue
            _, family, _, system, *values = [w.strip() for w in line.strip("|").split("|")]
            prefix = "BH9/" + system
            def side(label):
                return [prefix + role for role in (label, label+"1", label+"2", label+"3")
                        if prefix + role in geoms]
            rows.append(Reaction(prefix, "BH9/" + family, side("R"), side("P"), prefix+"TS",
                                 tuple(map(float, values))))
    return geoms, rows


def trip50(*, max_atoms=36):
    """Match the original corrected reference CSV to the authors' state-annotated XYZ.

    Several reactions reuse structures. Match by the published energy as well as
    identifier so that aliases and the role of every detached fragment are checked.
    """
    root = DATA / "TRIP50"
    geoms, energies, rows = {}, {}, []
    with (root / "trip50.extxyz").open() as stream:
        for line in stream:
            n = int(line)
            fields = dict(t.split("=", 1) for t in next(stream).split() if "=" in t)
            key = "TRIP50/" + fields["species_id"]
            geoms[key] = geometry([next(stream) for _ in range(n)], int(fields["charge"]),
                                  int(fields["spin_multiplicity"]), max_atoms=max_atoms)
            energies[key] = float(fields["energy_dlpno_ccsd_t"])
    with (root / "aliases.csv").open() as f:
        aliases = {r["alias"]: r["canonical"] for r in csv.DictReader(f)}
    with (root / "ct6c00144_si_001.csv").open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row["Theory"] != "Reference":
                continue
            system = row["Rxn Name"]
            roles = {}
            for role in ("R1", "R2", "TS", "P1", "P2"):
                energy = float(row[role])
                if not energy:
                    continue
                name = f"{system}-{role}"
                # Some coordinates are also present under their own original name.
                keys = ["TRIP50/" + name, "TRIP50/" + aliases.get(name, name)]
                matches = [k for k in keys if abs(energies.get(k, 0.) - energy) < 2e-6]
                if not matches:
                    matches = [k for k,v in energies.items() if abs(v-energy) < 2e-6]
                    if len(matches) != 1:
                        raise ValueError(f"Unresolved TRIP50 geometry {name}")
                roles[role] = matches[0]
            rows.append(Reaction("TRIP50/"+system, "TRIP50/"+row["Rxn Type"],
                          [v for k,v in roles.items() if k.startswith("R")],
                          [v for k,v in roles.items() if k.startswith("P")], roles["TS"],
                          tuple(float(row[k]) for k in
                                ("Forward Kinetics", "Reverse Kinetics", "Thermodynaics"))))
    return geoms, rows


def reactions(charges):
    """Validate stoichiometry and protect every formula on the fixed singlet panel."""
    from experiments.reactivity.thermochemistry import panel
    protected = set()
    for _, terms, _, _ in panel():
        for _, p in terms:
            g = geometry(p.read_text().splitlines()[2:], 0, 1)
            protected.add((tuple(sorted(g.z)), g.state))
    geoms, rows, excluded = {}, [], Counter()
    for loader in (bh9, trip50):
        structures, records = loader()
        geoms.update(structures)
        for row in records:
            keys = [row.ts, *row.reactants, *row.products]
            if not all(structures[k] is not None for k in keys):
                excluded["size_or_element"] += 1
                continue
            if any(structures[k].state[0] not in charges for k in keys):
                excluded["charge_outside_backbone"] += 1
                continue
            if any((tuple(sorted(structures[k].z)), structures[k].state) in protected for k in keys):
                excluded["fixed_panel_formula_and_state"] += 1
                continue
            for eq in row.equations():
                balance = Counter()
                for k, c in eq:
                    for element in structures[k].z:
                        balance[element] += c
                    balance["charge"] += c * structures[k].state[0]
                if any(balance.values()) or not row.reactants or not row.products:
                    raise ValueError(f"Unbalanced reaction {row.name}")
            f, b, e = row.reference
            if abs(f-b-e) > .021:
                raise ValueError(f"Inconsistent relative energies for {row.name}")
            rows.append(row)
    return geoms, rows, excluded
