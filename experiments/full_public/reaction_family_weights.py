"""Identify reaction families for training weights and retrospective error analysis.

BH9 and assigned RDB7 rows use published families. Unassigned RDB7 rows and
Grambow use a coarse atom-mapped connectivity edit signature. This captures
bond partners and fragment count changes without assuming a specific mechanism.
The calculator itself only receives geometry, total charge, and multiplicity.
"""

import csv
from collections import Counter
from functools import lru_cache

import numpy as np
from rdkit import Chem

from experiments.full_public.data import DATA
from experiments.reactivity.reaction_classes import lookup


def mapped_graph(smiles):
    params = Chem.SmilesParserParams()
    params.sanitize = False
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ValueError("Unreadable mapped reaction component")
    atoms = {a.GetAtomMapNum(): a.GetSymbol() for a in mol.GetAtoms()}
    if 0 in atoms or len(atoms) != mol.GetNumAtoms():
        return None
    bonds = {
        tuple(
            sorted((b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()))
        )
        for b in mol.GetBonds()
    }
    return atoms, bonds, len(Chem.GetMolFrags(mol))


def edit_family(reactant, product):
    left, right = mapped_graph(reactant), mapped_graph(product)
    if left is None or right is None or left[0] != right[0]:
        return "unmapped"
    atoms, before, nleft = left
    _, after, nright = right

    def partners(bonds):
        pairs = ["-".join(sorted((atoms[i], atoms[j]))) for i, j in bonds]
        return "+".join(sorted(pairs)) or "none"

    # Inverting the reaction exchanges endpoints and leaves its family intact.
    # Bond-order or electronic changes at fixed connectivity share one bin.
    edits = sorted((partners(before - after), partners(after - before)))
    return f"{min(nleft,nright)}↔{max(nleft,nright)} fragments, {edits[0]} ↔ {edits[1]}"


@lru_cache(maxsize=1)
def families():
    published = lookup()
    result = {("BH9", k): v for k, v in published.items() if k.startswith("BH9/")}
    for source, path in (
        ("RDB7", "RDB7/ccsdtf12_dz.csv"),
        ("Grambow", "Grambow_radicals/wb97xd3_rad.csv"),
    ):
        with (DATA / path).open() as stream:
            for row in csv.DictReader(stream):
                idx = row["idx"].zfill(6)
                key = "RDB7/rxn" + idx if source == "RDB7" else idx
                label = published.get(key, "unassigned")
                if label == "unassigned":
                    label = edit_family(row["rsmi"], row["psmi"])
                result[source, key] = label
    return result


def training_weights(data):
    """Flatten family frequencies partially, preserving each source's total weight."""
    _, _, sources, fold, kinds, cases = data
    result = np.ones(len(fold))
    labels = families()
    for source in ("RDB7", "BH9", "Grambow"):
        train = (fold >= 0) & (sources == source)
        rx = cases[train & (kinds == "reaction")]
        if not len(rx):
            continue
        freq = Counter(labels[source, k] for k in rx)
        # Four pseudo-observations temper the leverage of singleton classes.
        weights = np.array(
            [1 / np.sqrt(4 + freq[labels[source, k]]) for k in cases[train]]
        )
        result[train] = weights / weights.mean()
    return result
