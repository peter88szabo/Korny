"""Recover electronic supervision from the downloaded training archives.

RDB7 and Grambow supply atom-mapped endpoint SMILES. Atom maps locate formal
charge, radical electrons and pi valence. A labelled endpoint must reproduce
the element-labelled bond graph of the Cartesian structure. Formal radical
counts depend on a SMILES representation and are distinct from measured spin
densities. QM7-X separately supplies physical Hirshfeld partial charges.
"""

import csv
from collections import Counter, defaultdict
import pickle
import re
import sqlite3

import h5py
import numpy as np
from rdkit import Chem, RDLogger

from experiments.full_public import kernel_trials as kt
from experiments.full_public.data import H5, DB, fingerprint
from src.four_body import RADII

WORK = kt.WORK
PARAM = Chem.SmilesParserParams()
PARAM.removeHs = False


def molecule(smiles):
    mol = Chem.MolFromSmiles(smiles, PARAM)
    if mol is None:
        return []
    frags = Chem.GetMolFrags(mol)
    candidates = [list(range(mol.GetNumAtoms()))]
    if len(frags) > 1:
        candidates.extend(frags)
    indices = {a.GetIdx(): a.GetAtomMapNum() for a in mol.GetAtoms()}
    if sorted(indices.values()) != list(range(1, len(indices) + 1)):
        return []
    result = []
    for fragment in candidates:
        order = sorted(fragment, key=indices.get)
        atoms = [mol.GetAtomWithIdx(i) for i in order]
        z = np.array([a.GetAtomicNum() for a in atoms])
        q = [a.GetFormalCharge() for a in atoms]
        rad = [a.GetNumRadicalElectrons() for a in atoms]
        pi = [sum(b.GetBondTypeAsDouble() - 1 for b in a.GetBonds()) for a in atoms]
        pos = {idx: i for i, idx in enumerate(order)}
        graph = np.zeros((len(z), len(z)), bool)
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if i in pos and j in pos:
                graph[pos[i], pos[j]] = graph[pos[j], pos[i]] = True
        result.append((z, np.column_stack((q, rad, pi)), graph))
    return result


def match(g, candidates):
    """Verify mapped atom order and full adjacency, including explicit hydrogens."""
    dist = np.linalg.norm(g.r[:, None] - g.r[None, :], axis=-1)
    radius = np.array([RADII[int(z)] for z in g.z])
    ratio = dist / (radius[:, None] + radius)
    np.fill_diagonal(ratio, np.inf)
    matched = []
    for z, label, graph in candidates:
        if not np.array_equal(z, g.z) or abs(label[:, 0].sum() - g.state[0]) > 0.001:
            continue
        if any(np.array_equal(ratio < t, graph) for t in (1.26, 1.20, 1.32, 1.38)):
            matched.append(label)
    if matched and all(np.array_equal(matched[0], lab) for lab in matched[1:]):
        return matched[0]

    # Separated RDB products may have a different atom order. Use only exact
    # graph isomorphisms. If symmetry leaves unlike labels indistinguishable,
    # omit that assignment rather than assigning a spin to an arbitrary atom.
    def bare_graph(z, adj):
        mol = Chem.RWMol()
        for number in z:
            atom = Chem.Atom(int(number))
            atom.SetNoImplicit(True)
            mol.AddAtom(atom)
        for i, j in np.argwhere(np.triu(adj, 1)):
            mol.AddBond(int(i), int(j), Chem.BondType.SINGLE)
        return mol.GetMol()

    for z, label, graph in candidates:
        if sorted(z) != sorted(g.z) or abs(label[:, 0].sum() - g.state[0]) > 0.001:
            continue
        target = bare_graph(z, graph)
        for t in (1.26, 1.20, 1.32, 1.38):
            adj = ratio < t
            if graph.sum() != adj.sum():
                continue
            permutations = target.GetSubstructMatches(
                bare_graph(g.z, adj), uniquify=False, maxMatches=257
            )
            if permutations and len(permutations) < 257:
                possible = label[np.asarray(permutations)]
                if np.all(possible == possible[:1]):
                    matched.append(possible[0])
                break
    if matched and all(np.array_equal(matched[0], lab) for lab in matched[1:]):
        return matched[0]


def formal_labels():
    gs, rows, _ = kt.inputs()
    with (WORK / "ids.pkl").open("rb") as f:
        ids, _ = pickle.load(f)
    roles = defaultdict(set)
    for target in kt.TARGETS:
        d = kt.equation_data(target)
        for i, h in enumerate(d[3]):
            for j in d[0].indices[d[0].indptr[i] : d[0].indptr[i + 1]]:
                roles[j].add(int(h))
    tables = {}
    for src, path in (
        ("RDB7", "RDB7/wb97xd3.csv"),
        ("Grambow", "Grambow_radicals/wb97xd3_rad.csv"),
    ):
        with (kt.OUT.parent / path).open() as f:
            tables[src] = {int(row["idx"]): row for row in csv.DictReader(f)}
    count = Counter()
    seen = set()
    output = []
    for key, g in gs.items():
        parts = key.split("/")
        if parts[0] not in tables or parts[-1][0] not in ("r", "p"):
            continue
        idx = ids[key]
        if idx in seen:
            continue
        h = roles[idx]
        if not h or -1 in h:
            continue
        part = "dev" if h & {0, 1} else "fit"
        name = int(re.search(r"\d+", parts[-2])[0])
        data = tables[parts[0]][name]
        raw = match(g, molecule(data["rsmi" if parts[-1][0] == "r" else "psmi"]))
        if raw is None:
            count["unverified map"] += 1
            continue
        seen.add(idx)
        count[f"{parts[0]} {part} endpoints"] += 1
        count[f"{parts[0]} {part} atoms"] += len(raw)
        output.append((key, part, g.z, g.r, *g.state, raw))
    with (WORK / "atom-state-labels.pkl").open("wb") as f:
        pickle.dump(output, f)
    kt.write_json(WORK / "atom-state-labels.json", dict(count))
    print("Verified endpoint atom maps", dict(count), flush=True)


def hirshfeld():
    """Recover hCHG on the exact existing QM7-X split, checking geometry and net charge."""
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as db:
        rows = list(
            db.execute(
                "select fp,name,split,q,m from samples where source='QM7X' order by name"
            )
        )
    gs, _, _ = kt.inputs()
    with (WORK / "ids.pkl").open("rb") as f:
        ids, _ = pickle.load(f)
    excluded = set()
    for t in kt.TARGETS:
        d = kt.equation_data(t)
        excluded.update(d[0][d[3] < 2].indices)
    blocked = {
        (fingerprint(g.z, g.r), *g.state)
        for name, g in gs.items()
        if ids[name] in excluded
    }
    seen = set()
    output = []
    count = Counter()
    worst = 0.0
    with h5py.File(H5["QM7X"], "r") as archive:
        for fp, name, split, q, m in rows:
            key = (fp, q, m)
            if key in seen or key in blocked:
                continue
            seen.add(key)
            group = archive[re.search(r"m(\d+)", name)[1]][name]
            z = group["atNUM"][:].ravel().astype(int)
            r = group["atXYZ"][:]
            val = group["hCHG"][:].ravel()
            if (
                fingerprint(z, r) != fp
                or val.shape != z.shape
                or not np.isfinite(val).all()
            ):
                raise ValueError(f"QM7-X atomic-charge alignment failed for {name}")
            delta = abs(val.sum() - q)
            if delta > 0.025:
                count["sum disagreement"] += 1
                continue
            worst = max(worst, delta)
            output.append((name, split, z, r, q, m, val))
            count[split + " geometries"] += 1
            count[split + " atoms"] += len(z)
    with (WORK / "qm7x-charge-labels.pkl").open("wb") as f:
        pickle.dump(output, f)
    kt.write_json(
        WORK / "qm7x-charge-labels.json", dict(count, max_net_charge_error=worst)
    )
    print("Matched QM7-X hCHG", dict(count), "max charge-sum error", worst, flush=True)


if __name__ == "__main__":
    RDLogger.DisableLog("rdApp.warning")
    formal_labels()
    hirshfeld()
