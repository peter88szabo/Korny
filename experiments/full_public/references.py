"""Inventory the requested public reaction labels, electronic states and splits.

Use electronic totals from the matched quantum-chemistry output. Identify all
three stationary points of a reaction before computing either direction. Keep
each transition-state formula together across the barrier collections. Relative
PBE0, high-level electronic and radical DFT references have separate targets.
"""

import ast
import csv
import hashlib
import json
import pickle
import re
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from rdkit import Chem

from experiments.full_public.specified import OUT
from experiments.reactivity.high_level_fit import DATA as GMT
from experiments.reactivity.thermochemistry import EV_TO_KCAL
from experiments.reactivity import rdb7, spin_data
from experiments.reactivity.spin_data import DATA, Geometry
from experiments.reactivity.radical_curvatures import ARCHIVE as RAD
from src.units import HARTREE_EV

PERIODIC = Chem.GetPeriodicTable()


def fold(key):
    return (
        "validation"
        if int.from_bytes(
            hashlib.sha256(("public-reaction-v1/" + str(key)).encode()).digest()[:8],
            "big",
        )
        % 10
        == 0
        else "train"
    )


def xyz(lines, q, m):
    rows = [s.split() for s in lines if s.strip()]
    z = np.array([PERIODIC.GetAtomicNumber(s[0].capitalize()) for s in rows])
    if not len(z) or not z.all():
        raise ValueError("Invalid element in XYZ file")
    return Geometry(z, np.array([s[1:4] for s in rows], dtype=float), (int(q), int(m)))


def gmtkn():
    """Match every PBE0 row to its published equation, including ionic conventions."""
    refs = defaultdict(list)
    with (GMT / "PBEh-3c_reactions.csv").open() as f:
        for row in csv.DictReader(f):
            refs[row["Subset"]].append(row)
    gs = {}
    audit = Counter()
    rows = []
    with (DATA / "GMTKN55_PBE0/d2cdft.csv").open() as f:
        for row in csv.DictReader(f):
            if row["calctype"] != "PBE0":
                continue
            family = row["setname"]
            idx = int(row["rxnidx"])
            entry = refs[family][idx - 1]
            terms = []
            if (
                family != "WATER27"
                and abs(float(entry["ReferenceValue"]) - float(row["Eref"])) > 0.051
            ):
                audit["version_mismatch"] += 1
                continue
            folder = "BH76" if family == "BH76RC" else family
            for k, c in zip(
                ast.literal_eval(entry["Reaction"]),
                ast.literal_eval(entry["Stochiometry"]),
                strict=True,
            ):
                path = GMT / folder / k
                key = f"GMTKN55/{folder}/{k}"
                if key not in gs:
                    if not (path / "struc.xyz").is_file():
                        break
                    flags = [
                        int((path / name).read_text()) if (path / name).exists() else 0
                        for name in (".CHRG", ".UHF")
                    ]
                    gs[key] = xyz(
                        (path / "struc.xyz").read_text().splitlines()[2:],
                        flags[0],
                        1 + flags[1],
                    )
                terms.append((key, c))
            else:
                balance = Counter()
                for k, c in terms:
                    for z in gs[k].z:
                        balance[int(z)] += c
                nonzero = {z: v for z, v in balance.items() if v}
                if nonzero:
                    if set(nonzero) == {1} and (
                        family in ("PA26", "SIE4x4") or (family, idx) == ("G21IP", 1)
                    ):
                        # Add the omitted bare proton explicitly. The total
                        # electronic energy of H+ is exactly zero. In the
                        # atom-referenced convention its energy is -E(H).
                        key = "GMTKN55/H+"
                        gs[key] = Geometry(np.array([1]), np.zeros((1, 3)), (1, 1))
                        terms.append((key, -nonzero[1]))
                    else:
                        audit[f"unbalanced_{family}_{idx}_{nonzero}"] += 1
                        continue
                rows.append(
                    (
                        "pbe0",
                        "GMTKN55",
                        folder,
                        f"{family}/{idx}",
                        terms,
                        float(row["Ecalc"]) / EV_TO_KCAL,
                    )
                )
                continue
            audit["missing_geometry"] += 1
    return gs, rows, audit


def collect():
    OUT.mkdir(exist_ok=True)
    allg = {}
    rows = []
    removed = {}
    groups = {}

    def add(target, source, group, name, terms, y, kind):
        rows.append((target, source, fold(group), name, terms, float(y), kind))

    def add_reaction(source, gs, rec, target):
        roles = [rec.ts, *rec.reactants, *rec.products]
        if not all(gs.get(k) is not None for k in roles):
            return False
        formula = tuple(sorted(map(int, gs[rec.ts].z)))
        for eq, y, kind in zip(
            rec.equations(),
            rec.reference,
            ("forward", "reverse", "reaction"),
            strict=True,
        ):
            add(target, source, formula, rec.name, eq, y / EV_TO_KCAL, kind)
        groups[rec.name] = rec.family
        return True

    # BH9 includes heavier elements. Read their published atomic numbers
    # directly instead of applying the historical organic-pilot element mask.
    gs, recs, skip = rdb7.dataset(accept_verified_table=True)
    for rec in recs:
        add_reaction("RDB7", gs, rec, "high")
    allg.update(gs)
    removed["RDB7"] = dict(skip)
    print("RDB7", len(recs), flush=True)
    gs, recs = spin_data.bh9(max_atoms=None)
    with ZipFile(DATA / "BH9/ct2c00362_si_001.zip") as arc:
        for fn in arc.namelist():
            if "/XYZ_files/" in fn and fn.endswith(".xyz"):
                lines = arc.read(fn).decode().splitlines()
                gs["BH9/" + Path(fn).stem] = xyz(lines[2:], *map(int, lines[1].split()))
    for rec in recs:
        add_reaction("BH9", gs, rec, "high")
    allg.update(gs)
    print("BH9", len(recs), flush=True)
    gs, data, skip = gmtkn()
    allg.update(gs)
    removed["GMTKN55"] = dict(skip)
    for target, source, family, name, terms, y in data:
        add(target, source, ("GMT", family), name, terms, y, "energy_difference")
    print("GMTKN55", len(data), skip, flush=True)
    from experiments.reactivity.osos_fit import load

    gs, targets, eqs, inconsistent = load()
    removed["OSOS"] = {"inconsistent_parents": inconsistent}
    for k, g in gs.items():
        allg["OSOS/" + k] = g
        group = tuple(sorted(int(z) for z in g.z if z != 1))
        for target, y in zip(("pbe0", "high"), targets[k], strict=True):
            add(target, "OSOS", group, k, [("OSOS/" + k, 1)], y, "atomization")
    # A failed CC thermochemical cycle does not invalidate an independently
    # tabulated PBE0 energy. Retain all such bare-PBE0 structures too.
    with ZipFile(DATA / "OSOS/jp6c03915_si_002.zip") as arc:
        from io import StringIO

        abs_e = {
            k: float(e)
            for k, e in csv.reader(
                StringIO(arc.read("SI_Absolute_Energies/PBE0.csv").decode("utf-8-sig"))
            )
        }
        for fn in arc.namelist():
            if not (fn.startswith("SI_REF_DATA/XYZ/") and fn.endswith(".xyz")):
                continue
            k = Path(fn).stem
            if k in gs or k not in abs_e:
                continue
            lines = arc.read(fn).decode().splitlines()
            g = xyz(lines[2:], *map(int, lines[1].split()[-2:]))
            energy = (
                abs_e[k] - sum(abs_e[PERIODIC.GetElementSymbol(int(z))] for z in g.z)
            ) * HARTREE_EV
            key = "OSOS/" + k
            allg[key] = g
            add(
                "pbe0",
                "OSOS",
                tuple(sorted(int(z) for z in g.z if z != 1)),
                k,
                [(key, 1)],
                energy,
                "atomization",
            )
    for typ, entries in eqs.items():
        if typ == "atomization":
            continue
        for left, right, ref in entries:
            k = next(k for k in left + right if k != "H_atom")
            group = tuple(sorted(int(z) for z in gs[k].z if z != 1))
            terms = [("OSOS/" + n, -1) for n in left] + [
                ("OSOS/" + n, 1) for n in right
            ]
            for target, i in [("pbe0", 0), ("high", 1)]:
                y = sum(c * targets[n[5:]][i] for n, c in terms)
                add(
                    target,
                    "OSOS",
                    group,
                    "+".join(left) + ">" + "+".join(right),
                    terms,
                    y,
                    typ,
                )
    print("OSOS", len(gs), flush=True)
    logs = defaultdict(dict)
    skips = Counter()
    with tarfile.open(RAD, "r|gz") as arc:
        for member in arc:
            match = re.search(r"/rxn(\d+)/(ts|p|r)\d+\.log$", member.name)
            if not match:
                continue
            case, role = match.groups()
            s = arc.extractfile(member).read().decode(errors="replace")
            tail = s.rsplit("Standard Nuclear Orientation (Angstroms)", 1)[-1]
            coords = re.findall(
                r"^\s*\d+\s+([A-Z][a-z]?)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$",
                tail.split("Nuclear Repulsion Energy")[0],
                re.M,
            )
            energies = re.findall(
                r"Total energy in the final basis set\s*=\s*([-\d.]+)", tail, re.I
            )
            state = re.search(r"\$molecule\s+(-?\d+)\s+(\d+)\s", s, re.I)
            converged = "OPTIMIZATION CONVERGED" in s
            # A few files end during the Hessian, after both the geometry
            # optimization and the subsequent single-point SCF completed.
            if not (
                coords
                and energies
                and state
                and (converged or "Thank you very much for using Q-Chem" in tail)
            ):
                skips["incomplete_log"] += 1
                continue
            key = "Grambow/" + case + "/" + role
            g = xyz([" ".join(t) for t in coords], *map(int, state.groups()))
            allg[key] = g
            logs[case][role] = (key, float(energies[-1]) * HARTREE_EV)
    for case, roles in logs.items():
        if len(roles) != 3:
            skips["incomplete_reaction"] += 1
            continue
        (r, er), (p, ep), (t, et) = [roles[k] for k in ("r", "p", "ts")]
        group = tuple(sorted(map(int, allg[t].z)))
        for a, b, y, kind in [
            (r, t, et - er, "forward"),
            (p, t, et - ep, "reverse"),
            (r, p, ep - er, "reaction"),
        ]:
            add("wb97xd3", "Grambow", group, case, [(a, -1), (b, 1)], y, kind)
    removed["Grambow"] = dict(skips)
    print("Grambow", len(logs), skips, flush=True)
    keys = {k for *_, eq, y, kind in rows for k, _ in eq}
    allg = {k: allg[k] for k in keys}
    with (OUT / "reaction-data.pkl").open("wb") as f:
        pickle.dump((allg, rows, groups), f, protocol=5)
    counts = Counter((t, s, split, kind) for t, s, split, name, eq, y, kind in rows)
    info = dict(
        geometries=len(allg),
        excluded=removed,
        counts={"/".join(k): v for k, v in sorted(counts.items())},
    )
    (OUT / "reaction-inventory.json").write_text(json.dumps(info, indent=2) + "\n")
    print("reaction equations", len(rows), "geometry IDs", len(allg), flush=True)


if __name__ == "__main__":
    collect()
