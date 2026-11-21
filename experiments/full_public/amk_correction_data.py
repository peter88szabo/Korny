"""Import the authors' optimized structures from Zenodo 18714568.

Each geometry keeps its baseline method, reaction, role, coordinates and the
as-deposited energy. MACE predictions are model outputs, not QM force labels.
Only RDB7 records whose molecular state can be recovered from the original
reaction ledger enter the new source fit. Retain the other networks for audit.
"""

import hashlib
import re
import shlex
import sqlite3
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit.Chem import GetPeriodicTable

from experiments.full_public import kernel_trials as kt
from experiments.full_public.data import fingerprint

ROOT = kt.OUT.parent / "AMK_ML_corrections"
ARCHIVE = ROOT / "data_opt_qm_ml.tar"
DATABASE = ROOT / "stationary.sqlite"
DIGEST = "aa6b60d8d8969b456597e37233fb8445"
PERIODIC_TABLE = GetPeriodicTable()


def ledger():
    """Recover state and split from identified original RDB7 participants."""
    geoms, rows, _ = kt.inputs()
    roles = defaultdict(set)
    splits = {}
    held = set()
    for _, src, split, case, terms, _, kind in rows:
        if split == "validation":
            for k, _ in terms:
                g = geoms[k]
                held.add((fingerprint(g.z, g.r), *g.state))
        if src != "RDB7" or kind not in ("forward", "reverse"):
            continue
        splits[case] = split
        for key, sign in terms:
            role = (
                "transition_state"
                if sign > 0
                else {"forward": "reactant", "reverse": "product"}[kind]
            )
            g = geoms[key]
            roles[case, role, tuple(sorted(map(int, g.z)))].add(g.state)
    return roles, splits, held


def xyz_frames(stream):
    """Stream ASE extended XYZ frames, locating columns via their schema."""
    for line in stream:
        if not line.strip():
            continue
        n = int(line)
        meta = dict(
            v.split("=", 1) for v in shlex.split(stream.readline().decode()) if "=" in v
        )
        fields = meta["Properties"].split(":")
        offsets = {}
        width = 0
        for key, _, size in zip(fields[::3], fields[1::3], fields[2::3], strict=True):
            offsets[key] = (width, width + int(size))
            width += int(size)
        pos = slice(*offsets["pos"])
        z = np.empty(n, dtype="u1")
        r = np.empty((n, 3), dtype="<f8")
        for i in range(n):
            atoms = stream.readline().split()
            if len(atoms) != width:
                raise ValueError("Invalid extended XYZ atom row")
            z[i] = PERIODIC_TABLE.GetAtomicNumber(atoms[offsets["species"][0]].decode())
            r[i] = atoms[pos]
        if not np.all(np.isfinite(r)):
            raise ValueError("Nonfinite geometry")
        yield meta, z, r


def run():
    with ARCHIVE.open("rb") as f:
        if hashlib.file_digest(f, "md5").hexdigest() != DIGEST:
            raise ValueError("The archive differs from the publisher's checksum")
    roles, splits, held = ledger()
    tmp = DATABASE.with_suffix(".pending")
    tmp.unlink(missing_ok=True)
    with sqlite3.connect(tmp) as db, tarfile.open(ARCHIVE, "r|") as tar:
        db.execute("""CREATE TABLE frames(member TEXT, frame INTEGER, dataset TEXT, method TEXT,
            case_id TEXT, role TEXT, subset TEXT, split TEXT, q INTEGER, m INTEGER,
            fp BLOB, z BLOB, r BLOB, published_baseline_eV REAL, mace_prediction_eV REAL,
            PRIMARY KEY(member,frame))""")
        for item in tar:
            if not item.isfile() or not item.name.endswith(".xyz"):
                continue
            path = Path(item.name)
            dataset, method = path.parts[1], path.stem.split("_")[0]
            for i, (meta, z, r) in enumerate(xyz_frames(tar.extractfile(item))):
                raw_id = meta["reaction_id"]
                role = meta["structure_type"]
                state = (None, None)
                case = raw_id
                split = "external"
                fp = fingerprint(z, r)
                if dataset == "RDB7":
                    match = re.fullmatch(r"(rxn\d+)(?:\.xyz)?", raw_id)
                    case = "RDB7/" + match[1] if match else raw_id
                    states = roles.get((case, role, tuple(sorted(map(int, z)))), set())
                    split = "unmatched"
                    if len(states) == 1:
                        state = next(iter(states))
                        split = splits[case]
                        if split == "train" and (fp, *state) in held:
                            split = "protected"
                energy = float(meta["energy_" + method])
                prediction = (
                    float(meta["MACE_energy"]) if "MACE_energy" in meta else None
                )
                if not np.isfinite(energy):
                    raise ValueError(
                        f"Invalid baseline energy in {item.name} frame {i}"
                    )
                db.execute(
                    "INSERT INTO frames VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        item.name,
                        i,
                        dataset,
                        method,
                        case,
                        role,
                        meta["reaction_type"],
                        split,
                        *state,
                        fp,
                        z.tobytes(),
                        r.tobytes(),
                        energy,
                        prediction,
                    ),
                )
        db.execute("CREATE INDEX partitions ON frames(dataset,method,split,case_id)")
        db.execute("CREATE INDEX geometry ON frames(fp,q,m)")
        # Reoptimization can bring a shared fragment to the same coordinates
        # under several reaction IDs. Protect new held geometries at all levels.
        db.execute("""UPDATE frames AS a SET split='protected' WHERE a.split='train'
            AND EXISTS (SELECT 1 FROM frames b WHERE b.split='validation' AND
            b.fp=a.fp AND b.q=a.q AND b.m=a.m)""")
        counts = db.execute("""SELECT dataset,method,split,count(*),count(distinct case_id)
            FROM frames GROUP BY dataset,method,split ORDER BY dataset,method,split""").fetchall()
    tmp.replace(DATABASE)
    kt.write_json(
        ROOT / "inventory.json",
        {
            "archive_md5": DIGEST,
            "columns": ["dataset", "baseline", "split", "frames", "reaction_ids"],
            "counts": counts,
        },
    )
    print("Verified and imported baseline energies and geometries", counts, flush=True)


if __name__ == "__main__":
    run()
