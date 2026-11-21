"""Import AIMNet2-NSE and AutoMeKin electronic labels from the published archives.

Store source units, electronic method, molecular state, split and source IDs.
A joined reference table contains only actual endpoints or saddles. Arbitrary
scan frames describe relative energy surfaces and are never relabeled as TSs.
The frozen Korny development structures are protected against reuse by the
same orientation- and atom-order-independent geometry signature as its HDF data.
"""

import csv
import io
import json
import re
import sqlite3
import tarfile
import zipfile

import numpy as np
from rdkit import Chem

from experiments.full_public import kernel_trials as kt
from experiments.full_public.data import fingerprint
from src.units import HARTREE_EV

ROOT = kt.OUT.parent
OUT = ROOT / "paper_references"
AIM = ROOT / "AIMNet2_NSE"
CC = "DLPNO-CCSD(T)-F12/cc-pVDZ-F12"
DFT = "wB97X-D3/def2-TZVP"
WB = "wB97M-D3(BJ)/def2-TZVPP"
B3 = "B3LYP-D3(BJ)/def2-TZVP"
TPSS = "BASChem19 XYZ energies (TPSS-D3(BJ)/def2-TZVP per deposit)"
PT = Chem.GetPeriodicTable()


def xyz_geometry(lines):
    fields = [v.split() for v in lines if v.strip()]
    return np.array(
        [PT.GetAtomicNumber(x[0].capitalize()) for x in fields], np.uint8
    ), np.asarray([x[1:4] for x in fields], float)


def frames(handle):
    """Iterate complete XYZ records without loading a scan into memory."""
    for line in handle:
        if not line.strip() or line.strip() == ">":
            continue
        n = int(line.strip())
        comment = next(handle).strip()
        z, r = xyz_geometry([next(handle) for _ in range(n)])
        if n != len(z) or not np.isfinite(r).all():
            raise ValueError("Incomplete XYZ")
        yield comment, z, r


def put(
    db,
    source,
    system,
    group,
    name,
    role,
    method,
    q,
    m,
    z,
    r,
    energy,
    partition,
    frequencies=None,
):
    """Use eV internally. Identical geometries can have distinct spin/level/SCF roots."""
    if (int(z.sum()) - q - (m - 1)) % 2 or not np.isfinite(energy):
        raise ValueError((name, q, m, energy))
    sig = fingerprint(z, r)
    if sig in legacy:
        partition = "legacy_holdout"
    db.execute(
        "insert into geometries values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            name,
            source,
            system,
            group,
            role,
            method,
            q,
            m,
            z.tobytes(),
            np.ascontiguousarray(r, dtype="<f8").tobytes(),
            energy * HARTREE_EV,
            partition,
            sig,
            json.dumps(frequencies) if frequencies is not None else None,
            None,
        ),
    )


def basin(db):
    with (AIM / "BASChem19_summary_from_xyz.csv").open() as f:
        rows = list(csv.DictReader(f))
    with zipfile.ZipFile(AIM / "BASChem19_dataset.zip") as arc:
        names = set(arc.namelist())
        for rank, row in enumerate(rows, 1):
            system = row["Reaction"]
            root = "BASChem_19/" + system + "/"

            def state(key, root=root):
                return (
                    int(arc.read(root + key).decode().split()[-1])
                    if root + key in names
                    else 0
                )

            q, m = state("chrg.txt"), 1 + state("uhf.txt")
            split = "validation" if rank in (2, 7, 11, 17) else "train"
            ids = []
            vals = []
            for file, role in [("MIN", "reactant"), ("TS", "saddle")]:
                key = root + file + ".xyz"
                c, z, r = next(frames(io.StringIO(arc.read(key).decode())))
                e = float(c.split("=")[-1])
                vals.append(e)
                name = "BASChem19/" + system + "/" + file
                put(
                    db,
                    "BASChem19",
                    system,
                    system,
                    name,
                    role,
                    TPSS,
                    q,
                    m,
                    z,
                    r,
                    e,
                    split,
                )
                ids.append(name)
            np.testing.assert_allclose(
                (vals[1] - vals[0]) * 627.509,
                float(row["Activation_Barrier_kcal/mol"]),
                atol=0.002,
                rtol=0,
            )
            db.execute(
                "insert into channels values(?,?,?,?,?,?,?)",
                ("BASChem19", system, TPSS, system, ids[0], ids[1], None),
            )


def aim(db):
    with zipfile.ZipFile(AIM / "paper-data.zip") as arc:
        for path in arc.namelist():
            if not path.endswith(
                ("_trj.xyz", "/product.xyz", "/reactant.xyz", "NEB-TS_converged.xyz")
            ):
                continue
            system, case = path.split("/", 1)
            method = B3 if system == "AIBN" else WB
            # The UKS singlet can follow another electronic root at the same
            # multiplicity. Retain its identity explicitly in the archive.
            if "singlet_uks" in case:
                method += " (UKS singlet)"
            m = (
                2
                if system == "2-methylene-1_3-dioxepane"
                else 3
                if "triplet" in case
                else 1
            )
            split = "validation" if system == "benzopinacol" else "train"
            with arc.open(path) as handle:
                for i, (c, z, r) in enumerate(frames(io.TextIOWrapper(handle))):
                    match = re.search(r"\bE\s+([-+\d.Ee]+)\s*$", c)
                    if match is None:
                        continue
                    e = float(match[1])
                    put(
                        db,
                        "AIMNet2-NSE supplement",
                        system,
                        case.rsplit("/", 1)[0] if "/" in case else "IRC",
                        "AIMNet2-NSE/" + path + "/" + str(i),
                        "IRC" if m == 2 else "scan",
                        method,
                        0,
                        m,
                        z,
                        r,
                        e,
                        split,
                    )


def auto(db):
    for archive in sorted((ROOT / "AutoMeKin_2026").glob("*.tar")):
        with tarfile.open(archive) as tar:
            system = archive.stem
            cc_method = "CCSD(T)-F12/cc-pVDZ-F12" if system == "MEA" else CC
            partition = "validation" if system == "cBD-CN" else "train"
            names = tar.getnames()
            root = next(
                n.rsplit("/", 1)[0]
                for n in names
                if "/FINAL_HL_" in n and n.endswith("/RXNet")
            )
            summary = next(n for n in names if n.endswith("cc_energies.csv"))
            with tar.extractfile(summary) as f:
                cc = {v["label"]: v for v in csv.DictReader(io.TextIOWrapper(f))}
            # Recover the full-system input charge and spin directly from
            # independent published single point logs for this same network.
            states = set()
            for path in names:
                if "/CC_OUTPUTS/" not in path or not re.search(
                    r"/(?:TS|MIN)_\d+\.out$", path
                ):
                    continue
                body = tar.extractfile(path).read().decode(errors="replace")
                state = re.search(r"\*\s+xyz\s+(-?\d+)\s+(\d+)", body, re.IGNORECASE)
                if state:
                    states.add(tuple(map(int, state.groups())))
            if len(states) != 1:
                raise ValueError((system, states))
            q, m = states.pop()
            found = {}
            for table in ("min", "ts"):
                coord = sqlite3.connect(":memory:")
                coord.deserialize(tar.extractfile(root + "/" + table + ".db").read())
                for i, nat, e, geom, freq in coord.execute(
                    f"select id,natom,energy,geom,freq from {table}"
                ):
                    z, r = xyz_geometry(geom.splitlines())
                    if nat != len(z):
                        raise ValueError((system, table, i))
                    name = table.upper() + str(i)
                    values = [(DFT, float(e), m)]
                    ref = cc.get(table.upper() + "_" + str(i))
                    if ref and ref["status"] == "OK" and int(ref["mult"]) == m:
                        values.append((cc_method, float(ref["energy_Eh"]), m))
                    for method, energy, mult in values:
                        key = f"AutoMeKin/{system}/{method}/{name}"
                        fvals = (
                            [float(v) for v in freq.split()]
                            if freq and method == DFT
                            else None
                        )
                        put(
                            db,
                            "AutoMeKin",
                            system,
                            system,
                            key,
                            "saddle" if table == "ts" else "minimum",
                            method,
                            q,
                            mult,
                            z,
                            r,
                            energy,
                            partition,
                            fvals,
                        )
                        found[(method, name)] = key
                coord.close()
            # Use the deposited TS-to-endpoint network. A connected minimum–
            # minimum path defines three relative energies on the same surface.
            lines = tar.extractfile(root + "/RXNet").read().decode().splitlines()
            for line in lines:
                matched = re.match(
                    r"\s*(\d+)\s+[-.\d]+\s+MIN\s+(\d+)\s+<--->\s+MIN\s+(\d+)", line
                )
                if not matched:
                    continue
                ts, left, right = matched.groups()
                if left == right:
                    continue
                for method in (DFT, cc_method):
                    keys = [
                        found.get((method, key))
                        for key in ("MIN" + left, "TS" + ts, "MIN" + right)
                    ]
                    if all(keys):
                        db.execute(
                            "insert or ignore into channels values(?,?,?,?,?,?,?)",
                            ("AutoMeKin", system, method, ts, *keys),
                        )


def build():
    import pickle

    global legacy
    geoms, _, _ = kt.inputs()
    with (kt.WORK / "ids.pkl").open("rb") as f:
        _, names = pickle.load(f)
    held = set()
    for target in kt.TARGETS:
        d = kt.equation_data(target)
        held.update(d[0][d[3] == -1].indices.tolist())
    legacy = {fingerprint(geoms[names[i]].z, geoms[names[i]].r) for i in held}
    OUT.mkdir(exist_ok=True)
    dest = OUT / "electronic.sqlite"
    tmp = dest.with_suffix(".pending")
    if tmp.exists():
        tmp.unlink()
    with sqlite3.connect(tmp) as db:
        db.executescript("""
        create table geometries(id text primary key,source text,system text,series text,role text,method text,charge integer,mult integer,z blob,r blob,energy_eV real,split text,signature blob,frequencies_cm1 text,duplicate_of text);
        create table channels(source text,system text,method text,channel text,reactant text,ts text,product text,primary key(source,system,method,channel));
        create index bygeometry on geometries(method,charge,mult,signature);
        """)
        for reader in (basin, aim, auto):
            reader(db)
            print("Imported", reader.__name__, flush=True)
        # If rounded coordinates recur with incompatible energies, retain the
        # raw rows but exclude all conflicting labels. For consistent copies
        # select one representative, with held-out roles taking precedence.
        for method, q, m, sig, low, high in db.execute(
            "select method,charge,mult,signature,min(energy_eV),max(energy_eV) from geometries group by method,charge,mult,signature having count(*)>1"
        ).fetchall():
            records = db.execute(
                'select id,split from geometries where method=? and charge=? and mult=? and signature=? order by case split when "legacy_holdout" then 0 when "validation" then 1 else 2 end,id',
                (method, q, m, sig),
            ).fetchall()
            if high - low > 0.01:
                db.executemany(
                    "update geometries set split='conflicting' where id=?",
                    [(v[0],) for v in records],
                )
            else:
                db.executemany(
                    "update geometries set duplicate_of=?,split=? where id=?",
                    [(records[0][0], records[0][1], v[0]) for v in records[1:]],
                )
        inventory = db.execute(
            "select source,method,split,count(*),sum(duplicate_of is null) from geometries group by source,method,split"
        ).fetchall()
        channels = db.execute(
            "select source,method,count(*) from channels group by source,method"
        ).fetchall()
        print("Geometry and channel inventory", inventory, channels, flush=True)
    tmp.replace(dest)
    kt.write_json(
        OUT / "inventory.json",
        {
            "columns": ["source", "method", "split", "raw", "unique"],
            "geometries": inventory,
            "channels": channels,
        },
    )


if __name__ == "__main__":
    build()
