"""Run the PBE0 energy/force stage for the four requested HDF5 datasets.

Every valid parent is assigned once. All conformers, distortions, or scan
angles of that parent follow it. Identical geometry/state pairs in multiple
parents link those parents before splitting. Only the four named source files are opened.
"""

import hashlib
import sqlite3
import re

import h5py
import numpy as np

from experiments.full_public import backbone
from experiments.full_public.data import DATA, H5, fingerprint
from experiments.des15k.run import ATOMIC, electronic_labels

OUT = DATA / "requested_public"
DB = OUT / "pbe0.sqlite"
SOURCES = ("QM7X", "DES15K", "SPICE", "TorsionNet")


def prepare():
    OUT.mkdir(exist_ok=True)
    tmp = DB.with_suffix(".building")
    tmp.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp)
    conn.execute(
        "CREATE TABLE samples(source TEXT,name TEXT,parent TEXT,split TEXT,fp BLOB,q INTEGER,m INTEGER,z BLOB,r BLOB,energy REAL,force BLOB,ratios BLOB)"
    )
    for source in SOURCES:
        with h5py.File(H5[source]) as collection:
            for group, outer in collection.items():
                for name, data in outer.items():
                    z = data["atNUM"][:]
                    r = data["atXYZ"][:]
                    f = data["pbe0FOR"][:]
                    q, m = electronic_labels(data)
                    e = float(data["ePBE0"][0]) - sum(ATOMIC[a] for a in z)
                    if source == "QM7X":
                        parent = group
                    elif source == "DES15K":
                        parent = name.rsplit("-", 1)[0]
                    elif source == "SPICE":
                        parent = re.sub(r"\d+$", "", name)
                    else:
                        parent = name.rsplit("_conf_", 1)[0]
                    split = "train" if m is not None else "missing_spin"
                    if not (
                        np.isfinite(e) and np.isfinite(r).all() and np.isfinite(f).all()
                    ):
                        split = "invalid"

                    def blob(a):
                        return np.ascontiguousarray(a, dtype="<f8").tobytes()

                    conn.execute(
                        "INSERT INTO samples VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            source,
                            name,
                            parent,
                            split,
                            fingerprint(z, r),
                            int(q),
                            m,
                            z.astype("u1").tobytes(),
                            blob(r),
                            e,
                            blob(f),
                            blob(data["hRAT"][:].ravel()),
                        ),
                    )
        conn.commit()
    conn.execute("CREATE INDEX by_parent ON samples(source,parent)")
    conn.execute("CREATE UNIQUE INDEX by_name ON samples(source,name)")
    groups = {
        (s, p): (s, p)
        for s, p in conn.execute("SELECT DISTINCT source,parent FROM samples")
    }

    def representative(g):
        while groups[g] != g:
            groups[g] = groups[groups[g]]
            g = groups[g]
        return g

    prev = None
    last = None
    for fp, q, m, s, p in conn.execute(
        "SELECT fp,q,m,source,parent FROM samples WHERE m IS NOT NULL ORDER BY fp,q,m,source,parent"
    ):
        key = fp, q, m
        if prev == key:
            a, b = representative(last), representative((s, p))
            groups[max(a, b)] = min(a, b)
        last = (s, p)
        prev = key
    for s, p in groups:
        a, b = representative((s, p))
        fold = (
            int.from_bytes(
                hashlib.sha256(f"Korny-requested-v1/{a}/{b}".encode()).digest()[:8],
                "big",
            )
            % 10
        )
        split = "dev" if fold == 0 else "train"
        conn.execute(
            "UPDATE samples SET split=? WHERE source=? AND parent=? AND split NOT IN ('missing_spin','invalid')",
            (split, s, p),
        )
    conn.execute("CREATE INDEX by_split ON samples(split,source)")
    conn.commit()
    print(
        conn.execute(
            "SELECT source,split,count(*) FROM samples GROUP BY source,split ORDER BY source,split"
        ).fetchall(),
        flush=True,
    )
    assert not conn.execute(
        "SELECT 1 FROM samples WHERE split IN ('train','dev') GROUP BY fp,q,m HAVING count(DISTINCT split)>1 LIMIT 1"
    ).fetchone()
    conn.close()
    tmp.replace(DB)


if __name__ == "__main__":
    prepare()
    backbone.DB, backbone.OUT = DB, OUT
    backbone.run(5)
