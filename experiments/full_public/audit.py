"""Count unique fitted geometry/state pairs and test validation saddle overlap."""

import json
import sqlite3
import numpy as np
from experiments.full_public.specified import DB, OUT
from experiments.full_public.data import fingerprint, load_reaction_data
from experiments.reactivity.stationary_fit import POINTS, MASSES
from experiments.reactivity.larger_vibrations import references


def run():
    gs, rows, _ = load_reaction_data()
    actual = {k: (fingerprint(g.z, g.r.astype(float)), *g.state) for k, g in gs.items()}
    h5 = {
        (fp, q, m)
        for fp, q, m in sqlite3.connect(DB).execute(
            "SELECT fp,q,m FROM samples WHERE split='train'"
        )
    }
    h5val = {
        (fp, q, m)
        for fp, q, m in sqlite3.connect(DB).execute(
            "SELECT fp,q,m FROM samples WHERE split='dev'"
        )
    }
    pts = json.loads(POINTS.read_text())
    pts.update(references())
    geom = set(h5)
    for name in json.loads((OUT / "stationary.json").read_text())["training"]:
        p = pts[name]
        geom.add(
            (
                fingerprint(
                    np.array([MASSES[a][0] for a in p["atoms"]]),
                    np.array(p["positions_A"]),
                ),
                0,
                1,
            )
        )
    subsets = {}
    overlaps = {}
    counts = {}
    for t in ("pbe0", "high", "wb97xd3"):
        fitted = geom.copy() if t == "pbe0" else subsets["pbe0"].copy()
        for label, src, split, name, eq, ref, kind in rows:
            if label == t and split == "train":
                fitted.update(actual[k] for k, c in eq)
        subsets[t] = fitted
        counts[t] = len(fitted)
        saddles = [
            actual[eq[-1][0]]
            for label, src, split, name, eq, ref, kind in rows
            if label == t and split == "validation" and kind == "forward"
        ]
        overlaps[t] = dict(
            checked=len(saddles),
            seen=sum(k in fitted for k in saddles),
            hdf_validation_geometries_in_fitting=len(fitted & h5val),
        )
    out = dict(
        HDF5_unique_train=len(h5),
        HDF5_unique_validation=len(h5val),
        HDF5_overlap=len(h5 & h5val),
        across_all_checkpoints=len(set.union(*subsets.values())),
        training=counts,
        overlap=overlaps,
    )
    (OUT / "audit.json").write_text(json.dumps(out, indent=2) + "\n")
    print(out, flush=True)


if __name__ == "__main__":
    run()
