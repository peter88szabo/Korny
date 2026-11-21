"""Prepare gas-phase Leuven singlets/doublets with known levels and checked path labels.

Keep complete trajectory files on the same side by composition and electronic
state. The reserved H-shift composition enters exclusively through verified IRC
triples with the earlier training/test conformer clusters. Reserve the named
Gamma compositions and every public validation TS composition from private
training, including duplicate molecular geometries from the public holdout.
"""

import hashlib
import json
import pickle
import sqlite3
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
from experiments.full_public.kernel_trials import WORK, inputs, write_json
from experiments.full_public.data import fingerprint, OUT
from experiments.reactivity.leuven_hshift import CACHE, references, DIFF
from src.units import HARTREE_EV
from src.forcefield import ForceField
from src.dispersion.d3 import d3_energy_forces


def datalist():
    gs, rows, _ = inputs()
    public = {
        tuple(sorted(gs[r[4][-1][0]].z))
        for r in rows
        if r[2] == "validation" and r[-1] == "forward"
    }
    valfp = {
        (fingerprint(gs[k].z, gs[k].r), gs[k].state)
        for row in rows
        if row[2] == "validation"
        for k, c in row[4]
    }
    pts, clusters = references()
    hform = tuple(sorted(pts[0][2][0].z))
    gm = json.loads((OUT.parent / "comparison/gamma/metadata.json").read_text())
    gamma = {tuple(sorted(g["z"])) for g in gm["geometries"].values()}
    coords = []
    ids = {}
    labels = []
    paths = []
    skipped = Counter()
    seen = {}

    def geom(z, r, q, m):
        fp = fingerprint(z, r)
        ident = (fp, (q, m))
        if ident not in ids:
            ids[ident] = len(coords)
            coords.append((z, r, q, m))
        return ids[ident], ident

    def ingest(file, kind, level, q, m, bz, br, e):
        if not level.endswith("/gas") or m not in (1, 2):
            skipped["other_state_or_solvent"] += 1
            return
        z = np.frombuffer(bz, "u1").astype(int)
        r = np.frombuffer(br, "<f8").reshape(-1, 3)
        form = tuple(sorted(map(int, z)))
        if form in public or (m == 2 and (form in gamma or form == hform)):
            skipped["reserved_formulas"] += 1
            return
        i, ident = geom(z, r, q, m)
        if ident in valfp:
            skipped["public_validation_geometry"] += 1
            return
        key = (i, level)
        if key in seen:
            # Some trajectory energies are printed less precisely. Prefer
            # the matched stationary log, which is ingested first.
            skipped["repeated_geometry_level"] += 1
            return
        seen[key] = float(e)
        group = (level, form, q, m)
        fold = (
            int.from_bytes(
                hashlib.sha256(str((form, q, m)).encode()).digest()[:4], "big"
            )
            % 10
        )
        labels.append(
            (
                i,
                group,
                "validation" if fold == 0 else "train",
                e * HARTREE_EV,
                kind,
                file,
            )
        )

    with sqlite3.connect(f"file:{CACHE}/labels.sqlite?mode=ro", uri=True) as con:
        for rec in con.execute(
            "SELECT hash,kind,level,q,m,z,r,energy FROM labels ORDER BY hash,stage"
        ):
            ingest(*rec)
    with sqlite3.connect(f"file:{CACHE}/trajectories.sqlite?mode=ro", uri=True) as con:
        for rec in con.execute(
            "SELECT file,'trajectory',level,q,m,z,r,e_hartree FROM frames ORDER BY file,frame"
        ):
            ingest(*rec)
    # Leave an entire near-conformer cluster out and fit the rest of the paired
    # CC paths. No other observation of this composition enters at any level.
    previous = json.loads((CACHE / "hshift_group_cv.json").read_text())["groups"]
    withheld = set(previous[0]["test"])
    for name, _, points, cc in pts:
        case = [geom(p.z, p.r, 0, 2)[0] for p in points]
        paths.append(
            (name, "validation" if name in withheld else "train", case, DIFF @ cc)
        )
    for entry in gm["panels"]["Gamma_CaseII"]:
        names = [entry["reactants"][0], entry["ts"], entry["products"][0]]
        assert len(entry["reactants"]) == len(entry["products"]) == 1
        case = []
        for k in names:
            g = gm["geometries"][k]
            case.append(geom(np.array(g["z"]), np.array(g["r"]), g["q"], g["m"])[0])
        from experiments.reactivity.thermochemistry import EV_TO_KCAL

        paths.append(
            (
                entry["name"],
                "gamma_validation",
                case,
                np.array(entry["reference"]) / EV_TO_KCAL,
            )
        )
    counts = Counter(
        (role, "CC" if "ccsd" in group[0] else "DFT", group[-1])
        for i, group, role, e, k, f in labels
    )
    trained = {i for i, g, role, e, k, f in labels if role == "train"} | {
        i for n, role, ks, y in paths if role == "train" for i in ks
    }
    test = {i for i, g, role, e, k, f in labels if role != "train"} | {
        i for n, role, ks, y in paths if role != "train" for i in ks
    }
    assert not trained & test
    counts_by_m = Counter(coords[i][3] for i in trained)
    write_json(
        WORK / "leuven-counts.json",
        dict(
            labels={str(k): v for k, v in counts.items()},
            unique_training_states=dict(counts_by_m),
            unique_validation=len(test),
            paths=dict(Counter(row[1] for row in paths)),
            excluded=dict(skipped),
        ),
    )
    with (WORK / "leuven-data.pkl").open("wb") as f:
        pickle.dump((coords, labels, paths), f)
    print("Leuven inventory", dict(counts), counts_by_m, len(coords), flush=True)
    return coords


def prior_chunk(bounds):
    out = []
    for z, r, q, m in coords[bounds[0] : bounds[1]]:
        v = parent.design(z, r, q, m)[0]
        e = parent.prior(z, r, q, m)[0]
        if len(z) > 1:
            e += d3_energy_forces(z, r)[0]
        out.append(e + weights @ v)
    return bounds[0], np.array(out)


def run():
    global coords, parent, weights
    if (WORK / "leuven-data.pkl").exists():
        with (WORK / "leuven-data.pkl").open("rb") as f:
            coords = pickle.load(f)[0]
    else:
        coords = datalist()
    parent = ForceField.load(OUT / "joint-pbe0-reaction.npz")
    a, g, m = json.loads((WORK / "spin-choices.json").read_text())["high"]["best"]
    weights = np.stack(
        [
            parent.weights + np.load(WORK / f"spin-high-{h}-{a}-{g}-{m}.npy")
            for h in (0, 1)
        ]
        + [ForceField.load(WORK / "spin-high.npz").weights]
    )
    out = np.empty((len(coords), 3))
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context("fork")) as pool:
        tasks = [(i, min(len(coords), i + 64)) for i in range(0, len(coords), 64)]
        for i, vals in pool.map(prior_chunk, tasks, buffersize=14):
            out[i : i + len(vals)] = vals
    np.save(WORK / "leuven-prior.npy", out)
    print("Leuven PBE0+D3 and high-level parent corrections ready", flush=True)


if __name__ == "__main__":
    run()
