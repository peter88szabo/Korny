"""Fit the local upstream sGDML implementation and score Korny's saved holdouts.

Each model has one elemental composition, charge and multiplicity. The training
matrix contains bare PBE0 energies and all Cartesian forces for the selected
training geometries. Models may extrapolate between isomers. Keep every held-out
equation in the output ledger, including equations missing one or more energies.
"""
import csv
import hashlib
import json
import multiprocessing as mp
import pickle
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from rdkit import Chem
from scipy.spatial.distance import cdist

from experiments.full_public.data import DB, OUT, fingerprint
from experiments.reactivity.sgdml_hshift import GDMLPredict, GDMLTrain, connectivity
from experiments.reactivity.thermochemistry import EV_TO_KCAL as KCAL
from sgdml.utils.perm import find_perms
from src.dispersion.d3 import d3_energy_forces
from src.forcefield import ForceField

DEST = OUT / "sgdml-matched"
MODEL_DIR = DEST / "models"
WIDTH = 1.0
RIDGE = 1e-8


def state_key(z, q, m):
    return bytes(sorted(z)).hex() + f"_q{q}_m{m}"


def atom_order(z, r):
    """Map atom indices between archives using connectivity and internal distances.

    Graph-equivalent atoms are ordered by their element-resolved distance
    signatures. Both coordinates and forces always use the same permutation.
    """
    graph = connectivity(z, r)
    graph.UpdatePropertyCache(strict=False)
    ranks = Chem.CanonicalRankAtoms(graph, breakTies=False, includeChirality=False)
    d = cdist(r, r)
    env = np.concatenate([np.sort(d[:, z == a], axis=1) for a in np.unique(z)], axis=1)
    return np.array(sorted(range(len(z)), key=lambda i: (z[i], ranks[i], *env[i].round(8))))


def read_inventory():
    """Use Korny's persisted reaction split and reproduce its molecular dedup order."""
    with (OUT / "reaction-data.pkl").open("rb") as stream:
        geoms, rows, _ = pickle.load(stream)
    rows = [r for r in rows if r[2] == "validation"]
    keys = sorted({key for row in rows for key, _ in row[4]})
    reaction = defaultdict(list)
    forbidden = set()
    for key in keys:
        g = geoms[key]
        reaction[state_key(g.z, *g.state)].append(key)
        forbidden.add((fingerprint(g.z, g.r), *g.state))
    train, held = defaultdict(list), defaultdict(list)
    with sqlite3.connect(f"file:{DB}?mode=ro", uri=True) as db:
        for split, store in [("dev", held), ("train", train)]:
            seen = set()
            for i, src, name, fp, q, m, z in db.execute(
                "SELECT rowid,source,name,fp,q,m,z FROM samples WHERE split=? ORDER BY source,name", (split,)
            ):
                uid = (fp, q, m)
                if uid in seen or (split == "train" and uid in forbidden):
                    continue
                seen.add(uid)
                store[state_key(z, q, m)].append((i, src, name, fp))
            if split == "dev":
                forbidden.update(seen)
    return geoms, rows, reaction, train, held


def read_sample(db, i):
    bz, br, e, bf = db.execute("SELECT z,r,energy,force FROM samples WHERE rowid=?", (i,)).fetchone()
    z = np.frombuffer(bz, "u1").astype(int)
    r = np.frombuffer(br, "<f8").reshape(-1, 3)
    f = np.frombuffer(bf, "<f8").reshape(-1, 3)
    return z, r, e, f


def worker_init():
    global trainer, db
    trainer = GDMLTrain(max_memory=2, max_processes=1, use_torch=False)
    db = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def train_group(key):
    """Build one bounded dense energy/force kernel from training fingerprints only."""
    file = MODEL_DIR / (key + ".npz")
    if file.exists():
        return key
    candidates = sorted(TRAIN[key], key=lambda row: row[3])
    nat = len(bytes.fromhex(key.split("_")[0]))
    selected = candidates[:min(64, 1800 // (3 * nat + 1))]
    coords, forces, energies = [], [], []
    for i, _, _, _ in selected:
        z, r, e, f = read_sample(db, i)
        order = atom_order(z, r)
        coords.append(r[order])
        forces.append(f[order] * KCAL)
        energies.append(e * KCAL)
    coords = np.array(coords)
    z = np.sort(z)
    # Discover permutations on training geometries. Limit this pairwise search
    # to eight structures; the kernel solve includes every selected geometry.
    perms = find_perms(coords[:8], z, max_processes=1) if len(coords) > 1 else np.arange(nat)[None, :]
    if perms is None:
        perms = np.arange(nat)[None, :]
    assert np.all(z[perms] == z)
    data = dict(R=coords, z=z, F=np.array(forces), E=np.array(energies),
                name=np.array(key), theory=np.array("PBE0"))
    task = trainer.create_task(data, len(coords), data, 0, sig=WIDTH, lam=RIDGE,
                               use_E=True, use_E_cstr=True, perms=perms)
    model = trainer.train(task)
    if not np.isfinite(model["alphas_F"]).all():
        raise ValueError(f"Non-finite sGDML coefficients for {key}")
    model["training_rowids"] = np.array([row[0] for row in selected])
    model["training_available"] = len(candidates)
    temp = file.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **model)
    temp.replace(file)
    return key


def predict_group(key):
    """Compute one energy per distinct molecular term at Korny's saved coordinates."""
    with np.load(MODEL_DIR / (key + ".npz"), allow_pickle=True) as saved:
        pred = GDMLPredict(dict(saved), max_processes=1, use_torch=False)
    energies = {}
    for name in REACTION[key]:
        p = GEOMS[name]
        order = atom_order(p.z, p.r)
        e, _ = pred.predict(p.r[order].reshape(1, -1))
        bare = float(e.item() / KCAL)
        d3 = float(d3_energy_forces(p.z, p.r)[0]) if len(p.z) > 1 else 0.0
        if not np.isfinite([bare, d3]).all():
            raise ValueError(f"Non-finite prediction for {name}")
        energies[name] = (bare, d3)
    return energies


def score(rows, energies):
    """Write all original IDs and preserve distinct forward/reverse coverage."""
    models = {t: ForceField.load(OUT / f"joint-{t}-reaction.npz",
                                 dispersion="off" if t == "pbe0" else "d3")
              for t in {r[0] for r in rows}}
    kp = {}
    stats = defaultdict(list)
    totals = Counter()
    with (DEST / "reaction-equations.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["target","source","id","observable","reference_kcal","sgdml_kcal",
                         "korny_matched_kcal","sgdml_error_kcal","korny_error_kcal","missing_geometry_ids"])
        for target, src, _, name, terms, ref, kind in rows:
            bin_key = (target, src, kind)
            totals[bin_key] += 1
            missing = [k for k, c in terms if k not in energies]
            if missing:
                writer.writerow([target,src,name,kind,ref*KCAL,"","","","",json.dumps(missing)])
                continue
            s = sum(c * (energies[k][0] + (energies[k][1] if target != "pbe0" else 0))
                    for k, c in terms)
            for k, c in terms:
                if (target, k) not in kp:
                    g = GEOMS[k]
                    kp[target,k] = models[target].energy(g.z, g.r, *g.state)
            p = sum(c*kp[target,k] for k,c in terms)
            err_s, err_k = (s-ref)*KCAL, (p-ref)*KCAL
            stats[bin_key].append([err_s,err_k])
            writer.writerow([target,src,name,kind,ref*KCAL,s*KCAL,p*KCAL,err_s,err_k,""])
    for target, src, kind in list(totals):
        if kind == "forward":
            bk = (target,src,"both_barriers")
            for direction in ["forward","reverse"]:
                totals[bk] += totals[target,src,direction]
                stats[bk].extend(stats[target,src,direction])
    result = {}
    for k, count in sorted(totals.items()):
        errors = np.array(stats[k])
        result["/".join(k)] = dict(total=count, scored=len(errors),
                mae=np.abs(errors).mean(axis=0).tolist() if len(errors) else None,
                rmse=np.sqrt(np.mean(errors**2,axis=0)).tolist() if len(errors) else None)
    (DEST / "reaction-scores.json").write_text(json.dumps(result, indent=2) + "\n")
    for k, v in result.items():
        print(k, v, flush=True)


def run(molecular=False):
    global GEOMS, REACTION, TRAIN, HELD
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    GEOMS, rows, REACTION, TRAIN, HELD = read_inventory()
    if molecular:
        from experiments.full_public.check_spectra import panels
        from experiments.reactivity.stationary_fit import MASSES
        spectra = {state_key([MASSES[s][0] for s in p["atoms"]], 0, 1) for p in panels().values()}
        keys = sorted((HELD.keys() | spectra) & TRAIN.keys())
        print(f"Training molecular holdout models for {len(keys)} composition/state groups", flush=True)
        with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("fork"), initializer=worker_init) as pool:
            for i, key in enumerate(pool.map(train_group, keys), 1):
                if i % 50 == 0 or i == len(keys):
                    print(f"Molecular groups {i}/{len(keys)}", flush=True)
        return
    candidates = sorted(REACTION.keys() & TRAIN.keys(), key=lambda k: -len(REACTION[k]))
    print(f"{len(rows)} held-out equations, {len(candidates)} matched molecular compositions and states", flush=True)
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("fork"), initializer=worker_init) as pool:
        energies = {}
        for n, key in enumerate(pool.map(train_group, candidates), 1):
            if n == 1 or n % 10 == 0 or n == len(candidates):
                print(f"reaction models {n}/{len(candidates)}", flush=True)
        for terms in pool.map(predict_group, candidates):
            energies.update(terms)
    score(rows, energies)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecular-models", action="store_true")
    run(parser.parse_args().molecular_models)
