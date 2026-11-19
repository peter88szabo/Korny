"""Validate the deployed PBE0 reaction checkpoint, with switchable TS and D3.

Fit the auxiliary volume predictor exclusively to eligible training records from
the four requested HDF5 collections. On the disjoint development parents,
evaluate the same prediction with either correction against PBE0+MBD. All
reported force errors include the coordinate derivative of the correction.
"""

from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import sqlite3

import h5py
import numpy as np

from experiments.full_public.specified import OUT, DB, SOURCES
from experiments.full_public.data import H5
from experiments.full_public import volumes
from experiments.reactivity.thermochemistry import EV_TO_KCAL
from src.forcefield import ForceField
from src.dispersion.volumes import HirshfeldVolumes
from src.dispersion.ts import ts_from_log_ratios
from src.dispersion.d3 import d3_energy_forces


def train_volumes(conn):
    # Only the pure feature calculator is shared with the archive experiment.
    # Inputs and split come from the explicitly requested four-source ledger.
    volumes.OUT = OUT
    features = volumes.accelerator()
    species = volumes.SPECIES
    width = volumes.WIDTH
    sums = {s: np.zeros((width + 1, width + 1)) for s in species}
    means = {s: np.zeros(width) for s in species}
    blocks = {s: [] for s in species}
    counts = Counter()
    used = Counter()
    seen = set()

    def flush():
        for s, parts in blocks.items():
            if parts:
                a = np.vstack(parts)
                sums[s] += a.T @ a
                means[s] += a[:, :-1].sum(axis=0)
                counts[s] += len(a)
                parts.clear()

    for src, fp, q, m, bz, br, bv in conn.execute(
        "SELECT source,fp,q,m,z,r,ratios FROM samples WHERE split='train' ORDER BY source,name"
    ):
        if (fp, q, m) in seen:
            continue
        z = np.frombuffer(bz, "u1")
        r = np.frombuffer(br, "<f8").reshape(-1, 3)
        v = np.frombuffer(bv, "<f8")
        if not np.all(np.isfinite(v) & (v > 0)):
            continue
        seen.add((fp, q, m))
        used[src] += 1
        x = features(z, r, True)[2]
        a = np.column_stack((x, np.log(v)))
        for s in np.unique(z):
            blocks[s].append(a[z == s])
        if len(seen) % 512 == 0:
            flush()
    flush()
    coeff = {}
    for s in species:
        gram = sums[s][:-1, :-1].copy()
        var = np.diag(gram) / counts[s] - (means[s] / counts[s]) ** 2
        gram.flat[:: width + 1] += 0.01 * np.maximum(0.01, var)
        coeff[s] = np.linalg.solve(gram, sums[s][:-1, -1])
    HirshfeldVolumes(volumes.SPEC, coeff).save(OUT / "hirshfeld-requested.npz")
    (OUT / "volume-training.json").write_text(json.dumps(used, indent=2) + "\n")
    print("volume fit", dict(used), flush=True)


def init(prefix):
    global model, vol, handles, conn
    model = ForceField.load(OUT / (prefix + "pbe0-reaction.npz"))
    vol = HirshfeldVolumes.load(OUT / "hirshfeld-requested.npz")
    handles = {s: h5py.File(H5[s], "r") for s in SOURCES}
    conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def validate(task):
    src, keys = task
    totals = defaultdict(lambda: np.zeros(6))
    vol_abs = 0.0
    atoms = 0
    profiles = []
    for name in keys:
        parent, q, m, bz, br, e, bf = conn.execute(
            "SELECT parent,q,m,z,r,energy,force FROM samples WHERE source=? AND name=?",
            (src, name),
        ).fetchone()
        z = np.frombuffer(bz, "u1").astype(int)
        r = np.frombuffer(br, "<f8").reshape(-1, 3)
        f = np.frombuffer(bf, "<f8").reshape(-1, 3)
        g = handles[src][parent if src == "QM7X" else name][name]
        e0, fr = model.predict(z, r, q, m)
        logs, j = vol.evaluate(z, r)
        vol_abs += np.abs(np.exp(logs) - g["hRAT"][:].ravel()).sum()
        atoms += len(z)
        shift = float(g["ePBE0+MBD"][0] - g["ePBE0"][0])
        for label, de, df in [
            ("PBE0", 0.0, 0.0),
            ("PBE0+TS", *ts_from_log_ratios(z, r, logs, j)),
            ("PBE0+D3", *d3_energy_forces(z, r)),
        ]:
            de = e0 + de - e - (shift if label != "PBE0" else 0.0)
            df = fr + df - (g["totFOR"][:] if label != "PBE0" else f)
            totals[label] += [
                1,
                abs(de),
                de * de,
                r.size,
                np.abs(df).sum(),
                np.square(df).sum(),
            ]
        profiles.append((name, parent, e, e0))
    return src, dict(totals), vol_abs, atoms, profiles


def run(joint=False):
    conn = sqlite3.connect(DB)
    prefix = "joint-" if joint else ""
    if not joint or not (OUT / "hirshfeld-requested.npz").exists():
        train_volumes(conn)
    keys = defaultdict(list)
    seen = set()
    for s, n, fp, q, m in conn.execute(
        "SELECT source,name,fp,q,m FROM samples WHERE split='dev' ORDER BY source,name"
    ):
        if (fp, q, m) in seen:
            continue
        keys[s].append(n)
        seen.add((fp, q, m))
    conn.close()
    tasks = [
        (s, ns[i : i + 40]) for s, ns in keys.items() for i in range(0, len(ns), 40)
    ]
    sums = defaultdict(lambda: defaultdict(lambda: np.zeros(6)))
    ratio = Counter()
    atoms = Counter()
    refs = defaultdict(lambda: defaultdict(list))
    result = {}
    with ProcessPoolExecutor(
        max_workers=4,
        mp_context=mp.get_context("fork"),
        initializer=init,
        initargs=(prefix,),
    ) as pool:
        for s, metrics, v, n, prof in pool.map(validate, tasks, buffersize=8):
            for k, x in metrics.items():
                sums[s][k] += x
            ratio[s] += v
            atoms[s] += n
            for name, parent, e, p in prof:
                refs[s][parent].append((name, e, p))
            if int(sums[s]["PBE0"][0]) == len(keys[s]):
                table = {}
                for k, (count, ab, se, nf, af, sf) in sums[s].items():
                    table[k] = dict(
                        n=int(count),
                        energy_mae_kcal=EV_TO_KCAL * ab / count,
                        energy_rmse_kcal=EV_TO_KCAL * np.sqrt(se / count),
                        force_mae_kcal_A=EV_TO_KCAL * af / nf,
                        force_rmse_kcal_A=EV_TO_KCAL * np.sqrt(sf / nf),
                    )
                profiles = []
                ranges = []
                for rec in refs[s].values():
                    if len(rec) < 2:
                        continue
                    data = np.array([(e, p) for _, e, p in sorted(rec)])
                    profiles.extend(np.diff(data[1:] - data[:1], axis=1).ravel())
                    ranges.append(np.diff(np.ptp(data, axis=0)).item())
                table["relative_pbe0"] = dict(
                    n=len(profiles),
                    mae_kcal=float(np.abs(profiles).mean() * EV_TO_KCAL)
                    if profiles
                    else None,
                    n_profiles=len(ranges),
                    range_mae_kcal=float(np.abs(ranges).mean() * EV_TO_KCAL)
                    if ranges
                    else None,
                )
                table["hirshfeld_ratio_mae"] = ratio[s] / atoms[s]
                result[s] = table
                (OUT / (prefix + "final-hdf-validation.json")).write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                print(s, table, flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", action="store_true")
    run(parser.parse_args().joint)
