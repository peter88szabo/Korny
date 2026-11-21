"""Compare full internal Hessians at eight fixed, reference-optimized validation structures.

The panel is drawn by a stable hash of the structure ID, before reading errors
or Hessian spectra. It contains two endpoints and two transition states from
each of RDB7 and Grambow. Frequencies are evaluated at reference geometries, so
the maximum predicted force is also reported. A Hessian index at a nonstationary
model geometry by itself does not certify a transition state on the fitted PES.
"""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import multiprocessing as mp
import pickle
import re
import sqlite3
import tarfile
import numpy as np
from rdkit import Chem
from threadpoolctl import threadpool_limits
from experiments.full_public import kernel_trials as kt
from experiments.reactivity.radical_curvatures import ARCHIVE, final_hessian
from experiments.reactivity.stationary_fit import internal_transform
from experiments.reactivity.frequency_validation import TO_CM1
from src.forcefield import ForceField


def select():
    with (kt.WORK / "ids.pkl").open("rb") as handle:
        ids, _ = pickle.load(handle)
    allowed = {}
    for target in kt.TARGETS:
        d = kt.equation_data(target)
        train = set(d[0][d[3] >= 0].indices)
        held = set(d[0][d[3] == -1].indices)
        allowed[target] = held - train
    pool = defaultdict(list)
    with sqlite3.connect(
        f"file:{kt.WORK / 'rdb-dft-frames.sqlite'}?mode=ro", uri=True
    ) as db:
        for name, idx in db.execute(
            "select name,coord_id from stationary where converged=1 and hessian is not null and q=0 and m=1 and length(z) between 8 and 22"
        ):
            if idx in allowed["high"]:
                pool["RDB7", "saddle" if "/ts" in name else "minimum"].append(
                    (name, idx)
                )

        def choose(group):
            ranked = sorted(
                pool[group],
                key=lambda p: hashlib.sha256(
                    ("full-frequency-audit/" + p[0]).encode()
                ).digest(),
            )
            distinct = {}
            reactions = set()
            for n, i in ranked:
                case = n.split("/")[-2]
                if i in distinct or case in reactions:
                    continue
                distinct[i] = n
                reactions.add(case)
                if len(distinct) == 2:
                    break
            if len(distinct) != 2:
                raise ValueError(f"Too few archived structures in {group}")
            return distinct.values()

        jobs = []
        for role in ("minimum", "saddle"):
            for n in choose(("RDB7", role)):
                z, r, q, m, h = db.execute(
                    "select z,r,q,m,hessian from stationary where name=?", (n,)
                ).fetchone()
                z = np.frombuffer(z, np.uint8)
                r = np.frombuffer(r, float).reshape(-1, 3)
                jobs.append(
                    (
                        "RDB7",
                        role,
                        n,
                        z,
                        r,
                        (q, m),
                        np.frombuffer(h, float).reshape(r.size, r.size),
                    )
                )
    with (kt.WORK / "radical-modes.pkl").open("rb") as f:
        rad = pickle.load(f)
    for h, (name, z, _, state, converged, _) in rad:
        if h != -1 or not converged or state != (0, 2) or not 8 <= len(z) <= 22:
            continue
        match = re.search(r"/rxn(\d+)/(p|r|ts)\d+\.log$", name)
        if not match:
            continue
        idx = ids.get(f"Grambow/{match[1]}/{match[2]}")
        if idx in allowed["wb97xd3"]:
            pool["Grambow", "saddle" if match[2] == "ts" else "minimum"].append(
                (name, idx)
            )
    wanted = {
        n: role for role in ("minimum", "saddle") for n in choose(("Grambow", role))
    }
    with tarfile.open(ARCHIVE, "r|gz") as arc:
        for member in arc:
            if member.name in wanted:
                z, r, st, h = final_hessian(
                    arc.extractfile(member).read().decode(errors="replace")
                )
                jobs.append(("Grambow", wanted[member.name], member.name, z, r, st, h))
    assert len(jobs) == 8
    return jobs


def evaluate(item):
    src, role, name, z, r, st, reference = item
    ff = models[src]
    mass = np.array(
        [Chem.GetPeriodicTable().GetMostCommonIsotopeMass(int(s)) for s in z]
    )
    transform = internal_transform(mass, r)

    def frequencies(h):
        w, axes = np.linalg.eigh(transform.T @ h @ transform)
        return np.sign(w) * np.sqrt(abs(w)) * TO_CM1, axes

    truth, ref_modes = frequencies(reference)
    force = ff.predict(z, r, *st)[1]
    h = 0.001
    directions = np.eye(r.size).reshape(-1, *r.shape)
    dyn = np.stack(
        [
            (
                (ff.predict(z, r - h * u, *st)[1] - ff.predict(z, r + h * u, *st)[1])
                / (2 * h)
            ).ravel()
            for u in directions
        ]
    )
    est, modes = frequencies(0.5 * (dyn + dyn.T))
    return dict(
        source=src,
        role=role,
        id=name,
        atoms=len(z),
        multiplicity=int(st[1]),
        frequency_mae_cm1=float(abs(est - truth).mean()),
        most_negative_or_lowest_cm1=[float(truth[0]), float(est[0])],
        n_imag_below_minus50=[int(np.sum(truth < -50)), int(np.sum(est < -50))],
        lowest_mode_overlap=float(abs(modes[:, 0] @ ref_modes[:, 0])),
        max_force_eV_A=float(np.linalg.norm(force, axis=1).max()),
    )


def run(stage):
    global models
    jobs = select()
    models = {
        s: ForceField.load(kt.WORK / f"{stage}-{t}.npz", dispersion="d3")
        for s, t in (("RDB7", "high"), ("Grambow", "wb97xd3"))
    }
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("fork")) as pool:
        rows = list(pool.map(evaluate, jobs, buffersize=4))
    kt.write_json(kt.WORK / f"{stage}-full-stationary-audit.json", rows)
    for r in rows:
        print(
            r["id"],
            r["role"],
            round(r["frequency_mae_cm1"]),
            r["n_imag_below_minus50"],
            round(r["max_force_eV_A"], 2),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage")
    with threadpool_limits(1):
        run(parser.parse_args().stage)
