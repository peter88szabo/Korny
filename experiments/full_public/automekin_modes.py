"""Recover mass-normalized, signed vibrational modes from AutoMeKin's DFT archives.

Match each file to the independent DFT stationary-point database by network,
role, atom order, coordinates and frequencies. The stored displacement vectors
are Cartesian. Multiplying by the square roots of the atomic masses must give
orthogonal normal modes. This check distinguishes them from mass-weighted
vectors and from files for the archive's other electronic methods.
"""

import json
import re
import sqlite3
import tarfile
from collections import Counter

import numpy as np
from rdkit import Chem

from experiments.full_public import kernel_trials as kt
from experiments.full_public.paper_data import DFT, OUT, ROOT, xyz_geometry
from src.units import BOHR_A

DEST = OUT / "normal-modes"
FILE = re.compile(r"/(MIN|TS)(\d+)\.molden$")
PT = Chem.GetPeriodicTable()


def decode(body):
    _, raw_freq, raw_xyz, raw_disp = re.split(
        r"\[(?:FREQ|FR-COORD|FR-NORM-COORD)\]", body, flags=re.IGNORECASE
    )
    freq = np.fromstring(raw_freq, sep=" ")
    z, coord = xyz_geometry(raw_xyz.strip().splitlines())
    lines = re.split(r"(?im)^vibration\s+\d+\s*$", raw_disp)[1:]
    cart = np.array([np.fromstring(v, sep=" ") for v in lines])
    if cart.shape != (len(freq), 3 * len(z)) or len(freq) != 3 * len(z) - 6:
        raise ValueError("incomplete internal modes")
    if not np.all(np.isfinite(cart)) or not np.all(np.isfinite(freq)):
        raise ValueError("nonfinite normal modes")
    mass = np.array([PT.GetMostCommonIsotopeMass(int(s)) for s in z])
    weighted = cart * np.repeat(np.sqrt(mass), 3)
    norms = np.linalg.norm(weighted, axis=1)
    if np.any(norms < 1e-10):
        raise ValueError("zero mode vector")
    weighted /= norms[:, None]
    orth_error = np.abs(weighted @ weighted.T - np.eye(len(freq))).max()
    if orth_error > 0.008:
        raise ValueError("nonorthogonal mass-weighted modes")
    return (
        z,
        coord * BOHR_A,
        mass,
        freq,
        (cart / norms[:, None]).reshape(-1, len(z), 3),
        orth_error,
    )


def extract():
    DEST.mkdir(exist_ok=True)
    records, excluded = [], Counter()
    with sqlite3.connect(f"file:{OUT / 'electronic.sqlite'}?mode=ro", uri=True) as db:
        for archive in sorted((ROOT / "AutoMeKin_2026").glob("*.tar")):
            sys = archive.stem
            with tarfile.open(archive) as t:
                for member in t:
                    m = FILE.search(member.name)
                    if not m or "/FINAL_HL_" not in member.name:
                        continue
                    label = m[1] + str(int(m[2]))
                    name = f"AutoMeKin/{sys}/{DFT}/{label}"
                    state = db.execute(
                        "select role,split,charge,mult,z,r,frequencies_cm1,duplicate_of from geometries where id=?",
                        (name,),
                    ).fetchone()
                    if state is None or state[6] is None:
                        excluded[sys, "absent independent reference"] += 1
                        continue
                    role, split, q, mult, zraw, rraw, frequencies, dup = state
                    if split not in ("train", "validation") or dup is not None:
                        excluded[sys, "duplicate or excluded reference"] += 1
                        continue
                    try:
                        z, r, mass, freq, mode, orth = decode(
                            t.extractfile(member).read().decode()
                        )
                        if not np.array_equal(z, np.frombuffer(zraw, np.uint8)):
                            raise ValueError("different atom order")
                        ref_r = np.frombuffer(rraw, float).reshape(len(z), 3)
                        if np.max(np.abs(r - ref_r)) > 0.0001:
                            raise ValueError("different stationary geometry")
                        if not np.array_equal(freq, json.loads(frequencies)):
                            raise ValueError("different signed frequencies")
                    except ValueError as exc:
                        excluded[sys, str(exc)] += 1
                        continue
                    f = DEST / f"{sys}_{label}.npz"
                    with f.with_suffix(".pending").open("wb") as handle:
                        np.savez_compressed(
                            handle,
                            z=z,
                            r_A=ref_r,
                            mass_u=mass,
                            freq_cm1=freq,
                            modes=mode,
                            charge=q,
                            mult=mult,
                        )
                    f.with_suffix(".pending").replace(f)
                    records.append(
                        {
                            "file": f.name,
                            "id": name,
                            "system": sys,
                            "role": role,
                            "split": split,
                            "modes": len(freq),
                            "orthogonality_error": float(orth),
                        }
                    )
    if len(records) != len({r["id"] for r in records}):
        raise ValueError("Repeated stationary-point label")
    counts = Counter(r["split"] for r in records)
    kt.write_json(
        DEST / "manifest.json",
        {
            "method": DFT,
            "records": records,
            "excluded": {"/".join(k): v for k, v in sorted(excluded.items())},
        },
    )
    print(
        "Verified DFT stationary-point mode sets",
        counts,
        "signed mode counts",
        {k: sum(r["modes"] for r in records if r["split"] == k) for k in counts},
        "excluded",
        dict(excluded),
        flush=True,
    )


if __name__ == "__main__":
    extract()
