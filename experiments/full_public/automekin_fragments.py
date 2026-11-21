"""Recover separated AutoMeKin product energies from the original ORCA fragment jobs.

A PR label in RXNet may denote several fragments. The CC energy CSV reports
one number per fragment, sometimes using different individual spin states.
Require all original ORCA inputs, charges, geometries and successful matching
energies for a PR endpoint. Evaluate it as a sum of isolated fragments.
"""

import csv
import io
import re
import sqlite3
import tarfile
from collections import Counter

import numpy as np

from experiments.full_public import kernel_trials as kt
from experiments.full_public.paper_data import CC, OUT, ROOT, xyz_geometry
from src.units import HARTREE_EV

CC_LABEL = re.compile(r"/(PR_\d+_F\d+|MIN_\d+|TS_\d+)\.out$")
INPUT = re.compile(
    r"\*\s+xyz\s+(-?\d+)\s+(\d+)\s*\n((?:\s*\|\s*\d+>\s*[a-zA-Z]+\s+[\d.-]+[^\n]*\n)+)",
    re.IGNORECASE,
)
ENERGY = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?[\d.]+)")
ENDPOINT = re.compile(r"\b(MIN)\s+(\d+)|\b(PR)(\d+)\s*:")


def extract():
    geo = {}
    channels = []
    audit = []
    with sqlite3.connect(f"file:{OUT / 'electronic.sqlite'}?mode=ro", uri=True) as db:
        for archive in sorted((ROOT / "AutoMeKin_2026").glob("*.tar")):
            sys = archive.stem
            method = "CCSD(T)-F12/cc-pVDZ-F12" if sys == "MEA" else CC
            prefix = f"AutoMeKin/{sys}/{method}/"
            with tarfile.open(archive) as t:
                names = t.getnames()
                rx = next(
                    n for n in names if "/FINAL_HL_" in n and n.endswith("/RXNet")
                )
                csvfile = next(n for n in names if n.endswith("cc_energies.csv"))
                labels = {
                    v["label"]: v
                    for v in csv.DictReader(io.TextIOWrapper(t.extractfile(csvfile)))
                }
                wanted = {}
                for label in labels:
                    if m := re.fullmatch(r"PR_(\d+)_F(\d+)", label):
                        wanted.setdefault("PR" + m[1], []).append(label)
                old = db.execute(
                    "select id,z,r,charge,mult,energy_eV,role from geometries where source='AutoMeKin' and system=? and method=?",
                    (sys, method),
                )
                for name, zb, rb, q, m, e, role in old:
                    geo[name] = {
                        "z": np.frombuffer(zb, np.uint8).tolist(),
                        "r_A": np.frombuffer(rb, "<f8").reshape(-1, 3).tolist(),
                        "charge": q,
                        "mult": m,
                        "energy_eV": e,
                    }
                rawmin = 0
                for fn in names:
                    if not (m := CC_LABEL.search(fn)) or "/CC_OUTPUTS/" not in fn:
                        continue
                    label = m[1]
                    rec = labels[label]
                    if (
                        rec["status"] not in ("OK", "OK_RECOVERED")
                        or not rec["energy_Eh"]
                    ):
                        continue
                    s = t.extractfile(fn).read().decode(errors="replace")
                    inp = INPUT.search(s)
                    outs = ENERGY.findall(s)
                    if (
                        not inp
                        or not outs
                        or abs(float(outs[-1]) - float(rec["energy_Eh"])) > 2.0e-8
                    ):
                        continue
                    q, mu = int(inp[1]), int(inp[2])
                    if mu != int(rec["mult"]):
                        continue
                    z, r = xyz_geometry(
                        [line.split(">")[-1] for line in inp[3].splitlines()]
                    )
                    if (int(z.sum()) - q - (mu - 1)) % 2:
                        raise ValueError(fn)
                    key = prefix + (
                        label if label.startswith("PR_") else label.replace("_", "")
                    )
                    if not label.startswith("PR_"):
                        rawmin += 1
                    geo[key] = {
                        "z": z.tolist(),
                        "r_A": r.tolist(),
                        "charge": q,
                        "mult": mu,
                        "energy_eV": float(outs[-1]) * HARTREE_EV,
                    }
                allpr = {k: [prefix + n for n in sorted(v)] for k, v in wanted.items()}
                usable = {k: v for k, v in allpr.items() if all(n in geo for n in v)}
                count = Counter()
                for line in t.extractfile(rx).read().decode().splitlines()[2:]:
                    sides = [
                        (a or c) + (b or d) for a, b, c, d in ENDPOINT.findall(line)
                    ]
                    if len(sides) != 2 or not any(k.startswith("PR") for k in sides):
                        continue
                    count["fragment network rows"] += 1
                    if sides[0] == sides[1]:
                        continue
                    ts = prefix + "TS" + line.split()[0]
                    if ts not in geo:
                        continue
                    left, right = [
                        usable.get(k, []) if k.startswith("PR") else [prefix + k]
                        for k in sides
                    ]
                    if not left or not right or not all(n in geo for n in left + right):
                        continue

                    def composition(ids):
                        return Counter(z for n in ids for z in geo[n]["z"]), sum(
                            geo[n]["charge"] for n in ids
                        )

                    if composition(left) != composition([ts]) or composition(
                        right
                    ) != composition([ts]):
                        continue
                    channels.append(
                        {
                            "system": sys,
                            "method": method,
                            "channel": ts.split("/")[-1],
                            "ends": sides,
                            "reactant": left,
                            "ts": ts,
                            "product": right,
                            "split": "validation" if sys == "cBD-CN" else "train",
                        }
                    )
                    count["assigned channels"] += 1
                audit.append(
                    {
                        "system": sys,
                        "method": method,
                        "PR labels in CC CSV": len(wanted),
                        "PR with all ORCA geometries and charges": len(usable),
                        "validated original MIN/TS outputs": rawmin,
                        **count,
                    }
                )
    used = {k for c in channels for k in c["reactant"] + [c["ts"]] + c["product"]}
    # Some original output geometries differ slightly from the rounded network
    # database geometries. The raw ORCA geometry takes precedence above.
    dest = OUT / "fragment-channels.json"
    kt.write_json(
        dest,
        {
            "geometries": {k: geo[k] for k in sorted(used)},
            "channels": channels,
            "audit": audit,
        },
    )
    print("Verified coupled-cluster fragment channels", audit, flush=True)
    return dest


if __name__ == "__main__":
    extract()
