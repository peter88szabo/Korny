"""Compare full harmonic spectra at Korny's own optimized polyatomic geometries.

Read unscaled PBE0/cc-pVTZ stationary points from NIST CCCBDB. Save the
coordinates, frequencies and source URLs locally so a later fit sees the same
test. The electronic high-level checkpoint is evaluated with D3 by default.
"""
import argparse
import html
import json
from pathlib import Path
import re
from urllib.request import urlopen

import numpy as np

from experiments.reactivity.augmented import OUT, ROOT
from experiments.reactivity.frequency_validation import measure
from experiments.reactivity.stationary_fit import MASSES
from src.forcefield import ForceField

CASES = {"acetone": "67641", "propane": "74986", "benzene": "71432", "n-butane": "106978",
         "cyclopropane": "75194", "propene": "115071", "furan": "110009"}
SOURCE = ROOT / "dataset_for_training/NIST"


def _table(page, section):
    section = page.split(section+" calculated at PBE1PBE/cc-pVTZ", 1)[1]
    table = re.search(r"<table\b[^>]*>(.*?)</table>", section, re.I | re.S)[1]
    rows = []
    for tr in re.findall(r"<tr\b[^>]*>(.*?)</tr>", table, re.I | re.S):
        cells = re.findall(r"<td\b[^>]*>(.*?)</td>", tr, re.I | re.S)
        if cells:
            rows.append([html.unescape(re.sub("<[^>]+>", "", c)).strip() for c in cells])
    return rows


def references():
    """Require one complete unscaled spectrum, including degenerate partners."""
    points = {}
    SOURCE.mkdir(parents=True, exist_ok=True)
    for name, cas in CASES.items():
        url = f"https://cccbdb.nist.gov/energy3x.asp?method=54&basis=6&casno={cas}&charge=0"
        path = SOURCE/(cas+".html")
        if not path.exists():
            with urlopen(url, timeout=30) as stream:
                path.write_bytes(stream.read())
        page = path.read_bytes().decode("latin-1")
        gt = _table(page, "Geometric Data")
        atoms = [re.match("[A-Z][a-z]?", row[0])[0] for row in gt]
        ft = _table(page, "Vibrational Frequencies")
        freq = sorted(float(row[2]) for row in ft)
        if len(freq) != 3*len(atoms)-6:
            freq = sorted(float(row[2]) for row in ft for _ in range(2 if row[1].startswith("E") else 1))
        if len(freq) != 3*len(atoms)-6 or min(freq) <= 0:
            raise ValueError(f"Incomplete harmonic minimum for {name}")
        points[name] = dict(source=url, atoms=atoms, positions_A=[[float(v) for v in row[1:4]] for row in gt],
                            frequencies_cm1=freq)
    return points


def evaluate(model, points):
    cases = {}
    for name, point in points.items():
        z, mass = np.array([MASSES[a] for a in point["atoms"]]).T
        ref = np.array(point["frequencies_cm1"])
        r = np.array(point["positions_A"])
        case = dict(n_atoms=len(z), reference_cm1=ref.tolist(), source=point["source"])
        try:
            case.update(measure(model, z.astype(int), mass, r, ref))
            print(name, "bond A / angle deg / frequency cm-1",
                  *(round(case[k], 3) for k in ("bond_mae_A", "angle_mae_deg", "mae_cm1")), flush=True)
        except (RuntimeError, FloatingPointError) as exc:
            case["relaxation_failure"] = str(exc)
            print(name, str(exc), flush=True)
        cases[name] = case
    return cases


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--dispersion", choices=["off","d3"], default="d3")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = evaluate(ForceField.load(args.model, dispersion=args.dispersion), references())
    args.out.write_text(json.dumps(dict(checkpoint=args.model.name,dispersion=args.dispersion,cases=result),
                                   indent=2)+"\n")
