"""Decode the two Gamma series from the author's extractor manifests and ORCA records.

This read-only audit preserves the extractor's geometry/conformer labels and
fragment corrections. It records alternate stereochemical endpoint assignments
explicitly and never adds a row to the training split.
"""

import ast
from collections import Counter, defaultdict
import hashlib
from itertools import product
import json
import re
import sqlite3

import numpy as np
from experiments.reactivity.leuven_files import CACHE, ROOT
from experiments.reactivity.leuven_hshift import DIFF
from experiments.full_public.kernel_trials import WORK, KCAL, write_json
from src.forcefield import ForceField
from src.units import HARTREE_EV

SERIES = ("Case1/Gamma-Channel_Case-1", "Case2/CCSDT_Gamma-Channel")
SYMBOL = {"H": 1, "C": 6, "N": 7, "O": 8}
SPIN = {"OH": 2, "O2": 3, "HO2": 2, "O3": 1, "HCO": 2, "CH2OOH": 2}
LEVEL = "ORCA6.1.0/ccsd(t)-f12/ri/cc-pvdz-f12/gas"
GEOM_LEVEL = "ORCA6.1.0/m062x/aug-cc-pvtz/d3zero/gas"
HARTREE_KCAL = HARTREE_EV * KCAL


def manifest(script):
    """Read literals, including the signed fragment coefficients, without running the extractor."""
    tree = ast.parse(script.read_text())
    variables = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            key = ast.unparse(node.targets[0])
            if key in ("species_to_run", "partners"):
                variables[key] = ast.literal_eval(node.value)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == "SmallReactant":
            name, _, zp, e = (ast.literal_eval(v) for v in node.args[:4])
            geom = ast.literal_eval(
                next(v.value for v in node.keywords if v.arg == "geom")
            )
            variables.setdefault("fragments", {})[name] = dict(
                z=[SYMBOL[t[0]] for t in geom],
                r=[t[1:] for t in geom],
                q=0,
                m=SPIN[name],
                energy=e,
                ezpe=e + zp,
            )
    return variables


def stationary(folder, db):
    """Cross-check exported coordinates with the two stages of a converged compound job."""
    prefix = folder.relative_to(ROOT).as_posix() + "/"
    sql = """SELECT f.name,c.z,c.r,c.energy,c.q,c.m,o.r,o.freq
    FROM idx.files f JOIN labels c USING(hash) JOIN labels o ON o.hash=c.hash AND o.stage<c.stage
    WHERE f.name LIKE ? AND c.level=? AND o.kind='optimized' AND o.level=?
    AND o.freq IS NOT NULL AND c.z=o.z AND c.q=o.q AND c.m=o.m"""
    verified = defaultdict(list)
    for file, z, r, e, q, m, ro, freq in db.execute(
        sql, (prefix + "%", LEVEL, GEOM_LEVEL)
    ):
        parts = file[len(prefix) :].split("/")
        if len(parts) != 3 or not parts[1].startswith("conformer_"):
            continue
        rr = np.frombuffer(r, "<f8").reshape(-1, 3)
        if not np.allclose(
            rr, np.frombuffer(ro, "<f8").reshape(-1, 3), atol=2e-5, rtol=0
        ):
            continue
        freq = np.sort(json.loads(freq)[6:])
        saddle = parts[0].startswith("TS")
        if (freq < -1.0).sum() != int(saddle) or (saddle and freq[0] > -20.0):
            continue
        verified[parts[0], parts[1]].append(
            dict(
                z=np.frombuffer(z, "u1").tolist(),
                r=rr.tolist(),
                q=q,
                m=m,
                energy=e,
                file=file,
                imag_cm1=float(freq[0]) if saddle else None,
            )
        )
    records = defaultdict(list)
    for block in (folder / "all_conformers.txt").read_text().strip().split("\n\n"):
        lines = block.splitlines()
        if len(lines) < 2:
            continue
        name, conf = lines[0].split()
        val = re.match(
            r"E\(best\)\+ZPE = ([-.\d]+) Eh\s+Rel_CCSD\(T\)\+ZPE = ([-.\d]+) kcal/mol",
            lines[1],
        )
        if not val:
            continue
        geom = [
            line.split()
            for line in lines[2:]
            if line.split() and line.split()[0] in SYMBOL
        ]
        for g in verified.get((name, conf), ()):
            if g["z"] != [SYMBOL[t[0]] for t in geom]:
                continue
            if not np.allclose(
                g["r"], [list(map(float, t[1:])) for t in geom], rtol=0, atol=2e-5
            ):
                continue
            records[name].append(
                dict(g, conf=conf, ezpe=float(val[1]), relzpe=float(val[2]))
            )
            break
    # Use the author's CC+ZPE ordering on independently matched stationary records.
    return {s: min(rows, key=lambda x: x["ezpe"]) for s, rows in records.items()}


def endpoint(species, number, ts):
    """Prefer an exactly matching stereo suffix; otherwise expose the possible representatives."""
    choices = [
        s
        for s in species
        if re.match(rf"^Gamma-{number}(?![a-zA-Z0-9])", s) and "vdW" not in s
    ]

    def tags(name):
        return set(re.findall(r"(?:^|[-_])(ZZ|ZE|EE|EZ|R|S|RR|SS|RS)(?=$|[-_])", name))

    stereo = tags(ts)
    if stereo:
        exact = [s for s in choices if stereo <= tags(s)]
        if exact:
            return sorted(exact), False
    return sorted(choices), len(choices) > 1 or bool(
        stereo and not any(stereo <= tags(s) for s in choices)
    )


def run(model_file):
    model = ForceField.load(WORK / model_file, dispersion="d3")
    output = {}
    with sqlite3.connect(f"file:{CACHE}/labels.sqlite?mode=ro", uri=True) as db:
        db.execute("ATTACH ? AS idx", (f"file:{CACHE}/files.sqlite?mode=ro",))
        for series in SERIES:
            folder = ROOT / "Saving_VSC" / series
            script = folder / "extractor_2025_11_20a.py"
            cfg = manifest(script)
            points = stationary(folder, db)
            species = {s for s in cfg["species_to_run"] if not s.startswith("TS")}
            energies = {}

            def state(name):
                parts = [(name, 1, points[name])]
                parts.extend(
                    (k, c, cfg["fragments"][k])
                    for k, c in cfg["partners"].get(name, [])
                )
                atoms = Counter()
                q = 0
                electronic = ezpe = pred = 0.0
                for key, c, g in parts:
                    atoms.update({z: c * n for z, n in Counter(g["z"]).items()})
                    q += c * g["q"]
                    electronic += c * g["energy"]
                    ezpe += c * g["ezpe"]
                    if key not in energies:
                        energies[key] = model.energy(
                            np.array(g["z"]), np.array(g["r"]), g["q"], g["m"]
                        )
                    pred += c * energies[key]
                return (atoms, q), electronic, ezpe, pred

            results = []
            skips = {}
            for ts in (s for s in cfg["species_to_run"] if s.startswith("TS")):
                pat = re.match(r"TS[-_]Gamma-(\d+)-(\d+)(?:_|$)", ts)
                if not pat:
                    skips[ts] = "non-numeric or cross-series channel"
                    continue
                if ts not in points:
                    skips[ts] = (
                        "no exported, verified first-order saddle with matched CC geometry"
                    )
                    continue
                left, al = endpoint(species, pat[1], ts)
                right, ar = endpoint(species, pat[2], ts)
                if not left or not right:
                    skips[ts] = "endpoint absent from the extractor species list"
                    continue
                accepted = 0
                for a, b in product(left, right):
                    names = (a, ts, b)
                    if not all(s in points for s in names):
                        continue
                    states = [state(s) for s in names]
                    if not all(v[0] == states[1][0] for v in states):
                        continue
                    # The named reactions lie on a doublet surface. A sum of two fragment
                    # spins may couple to this multiplicity even when a fragment is a singlet.
                    if points[ts]["m"] != 2:
                        continue
                    for s in (a, b):
                        ms = {points[s]["m"] - 1}
                        for k, c in cfg["partners"].get(s, []):
                            if c < 0:
                                ms = set()
                                break
                            for _ in range(c):
                                m = SPIN[k] - 1
                                ms = {
                                    v
                                    for x in ms
                                    for v in range(abs(x - m), x + m + 1, 2)
                                }
                        if 1 not in ms:
                            break
                    else:
                        ref = DIFF @ np.array([v[1] for v in states]) * HARTREE_KCAL
                        zpe = DIFF @ np.array([v[2] for v in states]) * 627.509
                        archived = DIFF @ np.array([points[s]["relzpe"] for s in names])
                        np.testing.assert_allclose(zpe, archived, atol=0.012, rtol=0)
                        pred = DIFF @ np.array([v[3] for v in states]) * KCAL
                        results.append(
                            dict(
                                names=names,
                                conformers=[points[s]["conf"] for s in names],
                                files=[points[s]["file"] for s in names],
                                partners=[cfg["partners"].get(s, []) for s in names],
                                status="endpoint representative or alternatives"
                                if al or ar
                                else "names matched",
                                forward_reverse_reaction_electronic=ref.tolist(),
                                forward_reverse_reaction_with_zpe=archived.tolist(),
                                prediction_kcal=pred.tolist(),
                                imag_cm1=points[ts]["imag_cm1"],
                            )
                        )
                        accepted += 1
                if not accepted:
                    skips[ts] = (
                        "no verified, atom-, charge- and spin-balanced combination"
                    )
            output[series] = dict(
                script=str(script),
                sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                points=len(points),
                channels=results,
                skipped=skips,
            )
    write_json(WORK / "gamma-source-audit.json", dict(model=model_file, series=output))
    for s, entry in output.items():
        print(
            s,
            len(entry["channels"]),
            "assignments",
            len(set(r["names"][1] for r in entry["channels"])),
            "TS names",
            flush=True,
        )


if __name__ == "__main__":
    import sys

    run(sys.argv[1])
