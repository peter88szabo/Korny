"""Compare the final checkpoints at relaxed NIST minima, including butadiene."""

import json
import multiprocessing as mp
import re
from concurrent.futures import ProcessPoolExecutor

from experiments.full_public.specified import OUT
from experiments.reactivity.larger_vibrations import (
    SOURCE,
    _table,
    references,
    evaluate,
)
from experiments.reactivity.stationary_fit import POINTS
from src.forcefield import ForceField


def panels():
    points = json.loads(POINTS.read_text())
    points.update(references())
    # The first conformer on the archived butadiene page redirects the reader
    # to the second conformer. Select the block containing its actual tables.
    s = (SOURCE / "106990.html").read_bytes().decode("latin-1")

    def table(label):
        marker = label + " calculated at PBE1PBE/cc-pVTZ"
        return _table(marker + s.rsplit(marker, 1)[1], label)

    xyz = table("Geometric Data")
    freq = table("Vibrational Frequencies")
    points["1,3-butadiene"] = dict(
        atoms=[re.match("[A-Z][a-z]?", r[0])[0] for r in xyz],
        positions_A=[[float(x) for x in r[1:4]] for r in xyz],
        frequencies_cm1=sorted(float(r[2]) for r in freq),
        source="https://cccbdb.nist.gov/energy3x.asp?method=54&basis=6&casno=106990&charge=0",
    )
    assert len(freq) == 3 * len(xyz) - 6
    return {s: points[s] for s in ("propane", "benzene", "ethanol", "1,3-butadiene")}


def case(args):
    model, disp, name, p = args
    result = evaluate(ForceField.load(OUT / model, dispersion=disp), {name: p})[name]
    print(
        model, name, result.get("mae_cm1"), result.get("same_connectivity"), flush=True
    )
    return model, name, result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint", action="store_true")
    joint = parser.parse_args().joint
    prefix = "joint-" if joint else ""
    todo = [
        (m, d, n, p)
        for m, d in [
            ("pbe0-stationary-core.npz", "off"),
            ("pbe0-reaction.npz", "off"),
            ("high-reaction.npz", "d3"),
        ]
        for n, p in panels().items()
    ]
    if joint:
        todo = [
            (prefix + m, d, n, p)
            for m, d, n, p in todo
            if m != "pbe0-stationary-core.npz"
        ]
    allstats = {}
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("fork")) as pool:
        for m, n, ans in pool.map(case, todo):
            allstats.setdefault(m, {})[n] = ans
            (OUT / (prefix + "relaxed-frequency-validation.json")).write_text(
                json.dumps(allstats, indent=2) + "\n"
            )
