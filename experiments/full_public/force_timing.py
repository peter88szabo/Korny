"""Time one synchronized energy and all Cartesian forces on three fixed molecules.

Start one process per model. Changing positions between calls bypasses the ASE
results cache while preserving neighbor-list reuse, as in single-molecule MD.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dataset_for_training/requested_public"
RESULTS = OUT / "timings"
NAMES = ("methanol", "ethanol", "benzene")


def prepare():
    """Save identical input arrays for both packages' Python environments."""
    from experiments.full_public.check_spectra import panels
    points = json.loads((ROOT / "dataset_for_training/NIST/pbe0_cc_pvtz_stationary_points.json").read_text())
    points.update(panels())
    selected = {k: {f: points[k][f] for f in ("atoms", "positions_A")} for k in NAMES}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "systems.json").write_text(json.dumps(selected, indent=2))


def measure(model):
    systems = json.loads((RESULTS / "systems.json").read_text())
    if model.startswith("korny"):
        from src.forcefield import ForceField
        stage = model.removeprefix("korny-")
        start = perf_counter()
        calculator = ForceField.load(OUT / "kernel-trials" / f"{stage}-high.npz", dispersion="d3")
    else:
        from ase import Atoms
        from so3lr import So3lrCalculator
        start = perf_counter()
        calculator = So3lrCalculator(model=model, lr_cutoff=12., precision="float64")
    result = {"load_s": perf_counter() - start, "systems": {}}
    for index, (name, p) in enumerate(systems.items()):
        r = np.array(p["positions_A"])
        z = np.array([{"H": 1, "C": 6, "O": 8}[s] for s in p["atoms"]])
        if model.startswith("korny"):
            def evaluate(pos):
                return calculator.predict(z, pos, 0, 1)
        else:
            if index:
                # Each calculator owns its atom-count-specific neighbor workspace.
                calculator = So3lrCalculator(model=model, lr_cutoff=12., precision="float64")
            atoms = Atoms(numbers=z, positions=r, pbc=False, info={"charge": 0, "multiplicity": 1})
            atoms.calc = calculator

            def evaluate(pos):
                atoms.set_positions(pos)
                forces = atoms.get_forces()
                return atoms.get_potential_energy(), forces
        direction = np.random.default_rng(819).normal(size=r.shape)
        direction /= np.linalg.norm(direction)
        times, peaks = [], []
        for step in range(7):  # shape warmup, one settling call, five timed MD-like steps
            pos = r + step * 1e-4 * direction
            start = perf_counter()
            energy, force = evaluate(pos)
            force = np.asarray(force)
            elapsed = perf_counter() - start
            assert force.shape == r.shape and np.isfinite(force).all() and np.isfinite(energy)
            times.append(elapsed * 1000)
            peaks.append(float(np.linalg.norm(force)))
        assert np.ptp(peaks) > 1e-10
        item = {"atoms": len(z), "first_s": times[0] / 1000, "warm_ms": times[2:],
                "median_ms": float(np.median(times[2:]))}
        result["systems"][name] = item
        (RESULTS / f"{model}.json").write_text(json.dumps(result, indent=2))
        print(model, name, item, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    args = parser.parse_args()
    if args.model == "prepare":
        prepare()
    else:
        measure(args.model)
