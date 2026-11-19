"""Run from the repository root with python -m experiments.qm7x_pilot.run."""
import argparse
import csv
import json
import platform
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from src.forcefield import ForceField
from src.local import Molecules
from src.pair import PairBaseline

DATA = Path("dataset_for_training/QM7_X/8000_starting_set/8000.hdf5")
OUT = Path(__file__).resolve().parent
SPEC = dict(species=(1, 6, 7, 8, 16, 17), nr=3, lmax=2, ron=1.6, rc=3.6)
SETTINGS = dict(M=64, ell=6., scale_floor=0.1)
RIDGE = 1e-5
FORCE_WEIGHT = 4.  # Å, multiplying the per-geometry RMS force residual
SEED = 7108


def load_sample(path):
    rng = np.random.default_rng(SEED)
    records = []
    strata = defaultdict(list)
    elements = Counter()
    stereoisomers, parents = set(), set()
    geometry_count = 0
    energy_closure = force_closure = atomization_drift = 0.
    with h5py.File(path, "r") as data:
        for mol in sorted(data, key=int):
            group = data[mol]
            families = defaultdict(list)
            for key in sorted(group):
                parent, suffix = key.rsplit("-", 1)
                families[parent].append(suffix)
                stereoisomers.add(parent.rsplit("-", 1)[0])
            parents.update(families)
            geometry_count += len(group)
            parent = str(rng.choice(sorted(families)))
            suffixes = families[parent]
            displacements = [s for s in suffixes if s != "opt"]
            keys = [parent+"-opt"] + [parent+"-"+s for s in
                                      rng.choice(displacements, 4, replace=False)]
            reference = None
            for key in keys:
                g = group[key]
                z, r = g["atNUM"][()], g["atXYZ"][()]
                energy, force = -g["eAT"][()].item(), g["pbe0FOR"][()]
                pbe0 = g["ePBE0"][()].item()
                if reference is None:
                    reference = pbe0-energy
                    strata[17 in z].append(mol)
                    elements.update(set(map(int, z)))
                atomization_drift = max(atomization_drift, abs(pbe0-energy-reference))
                energy_closure = max(energy_closure, abs(
                    g["ePBE0+MBD"][()].item()-g["eMBD"][()].item()-pbe0))
                force_closure = max(force_closure, float(np.max(abs(
                    g["totFOR"][()]-g["vdwFOR"][()]-force))))
                if not all(np.isfinite(x).all() for x in (r, energy, force)):
                    raise ValueError(f"Non-finite label or geometry in {key}")
                records.append((mol, key, z, r, energy, force))
        properties = sorted(g.keys())
    partitions = {}
    for molecules in strata.values():
        ids = rng.permutation(molecules)
        n = round(.15*len(ids))
        for i, mol in enumerate(ids):
            partitions[str(mol)] = "test" if i < n else "validation" if i < 2*n else "train"
    audit = dict(molecules=len(partitions), geometries=geometry_count,
                 stereoisomer_ids=len(stereoisomers), conformer_ids=len(parents),
                 molecules_per_element=dict(elements), properties=properties,
                 sampled_energy_closure_max_eV=energy_closure,
                 sampled_force_closure_max_eV_A=force_closure,
                 sampled_atomic_reference_drift_max_eV=atomization_drift)
    if atomization_drift > 1e-4:
        raise ValueError("Energy changes of -eAT and ePBE0 disagree within a conformer")
    return records, partitions, audit


def counts(z):
    return np.array([np.count_nonzero(z == s) for s in SPEC["species"]])


def predict(z, r, offsets, model=None, pair=None):
    energy, force = counts(z) @ offsets, np.zeros_like(r)
    if pair is not None:
        ep, fp = pair.predict(z, r)
        energy += ep
        force += fp
    if model is not None:
        el, fl, _ = model.observables(z, r)
        energy += el
        force += fl
    return energy, force


def assess(records, model, offsets, pair):
    rows = []
    for mol, key, z, r, energy, force in records:
        prediction, predicted_force = predict(z, r, offsets, model, pair)
        diff = predicted_force-force
        rows.append(dict(molecule=mol, geometry=key, target_eV=energy,
                         prediction_eV=prediction, force_components=force.size,
                         force_abs_error_sum=float(abs(diff).sum()),
                         force_squared_error_sum=float(np.square(diff).sum())))
    errors = np.array([v["prediction_eV"]-v["target_eV"] for v in rows])
    anchors = {v["molecule"]: err for v, err in zip(rows, errors)
               if v["geometry"].endswith("-opt")}
    relative = [err-anchors[v["molecule"]] for v, err in zip(rows, errors)
                if not v["geometry"].endswith("-opt")]
    n = sum(v["force_components"] for v in rows)
    metrics = dict(energy_MAE_eV=float(abs(errors).mean()),
                   displacement_energy_MAE_eV=float(np.mean(np.abs(relative))),
                   force_MAE_eV_A=sum(v["force_abs_error_sum"] for v in rows)/n,
                   force_RMSE_eV_A=np.sqrt(sum(v["force_squared_error_sum"] for v in rows)/n))
    return metrics, rows


def call_latency(model, samples, offsets, pair):
    # Same warmed geometries for both heads. Include descriptor construction.
    geometries = [(v[2], v[3]) for v in samples[::5][:12]]
    for z, r in geometries:
        predict(z, r, offsets, model, pair)
    times = []
    for _ in range(3):
        for z, r in geometries:
            t = perf_counter()
            predict(z, r, offsets, model, pair)
            times.append((perf_counter()-t)*1000)
    return float(np.median(times))


def run(path, out):
    out.mkdir(parents=True, exist_ok=True)
    records, partitions, audit = load_sample(path)
    splits = {s: [v for v in records if partitions[v[0]] == s]
              for s in ("train", "validation", "test")}
    train = splits["train"]
    offsets = np.linalg.lstsq([counts(v[2]) for v in train], [v[4] for v in train],
                              rcond=None)[0]
    residuals = [(z, r, e-counts(z)@offsets, f) for _, _, z, r, e, f in train]
    config = dict(seed=SEED, specification=SPEC, sparse_settings=SETTINGS, ridge=RIDGE,
                  force_weight_A=FORCE_WEIGHT, target="-eAT", force_target="pbe0FOR",
                  composition_offsets_eV=offsets.tolist(), python=platform.python_version(),
                  numpy=np.__version__, h5py=h5py.__version__)
    report = dict(inventory=audit, config=config, split_molecules=dict(Counter(partitions.values())),
                  split_geometries={s:len(v) for s,v in splits.items()}, models={})
    print("Sample", report["split_geometries"], flush=True)
    predictions = []
    # Each file stores a complete local head. Composition offsets are shared.
    # Fit one pair baseline on training molecules, then use its E and F as offsets.
    pair_ridge = 0.05
    pair_file = out/"pair.npz"
    start = perf_counter()
    pair_model = PairBaseline(SPEC["species"], np.arange(.8, 3.5, .15), .18, 2.8, 3.6)
    pair_model.fit(residuals, ridge=pair_ridge, force_weight=FORCE_WEIGHT)
    pair_fit_time = perf_counter()-start
    pair_model.save(pair_file)
    pair_model = PairBaseline.load(pair_file)
    residuals_after_pair = []
    start = perf_counter()
    for z, r, e, f in residuals:
        ep, fp = pair_model.predict(z, r)
        residuals_after_pair.append((z, r, e-ep, f-fp))
    pair_subtraction_s = perf_counter()-start
    config["pair"] = dict(centers_A=pair_model.centers.tolist(), width_A=pair_model.width,
                          ron_A=pair_model.ron, rc_A=pair_model.rc, ridge=pair_ridge)
    for name, head, use_pair, cap in (
            ("composition", None, False, 0), ("gaussian", "gaussian", False, 0),
            ("tt12", "tt", False, 12), ("tt32", "tt", False, 32),
            ("pair", None, True, 0), ("pair_gaussian", "gaussian", True, 0),
            ("pair_tt12", "tt", True, 12)):
        model = None
        pair = pair_model if use_pair else None
        fit_data = residuals_after_pair if use_pair else residuals
        result = dict(training_s=pair_fit_time if use_pair else 0.,
                      checkpoint_bytes=pair_file.stat().st_size if use_pair else 0)
        if head:
            t = perf_counter()
            options = dict(order=2, cap=cap) if head == "tt" else {}
            model = Molecules([(z, r) for z, r, _, _ in fit_data], SPEC, heads=head,
                              **SETTINGS, **options)
            built = perf_counter()
            model.fit(fit_data, ridge=RIDGE, force_weight=FORCE_WEIGHT)
            fitted = perf_counter()
            file = out/(name+".npz")
            model.save_local(file)
            if name == "pair_gaussian":
                ForceField(pair, model, offsets, {0: 0.}, "PBE0 (-eAT)").save(out/"korny_qm7x_pbe0.npz")
            model = Molecules.load_local(file)
            result.update(options=options, build_s=built-t, regression_s=fitted-built)
            result["training_s"] += fitted-t + (pair_subtraction_s if use_pair else 0.)
            result["checkpoint_bytes"] += file.stat().st_size
        result["median_force_call_ms"] = call_latency(model, splits["validation"], offsets, pair)
        for s, samples in splits.items():
            result[s], rows = assess(samples, model, offsets, pair)
            predictions.extend(dict(model=name, split=s, **row) for row in rows)
        report["models"][name] = result
        print(name, result, flush=True)
    (out/"results.json").write_text(json.dumps(report, indent=2)+"\n")
    with (out/"predictions.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    run(args.data, args.out)
