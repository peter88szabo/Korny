"""Write the common-ID and external-set results from measured checkpoint artifacts."""

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from experiments.full_public import kernel_trials as kt
from experiments.full_public.aimnet_parent import ROOT as AIM
from experiments.full_public.download_references import CHUNK, source
from experiments.full_public.paired_baseline_audit import paired_intervals
from experiments.full_public.paper_data import OUT
from experiments.full_public.rdb19_irc import ROOT as RDB

PAN = kt.OUT / "so3lr-tauri-matched"
DEST = Path(__file__).resolve().parents[2] / "report/expanded-reference-fits.md"
LABELS = {
    37: "original curvature fit",
    41: "orbital curvature",
    42: "controlled midpoint refit",
    43: "paper reaction data",
    44: "pilot radical IRCs and p41 high-level surface",
    45: "full radical IRCs and p41 high-level surface",
    46: "pilot AIMNet2 pair transfer",
    47: "full AIMNet2 pair transfer",
    48: "size-dependent spin correction",
    49: "atom-resolved spin kernel",
    50: "network forces and measured normal modes",
    51: "short-range donor and acceptor environments",
    52: "joint calibration of four expanded-data corrections",
    53: "stationary-geometry DFT base and coupled-cluster correction",
    54: "transfer from reoptimized PM6, HF and r²SCAN-3c structures",
    55: "conservative neighbor-to-neighbor spin redistribution",
    56: "multiscale radical-path transfer",
    57: "local environment changes along radical paths",
    58: "Gaussian derivatives along reacting atomic environments",
    59: "full-release multiscale radical-path transfer",
    60: "reaction family inferred from local geometry",
    61: "denser atomic kernel on the full radical archive",
    62: "direct dense two-resolution fits on Korny references",
    65: "dense kernel with 5 kcal/mol Huber gap and midpoint losses",
    68: "doublet transfer from native energy, force and curvature fits",
    69: "Gaussian landmark correction",
    70: "Gaussian correction with additional singlet DFT supervision",
    71: "reaction-family-balanced Gaussian correction",
    72: "variational split-charge response",
    73: "Chebyshev modes",
    74: "graph diffusion modes",
    75: "Hermite oscillator modes",
    76: "Laguerre functions",
    77: "Legendre modes",
    78: "Jacobi modes",
    79: "Dirichlet sine modes",
    80: "Neumann cosine modes",
    81: "radial disk Bessel modes",
    82: "Krawtchouk modes",
    83: "Hahn modes",
    84: "Morse bond stretches and Legendre bends",
    85: "Legendre modes with pairwise coordinate products",
    86: "Chebyshev H-transfer correction with expanded reference data",
    87: "Krawtchouk H-transfer correction with expanded reference data",
    88: "Chebyshev H-transfer correction with a smooth angular window",
    89: "Krawtchouk H-transfer correction with a smooth angular window",
}


def read(p):
    return json.loads(p.read_text())


def common_rows(k, models):
    """Use the common active reaction IDs in both archived and newer score files."""
    with (PAN / f"p{k}-public-comparison.csv").open() as f:
        for a in csv.DictReader(f):
            if a["source"] not in kt.BARRIERS:
                continue
            try:
                scores = [float(a[n + " error_kcal"]) for n in models]
            except (ValueError, TypeError):
                continue
            if all(np.isfinite(scores)):
                yield a


def midpoint(k, name, models):
    errors = defaultdict(dict)
    for a in common_rows(k, models):
        record = errors[a["source"], a["id"]]
        if a["observable"] in record:
            raise ValueError("Duplicate reaction observable")
        record[a["observable"]] = float(a[name + " error_kcal"])
    if any(sum(s == source for s, _ in errors) != n for source, n in kt.COMMON_REACTIONS.items()) or any(
        set(d) != {"reaction", "forward", "reverse"} for d in errors.values()
    ):
        raise ValueError("Common ID set changed")
    return np.mean([abs((d["forward"] + d["reverse"]) / 2) for d in errors.values()])


def spin_comparison(stages, last):
    """Group the same common IDs by the archived transition-state multiplicity."""
    geoms, equations, _ = kt.inputs()
    spin = {(s, name): geoms[terms[-1][0]].state[1]
            for _, s, split, name, terms, _, kind in equations
            if s in kt.BARRIERS and kind == "forward" and split == "validation"}
    lines = ["## Comparison by transition-state multiplicity", "",
             "| Model | Spin | Reactions | Reaction | All barriers | Forward | Reverse |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for k in dict.fromkeys([49, 53, last]):
        if k not in stages:
            continue
        models = stages[k]["priority"]
        records = list(common_rows(k, models))
        for model in models if k == last else ("Korny",):
            values = defaultdict(list)
            for r in records:
                m = spin[r["source"], r["id"]]
                if m not in (1, 2):
                    raise ValueError("Unexpected multiplicity on the singlet/doublet panel")
                values[m, r["observable"]].append(abs(float(r[model + " error_kcal"])))
            for m, label in ((1, "singlet"), (2, "doublet")):
                rx, f, rev = (values[m, kind] for kind in ("reaction", "forward", "reverse"))
                name = f"Korny p{k}" if model == "Korny" else model
                lines.append(f"| {name} | {label} | {len(rx)} | {np.mean(rx):.2f} | "
                             f"{np.mean(f + rev):.2f} | {np.mean(f):.2f} | {np.mean(rev):.2f} |")
    return lines


def run():
    out = [
        "# Expanded-data Korny fits",
        "",
        "Measured through "
        + datetime.now().astimezone().isoformat(timespec="minutes")
        + ". The target is 5.0 kcal/mol MAE each for reaction energies and barriers, together with credible equilibrium and transition-state frequencies.",
        "",
        "## Reactions on the fixed common IDs",
        "",
        "The active panel contains 259 RDB7, 47 BH9 and 283 Grambow radical reactions. Each row uses the same 589 reactions and 1,178 directed barriers. The midpoint height equals the mean of the two directed barriers. Each Korny calculation includes D3(BJ) and the ATM term once. This repeatedly inspected panel measures development performance. The p49 backbone includes earlier TRIP50 training. The new corrections exclude TRIP50 from fitting and scoring.",
        "",
        "| Model | Reaction | Midpoint | All barriers | Forward | Reverse |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    stages = {
        k: read(PAN / f"p{k}-public-summary.json")["common"]
        for k in LABELS
        if (PAN / f"p{k}-public-summary.json").exists()
    }
    last = max(stages)
    if last >= 86:
        out[4:4] = ["The H-transfer fits also have reaction and barrier tables on RDB19-Rad and the held-out AutoMeKin network. [Expanded-source H-transfer benchmarks](hydrogen-transfer-kernels.md)", ""]
    radical_result = RDB / "multiscale-path67/held-path-curvature.json"
    if last >= 68 and radical_result.is_file():
        native_dir = radical_result.parent
        energy = read(native_dir / "all-held-validation.json")[0]["shared"]
        shape = read(radical_result)["summary"]["67"]
        mixed = stages[last]["priority"]["Korny"]
        previous = stages[49]["priority"]["Korny"]
        out[4:4] = [
            "## Latest results", "",
            f"On 626 held RDB19-Rad neutral-doublet paths, p67 reached {energy['reaction']:.2f} kcal/mol reaction MAE and {energy['barriers']:.2f} kcal/mol pooled barrier MAE. The force change across the saddle gave {shape['signed_mae_cm1']:.0f} cm⁻¹ signed directional error, with the downhill sign correct for {shape['correct_downhill']} of {shape['reference_downhill']} paths. This is a source-specific electronic potential with a fixed dispersion reference.", "",
            f"On the 589 mixed reaction IDs, p{last} reached {mixed['reaction']:.2f} kcal/mol for reactions and {mixed['barriers']:.2f} for barriers. The changes from p49 were {mixed['reaction']-previous['reaction']:+.3f} and {mixed['barriers']-previous['barriers']:+.3f} kcal/mol. The residual saddle forces and curvature errors remain important limits for reactive dynamics. [Eigenfunction trials, full per-source comparisons and frequency checks](operator-kernel-benchmarks.md)", "",
        ]

    def line(k, m, label):
        d = stages[k]["priority"][m]
        vals = [
            d["reaction"],
            midpoint(k, m, stages[k]["priority"]),
            d["barriers"],
            d["forward"],
            d["reverse"],
        ]
        out.append("| " + label + " | " + " | ".join(f"{v:.2f}" for v in vals) + " |")

    for k in stages:
        line(k, "Korny", f"Korny p{k}, {LABELS[k]}")
    for m in stages[last]["priority"]:
        if m != "Korny":
            line(last, m, m)
    if last >= 55:
        d = paired_intervals(f"p{last}-public")
        out += ["", f"Paired MAE changes for p{last} relative to p49 use the same 589 reactions. The 95% bootstrap intervals resample whole transition-state formulas within each source.", "",
                "| Observable | Change in MAE (kcal/mol) | 95% interval |", "|---|---:|---:|"]
        for key,name in (("reaction","Reaction"),("barriers","All barriers"),("forward","Forward"),("reverse","Reverse")):
            change = stages[last]["priority"]["Korny"][key] - stages[49]["priority"]["Korny"][key]
            out.append(f"| {name} | {change:+.3f} | [{d[key][0]:+.3f}, {d[key][1]:+.3f}] |")
    active_csv = PAN / f"p{last}-public-active-common.csv"
    active = list(common_rows(last, stages[last]["priority"]))
    with active_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=active[0].keys())
        writer.writeheader()
        writer.writerows(active)
    out += [
        "",
        "Energy MAEs are in kcal/mol. The high-level surface scores RDB7 and BH9. The ωB97X-D3 surface scores Grambow. [Errors and reference energies by common reaction ID](../dataset_for_training/requested_public/so3lr-tauri-matched/p"
        + str(last)
        + "-public-active-common.csv)",
        "",
        *spin_comparison(stages, last),
        "",
        "## Results by reaction dataset",
        "",
        "| Dataset | Model | Reactions | Reaction | All barriers | Forward | Reverse |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in kt.BARRIERS:
        for k in (49,52,62,65):
            if k == last or k not in stages:
                continue
            base = stages[k][dataset]["Korny"]
            out.append(
                f"| {dataset} | Korny p{k} | {base['n']} | {base['reaction']:.2f} | "
                f"{base['barriers']:.2f} | {base['forward']:.2f} | {base['reverse']:.2f} |"
            )
        for method, scores in stages[last][dataset].items():
            label = f"Korny p{last}" if method == "Korny" else method
            out.append(
                f"| {dataset} | {label} | {scores['n']} | {scores['reaction']:.2f} | "
                f"{scores['barriers']:.2f} | {scores['forward']:.2f} | {scores['reverse']:.2f} |"
            )
    out += [
        "",
        "## Frequency and saddle checks",
        "",
        "| Model | Propane | Benzene | Ethanol | 1,3-butadiene | Signed unstable-direction MAE | Correct curvature sign |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for k in stages:
        path = kt.WORK / f"p{k}-public-spectra.json"
        diag = kt.WORK / f"p{k}-public-diagnostics.json"
        if not path.exists() or not diag.exists():
            continue
        sp = read(path)
        h = read(diag)["radical"]["saddle"]
        vals = [
            sp[n].get("mae_cm1") for n in ["propane", "benzene", "ethanol", "1,3-butadiene"]
        ] + [h["mae_cm1"][0]]
        out.append(
            f"| p{k} | "
            + " | ".join(f"{v:.1f}" if v is not None else "Relaxation failed" for v in vals)
            + f" | {round(h['sign_agreement'][0] * h['n'])}/{h['n']} |"
        )
    out += [
        "",
        "All frequency MAEs are in cm⁻¹. The first four columns compare independently relaxed Korny minima with PBE0/cc-pVTZ. P44 inherits the p41 high-level surface. P48 and p49 inherit its singlet spectra because their spin corrections vanish at multiplicity one. The radical column measures signed curvature along one archived mode at each of 278 saddles.",
    ]
    geometry = (
        read(kt.WORK / f"p{last}-public-spectra.json")
        if (kt.WORK / f"p{last}-public-spectra.json").is_file()
        else {}
    )
    if geometry:
        out += [
            "",
            f"| Relaxed geometry, p{last} | Bond MAE (Å) | Angle MAE (°) | Aligned RMSD (Å) | Connectivity retained | Imaginary internal modes |",
            "|---|---:|---:|---:|---|---:|",
        ]
        for name, row in geometry.items():
            if "relaxation_failure" in row:
                out.append(f"| {name} | — | — | — | Relaxation failed | — |")
                continue
            out.append(
                f"| {name} | {row['bond_mae_A']:.4f} | {row['angle_mae_deg']:.2f} | "
                f"{row['aligned_rmsd_A']:.3f} | {'yes' if row['same_connectivity'] else 'no'} | "
                f"{sum(x < 0 for x in row['predicted_cm1'])} |"
            )
    out += [
        "",
        "| Full Hessian at reference geometry | Reference unstable modes | p37 | p41 |",
        "|---|---:|---:|---:|",
    ]
    for a, b in zip(
        read(kt.WORK / "p37-public-full-stationary-audit.json"),
        read(kt.WORK / "p41-public-full-stationary-audit.json"),
        strict=True,
    ):
        if a["id"] != b["id"]:
            raise ValueError("Mismatched Hessian audit")
        if a["role"] == "saddle":
            out.append(
                f"| {a['id']} | {a['n_imag_below_minus50'][0]} | {a['n_imag_below_minus50'][1]} | {b['n_imag_below_minus50'][1]} |"
            )
    out += [
        "",
        "The unstable-mode count uses a −50 cm⁻¹ threshold. All four p41 saddles retain excess negative curvature. The models also have residual forces at these archived geometries. Reactive MD and saddle optimization therefore require further validation.",
        "",
        "## Training material",
        "",
        "| Dataset | Training material | Reserved validation |",
        "|---|---|---|",
        "| Korny bare PBE0 backbone | 91,156 unique geometries across QM7-X 8000, DES15K, SPICE dipeptides and TorsionNet | 9,603 geometries |",
        "| RDB7 / BH9 / Grambow | 11,667 / 400 / 1,084 training reactions | 259 / 47 / 283 common validation reactions |",
        "| AIMNet2-NSE DFT supplement | 20,038 geometry/state observations, including 241 doublets and 8,531 triplets | 764 singlet/triplet benzopinacol frames |",
        "| BASChem19 | 15 forward barriers | 4 forward barriers |",
        "| AutoMeKin connected electronic channels | 538 DFT, 446 local CC and 51 canonical CC channels | 329 DFT and 265 local CC channels on the complete cBD-CN network |",
        "| AutoMeKin separated fragments | 62 additional CC channels with independently identified charged/spin-labelled components | 8 additional CC channels on cBD-CN |",
    ]
    for k in (44, 45):
        f = kt.WORK / f"p{k}-public-settings.json"
        if f.exists():
            d = read(f)
            out.append(
                f"| RDB19-Rad p{k} | {d['training_records']:,} doublet reactions, {d['training_frames']:,} distinct path positions across both directions | {d['validation_records']:,} reactions, {d['validation_frames']:,} positions |"
            )
    for tag in ("pilot", "full"):
        f = AIM / f"charges-{tag}/metrics.json"
        if f.exists():
            d = read(f)
            out.append(
                f"| AIMNet2 closed-shell {tag} | {d['training_geometries']:,} geometries | {d['heldout_geometries']:,} formula-held geometries, {d['heldout_atoms']:,} atoms |"
            )
    population_file = kt.OUT.parent / "AIMNet_NSE_2021/neighbor-spin55/populations.json"
    if population_file.exists():
        counts = read(population_file)["training_by_charge"]
        out.append(f"| p55 atom-resolved spin redistribution | {counts['-1'] + counts['1']:,} Ions-16 doublets and {counts['0']:,} neutral Lewis assignments | Formula-held ions, ChEMBL-20 ions and neutral radical centers |")
    modefile = OUT / "normal-modes/manifest.json"
    if modefile.is_file():
        modes = read(modefile)["records"]
        tr, ev = (
            [r for r in modes if r["split"] == v] for v in ("train", "validation")
        )
        out.append(
            f"| AutoMeKin DFT normal modes | {len(tr):,} stationary points, {sum(r['modes'] for r in tr):,} signed modes | {len(ev):,} cBD-CN structures, {sum(r['modes'] for r in ev):,} modes |"
        )
    out += [
        "",
        "The AIMNet2-NSE paper describes 13 million additional open-shell configurations and 200,000 reaction profiles. Its public links provided the five-system DFT supplement, BASChem19 and trained weights. A public download for the complete open-shell training extension is still unlocated. The original AIMNet2 closed-shell ωB97M parent has 20,108,565 geometries and 474,624,017 atoms. [AIMNet2-NSE article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12851018/) · [AIMNet2 parent archive](https://doi.org/10.1184/R1/27629937.v2) · [BASChem19](https://doi.org/10.1184/R1/29710664.v1)",
        "",
        "The RDB19-Rad import uses electronic energies and Cartesian forces from accepted Gaussian IRC points. Training uses path-relative energies and force components along short Cartesian displacements. The authors’ gas/solution Gibbs energies have separate fields. The two directions share a single transition state, counted once per retained reaction. The full deposit has 5,600 reaction IDs and 75 reaction-class labels. [RDB19-Rad data](https://zenodo.org/records/11493786)",
        "",
        "A separated AutoMeKin product enters the CC fit only when all fragment ORCA outputs supply charge, multiplicity, coordinates and energy. The importer verifies each electronic energy against the CSV and checks atom and charge balance across the channel. MEA uses canonical CCSD(T)-F12 and the four other networks use DLPNO-CCSD(T)-F12. [AutoMeKin archive](https://zenodo.org/records/22808731) · [Imported electronic channels and coverage](../dataset_for_training/paper_references/fragment-channels.json)",
    ]
    out += [
        "",
        "## Radical transfer beyond the original panel",
        "",
        "| External set and model | Reactions | Reaction | All barriers | Forward | Reverse |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def metrics(label, n, g, forward, reverse):
        out.append(
            f"| {label} | {n} | {g:.2f} | {(forward + reverse) / 2:.2f} | {forward:.2f} | {reverse:.2f} |"
        )

    for k in (44, 45):
        f = RDB / f"path-kernel{k}/deployed-endpoint-validation.json"
        if not f.exists():
            continue
        d = read(f)
        for name, v in d["mae_kcal"].items():
            metrics(
                f"RDB19-Rad {k} holdout, {name}",
                d["assigned"],
                v["reaction"],
                v["forward"],
                v["reverse"],
            )
    pilot = RDB / "path-kernel44/p49-deployed-endpoint-validation.json"
    if pilot.exists():
        d = read(pilot)
        v = d["mae_kcal"]["p49"]
        metrics(
            "RDB19-Rad 44 holdout, p49",
            d["assigned"],
            v["reaction"],
            v["forward"],
            v["reverse"],
        )
    f = OUT / "spin-extent48/external-validation.json"
    if f.is_file():
        d = read(f)
        for key in ("connected minima", "separated fragments"):
            q = d["sets"][key]
            v = np.array(q["mae_kcal"])
            for i, name in enumerate(d["models"]):
                metrics(f"cBD-CN CC {key}, {name}", q["n"], v[0, i], v[2, i], v[3, i])
        out += [
            "",
            "RDB19-Rad assignments match both terminal IRC geometries to the authors’ educt and product complex XYZ files. Values describe electronic barriers measured from these complexes. The 265 connected and eight fragmented cBD-CN channels all belong to one held-out chemical network.",
            "",
            "| Benzopinacol profile MAE, kcal/mol | Frames | p44 | p48 |",
            "|---|---:|---:|---:|",
        ]
        for key in (
            "benzopinacol",
            "benzopinacol within m=1",
            "benzopinacol within m=3",
        ):
            v = d["sets"][key]
            out.append(
                f"| {key} | {v['n']} | {v['mae_kcal'][0]:.1f} | {v['mae_kcal'][1]:.1f} |"
            )
        out += [
            "",
            "The complete scan uses the first singlet geometry as its energy origin. Each within-spin row uses its own first geometry. P48 normalizes three global spin contributions by the heavy-atom count and fits their amplitudes using training-only spin-state and reaction evidence. The cross-spin offset improved while the held-out triplet path remained inaccurate. [Spin fit and force checks](../experiments/full_public/trial_spin_extent.py)",
        ]
    if 49 in stages:
        v = read(OUT / "local-spin49/external-validation.json")
        out += [
            "",
            "| Held-out cBD-CN reference, p49 | Channels | Reaction | All barriers | Forward | Reverse |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for k, d in v["high"].items():
            rx, _mid, fw, rv = d["mae_kcal"]
            metrics(k, d["n"], rx, fw, rv)
        out += [
            "",
            f"Across both benzopinacol spin branches, the p49 MAE is {v['wb97xd3']['benzopinacol_profile_mae']:.1f} kcal/mol relative to the first singlet frame.",
        ]
    redox_path = OUT / "redox-environments/external-validation.json"
    if 51 in stages and redox_path.exists():
        results = read(redox_path)
        out += [
            "", "## Local chemical environments from vertical electronic energies", "",
            "P51 uses environment scores learned from vertical ionization energies and electron affinities to modulate short-range pair interactions. The correction and its derivative go to zero between separated fragments. Both Korny energy surfaces retain their original reaction-gap, midpoint, force and curvature supervision.", "",
            "| cBD-CN coupled-cluster channels | Model | Reactions | Reaction | All barriers | Forward | Reverse |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for name,d in results["high"].items():
            for model in ("p49","p51"):
                gap, _, fwd, rev=d[model+"_mae_kcal"]
                out.append(f"| {name} | {model} | {d['n']} | {gap:.2f} | {(fwd+rev)/2:.2f} | {fwd:.2f} | {rev:.2f} |")
        d=results["wb97xd3"]["benzopinacol"]
        out.extend(["",f"The benzopinacol scan MAE across {d['n']} frames is {d['p51_mae_kcal']:.1f} kcal/mol for p51 and {d['p49_mae_kcal']:.1f} for p49. [Kernel](../src/redox_environment.py) · [Reaction refit](../experiments/full_public/trial_redox_fit.py)"])
    if 52 in stages and (OUT / "joint-transfer52/external-validation.json").is_file():
        joint = read(kt.WORK / "p52-public-settings.json")
        out += ["", "## Joint fit with the expanded archives", "",
                f"P52 calibrates four corrections per electronic reference. The native radical data supply {joint['training_irc_pairs']:,} training reactions. AutoMeKin supplies {joint['training_stationary_points']:,} training stationary points. The AIMNet2 contribution transfers a pair-energy model estimated on the closed-shell parent archive. Ions-16 supplies vertical redox information.",
                "", "| Target surface | Additional full-release radical IRC correction | AIMNet2 pair correction | AutoMeKin Hessian correction | Ions-16 pair correction |",
                "|---|---:|---:|---:|---:|"]
        for t, weights in joint["amplitudes"].items():
            values = [weights[key] for key in ("full_minus_pilot_radical_IRC", "full_AIMNet2_parent", "AutoMeKin_Hessians", "Ions16_redox_pairs")]
            out.append(f"| {t} | " + " | ".join(f"{v:.3f}" for v in values) + " |")
        out += ["", "The following p52 results use the held cBD-CN network. The high-level branch uses coupled-cluster reference channels. The second branch is compared with the DFT channels.", "",
                "| Target | Channels | Reactions | Reaction | All barriers | Forward | Reverse |",
                "|---|---|---:|---:|---:|---:|---:|"]
        for t, sets in read(OUT / "joint-transfer52/external-validation.json").items():
            for channels, result in sets.items():
                gap, _, fwd, rev = result["gap_mid_fwd_rev_mae_kcal"]
                out.append(f"| {t} | {channels} | {result['n']} | {gap:.2f} | {(fwd+rev)/2:.2f} | {fwd:.2f} | {rev:.2f} |")
    if 68 in stages:
        fit = read(kt.WORK / "p68-public-settings.json")
        out += ["", "## Transfer of the native force and curvature fits", "",
                "P68 learns three coefficients on the Grambow training reactions and their stationary-point derivative constraints. The three inputs are scalar functions of one doublet geometry. They use the p61 native energy, the p64 minus p61 energy, and the p67 minus p64 energy. The last change includes both the measured saddle curvatures and the corresponding retuning of the Cartesian force weight. The high-level branch and relaxed singlet spectra come from p49. The target calculation includes explicit D3(BJ) and ATM once.", "",
                "| RDB19-Rad contribution | Coefficient on the Grambow surface |", "|---|---:|"]
        for label, weight in zip(("Native path energy", "Force-trained increment", "Curvature-trained increment"),
                                 fit["readout"], strict=True):
            out.append(f"| {label} | {weight:.5f} |")
    if 50 in stages and (OUT / "network-hessians50/external-validation.json").is_file():
        data = read(OUT / "network-hessians50/external-validation.json")
        out += [
            "",
            "## AutoMeKin force and curvature fit",
            "",
            "P50 takes two measured Hessian-vector products and a zero-force constraint per stationary point. One probe follows the softest mode. The second combines all remaining modes. The source DFT frequencies agree with the independent stationary-point table and the mass-weighted displacements are orthogonal. P50 uses the DFT forces and modes as auxiliary targets for its coupled-cluster energy surface. All cBD-CN channels and structures stayed outside model fitting and parameter selection.",
            "",
            "| cBD-CN electronic reference | Model | Reactions | Reaction | All barriers | Forward | Reverse |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for t, results in data.items():
            refname = "DLPNO-CCSD(T)-F12" if t == "high" else "ωB97X-D3"
            for label in ("connected", "separated"):
                if label not in results:
                    continue
                d = results[label]
                for model in ("p49", "p50"):
                    gap, _, fwd, rev = d[model + "_mae_kcal"]
                    out.append(
                        f"| {refname}, {label} | {model} | {d['n']} | {gap:.2f} | {(fwd + rev) / 2:.2f} | {fwd:.2f} | {rev:.2f} |"
                    )
        out += [
            "",
            "| Korny surface | cBD-CN structures | Count | Model | Soft-mode MAE (cm⁻¹) | Correct sign | Cartesian force MAE (eV/Å) |",
            "|---|---|---:|---|---:|---:|---:|",
        ]
        for t, results in data.items():
            for role in ("minimum", "saddle"):
                d = results[role]
                for idx, model in enumerate(("p49", "p50")):
                    out.append(
                        f"| {t} | {role} | {d['structures']} | {model} | {d['soft_direction_mae_cm1'][idx]:.1f} | {100 * d['sign_agreement'][idx]:.1f}% | {d['force_mae_eV_A'][idx]:.3f} |"
                    )
        out += [
            "",
            "The soft-mode measurement projects the predicted Hessian along the lowest published DFT eigenvector at the archived DFT geometry. Checking the number of unstable eigenvectors requires the full model Hessian. [Mode importer](../experiments/full_public/automekin_modes.py) · [Fitting and evaluation](../experiments/full_public/trial_network_hessians.py)",
        ]
    ion = kt.OUT.parent / "AIMNet_NSE_2021/spin-validation.json"
    if ion.is_file():
        d = read(ion)
        counts = d["training_rows_by_charge"]
        out += [
            "",
            "## Atom-resolved spin supervision",
            "",
            f"Ions-16 supplied {counts.get('-1', 0):,} training anion doublets and {counts.get('1', 0):,} cation doublets. The neutral training set supplied {counts.get('0', 0):,} radical-center labels from Korny’s existing mapped reactions.",
            "",
            "| Validation | Charge | Geometries | Atom spin MAE (e) | Equal-share MAE (e) | Most populated atom correct |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for v in d["metrics"]:
            out.append(
                f"| {v['split']} | {v['system_charge']:+d} | {v['geometries']:,} | {v['mae_e']:.4f} | {v['uniform_mae_e']:.4f} | {100 * v['top1_fraction']:.1f}% |"
            )
        out += [
            "",
            "The 2021 AIMNet-NSE release supplies PBE0/ma-def2-SVP spin-resolved NBO populations for ions. The neutral target is a mapped Lewis radical center. The predictor enforces the input molecular spin and permits signed atomic polarization. The ChEMBL-20 set and all its molecular formulas were excluded from Ions-16 training. The original 2021 authors used Ions-16 for testing, so Korny’s Ions-16 training rows cannot benchmark Korny against the 2021 method. [Original AIMNet-NSE data](https://zenodo.org/records/5007980) · [Authors’ field definitions](https://github.com/isayevlab/aimnetnse/blob/main/README.md)",
        ]
    redox = kt.OUT.parent / "AIMNet_NSE_2021/vertical-gaps/validation.json"
    if redox.exists():
        nums = read(redox)
        out.extend(
            [
                "",
                "## Vertical ionization energies and electron affinities",
                "",
                "The 2021 archive supplies charged doublets and neutral singlets at identical Cartesian coordinates. The auxiliary model learns the vertical ionization potential (cation minus neutral) and electron affinity (neutral minus anion) with independent coefficients. Both the analytical spin-weighted local environments and their molecular means enter the fit. Korny’s reaction and vibration tables report separate potential-energy-surface evaluations.",
                "",
                "| Quantity | Paired training geometries | Validation set | Pairs | MAE (eV) | Training-mean baseline (eV) |",
                "|---|---:|---|---:|---:|---:|",
            ]
        )
        for name, v in nums["metrics"].items():
            for key, label in (
                ("ions16_formula_holdout", "Ions-16, held formulas"),
                ("chembl20", "ChEMBL-20, independent"),
            ):
                a = v[key]
                out.append(
                    f"| {name} | {v['training_pairs']:,} | {label} | {a['pairs']:,} | {a['mae_eV']:.3f} | {a['mean_baseline_mae_eV']:.3f} |"
                )
        out.extend(
            [
                "",
                f"{nums['counts']['ions16 missing neutral']:,} Ions-16 ion records and {nums['counts']['chembl20 missing neutral']:,} ChEMBL-20 ion records lack the corresponding neutral reference. Their available atomic spin labels still contribute to the respective spin datasets. [Vertical-energy pairing and fit](../experiments/full_public/trial_vertical_gaps.py)",
            ]
        )
    out += [
        "",
        "## Closed-shell auxiliary fits",
        "",
        "| Atom-count groups | Held-out geometry count | Charge MAE (e) | Energy MAE (eV/atom) | Force-component MAE (eV/Å) |",
        "|---|---:|---:|---:|---:|",
    ]
    for tag, desc in (("pilot", "2–8 atoms"), ("full", "2–76 atoms")):
        f = AIM / f"charges-{tag}/metrics.json"
        e = AIM / f"potential-{tag}/metrics.json"
        if not (f.exists() and e.exists()):
            continue
        a = read(f)
        b = read(e)
        out.append(
            f"| {desc} | {a['heldout_geometries']:,} | {a['mae_e']:.4f} | {b['per_atom_energy_mae_eV']:.4f} | {b['force_component_mae_eV_A']:.4f} |"
        )
    out += [
        "",
        "The charge model conserves molecular charge through antisymmetric local transfers. The auxiliary radial energy fit uses source energies in eV per atom and two seeded force projections per training geometry. The force MAE uses every held-out Cartesian component. The auxiliary energy reference is dispersion-free ωB97M/def2-TZVPP. Its atom-energy offsets stay out of the reactive readouts. [Energy and force fit](../experiments/full_public/trial_parent_potential.py) · [Transfer to Korny](../experiments/full_public/trial_parent_transfer.py)",
        "",
        "## Archive status",
        "",
        "| Archive | Completed byte ranges (GB) | Total (GB) | Whole-file check |",
        "|---|---:|---:|---|",
    ]
    for alias in ("aimnet", "rdb19", "ions16"):
        path, _, num, _digest = source(alias)
        j = path.parent / "download-progress.json"
        if path.exists():
            count = num
            msg = "Deposited MD5 matched"
        elif j.exists():
            s = read(j)
            chunk = 8 * 1024**2 if alias == "ions16" else CHUNK
            count = s["prefix"] + sum(min(chunk, num - i) for i in s["complete"])
            msg = "Transfer in progress"
        else:
            count = 0
            msg = "Awaiting transfer"
        out.append(f"| {path.name} | {count / 1e9:.2f} | {num / 1e9:.2f} | {msg} |")
    out += [
        "",
        "The five AutoMeKin archives total 236,400,640 bytes and each matches its deposited MD5. Korny jobs share an 8 GB hard memory limit, a 6 GB reclaim threshold and zero swap allocation. [Original source inventory](aimnet2-nse-and-automekin.md) · [Full benchmark history](barrier-and-frequency-validation.md)",
    ]
    scale_file = RDB / "multiscale-path56/metrics.json"
    if scale_file.exists():
        d = read(scale_file)
        out += [
            "", "## Learning radical path shapes at two resolutions", "",
            f"The expanded RDB19-Rad snapshot adds {d['new_train_reactions']:,} training reactions. Each contributes seven sampled positions across the two IRC branches, including their common transition state. The fit uses the endpoint gap, midpoint height, two interior energies and two projected force labels. The signed errors use the archived educt/product geometries to orient every reaction. Kernel choice and ridge use three internal training-formula folds. The native evaluation uses the same {d['native_held_reactions']} direction-assigned held-out reactions at both training sizes.",
            "", "| Training reactions | Local kernel | Reaction MAE | Midpoint | All barriers | Forward | Reverse |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
        for r in d["fits"]:
            out.append(f"| {r['paths']:,} | {r['kernel']} | {r['reaction']:.2f} | {r['midpoint']:.2f} | {r['barriers']:.2f} | {r['forward']:.2f} | {r['reverse']:.2f} |")
        out += [
            "", "The native electronic errors are in kcal/mol. P56 transfers the shared and family-specific native energy readouts to the Grambow branch. The 589-reaction table above measures that transfer against Korny’s target references. [Path fitting code](../experiments/full_public/trial_path_scales.py)",
        ]
    full = RDB / "multiscale-path59"
    if (full / "later-holdout.json").exists() and (full / "metrics.json").exists():
        d = read(full / "later-holdout.json")
        n = read(full / "metrics.json")["train_reactions"]
        out += ["", f"The later-arriving validation panel contains {d['n']} doublet reaction IDs across {d['formulas']} held-out formulas. Those reaction IDs arrived after p56 was frozen. P59 used {n:,} training paths. The two rows score shared native RDB19-Rad energy readouts on identical IDs.", "",
                "| Native readout | Reaction | Midpoint | All barriers | Forward | Reverse |", "|---|---:|---:|---:|---:|---:|"]
        for key,label in (("snapshot","Frozen p56 native kernel"),("full","Full-release p59 native kernel")):
            row = d[key]
            out.append(f"| {label} | {row['reaction']:.2f} | {row['midpoint']:.2f} | {row['barriers']:.2f} | {row['forward']:.2f} | {row['reverse']:.2f} |")
    gate = RDB / "local-family60/native-evaluation.json"
    if gate.is_file():
        d = read(gate)
        out += ["", "## Inferring radical reaction families from one geometry", "",
                f"The same {d['held']} held-out RDB19-Rad reactions test the shared surface, a global linear mixture, an atom-local mixture inferred from geometry, and a diagnostic supplied with the archived family at prediction. The first three can evaluate individual geometries during MD. The native training partition has {d['training']:,} paths.", "",
                "| RDB19-Rad model | Reaction | Midpoint | All barriers | Forward | Reverse |", "|---|---:|---:|---:|---:|---:|"]
        for name,row in d["scores"].items():
            out.append(f"| {name} | {row['reaction']:.2f} | {row['midpoint']:.2f} | {row['barriers']:.2f} | {row['forward']:.2f} | {row['reverse']:.2f} |")
    native = []
    for stage in (59,61):
        root = RDB / f"multiscale-path{stage}"
        if all((root / name).is_file() for name in ("metrics.json","all-held-validation.json","later-holdout.json")):
            native.append((stage,read(root/"metrics.json"),read(root/"all-held-validation.json")[0],read(root/"later-holdout.json")))
    if native:
        out += ["", "## Full radical archive and kernel size", "",
                "The center-count comparison uses identical training reactions, held formulas and electronic targets. The shared kernel takes one geometry at a time. The supplied-family diagnostic additionally receives the archived mechanism.", "",
                "| Fit | Centers per resolution | Trained paths | Validation | Reactions | Reaction MAE | All barriers | Forward | Reverse |", "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
        for stage,m,a,d in native:
            for name,n,scores in (("All held, geometry only",a["n"],a["shared"]),("Later IDs, geometry only",d["n"],d["full"]),("All held, supplied family",a["n"],a["family"])):
                out.append(f"| p{stage} native | {m['columns_per_scale']} | {m['train_reactions']:,} | {name} | {n} | {scores['reaction']:.2f} | {scores['barriers']:.2f} | {scores['forward']:.2f} | {scores['reverse']:.2f} |")
    emitted=False
    for stage in (59,61,63,64,66,67):
        root=RDB/f"multiscale-path{stage}"
        if not all((root/k).is_file() for k in ("all-held-validation.json","native-forces.json")):
            continue
        if not emitted:
            out += ["","## Native Cartesian forces and reaction energies","",
                    "Every model in this table receives only a single neutral-doublet geometry. The energy test uses 626 held reactions. The force test uses all Cartesian components at the 626 transition states and 1,252 branch midpoints. Energy MAEs are in kcal/mol and force-component MAEs are in eV/Å. The standalone native checkpoints include the source dispersion and have an arbitrary energy zero. [Native checkpoint loader and force audit](../experiments/full_public/native_irc_pes.py)", "",
                    "| Native model | Reaction | All barriers | Forward | Reverse | Saddle force MAE | Midpath force MAE |", "|---|---:|---:|---:|---:|---:|---:|"]
            emitted=True
        f=read(root/"native-forces.json")
        m=read(root/"all-held-validation.json")[0]["shared"]
        out.append(f"| p{stage} | {m['reaction']:.2f} | {m['barriers']:.2f} | {m['forward']:.2f} | {m['reverse']:.2f} | {f['saddle']['mae_eV_A']:.3f} | {f['flanks']['mae_eV_A']:.3f} |")
    if emitted:
        out += ["",f"The mean absolute reference force components are {f['saddle']['zero_force_mae_eV_A']:.4f} eV/Å at the saddles and {f['flanks']['zero_force_mae_eV_A']:.4f} eV/Å at the branch midpoints. These full-vector checks probe directions outside the IRC tangent. A low reaction MAE alone cannot establish stationarity or MD accuracy."]
    constrained = RDB / "multiscale-path64/settings.json"
    if constrained.is_file():
        d = read(constrained)
        w = d["selected"]
        out += ["", f"P64 chooses the smallest training-fold directional force RMSE subject to a {d['training_energy_ceiling_kcal_mol']:.1f} kcal/mol MAE ceiling on each of the endpoint gap, forward barrier, and reverse barrier. Three internal folds group the {d['training']:,} training reactions by formula. Each path contributes {d['directions_per_path']} Cartesian force projections. The chosen weight was {w['weight']:g}. Training-fold MAEs were {w['inner_reaction']:.2f}, {w['inner_forward']:.2f}, and {w['inner_reverse']:.2f} kcal/mol in the same order. The final table scores the separate 626 held-out reactions."]
    secants = RDB / "multiscale-path64/held-path-curvature.json"
    if secants.exists():
        check = read(secants)
        eligible = [r for r in check["rows"] if "signed_cm1" in r]
        out += ["", "## Negative curvature at radical transition states", "",
                f"On {len(eligible)} of {len(check['rows'])} held paths, the first IRC steps point in opposing directions after mass-weighted rigid alignment. For each model and reference, the difference between their two Cartesian forces gives a directional curvature over the same chord. Conversion to a signed wavenumber gives the scale of the local error in cm⁻¹. This probes one finite direction at the archived saddle. The four molecular spectra above use full Hessians at relaxed minima. [Curvature audit](../experiments/full_public/audit_irc_curvature.py)",
                "", "| RDB19-Rad model | Signed directional MAE (cm⁻¹) | Correct among reference downhill directions |",
                "|---|---:|---:|"]
        summaries = dict(check["summary"])
        for stage in (66,67):
            new_path = RDB / f"multiscale-path{stage}/held-path-curvature.json"
            if not new_path.is_file():
                continue
            other = read(new_path)
            other_eligible = [r for r in other["rows"] if "signed_cm1" in r]
            np.testing.assert_array_equal([r["id"] for r in eligible], [r["id"] for r in other_eligible])
            np.testing.assert_allclose([r["signed_cm1"]["reference"] for r in eligible],
                                       [r["signed_cm1"]["reference"] for r in other_eligible],
                                       atol=1e-7, rtol=0)
            summaries.update(other["summary"])
        for k, m in summaries.items():
            out.append(f"| p{k} | {m['signed_mae_cm1']:.1f} | {m['correct_downhill']}/{m['reference_downhill']} |")
        for stage in (66,67):
            settings = RDB / f"multiscale-path{stage}/settings.json"
            if not settings.is_file():
                continue
            d = read(settings)
            w = d["selected"]
            if stage == 66:
                out += ["", f"P66 and p67 add one observed saddle-curvature projection for each of {d['training_curvatures']:,} eligible training reactions. On three internal formula folds, selection minimizes signed directional wavenumber MAE subject to a 5 kcal/mol ceiling on each energy observable and a 0.60 eV/Å projected force RMSE ceiling. The 626 validation paths are excluded from coefficient and weight fitting.", "",
                        "| Model | Force weight | Curvature weight | Training-fold signed MAE (cm⁻¹) |",
                        "|---|---:|---:|---:|"]
            out.append(f"| p{stage} | {w['weight']:g} | {w['curvature_weight']:g} | {w['inner_curvature_mae_cm1']:.1f} |")
    cross = [(k,RDB/f"multiscale-path{k}/direct-grambow.json") for k in (59,61,63,64,66,67)]
    if any(path.exists() for _,path in cross):
        out += ["", "## Direct transfer across radical archives", "",
                "Grambow evaluates a different electronic reference and reaction distribution. Each native energy readout below is frozen before evaluating Grambow. All energies use one geometry at a time and both directed barriers use the archived transition state.", "",
                "| Grambow model | Reactions | Reaction | All barriers | Forward | Reverse |", "|---|---:|---:|---:|---:|---:|"]
        for k,path in cross:
            if path.exists():
                data=read(path)
                m=data["scores"]
                out.append(f"| Uncalibrated RDB19-Rad native p{k} | {data['eligible']} | {m['reaction']:.2f} | {m['barriers']:.2f} | {m['forward']:.2f} | {m['reverse']:.2f} |")
        for name,m in stages[49]["Grambow"].items():
            label="Korny p49" if name=="Korny" else name
            out.append(f"| {label} | {m['n']} | {m['reaction']:.2f} | {m['barriers']:.2f} | {m['forward']:.2f} | {m['reverse']:.2f} |")
    temp = DEST.with_suffix(".pending")
    temp.write_text("\n".join(out) + "\n")
    temp.replace(DEST)


if __name__ == "__main__":
    run()
