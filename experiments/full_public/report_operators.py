"""Publish the operator-basis experiments from measured artifacts on the fixed Korny IDs."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.full_public import kernel_trials as kt
from experiments.full_public.paired_baseline_audit import paired_intervals
from experiments.full_public.reaction_family_weights import families
from experiments.full_public.trial_operator_kernels import ROOT, full_candidates

DEST = Path(__file__).resolve().parents[2] / "report/operator-kernel-benchmarks.md"
NAMES = dict(
    hermite="Hermite oscillator",
    laguerre="Laguerre",
    legendre="Legendre",
    chebyshev="Chebyshev",
    jacobi="Jacobi",
    dirichlet_sine="Dirichlet sine",
    neumann_cosine="Neumann cosine",
    bessel_disk="Bessel disk",
    krawtchouk="Krawtchouk",
    hahn="Hahn",
    graph_diffusion="Graph diffusion",
    matern52="Matérn 5/2",
    gaussian="Gaussian",
    rational_quadratic="Rational quadratic",
    exponential="Exponential",
)
KEYS = ("reaction", "forward", "reverse", "barriers")
PAN = kt.OUT / "so3lr-tauri-matched"


def read(path):
    return json.loads(path.read_text())


def row(name, d):
    return f"| {name} | " + " | ".join(f"{d[k]:.3f}" for k in KEYS) + " |"


def run():
    screen = read(ROOT / "summary.json")
    candidates = full_candidates()
    labels = {
        68: "Parent",
        69: "Gaussian",
        70: "Gaussian plus singlet DFT supervision",
        71: "Gaussian plus reaction-family balancing",
        72: "Split-charge response",
    }
    labels.update({stage: NAMES[c["family"]] for stage, c in candidates.items()})
    labels[84] = "Morse stretches and Legendre bends"
    labels[85] = "Legendre with pairwise coordinate products"
    summaries = {
        s: read(PAN / f"p{s}-public-summary.json")["common"]
        for s in labels
        if (PAN / f"p{s}-public-summary.json").exists()
    }
    mixed = summaries[68]["priority"]
    out = [
        "# Eigenfunction kernels for reaction energies and barriers",
        "",
        "Korny was fitted on 11,667 RDB7, 400 BH9, and 1,084 Grambow training reactions. The fixed common panel contains 259 RDB7, 47 BH9, and 283 Grambow reactions, giving 589 endpoint gaps and 1,178 directed barriers. High-level references score RDB7 and BH9. The Grambow references are ωB97X-D3. All deployable Korny rows include D3(BJ) and ATM once.",
        "",
        "## Formula-fold comparison",
        "",
        "For each of 11 operator families, two coordinate scales and four ridge strengths were compared. There are 80 features per central element and 320 molecular features. Each score averages six MAEs equally across the three sources and two observables (reaction energy and pooled barrier). Two formula folds contain 1,909 RDB7, 102 BH9, and 445 Grambow reactions. The earlier p12 baseline has separately fitted predictions for each of these folds. The full-data corrections use the p68 parent. The scores of these two experiments have different baselines and populations.",
        "",
        "| Family | Internal score (kcal/mol) | Scale multiplier | Ridge |",
        "|---|---:|---:|---:|",
        f"| Parent alone | {screen['control_macro']:.3f} | — | — |",
    ]
    ranked = list(screen["best"])
    ctl = ROOT / "radial-controls.json"
    if ctl.exists():
        controls = read(ctl)
        ranked += [
            min(
                (x for x in controls if x["family"] == fam),
                key=lambda x: x["macro_mae"],
            )
            for fam in dict.fromkeys(x["family"] for x in controls)
        ]
    for v in sorted(ranked, key=lambda x: x["macro_mae"]):
        out.append(
            f"| {NAMES[v['family']]} | {v['macro_mae']:.3f} | {v['width']:.2f} | {v['ridge']:g} |"
        )
    out += [
        "",
        "The ordinary radial profiles use a center-Gram ridge, equivalent to a regularized Nyström basis. Their scale multiplier sets a radial half-height distance. For the one-dimensional operator modes, scale controls the spread of the projection coordinates. The graph scale controls affinity bandwidth. All use training landmarks; the profile multipliers and operator multipliers parameterize different distances.",
        "",
        "## Fixed mixed panel",
        "",
        "Entries are MAEs in kcal/mol. Chebyshev and graph diffusion were chosen on the formula folds before scoring this panel. The other families are exploratory comparisons. The operator fits start from p68. They share the same 91,156 bare-PBE0 anchor geometries, directional force and curvature constraints, NIST training minima, and Grambow saddle constraints.",
        "",
        "| Method | Reaction | Forward barrier | Reverse barrier | All barriers |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage, d in summaries.items():
        out.append(row(f"Korny p{stage} · {labels[stage]}", d["priority"]["Korny"]))
    for name, d in mixed.items():
        if name != "Korny":
            out.append(row(name, d))
    out += [
        "",
        "## Dataset breakdown",
        "",
        "| Dataset | Method | Reaction | Forward barrier | Reverse barrier | All barriers |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for source in kt.BARRIERS:
        for stage in summaries:
            out.append(
                row(
                    f"{source} | Korny p{stage} · {labels[stage]}",
                    summaries[stage][source]["Korny"],
                )
            )
        for name, d in summaries[68][source].items():
            if name != "Korny":
                out.append(row(source + " | " + name, d))
    out += [
        "",
        "Paired changes versus p68 use identical IDs. The intervals resample whole transition-state formulas within each source (4,000 resamples). Repeated inspection makes the common panel a development benchmark.",
        "",
        "| Preselected model | Observable | MAE change | 95% interval, kcal/mol |",
        "|---|---|---:|---:|",
    ]
    for stage in (73, 74):
        if stage not in summaries:
            continue
        interval = paired_intervals(f"p{stage}-public", parent="p68-public")
        for k in KEYS:
            change = summaries[stage]["priority"]["Korny"][k] - mixed["Korny"][k]
            lo, hi = interval[k]
            out.append(f"| p{stage} | {k} | {change:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    out += [
        "",
        "## Geometry and frequencies",
        "",
        "The four full frequency MAEs are in cm⁻¹ against PBE0/cc-pVTZ at independently relaxed minima. The signed saddle error compares curvature along the reference unstable direction on 278 Grambow structures.",
        "",
        "| Model | Propane | Benzene | Ethanol | 1,3-butadiene | Equal-molecule mean | Signed saddle MAE | Downhill sign |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in summaries:
        sp = kt.WORK / f"p{stage}-public-spectra.json"
        diag = kt.WORK / f"p{stage}-public-diagnostics.json"
        if not diag.exists():
            continue
        results = read(sp) if sp.exists() else {}
        vals = [
            results.get(n, {}).get("mae_cm1")
            for n in ("propane", "benzene", "ethanol", "1,3-butadiene")
        ]
        h = read(diag)["radical"]["saddle"]
        out.append(
            f"| p{stage} | "
            + " | ".join(
                f"{v:.1f}" if v is not None else "—"
                for v in vals + [float(np.mean(vals)) if None not in vals else None]
            )
            + f" | {h['mae_cm1'][0]:.1f} | {round(h['sign_agreement'][0]*h['n'])}/{h['n']} |"
        )
    out += [
        "",
        "| Model | Largest bond-length MAE (Å) | Largest angle MAE (degrees) | Connectivity retained | Minima with no imaginary internal modes |",
        "|---|---:|---:|---:|---:|",
    ]
    for stage in summaries:
        sp = kt.WORK / f"p{stage}-public-spectra.json"
        if sp.exists():
            geoms = list(read(sp).values())
            valid = [v for v in geoms if "bond_mae_A" in v]
            if valid:
                bonds = max(v["bond_mae_A"] for v in valid)
                angles = max(v["angle_mae_deg"] for v in valid)
                connected = sum(v["same_connectivity"] for v in valid)
                stable = sum(min(v["predicted_cm1"]) >= 0 for v in valid)
                out.append(
                    f"| p{stage} | {bonds:.4f} | {angles:.2f} | {connected}/{len(geoms)} | {stable}/{len(geoms)} |"
                )
    out += [
        "",
        "## What the basis functions represent",
        "",
        r"A finite kernel has the form $k(x,y)=\sum_j w_j\phi_j(x)\phi_j(y)$ with $w_j\geq0$. Korny evaluates molecular features by summing atom-centered mode amplitudes after subtracting the isolated-atom value. Permutation and rigid-motion invariance come from the atomic descriptors. The energy coefficients are real and the analytic Cartesian forces are derivatives of the same energy.",
        "",
        "The continuous families use five modes on each of 16 fixed projections of the training environment descriptors. Hermite oscillator functions include a Gaussian envelope. Laguerre functions include an exponential envelope on the positive half-line. Legendre, Chebyshev, and Jacobi modes use a bounded coordinate. The interval Laplacian gives sine or cosine modes with Dirichlet or Neumann boundary conditions. The radial disk Laplacian gives Bessel J₀ modes with zeros at the boundary. The operator eigenfunctions are known on their native domains; projection, weighting, and summing over atoms induce the kernel on molecular geometry. [NIST orthogonal polynomials](https://dlmf.nist.gov/18.3) · [NIST Bessel zeros](https://dlmf.nist.gov/10.21) · [Solin and Särkkä](https://arxiv.org/abs/1401.5508)",
        "",
        "Krawtchouk and Hahn modes come from two finite birth–death chains on 13 sites. Binomial averaging extends each discrete eigenfunction smoothly in the geometry descriptor. Graph diffusion instead uses the nonconstant eigenvectors of a reversible random walk on 256 training environments per element, with smooth affinity-weighted extension to new geometries. These features require no reaction label at prediction time. [NIST discrete orthogonal polynomials](https://dlmf.nist.gov/18.19) · [Coifman and Lafon, diffusion maps](https://doi.org/10.1016/j.acha.2006.04.006)",
        "",
        "P84 changes the coordinate as well as the operator. Four Morse oscillator bound-state functions act on each bond length scaled by the sum of its covalent radii. A product of two radial ground-state functions multiplies the first three Legendre modes of the included angle. Element-group channels distinguish participating atoms. A smooth cutoff switches off each contribution at dissociation. The resulting 342 explicit features take part in the same fit and independent geometry checks. [Morse’s diatomic oscillator](https://doi.org/10.1103/PhysRev.34.57)",
        "",
        "P70 adds 11,667 RDB7 singlet DFT training gaps and midpoint heights to the DFT branch at weight 0.25. P71 partially equalizes family frequencies within each training source. P72 uses the geometry-dependent energy response of variational atom-to-atom charge flow, with smooth bond conductance and softened electrostatics. Its induced flows sum to zero on each disconnected fragment. The baseline carries the total charge and spin. [Split-charge equilibration](https://doi.org/10.1063/1.2346671)",
        "",
        "## Where the p68 parent incurs reaction-energy error",
        "",
        "The three families contributing the largest total absolute endpoint error in each source are listed below. Unassigned reactions use atom-mapped changes in connectivity grouped by element pair and fragment count. These coarse groups can contain several mechanisms.",
        "",
        "| Dataset | Family or edits | Training | Held | Reaction | Forward | Reverse |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels_by_id = families()
    counts = defaultdict(int)
    for t in kt.TARGETS:
        _, _, sources, fold, kind, ids = kt.equation_data(t)
        train = (fold >= 0) & (kind == "reaction")
        for source, k in zip(sources[train], ids[train]):
            if (source, k) in labels_by_id:
                counts[source, labels_by_id[source, k]] += 1
    data = defaultdict(lambda: defaultdict(list))
    with (PAN / "p68-public-active-common.csv").open() as stream:
        for r in csv.DictReader(stream):
            s, k = r["source"], r["id"]
            data[s, labels_by_id[s, k]][r["observable"]].append(
                abs(float(r["Korny error_kcal"]))
            )
    for s in kt.BARRIERS:
        top = sorted(
            ((f, d) for (src, f), d in data.items() if src == s),
            key=lambda x: -sum(x[1]["reaction"]),
        )[:3]
        for f, d in top:
            out.append(
                f"| {s} | {f} | {counts[s,f]} | {len(d['reaction'])} | "
                + " | ".join(f"{np.mean(d[k]):.2f}" for k in KEYS[:3])
                + " |"
            )
    if all(k in summaries for k in range(73, 85)):
        tested = {k: s["priority"]["Korny"] for k, s in summaries.items() if k >= 73}
        endpoint = min(tested, key=lambda k: tested[k]["reaction"])
        pooled = min(tested, key=lambda k: tested[k]["barriers"])
        n_forward = sum(
            v["forward"] < mixed["Korny"]["forward"] for v in tested.values()
        )
        n_reverse = sum(
            v["reverse"] < mixed["Korny"]["reverse"] for v in tested.values()
        )
        out[2:2] = [
            f"On the same 589 reactions, {labels[endpoint]} had the lowest reaction MAE ({tested[endpoint]['reaction']:.3f} kcal/mol). {labels[pooled]} had the lowest pooled barrier MAE ({tested[pooled]['barriers']:.3f} kcal/mol). The parent scored {mixed['Korny']['reaction']:.3f} and {mixed['Korny']['barriers']:.3f}. These minima summarize a repeatedly inspected panel.",
            "",
            f"Of the {len(tested)} eigenfunction fits, {n_forward} improved forward barriers and {n_reverse} improved reverse barriers relative to the parent. The 5 kcal/mol mixed-panel goal remains open. The graph operator ranked first on the development folds, while its mixed-panel reaction MAE rose to {tested[74]['reaction']:.3f}.",
            "",
        ]
    if 85 in summaries:
        a = summaries[77]["priority"]["Korny"]
        b = summaries[85]["priority"]["Korny"]
        out += [
            "",
            "## Does a larger function space improve energies?",
            "",
            "At fixed input scaling the five Legendre, Chebyshev, Jacobi, Hahn and Krawtchouk modes used here all span the same centered polynomial space through degree five on each projected axis. The binomial lift of a degree-n discrete polynomial is a polynomial of degree at most n in its success probability. Projection of their centered columns onto the Legendre space gave relative residuals below 7 × 10⁻¹⁶ on 201 points. Their coefficient penalties, mode normalization and selected length scales can still give different predictions.",
            "",
            "P85 retains all 320 p77 Legendre features and adds the 120 distinct products between the first-degree modes on its 16 coordinates for each of four central elements. It has 800 features. The coordinate scales, ridge, parent, training equations and force/frequency constraints are identical to p77. Both fits were scored on the same reaction IDs.",
            "",
            "| Model | Reaction | Forward barrier | Reverse barrier | All barriers |",
            "|---|---:|---:|---:|---:|",
            row("P77 additive", a),
            row("P85 coupled", b),
            "",
            f"Adding 480 cross-coordinate products changed the reaction MAE by {b['reaction'] - a['reaction']:+.3f} kcal/mol and the pooled barrier MAE by {b['barriers'] - a['barriers']:+.3f} kcal/mol at the same regularization strength.",
        ]
    temp = DEST.with_suffix(".pending")
    temp.write_text("\n".join(out) + "\n")
    temp.replace(DEST)


if __name__ == "__main__":
    run()
