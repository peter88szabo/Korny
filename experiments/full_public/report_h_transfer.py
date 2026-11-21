"""Audit single-H transfers on exactly the shared Korny and SO3LR reaction IDs.

Mapped endpoints identify the H, donor and acceptor in RDB7 and Grambow.
Assign a measured distance only when the geometry's atom ordering also passes
the mapped element and endpoint hydrogen-neighbour checks. BH9 supplies a
published HAT family and contributes to energy statistics independently.
"""

import csv
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiments.full_public import kernel_trials as kt
from experiments.full_public.reaction_family_weights import families, mapped_graph
from experiments.full_public.data import DATA
from rdkit import Chem

PAN = kt.OUT / "so3lr-tauri-matched"
DEST = Path(__file__).resolve().parents[2] / "report/hydrogen-transfer-kernels.md"
KEYS = ("reaction", "forward", "reverse")
COMPARATORS = ("Original SO3LR", "Tauri-M-2606", "Tauri-L-2606")
MODELS = {
    73: "Chebyshev",
    82: "Krawtchouk",
    86: "Chebyshev + H bonds",
    87: "Krawtchouk + H bonds",
    88: "Chebyshev + angular H bonds",
    89: "Krawtchouk + angular H bonds",
}


@lru_cache(maxsize=1)
def transfers():
    """Identify pure connectivity exchanges, allowing bond-order changes."""
    found = {
        key: None
        for key, label in families().items()
        if key[0] == "BH9" and label == "BH9/Hydrogen atom transfer"
    }
    for src, path in (
        ("RDB7", "RDB7/ccsdtf12_dz.csv"),
        ("Grambow", "Grambow_radicals/wb97xd3_rad.csv"),
    ):
        with (DATA / path).open() as inp:
            for row in csv.DictReader(inp):
                lhs, rhs = (mapped_graph(row[k]) for k in ("rsmi", "psmi"))
                if lhs is None or rhs is None or lhs[0] != rhs[0]:
                    continue
                atoms, before, _ = lhs
                lost, gained = before - rhs[1], rhs[1] - before
                if len(lost) != 1 or len(gained) != 1:
                    continue
                old, new = map(set, (next(iter(lost)), next(iter(gained))))
                shared = old & new
                if len(shared) != 1:
                    continue
                (h,) = shared
                (donor,) = old - shared
                (acceptor,) = new - shared
                if atoms[h] != "H" or "H" in (atoms[donor], atoms[acceptor]):
                    continue
                case = ("RDB7/rxn" if src == "RDB7" else "") + row["idx"].zfill(6)
                found[src, case] = h, donor, acceptor, lhs, rhs
    return found


def panel(stage):
    """Use only rows with finite errors from every comparator, preserving IDs."""
    values = defaultdict(dict)
    with (PAN / f"p{stage}-public-comparison.csv").open() as inp:
        for r in csv.DictReader(inp):
            cols = [r[m + " error_kcal"] for m in ("Korny", *COMPARATORS)]
            if all(cols) and np.isfinite(np.asarray(cols, float)).all():
                values[r["source"], r["id"]][r["observable"]] = (
                    float(r["reference_kcal"]),
                    *map(float, cols),
                )
    assert len(values) == 589 and all(set(v) == set(KEYS) for v in values.values())
    return values


def metrics(cases, rows, col=1):
    err = np.array([[rows[k][obs][col] for obs in KEYS] for k in cases])
    rx, fw, rv = np.abs(err).mean(0)
    return (rx, fw, rv, (fw + rv) / 2)


def aligned(geom, graph):
    """Verify the map-number ordering using every singly heavy-bonded H."""
    atoms, bonds, _ = graph
    z, r = geom.z, geom.r
    symbols = Chem.GetPeriodicTable()
    if sorted(atoms) != list(range(1, len(z) + 1)) or any(
        atoms[i] != symbols.GetElementSymbol(int(z[i - 1])) for i in atoms
    ):
        return False
    heavy = np.flatnonzero(z != 1)
    for h in np.flatnonzero(z == 1):
        partners = [b if a == h + 1 else a for a, b in bonds if h + 1 in (a, b)]
        if len(partners) != 1 or atoms[partners[0]] == "H":
            continue
        if heavy[np.argmin(np.linalg.norm(r[heavy] - r[h], axis=1))] != partners[0] - 1:
            return False
    return True


def structures(cases):
    """Measure covalent stretches at R, TS and P in verified atom-map order."""
    gs, eq, _ = kt.inputs()
    names = defaultdict(dict)
    for _, src, split, case, terms, _, kind in eq:
        if split == "validation" and (src, case) in cases:
            names[src, case][kind] = terms
    data = []
    for key in sorted(cases):
        assignment = transfers()[key]
        if assignment is None:
            continue
        h, donor, acceptor, left, right = assignment
        rx = names[key]["reaction"]
        a = [gs[n] for n, c in rx if c == -1]
        b = [gs[n] for n, c in rx if c == 1]
        ts = gs[names[key]["forward"][-1][0]]
        if (
            len(a) != 1
            or len(b) != 1
            or not aligned(a[0], left)
            or not aligned(b[0], right)
        ):
            continue
        if not np.array_equal(ts.z, a[0].z):
            continue
        i, j, k = donor - 1, h - 1, acceptor - 1
        # Verify that spectator H assignments also agree at the saddle. This
        # cross-check allows the transferring H to lie nearer either partner.
        heavy = np.flatnonzero(ts.z != 1)
        spectator_ok = all(
            (
                heavy[np.argmin(np.linalg.norm(ts.r[heavy] - ts.r[u], axis=1))]
                == heavy[np.argmin(np.linalg.norm(a[0].r[heavy] - a[0].r[u], axis=1))]
            )
            for u in np.flatnonzero(ts.z == 1)
            if u != j
        )
        if not spectator_ok:
            continue
        lengths = [
            np.linalg.norm(g.r[[i, k]] - g.r[j], axis=1) for g in (a[0], ts, b[0])
        ]
        v = ts.r[[i, k]] - ts.r[j]
        cosine = np.dot(v[0], v[1]) / np.prod(lengths[1])
        data.append(
            (
                key,
                left[0][donor] + "→" + left[0][acceptor],
                *np.ravel(lengths),
                np.degrees(np.arccos(np.clip(cosine, -1, 1))),
            )
        )
    return data


def radical_structures():
    """Follow each identifiable transferred H through both RDB19-Rad IRC branches.

    The release supplies atom-ordered educt/product complexes and the TS. A
    heavy-to-heavy transfer must change the nearest heavy partner for exactly
    one H actually in covalent contact at both endpoints. Distances along the
    first accepted IRC step are invariant to a whole-molecule reorientation.
    """
    from experiments.full_public.h_transfer_sources import RADICAL, IRC
    from experiments.full_public.trial_h_transfer import ROOT, bases
    import json

    rec = json.loads((RADICAL / "snapshot.json").read_text())
    radii = next(iter(bases().values())).radii
    result = []
    skip = Counter()
    for item in rec:
        if item["family"] not in ("H Atom Abstraction", "intra_H_migration"):
            continue
        rx = item["rxn"]
        split = "validation" if item["heldout"] else "train"
        source = IRC / "source-endpoints"
        labels = []
        for name in ("IRC_educt", "ts", "IRC_product"):
            with np.load(source / f"{rx}-{name}.npz") as f:
                labels.append((f["z"].astype(int), f["r_A"].astype(float)))
        z, ts = labels[1]
        if any(not np.array_equal(z, zz) for zz, r in labels):
            skip[split, "atom order"] += 1
            continue
        hs, heavy = np.flatnonzero(z == 1), np.flatnonzero(z != 1)
        if len(heavy) < 2 or not len(hs):
            skip[split, "insufficient heavy atoms"] += 1
            continue
        rr = labels[0][1]
        rp = labels[2][1]
        near = []
        for r in (rr, rp):
            d = np.linalg.norm(r[hs, None] - r[heavy], axis=2)
            js = d.argmin(1)
            a = heavy[js]
            near.append(
                (a, d[np.arange(len(hs)), js] < 1.30 * (radii[1] + radii[z[a]]))
            )
        changed = np.flatnonzero((near[0][0] != near[1][0]) & near[0][1] & near[1][1])
        if len(changed) != 1:
            skip[split, "ambiguous or non-heavy transfer"] += 1
            continue
        h = hs[changed[0]]
        donor, acceptor = (v[0][changed[0]] for v in near)

        def lengths(r):
            return np.linalg.norm(r[[donor, acceptor]] - r[h], axis=1)

        lts = lengths(ts)
        v = ts[[donor, acceptor]] - ts[h]
        angle = np.degrees(np.arccos(np.clip(v[0] @ v[1] / lts.prod(), -1, 1)))
        steps = []
        for side in ("forw", "rev"):
            with np.load(IRC / f"{rx}-{side}.npz") as f:
                r = f["r_A"][:2].astype(float)
            q = [np.dot(lengths(x), [1.0, -1.0]) for x in r]
            steps.append(q[1] - q[0])
        result.append(
            dict(
                reaction=rx,
                split=split,
                family=item["family"],
                elements=[int(z[donor]), int(z[acceptor])],
                lengths_A=np.r_[lengths(rr), lts, lengths(rp)].tolist(),
                angle_deg=float(angle),
                opposite_initial_steps=bool(steps[0] * steps[1] < 0),
            )
        )
    kt.write_json(
        ROOT / "radical-h-structures.json",
        {
            "observations": result,
            "excluded": [[s, why, n] for (s, why), n in skip.items()],
        },
    )
    for split in ("train", "validation"):
        rows = [r for r in result if r["split"] == split]
        print(
            split,
            len(rows),
            "verified H transfers; median TS angle",
            np.median([r["angle_deg"] for r in rows]),
            "first IRC steps straddle TS",
            sum(r["opposite_initial_steps"] for r in rows),
            flush=True,
        )


def run():
    import json
    from experiments.full_public.trial_h_transfer import ROOT, ANGULAR
    from experiments.full_public.h_transfer_sources import RADICAL

    def read(p):
        return json.loads(p.read_text())

    def mrow(label, values):
        return "| " + label + " | " + " | ".join(f"{v:.3f}" for v in values) + " |"

    rows = {s: panel(s) for s in MODELS}
    for d in rows.values():
        assert d.keys() == rows[73].keys()
        assert all(d[k][o][0] == rows[73][k][o][0] for k in d for o in KEYS)
    keys = set(rows[73])
    hkeys = keys & transfers().keys()
    names = {s: f"p{s} {n}" for s, n in MODELS.items()}
    held = {f: read(ROOT / f"{f}-validation.json") for f in ("chebyshev", "krawtchouk")}
    angular = {f: read(ANGULAR / f"{f}-validation.json") for f in held}
    out = [
        "# H-transfer kernels on the expanded reaction data",
        "",
        "The p86 Chebyshev and p87 Krawtchouk fits extend p73 and p82 respectively. The p88 and p89 controls refit the same parents on the same training examples with an additional angular window. The updated general checkpoints include D3(BJ) and ATM once. The dedicated radical checkpoints learn the archive's dispersion-inclusive electronic energy. Energy MAEs are in kcal/mol.",
        "",
        "## Training coverage",
        "",
        "| Source | Training | Held out | Quantity entering the fit |",
        "|---|---:|---:|---|",
        "| RDB7 / BH9 / Grambow | 11,667 / 400 / 1,084 | 259 / 47 / 283 | Endpoint gaps and midpoint heights |",
        "| H transfers within RDB7 / BH9 / Grambow | 2,052 / 77 / 366 | 53 / 13 / 83 | Mapped exchanges or the published BH9 HAT family, included above |",
        "| RDB19-Rad | 4,974 paths | 626 paths | Electronic gaps and midpoint heights on both endpoint complexes and their common saddle |",
        "| RDB19-Rad H-transfer subset | 1,254 | 172 | H atom abstractions and intramolecular H migrations, included above |",
        "| AutoMeKin connected DFT channels | 538 | 329 | Electronic gaps and midpoint heights |",
        "| AutoMeKin connected CC channels | 51 canonical + 446 local | 265 local | Electronic gaps and midpoint heights |",
        "| AutoMeKin fragmented CC channels | 40 canonical + 22 local | 8 local | Endpoint fragment energies summed by channel |",
        "| AIMNet2-NSE supplement | 11,507 singlet/doublet geometries | 369 singlet geometries | Relative energy within each scan or IRC series and native Hamiltonian |",
        "| BASChem19 | 15 reactions | 4 reactions | Forward barriers |",
        "| Reoptimized RDB7 (PM6 / HF / r²SCAN-3c) | 9,584 / 10,438 / 11,310 reaction IDs | 207 / 231 / 252 | Three training-only energy directions enter the coefficient prior |",
        "",
        "The reoptimized archives overlap RDB7 reaction IDs. AutoMeKin has multiple methods on many of the same channels. The RDB19 counts correspond to 16,800 stationary geometries over both splits, with three per path. Interior IRC steps were examined separately for the H-transfer audit. The two fits also retain the 91,156 bare-PBE0 anchor geometries and the NIST and Grambow derivative constraints. Only the locally deposited AIMNet2-NSE supplement entered this fit.",
        "",
        "## Validation on the downloaded reaction data",
        "",
        "These rows use the deployed coefficients. The native electronic level used to score each row is shown. The cBD-CN network is held out in its entirety. RDB19-Rad holds out complete molecular formulas.",
        "",
        "| Dataset | Model | N | Reaction | Forward | Reverse | All barriers |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    kat = ("reaction", "forward", "reverse", "barriers")
    for source in held["chebyshev"]:
        for family, parent, stage in (("chebyshev", 73, 86), ("krawtchouk", 82, 87)):
            for key, s in (("parent", parent), ("fit", stage)):
                d = held[family][source][key]
                if "barriers" in d:
                    out.append(
                        mrow(
                            source.replace(" | ", " · ")
                            + " | "
                            + names[s]
                            + " | "
                            + str(d["n"]),
                            [d[k] for k in kat],
                        )
                    )
            d = angular[family][source]["fit"]
            if "barriers" in d:
                out.append(
                    mrow(
                        source.replace(" | ", " · ")
                        + " | "
                        + names[stage + 2]
                        + " | "
                        + str(d["n"]),
                        [d[k] for k in kat],
                    )
                )
    out += [
        "",
        "The four withheld BASChem19 reactions provide forward barriers. The archived benzopinacol singlet scan uses deviations from its mean energy over 369 frames.",
        "",
        "| Dataset and observable | p73 | p86 | p88 | p82 | p87 | p89 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for src, d in held["chebyshev"].items():
        if "single" in d["fit"]:
            label = (
                "BASChem19 forward barrier"
                if src.startswith("BASChem")
                else "AIMNet2-NSE singlet relative scan energy"
            )
            out.append(
                mrow(
                    label,
                    [
                        v
                        for f in held
                        for v in (
                            held[f][src]["parent"]["single"],
                            held[f][src]["fit"]["single"],
                            angular[f][src]["fit"]["single"],
                        )
                    ],
                )
            )
    out += [
        "",
        "## Comparison on identical public IDs",
        "",
        "| Panel | Model | N | Reaction | Forward | Reverse | All barriers |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    panels = {"All public reactions": keys, "H transfer": hkeys}
    panels.update(
        {src + " H transfer": {k for k in hkeys if k[0] == src} for src in kt.BARRIERS}
    )
    for label, ids in panels.items():
        for s, d in rows.items():
            out.append(
                mrow(
                    label + " | " + names[s] + " | " + str(len(ids)),
                    metrics(sorted(ids), d),
                )
            )
        for j, m in enumerate(COMPARATORS, 2):
            out.append(
                mrow(
                    label + " | " + m + " | " + str(len(ids)),
                    metrics(sorted(ids), rows[73], j),
                )
            )
    out += [
        "",
        "## Forming and breaking bond lengths",
        "",
        "D denotes the donor and A the acceptor in the reference reaction direction. Lengths (Å) and angles (degrees) are medians of individual measurements. RDB7 and Grambow require an explicitly mapped single H switch between two heavy atoms and a matching ordering of the archived stationary geometries. BH9 contributes its published HAT labels to the energy tables. RDB19 uses the author's atom-ordered educt and product complexes and verifies the changing heavy partner at both ends.",
        "",
        "| Archive and split | N | D–H reactant | H–A reactant | D–H saddle | H–A saddle | D–H product | H–A product | D–H–A saddle angle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    structs = structures(hkeys)
    for src in ("RDB7", "Grambow"):
        vals = [r[2:] for r in structs if r[0][0] == src]
        out.append(
            mrow(src + " validation | " + str(len(vals)), np.median(vals, axis=0))
        )
    raw = read(ROOT / "radical-h-structures.json")["observations"]
    for split in ("train", "validation"):
        sub = [r for r in raw if r["split"] == split]
        out.append(
            mrow(
                "RDB19-Rad " + split + " | " + str(len(sub)),
                np.median([r["lengths_A"] + [r["angle_deg"]] for r in sub], axis=0),
            )
        )
    records = [r for r in raw if r["split"] == "validation"]
    out += [
        "",
        f"The two initial IRC steps moved the bond difference r(D–H) − r(H–A) in opposite directions on {sum(r['opposite_initial_steps'] for r in records)} of {len(records)} RDB19-Rad validation H transfers. This is a property of the archived reference paths. It does not establish that a Korny trajectory reaches the two endpoints.",
        "",
        "## Frequencies and relaxed geometry",
        "",
        "| Model | Propane | Benzene | Ethanol | 1,3-butadiene | Equal-molecule mean (cm⁻¹) | Connectivity retained | Grambow saddle sign |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in MODELS:
        f = read(kt.WORK / f"p{s}-public-spectra.json")
        v = [
            f[m]["mae_cm1"] for m in ("propane", "benzene", "ethanol", "1,3-butadiene")
        ]
        d = read(kt.WORK / f"p{s}-public-diagnostics.json")["radical"]["saddle"]
        out.append(
            "| "
            + names[s]
            + " | "
            + " | ".join(f"{x:.1f}" for x in [*v, np.mean(v)])
            + f" | {sum(i['same_connectivity'] for i in f.values())}/4 | {round(d['n']*d['sign_agreement'][0])}/{d['n']} |"
        )
    native = read(RADICAL / "all-held-validation.json")[0]["shared"]
    out += [
        "",
        "## Dedicated neutral-doublet potential",
        "",
        "Here the parent is p67, trained at the RDB19-Rad electronic level. Both extensions use 4,974 training paths, two flank energy and projected-force measurements per path, and the change in force across the saddle where available. Each row uses a scalar potential with analytic forces. Curvature is a signed directional wavenumber from the two initial IRC steps. It is measured in cm⁻¹.",
        "",
        "| Held-out RDB19-Rad panel | Model | N | Reaction | Forward | Reverse | All barriers | Saddle curvature MAE | Downhill sign |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    native_h = read(ROOT / "native-rdb19-validation.json")
    for panel_name in ("all", "H transfers"):
        for model_name, entries in native_h.items():
            d = entries[panel_name]
            out.append(
                mrow(
                    panel_name + " | " + model_name + " | " + str(d["n"]),
                    [d[k] for k in kat],
                )
                + f" {d['signed_cm1']:.1f} | {d['downhill']}/{d['n']} |"
            )
    from experiments.full_public.paired_baseline_audit import paired_intervals

    out += [
        "",
        "Changes on the 589 common IDs were paired by reaction. Intervals resample whole transition-state formulas within each source 4,000 times.",
        "",
        "| Angular checkpoint and parent | Quantity | MAE change (kcal/mol) | 95% interval |",
        "|---|---|---:|---:|",
    ]
    for stage, parent in ((88, 73), (89, 82)):
        intervals = paired_intervals(f"p{stage}-public", parent=f"p{parent}-public")
        diff = np.subtract(
            metrics(sorted(keys), rows[stage]), metrics(sorted(keys), rows[parent])
        )
        for i, observable in enumerate(kat):
            lo, hi = intervals[observable]
            out.append(
                f"| p{stage} vs p{parent} | {observable} | {diff[i]:+.3f} | [{lo:+.3f}, {hi:+.3f}] |"
            )
    cc_source = next(k for k in held["krawtchouk"] if k.startswith("AutoMeKin | DLPNO"))
    cc = held["krawtchouk"][cc_source]
    native_comparison = native_h["chebyshev"]["H transfers"]
    out[2:2] = [
        f"On 265 coupled-cluster channels in the held-out AutoMeKin cBD-CN network, p87 reduced pooled barrier MAE from {cc['parent']['barriers']:.2f} to {cc['fit']['barriers']:.2f} kcal/mol. On the 172 held RDB19-Rad H transfers, the dedicated p67 + Chebyshev model reached {native_comparison['reaction']:.2f} reaction and {native_comparison['barriers']:.2f} pooled barrier MAE. The general checkpoints remain above 5 kcal/mol on the mixed panel.",
        "",
    ]
    out += [
        "",
        "## Interpretation and fitting details",
        "",
        f"The earlier p67 radical-specific Korny potential scored {native['reaction']:.3f} reaction, {native['forward']:.3f} forward, {native['reverse']:.3f} reverse and {native['barriers']:.3f} pooled barrier MAE on the same 626 RDB19-Rad IDs. Its reference and training objective specialize it to this neutral-doublet dataset.",
        "",
        "Each new geometry feature measures the difference and sum of two smooth covalent contacts to one H and their included angle. Both contact distances are scaled by covalent radii. The basis is symmetric under exchange of equivalent heavy ends and has analytic Cartesian derivatives. A 336-coefficient correction is fitted for each target branch. All training channel records enter as electronic differences against the corresponding frozen parent, including the previously downloaded sources. A separate regularized coefficient offset for each native method is eliminated analytically during fitting. The deployed potential uses the shared coefficients. The AIMNet2 scan offsets are centered separately by series, state and Hamiltonian. The prior uses three directions learned on reoptimized training reactions. The new descriptor and expanded supervision changed together, so their contributions cannot be assigned individually from these four models.",
        "",
        "The p88/p89 angular window switches the extra contribution smoothly from zero at 110° to full strength at 160°. Its derivative is included in the Cartesian forces. The angle choice was motivated by the structural audit, which included previously inspected validation geometries. The comparisons are development results.",
        "",
        "[Earlier kernel comparisons](operator-kernel-benchmarks.md) · [Dataset inventories and primary source records](aimnet2-nse-and-automekin.md) · [H-transfer feature implementation](../src/hydrogen_transfer.py) · [Native data fitting and evaluation](../experiments/full_public/h_transfer_sources.py)",
    ]
    tmp = DEST.with_suffix(".pending")
    tmp.write_text("\n".join(out) + "\n")
    tmp.replace(DEST)


if __name__ == "__main__":
    run()
