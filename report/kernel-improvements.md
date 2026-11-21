# Improving Korny for reaction energies, barriers and reactive dynamics

Korny's biggest measured error is the height of the RDB7 saddle relative to its two endpoints. On all 259 held-out RDB7 reactions, Korny underestimates that height by 15.03 kcal/mol on average. Its mean absolute error for the same quantity is 26.48 kcal/mol. I would keep the linear energy readout, tune the trade-off between the PBE0 surface and the higher-level reaction correction, then test a larger reactive basis and a four-body representation. These figures describe the original checkpoint. The completed experiments and same-ID comparisons with original SO3LR and both Tauri models are in the [reactive kernel experiment report](reactive-kernel-experiments.md).

## Matched results with dispersion

**Korny included PBE0-D3(BJ), including the three-body ATM term, in every reaction and barrier below.** The high-level and radical corrections were fitted with D3 present. The evaluator adds D3 once to each separately evaluated fragment and transition state. Original SO3LR uses its own built-in dispersion. Both models used the same coordinates, stoichiometries and reference energies. Each cell gives **Korny / original SO3LR** MAE in kcal/mol. [Korny fit](../experiments/full_public/fit_reactions.py) · [D3 implementation](../src/dispersion/d3.py) · [Matched evaluator](../experiments/full_public/so3lr_validation.py)

| Dataset | Korny training reactions | Original validation reactions | Reactions scored by both | Reaction energy | All barriers | Forward | Reverse |
|---|---:|---:|---:|---:|---:|---:|---:|
| RDB7 | 11,667 | 259 | 259 | 25.54 / 13.59 | 30.03 / 11.38 | 32.30 / 12.12 | 27.77 / 10.64 |
| BH9 | 400 | 49 | 47 | 25.56 / 15.67 | 21.89 / 13.77 | 19.45 / 15.64 | 24.33 / 11.90 |
| TRIP50 | 46 | 4 | 2 | 55.35 / 34.31 | 38.93 / 17.15 | 57.50 / 10.87 | 20.37 / 23.44 |
| Grambow radicals | 1,084 | 283 | 283 | 15.47 / 13.88 | 18.53 / 15.71 | 20.19 / 13.97 | 16.86 / 17.46 |
| **Total or pooled MAE** | **13,197** | **595** | **591** | **20.82 / 13.97** | **23.91 / 13.67** | **25.56 / 13.28** | **22.25 / 14.05** |

Each training reaction supplies three energy differences, with two independent combinations. The first three datasets supplied 12,113 training reactions to the high-level branch. Grambow supplied 1,084 to the separate wB97X-D3 branch. RDB7 uses CCSD(T)-F12a electronic energies. BH9 and TRIP50 use their published high-level differences. The pooled comparison weights 591 forward and 591 reverse barriers equally. The four unscored reactions contain silicon, outside original SO3LR's element coverage. The SO3LR cutoff was 12 Å throughout for consistency with previously saved predictions. [Split counts](../dataset_for_training/requested_public/reaction-inventory.json) · [Per-reaction files, excluded IDs and cutoff details](so3lr-matched-validation.md)

The original SO3LR checkpoint received **zero local training or fine-tuning reactions** in this comparison. Its authors describe pretraining on approximately 2.7 million GEMS fragments, 1 million QM7-X structures, 60,000 Aquamarine structures, 33,000 SPICE dipeptides and 15,000 DES dimers. These are approximate molecular-structure counts. Exact overlap of pretraining geometries with this validation panel has not been established. [Original SO3LR training data description](https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/679bf19781d2151a02991c58/original/SO3LR_ChemRxiv_v2.pdf)

## Additional supervision used by Korny

| Dataset | Training | Validation | What the counts represent |
|---|---:|---:|---|
| QM7-X 8000 | 39,592 | 3,838 | Distinct geometries with bare PBE0 energies and forces |
| DES15K | 11,407 | 1,301 | Distinct geometries with bare PBE0 energies and forces |
| SPICE dipeptides | 30,599 | 3,250 | Distinct geometries with bare PBE0 energies and forces |
| TorsionNet500 | 9,558 | 1,214 | Distinct geometries with bare PBE0 energies and forces |
| GMTKN55 PBE0 | 1,392 | 113 | Thermochemical energy-difference equations |
| OSOS H abstraction | 3,253 | 102 | Reaction equations at each of the two reference levels |
| OSOS H shift | 2,221 | 67 | Reaction equations at each reference level |
| OSOS isomerization | 893 | 29 | Reaction equations at each reference level |
| OSOS atomization | 4,291 PBE0 / 4,281 high-level | 138 PBE0 / 137 high-level | Molecular atomization labels |
| NIST | 14 | 4 | Stationary geometries and harmonic spectra |
| Grambow curvature | 3,234 | 844 | Structures with usable Cartesian Hessians in the Grambow split |
| Short-range repulsion | 320,028 DMol grid points | 16,524 independent MP2 points | Diatomic repulsion data |

The HDF5 total is **91,156 training geometries and 9,603 validation geometries**. Across all three deployed branches, the union of molecular training geometry/state pairs is 127,635. OSOS contributes 6,367 training reaction equations per level across the three named classes. Repeated labels at two levels describe the same chemistry. [Geometry audit](../dataset_for_training/requested_public/audit.json) · [Training and validation inventory](full-public-fit.md)

## What is wrong with the barriers?

For each reaction define the endpoint gap as the product energy minus the reactant energy, and the midpoint barrier as the saddle energy minus the mean of those two endpoint energies. The midpoint barrier is the average of the forward and reverse barriers **before** taking an absolute error. It measures how high the model places the saddle relative to both endpoints. A negative signed error means the predicted midpoint barrier is too low. This statistic alone cannot attribute the error to one of the three geometries.

| Dataset | Matched reactions | Midpoint barrier MAE, Korny / SO3LR | Midpoint barrier signed mean, Korny / SO3LR |
|---|---:|---:|---:|
| RDB7 | 259 | 26.48 / 9.02 | −15.03 / +3.29 |
| BH9 | 47 | 18.69 / 11.41 | +2.27 / −3.76 |
| TRIP50 | 2 | 29.82 / 6.92 | +29.82 / −6.28 |
| Grambow radicals | 283 | 16.96 / 14.52 | −11.36 / −10.88 |

All entries are kcal/mol. Both models underestimate the midpoint barrier on the Grambow doublets. The larger RDB7 difference occurs on a neutral-singlet panel. Improving open-shell features alone would leave that problem unresolved. The reference forward-minus-reverse identity closes to tabulation precision, with discrepancies up to 0.01 kcal/mol in the matched BH9 rows. [RDB7 equations](../dataset_for_training/requested_public/so3lr-original-matched/RDB7-equations.csv) · [BH9](../dataset_for_training/requested_public/so3lr-original-matched/BH9-equations.csv) · [TRIP50](../dataset_for_training/requested_public/so3lr-original-matched/TRIP50-equations.csv) · [Grambow](../dataset_for_training/requested_public/so3lr-original-matched/Grambow-equations.csv)

## What successful methods do differently

| Published method | What the authors demonstrated | What I would test in Korny |
|---|---|---|
| Difference-SLATM with kernel ridge regression | Using product-minus-reactant structural descriptors, the kernel baseline achieved **6.89 ± 0.20 kcal/mol** barrier MAE on GDB7-22-TS across ten random 80/10/10 splits, with explicit hydrogens and no atom mapping. The study also examined harder extrapolative splits. | Use endpoint-difference descriptors as a reaction-only diagnostic on Korny's formula-held-out split. They require a known product. For MD, learn the endpoint-gap and midpoint-barrier targets as differences of the same single-geometry potential. [3DReact paper, Table 1](https://arxiv.org/pdf/2312.08307) |
| FCHL19 | The authors optimized the nonlinear parameters of a discretized, element-resolved local representation, then used kernels for molecular energies and forces. Their molecular energy and force benchmarks test different observables from reaction barriers. | Compare an element-resolved FCHL-style radial and angular representation against Korny's compressed one. Refit the linear coefficients after each change in radial width, angular scaling or cutoff. [Christensen et al.](https://arxiv.org/abs/1909.01946) |
| Linear atomic cluster expansion (ACE) | Four- and five-body symmetric polynomial bases allowed linear molecular force fields to be tested on normal modes, torsion scans and bond breaking. These are directly relevant to Korny's MD objectives. | Add a small four-body invariant basis. Test whether environments with similar pair distances and bond angles become distinguishable through torsion and out-of-plane geometry. ACE provides an explicit feature map compatible with a linear readout. [Kovács et al.](https://doi.org/10.1021/acs.jctc.1c00647) |
| FLARE++ | A sparse Gaussian process used uncertainty to select environments during reactive hydrogen dissociation and recombination on platinum. A polynomial mapping made prediction cost independent of training-set size in that model. | Select additional centers from **existing, labelled training data** using novelty and out-of-fold force or energy errors. Consider a polynomial kernel if the enlarged basis makes MD calls too slow. [Vandermause et al.](https://arxiv.org/abs/2106.01949) |

The 6.89 kcal/mol literature result is a particularly useful comparison to design an experiment around. It uses known endpoints and a different split and target protocol. It supplies no measured expectation for Korny's 595-reaction holdout or for forces along an unknown reaction path.

## What to change first

| Priority | Change | Reason and controlled test |
|---|---|---|
| 1 | Make the endpoint gap and midpoint barrier independently weightable. Tune the penalty anchoring the high-level correction to the PBE0 surface. | The earlier RDB7 correction gave 11.99 reaction and 18.00 barrier MAE. The joint model gives 25.54 and 30.03 while preserving a much better molecular surface. Several settings changed between fits. With the current basis fixed, sweep the anchor around its present 0.15 value and the ratio of the two reaction weights. Recompute both directional barriers from the same scalar potential. [Joint fit](../experiments/full_public/fit_reactions.py) · [Earlier results](full-public-fit.md) |
| 2 | Expand and improve placement of the reactive centers. | The deployed reaction kernel uses **371 centers** across eight elements, with **64 each for H, C, N and O**. Basis construction takes up to 24 candidate geometries per family before selecting centers. Try 64, 128 and 256 centers per common element, keeping data and regularization controls comparable. Include forming and breaking bonds, shared H, peroxy radicals and equilibrium environments. Use KinBot classes to balance training candidates. An MD step only needs geometry, charge and multiplicity. [Basis selection](../experiments/reactivity/matern_fit.py) · [Earlier class-aware comparison](reaction-benchmarks.md) |
| 3 | Improve the local structure map and tune nonlinear scales. | The deployed reaction kernel uses two radial basis functions, angular order up to two, a 3.2 Å cutoff, and at most 12 projected components in each of the radial and angular blocks. The double Matérn has saved length and angular scales of 1.0 and 1.0. Test more radial resolution and angular order, then a four-body block. Compare each at fixed center count. Use smooth cutoffs and symmetry-invariant features so gradients and Hessians remain usable. [Kernel implementation](../src/matern.py) · [Current high-level checkpoint](../dataset_for_training/requested_public/joint-high-reaction.npz) |
| 4 | Improve electronic-state dependence and saddle curvature where the data support it. | The global doublet and triplet terms in the reaction correction see only every fourth local-kernel feature. Test a learned or chemically selected spin-sensitive basis around radical centers, and use the archived Grambow training Hessians for unstable and low-frequency bending directions. Score the signed imaginary mode, mode curvature, and relaxed geometry alongside barrier MAE. Test separated-fragment energy limits for the relevant spin channels. [State-dependent features](../src/reaction_surface.py) · [Curvature supervision](../experiments/full_public/radical_hessians.py) |

Korny is nonlinear in geometry already. At fixed descriptor parameters and centers, its energy corrections are linear in their coefficients. For nonlinear tuning, use an outer search over feature parameters and **solve the regularized linear problem again at every setting**. An earlier pilot already compared 12 combinations of Gaussian, Matérn and double Matérn kernels with scale and ridge settings. It used 159 training reactions and 50 development reactions. The deployed joint fit retained fixed kernel scales. The next experiment would tune them on grouped folds of its own training data. Changing the feature basis also requires rebuilding its force and curvature matrices. [Pilot results](../experiments/reactivity/kernels_fit.json) · [Joint force equations](../experiments/full_public/joint_observations.py)

Use internal folds grouped by transition-state formula, keeping both barrier directions and shared trajectories together. Report per-dataset reaction, forward and reverse MAEs, low-frequency bends and saddle modes, optimized bond and angle errors, and force time per MD step. Compare the same geometry and frequency cases at each stage. The 595 published validation IDs have already been inspected. A final independent panel is needed to claim generalization below 5 kcal/mol. The existing data are sufficient for the proposed comparisons; further reference calculations would be justified by a specific uncovered environment or reaction-path region.

## Dispersion in production

For the reported high-level and radical checkpoints use `ForceField.load(path, dispersion="d3")`, or set `korny_dispersion="d3"` in Smite. The generic loader and Smite adapter currently default to `off` when the option is omitted. Bare PBE0 remains the training and diagnostic reference for the backbone. For separately labelled PBE0+MBD validation, the existing report gives the bare model with TS or D3 added independently. Switching a fitted reaction branch from D3 to TS would require a separate validation. [Force-field loader](../src/forcefield.py) · [Dispersion validation](full-public-fit.md)
