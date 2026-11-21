# Reactive kernel experiments on the Korny public validation split

The **combined kernel (p11-public)** reaches **11.35 kcal/mol reaction MAE and 14.24 kcal/mol pooled barrier MAE** on the 591 reaction IDs common to Korny, original SO3LR, Tauri-M and Tauri-L. The 10.0 kcal/mol target remains open for both pooled metrics. All energy MAEs are in kcal/mol. All reaction predictions include PBE0-D3(BJ) with ATM once per geometry. The underlying PBE0 fit used dispersion-free energy and force targets. Grambow uses a separate wB97X-D3-targeted correction. The two branches represent different electronic reference levels.

## Reaction energies and barriers

| Dataset | Training reactions | Validation reactions | Reaction energy | All barriers | Forward | Reverse |
|---|---:|---:|---:|---:|---:|---:|
| RDB7 | 11,667 | 259 | 11.55 | 16.23 | 15.85 | 16.62 |
| BH9 | 400 | 49 | 17.39 | 17.31 | 15.13 | 19.49 |
| Grambow | 1,084 | 283 | 9.78 | 11.72 | 12.39 | 11.06 |
| TRIP50 | 46 | 4 | 40.46 | 21.76 | 16.53 | 26.99 |
| RDB7, BH9 and Grambow | 13,151 | 591 | 11.19 | 14.16 | 14.13 | 14.19 |
| All four panels | 13,197 | 595 | 11.38 | 14.21 | 14.15 | 14.28 |

Each reaction contributes two directional barriers. The two directions share a saddle and their signed difference is the reaction energy, up to tabulation precision. The priority pool includes all BH9 reactions. TRIP50 was scored separately and excluded from hyperparameter selection.

## Which changes moved the errors

| Model | Priority reactions | Reaction energy | All barriers | Forward | Reverse |
|---|---:|---:|---:|---:|---:|
| Endpoint-gap and midpoint-barrier weighting | 591 | 17.21 | 18.58 | 19.16 | 18.01 |
| Spin-prioritized selection and NIST derivative constraints | 591 | 16.99 | 23.22 | 24.34 | 22.10 |
| More atom-centered kernels | 591 | 16.26 | 22.80 | 23.72 | 21.88 |
| Tuned Matérn and angular scales | 591 | 16.09 | 22.83 | 23.76 | 21.89 |
| Four-body torsion and out-of-plane terms | 591 | 15.66 | 22.60 | 23.57 | 21.62 |
| Bond length and endpoint coordination | 591 | 15.73 | 20.23 | 20.82 | 19.64 |
| 4× reaction weights | 591 | 13.83 | 18.48 | 18.58 | 18.38 |
| Larger bond basis, 16× reaction weights | 591 | 13.37 | 17.46 | 17.68 | 17.25 |
| Element-resolved approach angles | 591 | 11.23 | 15.45 | 15.46 | 15.43 |
| 256 angular features and unstable-mode weight trials | 591 | 12.30 | 14.96 | 14.93 | 15.00 |
| Isomer-relative electronic energy supervision | 591 | 12.30 | 14.62 | 14.60 | 14.64 |
| 50/50 combined angular kernel | 591 | 11.19 | 14.16 | 14.13 | 14.19 |

The first weight sweep favored a 0.003 PBE0 anchor and sixfold midpoint weight but distorted molecular curvature. Each subsequent comparison selected settings on two inner folds grouped by transition-state formula. Selection used RDB7 and BH9 singlets/doublets and a separate Grambow development score. The larger 4.2 Å atom-centered descriptor lost to the 3.2 Å representation. Huber reweighting at 5 and 10 kcal/mol lost to least squares with higher reaction weights. Repeated inspection makes the outer IDs a regression benchmark. An untouched set is needed for an independent final accuracy claim.

The force-field terms sum over atom or bond environments. Bond features encode scaled distance, smooth H/C/N/O coordination at both ends and, in the angular version, element-resolved neighbor projections along the bond. Symmetrizing homonuclear bonds removes atom-order dependence. The Jacobian includes differentiation of the neighbor directions and the central bond axis. Detached fragments have additive energies for these terms. Charge and multiplicity enter the parent potential. The new bond descriptor itself depends on element and geometry. Tests cover Cartesian finite differences, permutations, rotations and separated fragments. [Bond features](../src/bond_environment.py) · [Training driver](../experiments/full_public/trial_fit.py)

The later training experiment recovered electronic energies for individual stationary structures from the RDB7 Molpro and Grambow Q-Chem logs. The per-geometry energies reproduce the saved reaction and barrier equations. After excluding held-out geometry IDs and inconsistent duplicates, the extra objective centered electronic energies within the same reference method, atomic composition, charge and spin. RDB7 supplied 26,155 structures in 212 groups, and Grambow supplied 2,236 in 49 groups. The centered observations have 25,943 and 2,187 degrees of freedom, respectively. Some information overlaps the original reaction equations. Changing the soft-mode weight also affects the radical comparison. [Electronic-energy reader and split logic](../experiments/full_public/trial_isomers.py)

The 144- and 256-feature banks use distinct random draws. Increasing their size improved the development score and held-out barriers but worsened the held-out reaction energies. The final candidate uses an equal average of the earlier 144-feature model and the 256-feature model trained with cross-reaction isomer energies. The mixing coefficient was fixed before scoring the average. The implementation merges the features into one scalar potential with a single evaluation of the shared backbone and D3. Both branches reproduced the component-average energies and analytic forces to 10⁻⁹ on eight archived structures. [Checkpoint construction](../experiments/full_public/trial_average.py)

## Korny against every available comparator

Each row in a panel uses exactly the same reaction IDs. The full four-model intersection contains 591 reactions and 1,182 directional barriers. Every MAE is in kcal/mol. “Korny (144 angular)” is the earlier reaction-energy candidate. “Korny (256 angular)” refers to the capacity experiment. The latest trial selection is labelled by its checkpoint.

| Panel | Model | Reactions | Reaction energy | All barriers | Forward | Reverse |
|---|---|---:|---:|---:|---:|---:|
| All matched | Korny (144 angular) | 591 | 11.41 | 15.54 | 15.53 | 15.54 |
| All matched | Korny (256 angular) | 591 | 12.47 | 15.05 | 15.02 | 15.08 |
| All matched | Korny (isomer supervision) | 591 | 12.44 | 14.69 | 14.67 | 14.71 |
| All matched | Korny (p11-public) | 591 | 11.35 | 14.24 | 14.20 | 14.28 |
| All matched | Original SO3LR | 591 | 13.97 | 13.67 | 13.28 | 14.05 |
| All matched | SO3LR Tauri-M-2606 | 591 | 6.16 | 8.08 | 8.27 | 7.90 |
| All matched | SO3LR Tauri-L-2606 | 591 | 4.48 | 6.02 | 5.98 | 6.06 |
| RDB7 | Korny (144 angular) | 259 | 12.17 | 18.37 | 18.36 | 18.38 |
| RDB7 | Korny (256 angular) | 259 | 11.97 | 16.91 | 16.16 | 17.65 |
| RDB7 | Korny (p11-public) | 259 | 11.55 | 16.23 | 15.85 | 16.62 |
| RDB7 | Original SO3LR | 259 | 13.59 | 11.38 | 12.12 | 10.64 |
| RDB7 | SO3LR Tauri-M-2606 | 259 | 6.67 | 9.69 | 10.15 | 9.24 |
| RDB7 | SO3LR Tauri-L-2606 | 259 | 5.41 | 6.90 | 6.93 | 6.88 |
| BH9 | Korny (144 angular) | 47 | 17.24 | 17.32 | 16.12 | 18.52 |
| BH9 | Korny (256 angular) | 47 | 16.57 | 17.88 | 15.82 | 19.94 |
| BH9 | Korny (p11-public) | 47 | 16.84 | 17.40 | 15.40 | 19.40 |
| BH9 | Original SO3LR | 47 | 15.67 | 13.77 | 15.64 | 11.90 |
| BH9 | SO3LR Tauri-M-2606 | 47 | 7.20 | 8.45 | 9.22 | 7.69 |
| BH9 | SO3LR Tauri-L-2606 | 47 | 4.99 | 7.13 | 7.12 | 7.13 |
| Grambow | Korny (144 angular) | 283 | 9.25 | 12.47 | 12.74 | 12.19 |
| Grambow | Korny (256 angular) | 283 | 11.77 | 12.70 | 13.70 | 11.70 |
| Grambow | Korny (p11-public) | 283 | 9.78 | 11.72 | 12.39 | 11.06 |
| Grambow | Original SO3LR | 283 | 13.88 | 15.71 | 13.97 | 17.46 |
| Grambow | SO3LR Tauri-M-2606 | 283 | 5.49 | 6.55 | 6.42 | 6.69 |
| Grambow | SO3LR Tauri-L-2606 | 283 | 3.53 | 5.04 | 4.94 | 5.14 |
| TRIP50 | Korny (144 angular) | 2 | 83.05 | 41.52 | 30.73 | 52.32 |
| TRIP50 | Korny (256 angular) | 2 | 80.44 | 40.22 | 34.70 | 45.74 |
| TRIP50 | Korny (p11-public) | 2 | 77.81 | 38.91 | 29.99 | 47.82 |
| TRIP50 | Original SO3LR | 2 | 34.31 | 17.15 | 10.87 | 23.44 |
| TRIP50 | SO3LR Tauri-M-2606 | 2 | 9.75 | 6.91 | 3.74 | 10.08 |
| TRIP50 | SO3LR Tauri-L-2606 | 2 | 7.13 | 3.56 | 3.18 | 3.95 |

The two Tauri checkpoints each cover all 595 Korny reactions. Original SO3LR lacks silicon and scored 47 of 49 BH9 reactions and 2 of 4 TRIP50 reactions. On all four TRIP50 cases the errors differ substantially from the two-case intersection. The following comparison keeps all 595 cases for the three methods that cover them.

| Panel | Model | Reactions | Reaction energy | All barriers | Forward | Reverse |
|---|---|---:|---:|---:|---:|---:|
| All available | Korny (p11-public) | 595 | 11.38 | 14.21 | 14.15 | 14.28 |
| All available | SO3LR Tauri-M-2606 | 595 | 6.24 | 8.10 | 8.28 | 7.91 |
| All available | SO3LR Tauri-L-2606 | 595 | 4.58 | 6.05 | 6.02 | 6.08 |
| TRIP50 | Korny (p11-public) | 4 | 40.46 | 21.76 | 16.53 | 26.99 |
| TRIP50 | SO3LR Tauri-M-2606 | 4 | 22.16 | 12.10 | 10.50 | 13.69 |
| TRIP50 | SO3LR Tauri-L-2606 | 4 | 22.09 | 11.04 | 12.78 | 9.31 |

### Same IDs as the available sGDML predictions

The previous sGDML calculation yielded six reaction energies, 29 forward barriers and six reverse barriers from RDB7. Every method in this table is scored on that same subset for the respective observable. No additional sGDML calculation was needed.

| Model | Reaction energy (6) | All barriers (35) | Forward (29) | Reverse (6) |
|---|---:|---:|---:|---:|
| Korny (p11-public) | 8.61 | 10.01 | 10.23 | 8.97 |
| Original SO3LR | 2.98 | 7.35 | 8.07 | 3.86 |
| SO3LR Tauri-M-2606 | 4.42 | 6.03 | 6.41 | 4.22 |
| SO3LR Tauri-L-2606 | 3.78 | 4.91 | 5.16 | 3.67 |
| Stock sGDML + D3 | 37.52 | 93.49 | 95.68 | 82.90 |

Both Tauri models used the bundled June 2026 checkpoints, float64 CPU inference, a 12 Å long-range cutoff and per-geometry charge and multiplicity. Each used 92 earlier predictions verified by ID, coordinates, atoms and state, and 1,806 fresh geometry predictions. The model energy contains its own dispersion. Detached fragments receive separate model calls. Possible overlap of pretrained model data with the benchmarks is unknown. [All per-equation signed errors and references](../dataset_for_training/requested_public/so3lr-tauri-matched/p11-public-comparison.csv) · [Full and common-set statistics](../dataset_for_training/requested_public/so3lr-tauri-matched/p11-public-summary.json) · [Original SO3LR protocol](so3lr-matched-validation.md) · [sGDML coverage](sgdml-matched-validation.md) · [Tauri matched evaluator](../experiments/full_public/tauri_validation.py)

## Geometries and frequencies

These four NIST PBE0/cc-pVTZ molecules were held out from the 14-molecule stationary-structure and frequency training set. Each model relaxed its own minimum before the Cartesian Hessian calculation. Frequencies compare sorted vibrational wavenumbers, so individual bending/stretching assignments require mode vectors.

| Relaxed molecule | Original MAE, cm⁻¹ | New MAE, cm⁻¹ | Bond MAE, Å | Angle MAE, degrees | Connectivity retained |
|---|---:|---:|---:|---:|---|
| propane | 115.8 | 87.9 | 0.003 | 2.78 | True |
| benzene | 157.1 | 93.0 | 0.009 | 0.32 | True |
| ethanol | 120.8 | 85.7 | 0.009 | 2.70 | True |
| 1,3-butadiene | 196.5 | 122.7 | 0.011 | 3.71 | True |

| Grambow projected Hessian direction | Structures | Softest signed-frequency MAE, cm⁻¹ | Correct soft-mode sign | Highest-frequency MAE, cm⁻¹ |
|---|---:|---:|---:|---:|
| endpoint | 566 | 86.6 | 84.6% | 52.2 |
| saddle | 278 | 808.1 | 77.3% | 59.7 |

The combined model retains the reference connectivity on all four relaxed NIST systems. For Grambow saddles its unstable-direction sign is correct on 215 of 278 structures. The saddle metric measures curvature along the archived unstable mode at the reference geometry. A model can give a good barrier energy and a poor local imaginary frequency. Accurate reaction dynamics and transition-state frequencies remain unestablished.

## Controlled isoprene inclusion test

Both models below used the 128-center four-body architecture, the same coefficients in the objective, and the same IDs. The private branch added 14,896 distinct singlet and 31,419 doublet training geometry/states, including nine paired IRC training paths. It included 46,326 DFT and 751 CC energy records, with separate DFT level corrections. The seven held-out IRC paths and two earlier matched Gamma channels contributed no training equations. Formula, state and duplicate screening reserved 90,186 raw private records, which include trajectories. Geometry and record counts differ.

| Training set | Reactions | Reaction energy | All barriers | Forward | Reverse |
|---|---:|---:|---:|---:|---:|
| Public only | 591 | 15.66 | 22.60 | 23.57 | 21.62 |
| Public + Leuven | 591 | 15.77 | 22.71 | 23.68 | 21.74 |

The paired, within-source formula bootstrap gave a 95% interval for the change upon inclusion of [-0.28, 0.34] kcal/mol for reactions and [-0.04, 0.32] for barriers. It resampled 10 RDB7, 35 BH9 and 9 Grambow formulas and preserved source weights. The small positive point estimates are compatible with no general effect. Subsequent general fits used only public labels.

## Gamma channel assignments from the author’s Python extractors

Both Case I and Case II define the species lists, signed fragment partners, fragment coordinates and energies inside their November 2025 Python extractors. Case I uses Gamma-2_ZZ as its reporting origin and Case II uses Gamma-2_ZZ_S. The scripts rank conformers by CCSD(T)+ZPE with a DFT+ZPE fallback. For the electronic force-field diagnostic we kept only CCSD(T)-F12/cc-pVDZ-F12 records paired with a frequency-checked M06-2X-D3/aug-cc-pVTZ optimization at matching coordinates. We used the lowest CC+ZPE conformer passing these checks, its electronic CC energy for the PES and its ZPE-corrected energy only for comparison to the original table. All reconstructed ZPE differences agree to within 0.012 kcal/mol of the tabulated differences.

| Series and endpoint choice | Electronic ΔE | Forward | Reverse | ΔE with ZPE | Forward with ZPE | Reverse with ZPE |
|---|---:|---:|---:|---:|---:|---:|
| Case1, Gamma-2_ZZ → Gamma-4-ZZ-R | -1.82 | 14.26 | 16.07 | -3.12 | 10.85 | 13.97 |
| Case1, Gamma-2_ZZ → Gamma-4-ZZ-S | -5.61 | 14.26 | 19.87 | -6.49 | 10.85 | 17.34 |
| Case1, Gamma-2_ZZ → Gamma-4_extraHbonds | -4.36 | 14.26 | 18.62 | -5.09 | 10.85 | 15.94 |
| Case2, Gamma-2_ZZ_S → Gamma-4_ZZ_R | -2.42 | 15.01 | 17.43 | -3.11 | 11.71 | 14.82 |
| Case2, Gamma-4_ZZ_R → Gamma-5_ZZ_R + OH | -4.79 | 20.14 | 24.93 | -8.01 | 18.14 | 26.15 |

The two manifests yielded 4 Case I and 12 Case II distinct saddle directory names with candidate complete assignments. Some are reruns or use alternative endpoints. Case I TS-2-4 identifies species 2 → 4, but its name alone leaves three Gamma-4 series in the manifest. In Case II the extractor lists a Gamma-2 ZZ/S representative and a Gamma-4 ZZ/R representative for the comparison. A matching first-order saddle does not by itself prove which stereoisomer its IRC reaches. Gamma-5 requires OH and Gamma-7 requires HO₂. Isoprene stays a separate chemistry diagnostic. [Per-conformer source files, alternatives and omitted channels](../dataset_for_training/requested_public/kernel-trials/gamma-source-audit.json) · [Read-only extraction](../experiments/full_public/leuven_channels.py)

## Published 6.89 kcal/mol kernel baseline

The authors’ EquiReact repository was checked out at revision 0f2702b1235a2184cfef8f9f4e126e8c4aff2d18. Their baseline builds global SLATM spectra for each participant, sums fragments on each side, and takes products minus reactants with explicit hydrogens. For GDB they use a Manhattan-distance Laplacian kernel with γ = 0.01 and ridge = 10⁻¹⁰. Scoring the ten saved prediction files against their GDB barrier column gives 6.89 ± 0.20 kcal/mol for random splits, 10.97 ± 1.01 for scaffold splits and 8.06 ± 0.16 for ascending-size splits. These are forward-barrier errors from known endpoints. Korny instead evaluates the same scalar potential at the two endpoints and saddle, which also yields forces. Nonlinear geometry scales are fitted by refitting the linear coefficients at each candidate setting. [Original code](https://github.com/lcmd-epfl/EquiReact/tree/0f2702b1235a2184cfef8f9f4e126e8c4aff2d18/baseline_slatm) · [Paper](https://arxiv.org/pdf/2312.08307)

## Fitting time

| Completed CPU experiment | Wall time for the fit sweep |
|---|---:|
| Bond coordination features, two kernel widths | 30 s |
| Robust versus conventional loss | 38 s |
| Larger bond basis and reaction weights | 77 s |
| Direction-dependent bond kernel and reaction weights | 71 s |
| 256-feature and saddle-curvature trials | 147 s |
| Cross-reaction isomer-energy trials | 217 s |
| Merging and checking the final checkpoint pair | 3 s |

Times run from log creation to the last fit message on this machine, with geometry features already cached. Feature preparation and relaxed-frequency validation are additional. The shared experiment slice reported a peak of 22.01 GB against a 26 GB hard cap. Smite's Korny interface reproduced the final checkpoint energies and forces on singlet water and doublet HO₂ with D3 enabled.

## Data and local checkpoints

The PBE0 parent used 39,592 QM7-X 8000, 11,407 DES15K, 30,599 SPICE dipeptide and 9,558 TorsionNet training geometries. Each of the 91,156 structures supplied both an energy anchor and a deterministic projected force anchor when enlarging the basis. The parent fit retains its original full force supervision. Fourteen NIST training geometries supplied stationary/curvature constraints. The radical objective used two measured Hessian directions at each of 3,234 training structures. OSOS, GMTKN55 PBE0 and short-range repulsion contributed at the stages and levels documented in the full inventory. The additional kernels were evaluated under the shared 26 GB limit. [Full training counts and method labels](full-public-fit.md) · [Priorities and scientific references](kernel-improvements.md)

Load the [high-level checkpoint](../dataset_for_training/requested_public/kernel-trials/p11-public-high.npz) or [Grambow-aligned checkpoint](../dataset_for_training/requested_public/kernel-trials/p11-public-wb97xd3.npz) using `ForceField.load(path, dispersion="d3")` and pass the molecular charge and multiplicity. These experimental checkpoints are stored separately from the deployed model. [All validation metrics](../dataset_for_training/requested_public/kernel-trials/p11-public.json) · [Relaxed structures, full spectra and frequency errors](../dataset_for_training/requested_public/kernel-trials/p11-public-spectra.json)
