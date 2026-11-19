# Full public-data fit and validation

The joint PBE0 fit uses electronic energies, Cartesian forces, equilibrium geometries and harmonic frequencies. The three deployed checkpoints share the same PBE0 backbone. The high-level and radical checkpoints add separate reaction calibrations. All four tested molecules retained the reference bonding pattern after relaxation on both final checkpoints. Chemical reaction and barrier MAEs still exceed 5 kcal/mol on each barrier panel.

## Reaction energies and barriers

Each validation reaction contributes one reaction energy and one barrier in each direction. The two directions and all conformers with the same transition-state formula stay in the same split across the barrier collections. Shared endpoint fragments can occur in different reactions. The exact geometry and electronic state of each of the 595 validation saddles was absent from its model’s training data. Errors are kcal/mol.

| Dataset | Train reactions | Validation reactions | Reaction MAE | Forward MAE | Reverse MAE | Both barriers MAE | Reaction RMSD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RDB7 | 11667 | 259 | 25.54 | 32.30 | 27.77 | 30.03 | 34.88 |
| BH9 | 400 | 49 | 25.27 | 19.30 | 23.51 | 21.41 | 32.35 |
| TRIP50 | 46 | 4 | 29.47 | 30.54 | 10.27 | 20.40 | 42.19 |
| Grambow | 1084 | 283 | 15.47 | 20.19 | 16.86 | 18.53 | 20.66 |

RDB7 uses electronic CCSD(T)-F12a energy differences. BH9 and TRIP50 use their published high-level reference differences. Grambow uses the electronic wB97X-D3 energies printed for the actual minima and saddles. The high-level and Grambow checkpoints were calibrated and validated with PBE0-D3(BJ) switched on. Four TRIP50 validation reactions give a particularly uncertain mean.

| Dataset | Observable | PBE0 train | PBE0 validation | PBE0 MAE | PBE0 RMSD | High-level MAE |
| --- | --- | --- | --- | --- | --- | --- |
| GMTKN55 | energy difference | 1392 | 113 | 29.96 | 43.46 | — |
| OSOS | H abstraction | 3253 | 102 | 8.90 | 11.33 | 9.71 |
| OSOS | H shift | 2221 | 67 | 6.92 | 9.92 | 6.27 |
| OSOS | isomerization | 893 | 29 | 12.95 | 17.04 | 15.82 |
| OSOS | atomization | 4291 | 138 | 27.49 | 42.57 | 29.30 |

All 1,505 GMTKN55 PBE0 equations are included. Explicit H⁺ completes the relevant proton and dissociation conventions. One inconsistent OSOS parent thermochemical cycle was excluded from the high-level labels. Its independently published PBE0 labels remain in the PBE0 fit. All 11,926 RDB7 channels are included. For rxn009211, the product log ends during the footer, and the printed reference, pair-correlation and triples contributions independently reproduce its final energy.

## Molecular energies, forces and optional dispersion

Molecule and fragment IDs define the HDF5 partition. Linked duplicate geometry/state pairs belong to the same side. A structural fingerprint uses element-labelled pair distances rounded to 0.0001 Å. The four HDF5 files contained 101,993 records. Six DES15K records lacked usable spin labels. After grouping and deduplication, 91,156 distinct eligible geometry/state pairs supplied 7,073,877 Cartesian force components, with 9,603 further pairs for validation.

| Dataset | Train | Validation | PBE0 E | PBE0 force | With TS E | With TS force | With D3 E | With D3 force |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| QM7X | 39592 | 3838 | 12.65 | 11.31 | 13.72 | 11.33 | 12.25 | 11.31 |
| DES15K | 11407 | 1301 | 14.03 | 5.87 | 15.24 | 5.87 | 13.93 | 5.87 |
| SPICE | 30599 | 3250 | 10.55 | 7.03 | 13.58 | 7.03 | 10.94 | 7.03 |
| TorsionNet | 9558 | 1214 | 11.77 | 5.35 | 14.63 | 5.37 | 11.37 | 5.36 |

These are MAEs in kcal/mol for whole-molecule energy and kcal/mol/Å for individual force components. The bare column uses the separate PBE0 references. The TS and D3 columns use the released PBE0+MBD references. A separate volume model learned log Hirshfeld ratios from all 91,156 HDF5 training geometries. The 500 TorsionNet fragments supply 24 scan entries each, with 444 fragments in training and 56 in validation. Repeated coordinates do not add distinct examples. The final bare model’s mean absolute error for the maximum-minus-minimum scan range was 5.77 kcal/mol.

## Relaxed geometries and frequencies

Fourteen NIST PBE0/cc-pVTZ minima supply stationarity and squared-frequency targets. The frequency fit uses internal, mass-weighted coordinates. Reference mode vectors are unavailable in these tables, so the parent model supplies mode directions for the final linear frequency constraints. Each validation starts from the reference geometry, optimizes on the model, and diagonalizes the model Hessian at its own minimum. Translations and rotations are projected out.

| Molecule | Modes | PBE0 bonds (Å) | PBE0 angles (°) | PBE0 frequencies (cm⁻¹) | High-level bonds (Å) | High-level angles (°) | High-level frequencies (cm⁻¹) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| propane | 27 | 0.002 | 3.09 | 101 | 0.003 | 4.04 | 116 |
| benzene | 30 | 0.001 | 0.01 | 261 | 0.009 | 0.33 | 157 |
| ethanol | 21 | 0.023 | 4.07 | 247 | 0.012 | 3.59 | 121 |
| 1,3-butadiene | 24 | 0.011 | 2.37 | 214 | 0.013 | 4.65 | 197 |

Both final models retained the reference connectivity for all four molecules. High-level spectra here are compared against PBE0 reference frequencies. They measure cross-method agreement. The butadiene page contains multiple conformer sections; the archived section with a complete coordinate and frequency table supplies the fourth validation case.

### Radical curvature at reference geometries

Grambow supplied 4,078 complete Cartesian Hessians for 3,234 training structures and 844 held-out structures. The fit uses the softest internal reference mode, including the unstable direction at a saddle, and the highest-frequency mode. Converged optimization geometries also supply zero-gradient constraints. Twenty-three incomplete Hessians contribute no curvature labels. All 1,367 complete electronic reaction energies and barriers participate in the energy split.

| Reference direction | Held-out structures | Projected frequency MAE (cm⁻¹) | Correct curvature sign |
| --- | --- | --- | --- |
| endpoint · highest | 566 | 44 | 100.0% |
| endpoint · softest | 566 | 95 | 67.8% |
| saddle · highest | 278 | 42 | 100.0% |
| saddle · softest | 278 | 871 | 71.2% |

These projections measure stiffness along the indicated reference directions at the published geometry. Full optimized normal-mode spectra are reported for the NIST panel above.

## Short-range repulsion

The screened-core fit used all 320,028 tabulated DMol radii from 4,278 unordered element pairs (H through U). Two independent amplitudes per pair were refitted with the published decay lengths and a unit screening value at zero separation. Three pairs retained the published amplitudes to maintain monotonicity. Independent MP2 validation used 16,524 points from 324 unordered pairs. The screening-factor MAE was 0.00453. The interaction-energy MAE was 11.48 eV over the 1,644 points in the 30–1,000 eV repulsive regime. The molecular checkpoints retain the published NLH backbone core for the eight HDF5 elements and use the refit for additional benchmark elements.

## What changed with joint supervision

The first reaction-only correction reached lower reaction MAEs but severely distorted the PBE0 surface. On the HDF5 validation, that intermediate corrected PBE0 model had energy MAEs of 170.77 (DES15K), 75.98 (QM7-X), 498.38 (SPICE), and 323.76 (TorsionNet) kcal/mol. The final regression includes the whole HDF5 energy/force objective and the NIST geometry/frequency targets. Its higher-level branches regularize deviations from the resulting PBE0 surface across the observed HDF5 geometry and force directions. The Grambow branch also uses its method-matched Hessians.

| Dataset | First correction reaction MAE | First correction barrier MAE | Joint reaction MAE | Joint barrier MAE |
| --- | --- | --- | --- | --- |
| RDB7 | 11.99 | 18.00 | 25.54 | 30.03 |
| BH9 | 18.24 | 15.21 | 25.27 | 21.41 |
| TRIP50 | 41.62 | 26.28 | 29.47 | 20.40 |
| Grambow | 8.47 | 11.33 | 15.47 | 18.53 |

The validation panels were inspected during this work. The reported comparisons describe these splits; a blind external test would be needed for an independent generalization estimate.

## Checkpoints and timings

Across the three final checkpoints, 127,635 distinct molecular geometry/state pairs entered training. This union excludes held-out data, artificial finite-difference displacements and the independent diatomic screening grid. Including its backbone, the bare PBE0 checkpoint used 97,617 pairs, the high-level checkpoint 125,399 and the radical checkpoint 99,853. Overlaps between targets count once in the union.

- **Bare PBE0** — dispersion off for PBE0 energies, geometries and frequencies. [joint-pbe0-reaction.npz](../dataset_for_training/requested_public/joint-pbe0-reaction.npz)
- **High-level** — D3(BJ) on for the reported RDB7, BH9, TRIP50 and OSOS comparisons. [joint-high-reaction.npz](../dataset_for_training/requested_public/joint-high-reaction.npz)
- **Radical wB97X-D3** — D3(BJ) on for the reported Grambow comparisons. [joint-wb97xd3-reaction.npz](../dataset_for_training/requested_public/joint-wb97xd3-reaction.npz)
- **Hirshfeld ratios** — independent TS input for the eight HDF5 elements. [hirshfeld-requested.npz](../dataset_for_training/requested_public/hirshfeld-requested.npz)
- **Screened core** — standalone table covering all 4,278 pairs. [short-range-pairs.npz](../dataset_for_training/requested_public/short-range-pairs.npz)

The initial backbone fit took 148 s and its validation 14 s. Compressing the complete HDF5 energy/force equations into the final kernel basis took 460 s. Final reaction feature assembly, coefficient solves and reaction scoring took 63.6 s together. These stage timings exclude the separate Hessian, optimization and dispersion checks. Numerical jobs shared a 26 GB memory limit. Smite reproduced all three checkpoint energies and forces on a molecular doublet; finite-difference force checks also included singlet, doublet, triplet and heavy-element geometries.

Reproduction scripts are in [experiments/full_public](../experiments/full_public). Full numerical results and per-mode spectra are available in [reaction validation](../dataset_for_training/requested_public/joint-reaction-validation.json), [HDF5 validation](../dataset_for_training/requested_public/joint-final-hdf-validation.json), [relaxed frequencies](../dataset_for_training/requested_public/joint-relaxed-frequency-validation.json) and [radical curvatures](../dataset_for_training/requested_public/projected-curvature-validation.json).
