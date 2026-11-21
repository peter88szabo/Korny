# Reaction and frequency validation after the combined kernel

## Current fitting results

The goal is 5.0 kcal/mol MAE each for reaction energies and barriers, with credible equilibrium and saddle frequencies. Korny p49 has the lowest pooled barrier error among the completed trials in this series. It gives **9.36 kcal/mol for reaction energies and 10.67 kcal/mol for barriers**. The fixed common panel contains 591 reactions and 1,182 directed barriers. Each Korny prediction includes D3(BJ) with ATM once.

| Model | Reaction | All barriers | Forward | Reverse |
|---|---:|---:|---:|---:|
| Korny p28, Hirshfeld and radical environments | 9.76 | 11.20 | 11.19 | 11.21 |
| Korny p32, QM7-X force response | 9.75 | 11.18 | 11.11 | 11.25 |
| Korny p33, spin-resolved orbital populations | 9.50 | 10.83 | 10.76 | 10.90 |
| Korny p34, optimized Fourier spectrum | 9.72 | 11.07 | 11.09 | 11.05 |
| Korny p35, frozen Fourier spectrum | 9.69 | 11.08 | 11.09 | 11.06 |
| Korny p36, spin-resolved bond populations | 9.39 | 10.77 | 10.68 | 10.85 |
| Korny p37, bond populations and measured force response | 9.40 | 10.75 | 10.61 | 10.89 |
| Korny p38, 12,288 compact local centers | 9.91 | 11.14 | 11.20 | 11.08 |
| Korny p39, environment-dependent X-H springs | 9.39 | 10.77 | 10.63 | 10.91 |
| Korny p40, reactive stationary forces and Hessian-vector products | 9.79 | 11.99 | 12.02 | 11.96 |
| Korny p41, orbital curvature recalibration | 9.38 | 10.79 | 10.64 | 10.94 |
| Korny p42, controlled midpoint refit | 9.41 | 10.72 | 10.58 | 10.85 |
| Korny p43, published spin scans and reaction networks | 9.42 | 10.70 | 10.57 | 10.82 |
| Korny p44, pilot radical IRC transfer and p41 high-level model | 9.42 | 10.74 | 10.61 | 10.88 |
| Korny p46, pilot parent energy and force transfer | 9.47 | 10.77 | 10.67 | 10.87 |
| Korny p48, size-normalized spin contributions | 9.43 | 11.01 | 10.93 | 11.09 |
| Korny p49, atom-resolved spin kernel | 9.36 | 10.67 | 10.55 | 10.79 |
| Original SO3LR | 13.97 | 13.67 | 13.28 | 14.05 |
| Tauri-M-2606 | 6.16 | 8.08 | 8.27 | 7.90 |
| Tauri-L-2606 | 4.48 | 6.02 | 5.98 | 6.06 |

All energy errors are in kcal/mol. The high-level checkpoint scores RDB7, BH9 and TRIP50. The wB97X-D3 checkpoint scores Grambow radicals. These two fitted potential-energy surfaces have different electronic reference levels. Repeated inspection of this benchmark makes it a development panel.

| Relaxed frequency MAE, cm⁻¹ | Propane | Benzene | Ethanol | 1,3-butadiene | Unstable direction, 278 radical saddles | Correct saddle sign |
|---|---:|---:|---:|---:|---:|---:|
| Korny p22 | 91.6 | 91.3 | 102.1 | 94.9 | 991.8 | 202/278 |
| Korny p28 | 86.5 | 186.5 | 150.2 | 174.2 | 804.8 | 208/278 |
| Korny p32 | 106.7 | 190.6 | 112.5 | 160.7 | 804.8 | 208/278 |
| Korny p33 | 89.5 | 188.4 | 157.2 | 184.5 | 801.4 | 211/278 |
| Korny p36 | 89.3 | 201.6 | 154.8 | 186.5 | 796.6 | 212/278 |
| Korny p37 | 107.3 | 200.3 | 105.2 | 163.2 | 796.6 | 212/278 |
| Korny p38 | 150.3 | 154.2 | 118.9 | 242.6 | 806.6 | 212/278 |
| Korny p39 | 91.4 | 222.5 | 90.0 | 163.6 | 796.6 | 212/278 |
| Korny p40 | 115.9 | 213.7 | 159.9 | 156.1 | 809.1 | 211/278 |
| Korny p41 | 89.4 | 151.2 | 104.1 | 156.3 | 795.6 | 212/278 |
| Korny p42 | 108.7 | 203.8 | 99.5 | 170.2 | 794.8 | 212/278 |
| Korny p43 | 103.2 | 220.4 | 117.0 | 177.4 | 796.5 | 213/278 |
| Korny p49 | 89.4 | 151.2 | 104.1 | 156.3 | 795.3 | 212/278 |

The four NIST reference spectra are PBE0/cc-pVTZ. Their predictions use independently relaxed Korny minima. The saddle measurement compares signed wavenumbers along the published Grambow unstable directions. It does not establish the full Hessian index or successful MD barrier crossing.

| Reaction dataset | Training reactions | Common validation reactions |
|---|---:|---:|
| RDB7 | 11,667 | 259 |
| BH9 | 400 | 47 |
| Grambow radicals | 1,084 | 283 |
| TRIP50 | 46 | 2 |
| **Total** | **13,197** | **591** |

The PBE0 backbone used 91,156 unique training geometries. There were 39,592 QM7-X 8000, 11,407 DES15K, 30,599 SPICE dipeptide and 9,558 TorsionNet geometries. The equilibrium constraints used 14 training minima. The radical derivative training used 3,234 stationary structures. Korny supports 595 reaction validation IDs, of which 591 also have original SO3LR and Tauri predictions. [Full split and inventory](full-public-fit.md)

### What changed in the new fits

- P28 recovered Hirshfeld partial charges for 545,198 QM7-X training atoms and formal-charge, radical and pi-valence labels on 11,835 RDB7 and 675 Grambow training endpoints. Its auxiliary predictor reached 0.0159 e Hirshfeld charge MAE on 51,308 development atoms and identified the labeled atom in 333 of 467 single-radical development structures.
- P29 used complete projected Hessians on 14 training minima. P30 added competing-bond features and extra signed-curvature supervision. Both had higher errors than p28 on the common energy panel. P31 combined spin-conditioned priors with 11,667 paired RDB7 DFT and coupled-cluster reactions. Its 14.51 kcal/mol reaction and 16.71 kcal/mol barrier MAEs ruled it out for this benchmark.
- P32 estimated local force responses from 392 QM7-X training conformers using their 39,592 bare PBE0 force geometries. On withheld distortions of those conformers, the median explained force variation was 83%. The fit retained 995 directions from 261 conformers and calibrated its high-level surface. Its radical surface is p28.
- P33 uses molecular charge and multiplicity to fill s and p orbitals in a smooth empirical matrix. The kernel sees atomic populations of occupied, unpaired and frontier orbitals. The model uses a single molecular chemical potential per spin. Ionic dissociation and long-range charge transfer still require checking before reactive MD use.

### Reaction gaps and transition-state midpoint heights

Energy errors are in kcal/mol on the same 591 complete common reaction IDs. The midpoint is the mean of the two directed barriers. Nearly all reference cycles close exactly, with discrepancies up to 0.01 kcal/mol from rounded reference values.

| Model | Reaction | Midpoint height | Pooled barriers | Forward | Reverse |
|---|---:|---:|---:|---:|---:|
| Korny p37 | 9.40 | 9.41 | 10.75 | 10.61 | 10.89 |
| Korny p41 | 9.38 | 9.50 | 10.79 | 10.64 | 10.94 |
| Korny p42 | 9.41 | 9.36 | 10.72 | 10.58 | 10.85 |
| Korny p43 | 9.42 | 9.36 | 10.70 | 10.57 | 10.82 |
| Korny p44 | 9.42 | 9.43 | 10.74 | 10.61 | 10.88 |
| Korny p46 | 9.47 | 9.44 | 10.77 | 10.67 | 10.87 |
| Korny p48 | 9.43 | 9.79 | 11.01 | 10.93 | 11.09 |
| Korny p49 | 9.36 | 9.38 | 10.67 | 10.55 | 10.79 |
| Original SO3LR | 13.97 | 11.84 | 13.67 | 13.28 | 14.05 |
| Tauri-M-2606 | 6.16 | 7.35 | 8.08 | 8.27 | 7.90 |
| Tauri-L-2606 | 4.48 | 5.50 | 6.02 | 5.98 | 6.06 |
- P34 optimized 10,752 Fourier frequency parameters using training reactions and P35 kept the same initial frequencies fixed. Both then fitted the linear energy readout. Optimizing the frequencies improved pooled barrier MAE by 0.004 kcal/mol and increased reaction MAE by 0.026 kcal/mol relative to the fixed arm.
- P37 applied the PBE0 force-response calibration to p36, using the same 995 training directions from 261 conformers. This reduced the ethanol and butadiene errors while leaving the pooled energy MAEs close to p36.
- The p38 compact kernel added 12,288 local centers to p36. Its pooled barrier MAE rose from 10.77 to 11.14 kcal/mol on the common panel. Full finite-difference and checkpoint reload checks agreed with the stored energy and analytic forces.
- P42 reproduces both p37 readouts with weights 48 for the midpoint and 36 for the reaction gap. Its only planned change doubles the midpoint weight to 96. The force, curvature, ridge, data and basis settings retain their p37 values.
- P40 calibrates Cartesian forces and full force changes along four reference vibrational modes at 576 archived training stationary points. The selection includes 144 minima and 144 transition states from each of RDB7 and Grambow. The four added calibration folds group structures by molecular formula.
- The p39 length and spring archive contains 67,156 C–H, 5,292 N–H and 3,277 O–H measurements from 12,051 RDB7 training endpoints. Training-formula holdouts gave equilibrium-length errors of 0.0034, 0.0034 and 0.0026 Å, respectively. The correction was calibrated against 1,442 direct force constants at 496 sampled endpoints.
- P36 adds rotation-invariant measures of electron sharing from off-diagonal orbital density matrix blocks. It distinguishes bond sharing along the internuclear axis, perpendicular contributions and unpaired-electron sharing.

### Paired changes on validation formulas

The intervals resample transition-state formulas within each source, keeping all directions of a reaction together and fixing the source weights. Values are 95% bootstrap intervals for the change in MAE, in kcal/mol. Negative values favor the first model.

| Comparison | Reaction change | Pooled barrier change |
|---|---:|---:|
| p33 minus p28 | -0.44 to -0.08 | -0.50 to -0.22 |
| p34 minus p35 | -0.07 to +0.12 | -0.07 to +0.04 |
| p36 minus p33 | -0.21 to -0.01 | -0.14 to +0.00 |
| p37 minus p36 | -0.05 to +0.06 | -0.08 to +0.08 |
| p38 minus p36 | -0.38 to +1.28 | -0.06 to +0.68 |
| p39 minus p37 | -0.01 to +0.00 | -0.00 to +0.09 |
| p40 minus p37 | -0.01 to +0.91 | +0.38 to +2.54 |
| p41 minus p37 | -0.14 to +0.14 | -0.06 to +0.12 |
| p42 minus p37 | -0.03 to +0.07 | -0.06 to +0.01 |
| p43 minus p42 | -0.06 to +0.10 | -0.07 to +0.03 |
| p44 minus p43 | -0.12 to +0.15 | -0.05 to +0.13 |
| p46 minus p44 | -0.02 to +0.14 | +0.01 to +0.06 |
| p48 minus p44 | -0.37 to +0.48 | -0.03 to +0.68 |
| p49 minus p43 | -0.24 to +0.13 | -0.17 to +0.11 |
| p49 minus p44 | -0.17 to +0.02 | -0.18 to +0.02 |
| p49 minus p48 | -0.57 to +0.33 | -0.77 to -0.03 |

### Frequency bands for p37 at its relaxed minima

| Molecule | Below 800 cm⁻¹ | 800–1,800 cm⁻¹ | Above 2,600 cm⁻¹ | Imaginary internal frequencies |
|---|---:|---:|---:|---:|
| propane | 61.6 (4) | 58.9 (15) | 220.8 (8) | 0 |
| benzene | 32.9 (6) | 190.4 (18) | 397.5 (6) | 0 |
| ethanol | 22.4 (3) | 84.5 (12) | 188.0 (6) | 0 |
| 1,3-butadiene | 106.3 (5) | 75.2 (13) | 401.3 (6) | 0 |

Each cell gives the MAE in cm⁻¹ followed by the number of modes. Modes are paired by sorted frequency. Benzene’s six C–H stretches are collectively too soft by 397 cm⁻¹ on average.

### Full Hessians at eight archived stationary geometries

These geometries were selected before evaluating p37. The reference Hessians come from the archived DFT calculations. The RDB7 high-level energy model and the archived frequency reference use different electronic levels. The Grambow radical model and its reference use the same level. Frequencies are compared at the original geometries. Korny forces there are appreciable, so these spectra do not describe reoptimized Korny stationary points. Counts use a threshold of −50 cm⁻¹.

| Source and geometry | Role | Reference unstable modes | Predicted unstable modes | Spectrum MAE (cm⁻¹) | Maximum predicted force (eV/Å) |
|---|---|---:|---:|---:|---:|
| RDB7 p001245_0 | minimum | 0 | 0 | 98 | 2.18 |
| RDB7 r005516 | minimum | 0 | 2 | 138 | 1.21 |
| RDB7 ts003164 | saddle | 1 | 2 | 210 | 6.03 |
| RDB7 ts008349 | saddle | 1 | 6 | 816 | 5.67 |
| Grambow ts001126.log | saddle | 1 | 4 | 217 | 3.25 |
| Grambow r000569.log | minimum | 0 | 4 | 166 | 3.41 |
| Grambow p000952.log | minimum | 0 | 6 | 214 | 3.39 |
| Grambow ts001137.log | saddle | 1 | 4 | 229 | 1.35 |

At RDB7 ts008349, the reference curvature along the most unstable p37 mode is +46.1 eV/Å². At a displacement of 0.00025 Å, the model energy gives -128.4 eV/Å² and its force derivative -128.4 eV/Å². The orbital-moment kernel alone contributes -175.8 eV/Å². This singles out one kernel contribution at one structure, with the same direction and coordinates throughout.

### P40 on the same eight Hessian validation geometries

| Source and geometry | Reference unstable | P37 unstable | P40 unstable | P37 frequency MAE (cm⁻¹) | P40 frequency MAE (cm⁻¹) |
|---|---:|---:|---:|---:|---:|
| RDB7 p001245_0 | 0 | 0 | 0 | 98 | 82 |
| RDB7 r005516 | 0 | 2 | 2 | 138 | 143 |
| RDB7 ts003164 | 1 | 2 | 2 | 210 | 214 |
| RDB7 ts008349 | 1 | 6 | 7 | 816 | 839 |
| Grambow ts001126.log | 1 | 4 | 4 | 217 | 216 |
| Grambow r000569.log | 0 | 4 | 3 | 166 | 149 |
| Grambow p000952.log | 0 | 6 | 6 | 214 | 216 |
| Grambow ts001137.log | 1 | 4 | 4 | 229 | 225 |

[Per-reaction errors and reference energies for p49](../dataset_for_training/requested_public/so3lr-tauri-matched/p49-public-comparison.csv) · [Validation statistics](../dataset_for_training/requested_public/so3lr-tauri-matched/p49-public-summary.json)

<!-- end current fitting results -->

The same 591 reaction IDs have predictions from every method. Each reaction has one endpoint energy difference and two barriers. Korny includes D3(BJ) with ATM and uses the high-level readout for RDB7, BH9 and TRIP50, and the wB97X-D3-aligned readout for Grambow. The original fixed split and the two grouped inner development folds are retained. Repeated use of the outer benchmark supports tracking progress. A final prediction claim requires testing on fresh reactions.

| Model | Reactions | Reaction MAE | All barriers | Forward | Reverse |
|---|---:|---:|---:|---:|---:|
| Korny p11, combined Fourier kernels | 591 | 11.35 | 14.24 | 14.20 | 14.28 |
| Korny p12, longer-range residual | 591 | 11.18 | 13.88 | 13.72 | 14.04 |
| Korny p13, two-hop bond residual | 591 | 11.28 | 13.58 | 13.54 | 13.62 |
| Original SO3LR | 591 | 13.97 | 13.67 | 13.28 | 14.05 |
| Tauri-M-2606 | 591 | 6.16 | 8.08 | 8.27 | 7.90 |
| Tauri-L-2606 | 591 | 4.48 | 6.02 | 5.98 | 6.06 |

Energy MAEs are in kcal/mol. The common panel contains 259 RDB7, 47 BH9, 283 Grambow and two TRIP50 reactions. The training counts were 11,667 RDB7, 400 BH9, 1,084 Grambow and 46 TRIP50. The 91,156 PBE0 training geometries retained energy and projected force constraints. Fourteen NIST training minima and 3,234 radical training structures supplied equilibrium and curvature constraints.

## Korny p13 across all supported IDs

| Panel | Reactions | Reaction MAE | All barriers | Forward | Reverse |
|---|---:|---:|---:|---:|---:|
| RDB7 | 259 | 11.35 | 14.87 | 14.53 | 15.20 |
| BH9 | 49 | 18.12 | 17.12 | 15.43 | 18.81 |
| Grambow | 283 | 9.74 | 11.65 | 12.16 | 11.14 |
| TRIP50 | 4 | 42.92 | 21.90 | 19.83 | 23.96 |
| priority | 591 | 11.14 | 13.51 | 13.47 | 13.56 |
| all | 595 | 11.35 | 13.57 | 13.51 | 13.63 |

The priority row covers RDB7, BH9 and Grambow with all 49 Korny-supported BH9 channels. Its 591 IDs differ from the four-method common panel.

## Relaxed geometries and harmonic frequencies

| Molecule | p11 MAE, cm⁻¹ | p12 MAE, cm⁻¹ | p13 MAE, cm⁻¹ | p13 bond MAE, Å | p13 angle MAE, degrees |
|---|---:|---:|---:|---:|---:|
| propane | 87.9 | 89.9 | 105.5 | 0.0074 | 2.45 |
| benzene | 93.0 | 95.9 | 120.0 | 0.0076 | 0.25 |
| ethanol | 85.7 | 88.0 | 116.9 | 0.0082 | 1.97 |
| 1,3-butadiene | 122.7 | 116.4 | 111.8 | 0.0147 | 3.62 |

All four optimized structures retained the archived bond connectivity. The reference spectra are PBE0/cc-pVTZ at NIST equilibrium geometries. Korny frequencies come from its own relaxed geometries.

| Korny checkpoint | Endpoint lowest-mode MAE, cm⁻¹ | Endpoint highest-mode MAE, cm⁻¹ | Saddle imaginary-mode MAE, cm⁻¹ | Saddle highest-mode MAE, cm⁻¹ | Correct saddle sign / 278 |
|---|---:|---:|---:|---:|---:|
| p11 | 86.6 | 52.2 | 808.1 | 59.7 | 215 |
| p12 | 87.2 | 52.3 | 810.2 | 58.9 | 214 |
| p13 | 86.7 | 69.7 | 791.4 | 83.1 | 215 |

Grambow curvatures sample two archived eigenvectors at each of 566 held-out endpoints and 278 saddles. They measure the predicted Hessian along those directions.

The p12 fit added a 4.2 Å atomic kernel. The p13 fit added 1,464 local Gaussian centers on coordination, a shorter distance window sensitive to bond contraction, and second-neighbor coordination. Centers came from training geometries. Two nonlinear lengths and three regularization settings were compared on inner formulas. Both readouts are scalar energies with analytic Cartesian forces. Reloaded checkpoints passed energy and directional-force checks.

[Earlier experiments and comparator protocol](reactive-kernel-experiments.md) · [Single-molecule energy and force timings](energy-force-timing.md) · [Per-reaction matched ledger](../dataset_for_training/requested_public/so3lr-tauri-matched/p13-public-comparison.csv)

## Kernel fits after p13

Every row below uses the same 591 reactions and 1,182 directed barriers. Positive reaction energies mean products above reactants. The forward and reverse barriers use the same transition-state energy. All Korny models include D3(BJ) plus ATM once. Each checkpoint pair uses a high-level surface for RDB7, BH9 and TRIP50 and a separately fitted wB97X-D3 surface for Grambow. The extensively reused validation panel tracks development. Independent evidence will require new reaction families.

| Model | Reaction MAE | All barriers | Forward | Reverse |
|---|---:|---:|---:|---:|
| Korny p12, Longer-range atomic kernel | 11.18 | 13.88 | 13.72 | 14.04 |
| Korny p14, Joint two-hop and four-body fit | 11.58 | 13.58 | 13.32 | 13.84 |
| Korny p15, Common parent across electronic levels | 27.80 | 25.62 | 25.98 | 25.27 |
| Korny p16, Learned bond metric | 11.00 | 13.08 | 12.97 | 13.20 |
| Korny p17, Bond metric with curvature anchors | 11.01 | 13.25 | 13.25 | 13.25 |
| Korny p18, Radial and angular density channels | 10.81 | 12.80 | 12.78 | 12.83 |
| Korny p19, Three-bond chemical environments | 10.70 | 12.52 | 12.53 | 12.52 |
| Korny p20, Ring and bond occupancies | 12.71 | 13.80 | 13.73 | 13.88 |
| Korny p21, Reaction-center participation | 11.12 | 13.15 | 12.94 | 13.36 |
| Korny p22, Joint kernels with radical trajectory forces | 11.42 | 13.46 | 13.59 | 13.34 |
| Korny p23, Out-of-fold kernel weighting | 10.31 | 12.13 | 12.06 | 12.20 |
| Korny p24, Independent kernels with full Hessian constraints | 10.27 | 11.82 | 11.78 | 11.86 |
| Korny p25, Periodic-group polynomial kernel | 10.03 | 11.48 | 11.42 | 11.54 |
| Korny p26, Auxiliary DFT-energy interpolation | 10.20 | 11.72 | 11.65 | 11.78 |
| Korny p27, Local orbital moments | 9.96 | 11.30 | 11.35 | 11.25 |
| Korny p28, Charge and radical environments | 9.76 | 11.20 | 11.19 | 11.21 |
| Korny p29, Joint electronic features and full Hessians | 10.44 | 12.53 | 12.52 | 12.54 |
| Korny p30, Bond competition and signed saddle curvature | 10.40 | 12.26 | 12.42 | 12.11 |
| Korny p31, Paired electronic reference levels | 14.51 | 16.71 | 16.97 | 16.44 |
| Korny p32, Measured PBE0 force changes | 9.75 | 11.18 | 11.11 | 11.25 |
| Korny p33, Spin-resolved orbital occupations | 9.50 | 10.83 | 10.76 | 10.90 |
| Korny p34, Optimized Fourier frequencies | 9.72 | 11.07 | 11.09 | 11.05 |
| Korny p35, Frozen Fourier frequencies | 9.69 | 11.08 | 11.09 | 11.06 |
| Korny p36, Spin-resolved bond populations | 9.39 | 10.77 | 10.68 | 10.85 |
| Korny p37, Bond populations with measured PBE0 force response | 9.40 | 10.75 | 10.61 | 10.89 |
| Korny p38, 12,288 compact local centers | 9.91 | 11.14 | 11.20 | 11.08 |
| Korny p39, Environment-dependent X-H springs | 9.39 | 10.77 | 10.63 | 10.91 |
| Korny p40, Reactive stationary forces and Hessian-vector products | 9.79 | 11.99 | 12.02 | 11.96 |
| Korny p42, Controlled doubling of the midpoint weight | 9.41 | 10.72 | 10.58 | 10.85 |
| Original SO3LR | 13.97 | 13.67 | 13.28 | 14.05 |
| Tauri-M-2606 | 6.16 | 8.08 | 8.27 | 7.90 |
| Tauri-L-2606 | 4.48 | 6.02 | 5.98 | 6.06 |

All energy errors are in kcal/mol. The public reaction training set contains 11,667 RDB7, 400 BH9, 1,084 Grambow and 46 TRIP50 reactions. The 91,156 PBE0 backbone training geometries comprise 39,592 QM7-X 8000, 11,407 DES15K, 30,599 SPICE dipeptide and 9,558 TorsionNet structures. The fit also uses 14 training minima with frequencies and 3,234 radical structures with derivative observations. Each candidate uses the original validation IDs. Kernel choices and objective weights use two inner formula folds. The public model excludes the Leuven data following the earlier controlled inclusion test. [Split and training inventory](full-public-fit.md)

### Per-dataset energy and barrier comparison

| Dataset | Common reactions | Model | Reaction MAE | All barriers | Forward | Reverse |
|---|---:|---|---:|---:|---:|---:|
| RDB7 | 259 | Korny p37 | 7.72 | 9.53 | 8.94 | 10.11 |
| RDB7 | 259 | Korny p42 | 7.75 | 9.53 | 8.95 | 10.11 |
| RDB7 | 259 | Original SO3LR | 13.59 | 11.38 | 12.12 | 10.64 |
| RDB7 | 259 | Tauri-M-2606 | 6.67 | 9.69 | 10.15 | 9.24 |
| RDB7 | 259 | Tauri-L-2606 | 5.41 | 6.90 | 6.93 | 6.88 |
| BH9 | 47 | Korny p37 | 17.20 | 16.36 | 15.58 | 17.14 |
| BH9 | 47 | Korny p42 | 17.26 | 16.37 | 15.52 | 17.22 |
| BH9 | 47 | Original SO3LR | 15.67 | 13.77 | 15.64 | 11.90 |
| BH9 | 47 | Tauri-M-2606 | 7.20 | 8.45 | 9.22 | 7.69 |
| BH9 | 47 | Tauri-L-2606 | 4.99 | 7.13 | 7.12 | 7.13 |
| Grambow | 283 | Korny p37 | 9.17 | 10.74 | 11.16 | 10.33 |
| Grambow | 283 | Korny p42 | 9.16 | 10.68 | 11.10 | 10.25 |
| Grambow | 283 | Original SO3LR | 13.88 | 15.71 | 13.97 | 17.46 |
| Grambow | 283 | Tauri-M-2606 | 5.49 | 6.55 | 6.42 | 6.69 |
| Grambow | 283 | Tauri-L-2606 | 3.53 | 5.04 | 4.94 | 5.14 |
| TRIP50 | 2 | Korny p37 | 74.84 | 37.42 | 31.27 | 43.56 |
| TRIP50 | 2 | Korny p42 | 74.63 | 37.31 | 31.29 | 43.33 |
| TRIP50 | 2 | Original SO3LR | 34.31 | 17.15 | 10.87 | 23.44 |
| TRIP50 | 2 | Tauri-M-2606 | 9.75 | 6.91 | 3.74 | 10.08 |
| TRIP50 | 2 | Tauri-L-2606 | 7.13 | 3.56 | 3.18 | 3.95 |

### Relaxed NIST frequencies and transition-state curvature

| Korny checkpoint | Propane | Benzene | Ethanol | 1,3-butadiene | Grambow saddle unstable mode | Correct saddle sign |
|---|---:|---:|---:|---:|---:|---:|
| p12 | 89.9 | 95.9 | 88.0 | 116.4 | 810.2 | 214/278 |
| p14 | 68.3 | 80.3 | 772.9 | 76.6 | 892.6 | 207/278 |
| p15 | 166.2 | 81.0 | 133.2 | 85.6 | 902.3 | 215/278 |
| p16 | 77.3 | 319.8 | 229.1 | 131.5 | 806.7 | 214/278 |
| p17 | 93.5 | 153.6 | 128.9 | 113.3 | 803.0 | 209/278 |
| p18 | 127.8 | 141.8 | 215.4 | 236.0 | 780.6 | 213/278 |
| p19 | 128.7 | 215.6 | 259.7 | 191.6 | 791.1 | 215/278 |
| p20 | 89.9 | 97.8 | 88.0 | 114.7 | 1209.0 | 189/278 |
| p21 | 86.8 | 119.4 | 85.2 | 124.4 | 867.7 | 206/278 |
| p22 | 91.6 | 91.3 | 102.1 | 94.9 | 991.8 | 202/278 |
| p23 | 94.4 | 138.7 | 152.4 | 237.9 | 801.1 | 212/278 |
| p24 | 77.8 | 119.7 | 130.6 | 138.4 | 795.5 | 212/278 |
| p25 | 80.8 | 150.5 | 126.2 | 182.7 | 798.7 | 211/278 |
| p26 | 84.6 | 203.8 | 131.7 | 137.6 | 783.1 | 214/278 |
| p27 | 79.4 | 160.8 | 107.7 | 178.4 | 794.0 | 209/278 |
| p28 | 86.5 | 186.5 | 150.2 | 174.2 | 804.8 | 208/278 |
| p29 | 155.2 | 218.4 | 168.2 | 142.3 | 1000.7 | 199/278 |
| p30 | 129.9 | 227.9 | 171.3 | 182.2 | 987.1 | 202/278 |
| p31 | 219.8 | 827.0 | 486.2 | 364.5 | 818.6 | 205/278 |
| p32 | 106.7 | 190.6 | 112.5 | 160.7 | 804.8 | 208/278 |
| p33 | 89.5 | 188.4 | 157.2 | 184.5 | 801.4 | 211/278 |
| p34 | 87.8 | 182.3 | 149.6 | 175.6 | 810.1 | 208/278 |
| p35 | 88.1 | 180.8 | 146.1 | 177.9 | 809.5 | 207/278 |
| p36 | 89.3 | 201.6 | 154.8 | 186.5 | 796.6 | 212/278 |
| p37 | 107.3 | 200.3 | 105.2 | 163.2 | 796.6 | 212/278 |
| p38 | 150.3 | 154.2 | 118.9 | 242.6 | 806.6 | 212/278 |
| p39 | 91.4 | 222.5 | 90.0 | 163.6 | 796.6 | 212/278 |
| p40 | 115.9 | 213.7 | 159.9 | 156.1 | 809.1 | 211/278 |
| p42 | 108.7 | 203.8 | 99.5 | 170.2 | 794.8 | 212/278 |

Frequency MAEs are in cm⁻¹. NIST compares sorted frequencies after an independent Korny relaxation. The Grambow column measures signed curvature along the published unstable eigenvector at each of 278 held-out reference saddles. It tests a different physical quantity from an equilibrium spectrum.

| p42 relaxed geometry | Bond MAE, Å | Angle MAE, degrees | Aligned RMSD, Å | Connectivity retained |
|---|---:|---:|---:|---|
| propane | 0.0046 | 1.93 | 0.1419 | Yes |
| benzene | 0.0107 | 0.36 | 0.0084 | Yes |
| ethanol | 0.0289 | 2.72 | 0.1160 | Yes |
| 1,3-butadiene | 0.0149 | 3.08 | 0.3198 | Yes |

The three-bond kernel improved energies, while its ethanol frequency error rose to 259.7 cm⁻¹. The gated kernel learned bond participation from 4,478 endpoint and 3,827 saddle geometries outside both development folds. It improved the BH9 pooled barrier MAE to 16.19 kcal/mol and the ethanol frequency MAE to 85.2 cm⁻¹, with a larger pooled barrier error across the common panel.

The Grambow training archive yielded 36,507 distinct optimization geometries within 3,252 training paths, including 35,423 force arrays. All 849 outer validation paths were reserved before extraction. The potential energy along each optimization is referenced to its own stationary structure. These extra observations first enter the p22 fit.

[Latest matched errors by reaction](../dataset_for_training/requested_public/so3lr-tauri-matched/p42-public-comparison.csv) · [Complete validation statistics](../dataset_for_training/requested_public/so3lr-tauri-matched/p42-public-summary.json) · [Relaxed geometries and spectra](../dataset_for_training/requested_public/kernel-trials/p42-public-spectra.json)

[Expanded data, common-ID results and download status](expanded-reference-fits.md)
