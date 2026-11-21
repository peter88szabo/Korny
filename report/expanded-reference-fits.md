# Expanded-data Korny fits

Measured through 2026-11-21T13:37+01:00. The target is 5.0 kcal/mol MAE each for reaction energies and barriers, together with credible equilibrium and transition-state frequencies.

## Latest results

On 626 held RDB19-Rad neutral-doublet paths, p67 reached 4.02 kcal/mol reaction MAE and 3.82 kcal/mol pooled barrier MAE. The force change across the saddle gave 237 cm⁻¹ signed directional error, with the downhill sign correct for 595 of 626 paths. This is a source-specific electronic potential with a fixed dispersion reference.

On the 589 mixed reaction IDs, p89 reached 8.97 kcal/mol for reactions and 10.53 for barriers. The changes from p49 were -0.162 and -0.044 kcal/mol. The residual saddle forces and curvature errors remain important limits for reactive dynamics. [Eigenfunction trials, full per-source comparisons and frequency checks](operator-kernel-benchmarks.md)

The H-transfer fits also have reaction and barrier tables on RDB19-Rad and the held-out AutoMeKin network. [Expanded-source H-transfer benchmarks](hydrogen-transfer-kernels.md)

## Reactions on the fixed common IDs

The active panel contains 259 RDB7, 47 BH9 and 283 Grambow radical reactions. Each row uses the same 589 reactions and 1,178 directed barriers. The midpoint height equals the mean of the two directed barriers. Each Korny calculation includes D3(BJ) and the ATM term once. This repeatedly inspected panel measures development performance. The p49 backbone includes earlier TRIP50 training. The new corrections exclude TRIP50 from fitting and scoring.

| Model | Reaction | Midpoint | All barriers | Forward | Reverse |
|---|---:|---:|---:|---:|---:|
| Korny p37, original curvature fit | 9.17 | 9.42 | 10.66 | 10.54 | 10.78 |
| Korny p41, orbital curvature | 9.16 | 9.51 | 10.70 | 10.57 | 10.83 |
| Korny p42, controlled midpoint refit | 9.19 | 9.38 | 10.63 | 10.51 | 10.74 |
| Korny p43, paper reaction data | 9.20 | 9.38 | 10.61 | 10.50 | 10.71 |
| Korny p44, pilot radical IRCs and p41 high-level surface | 9.20 | 9.44 | 10.65 | 10.54 | 10.77 |
| Korny p45, full radical IRCs and p41 high-level surface | 9.21 | 9.44 | 10.66 | 10.55 | 10.77 |
| Korny p46, pilot AIMNet2 pair transfer | 9.25 | 9.45 | 10.68 | 10.60 | 10.76 |
| Korny p47, full AIMNet2 pair transfer | 9.29 | 9.45 | 10.69 | 10.61 | 10.77 |
| Korny p48, size-dependent spin correction | 9.35 | 9.79 | 10.99 | 10.94 | 11.04 |
| Korny p49, atom-resolved spin kernel | 9.14 | 9.40 | 10.58 | 10.46 | 10.69 |
| Korny p50, network forces and measured normal modes | 9.42 | 9.64 | 10.86 | 10.75 | 10.97 |
| Korny p51, short-range donor and acceptor environments | 9.18 | 9.52 | 10.70 | 10.57 | 10.83 |
| Korny p52, joint calibration of four expanded-data corrections | 9.44 | 9.68 | 10.90 | 10.80 | 11.01 |
| Korny p53, stationary-geometry DFT base and coupled-cluster correction | 9.03 | 9.55 | 10.62 | 10.54 | 10.70 |
| Korny p54, transfer from reoptimized PM6, HF and r²SCAN-3c structures | 9.37 | 9.60 | 10.79 | 10.70 | 10.88 |
| Korny p55, conservative neighbor-to-neighbor spin redistribution | 9.18 | 9.40 | 10.60 | 10.51 | 10.69 |
| Korny p56, multiscale radical-path transfer | 9.14 | 9.41 | 10.59 | 10.48 | 10.70 |
| Korny p57, local environment changes along radical paths | 9.15 | 9.45 | 10.67 | 10.50 | 10.84 |
| Korny p58, Gaussian derivatives along reacting atomic environments | 9.24 | 9.44 | 10.62 | 10.36 | 10.88 |
| Korny p59, full-release multiscale radical-path transfer | 9.12 | 9.40 | 10.58 | 10.46 | 10.69 |
| Korny p60, reaction family inferred from local geometry | 9.22 | 9.43 | 10.62 | 10.54 | 10.70 |
| Korny p61, denser atomic kernel on the full radical archive | 9.16 | 9.40 | 10.58 | 10.46 | 10.70 |
| Korny p62, direct dense two-resolution fits on Korny references | 9.14 | 9.54 | 10.74 | 10.65 | 10.84 |
| Korny p65, dense kernel with 5 kcal/mol Huber gap and midpoint losses | 9.14 | 9.55 | 10.75 | 10.65 | 10.84 |
| Korny p68, doublet transfer from native energy, force and curvature fits | 9.13 | 9.40 | 10.57 | 10.45 | 10.69 |
| Korny p69, Gaussian landmark correction | 9.09 | 9.38 | 10.54 | 10.36 | 10.71 |
| Korny p70, Gaussian correction with additional singlet DFT supervision | 9.52 | 9.41 | 10.78 | 10.54 | 11.01 |
| Korny p71, reaction-family-balanced Gaussian correction | 9.09 | 9.34 | 10.52 | 10.36 | 10.69 |
| Korny p72, variational split-charge response | 9.19 | 9.42 | 10.61 | 10.53 | 10.69 |
| Korny p73, Chebyshev modes | 9.06 | 9.34 | 10.47 | 10.23 | 10.72 |
| Korny p74, graph diffusion modes | 9.22 | 9.42 | 10.60 | 10.43 | 10.77 |
| Korny p75, Hermite oscillator modes | 9.04 | 9.35 | 10.48 | 10.23 | 10.73 |
| Korny p76, Laguerre functions | 9.08 | 9.39 | 10.55 | 10.23 | 10.87 |
| Korny p77, Legendre modes | 9.05 | 9.34 | 10.47 | 10.23 | 10.72 |
| Korny p78, Jacobi modes | 8.96 | 9.43 | 10.58 | 10.35 | 10.82 |
| Korny p79, Dirichlet sine modes | 9.02 | 9.40 | 10.53 | 10.26 | 10.80 |
| Korny p80, Neumann cosine modes | 9.02 | 9.37 | 10.50 | 10.25 | 10.74 |
| Korny p81, radial disk Bessel modes | 9.01 | 9.38 | 10.54 | 10.28 | 10.80 |
| Korny p82, Krawtchouk modes | 9.00 | 9.37 | 10.51 | 10.28 | 10.75 |
| Korny p83, Hahn modes | 9.05 | 9.39 | 10.53 | 10.27 | 10.79 |
| Korny p84, Morse bond stretches and Legendre bends | 9.06 | 9.39 | 10.56 | 10.39 | 10.73 |
| Korny p85, Legendre modes with pairwise coordinate products | 9.20 | 9.42 | 10.60 | 10.36 | 10.84 |
| Korny p86, Chebyshev H-transfer correction with expanded reference data | 9.06 | 9.59 | 10.70 | 10.43 | 10.98 |
| Korny p87, Krawtchouk H-transfer correction with expanded reference data | 9.00 | 9.70 | 10.79 | 10.52 | 11.06 |
| Korny p88, Chebyshev H-transfer correction with a smooth angular window | 9.03 | 9.36 | 10.49 | 10.24 | 10.73 |
| Korny p89, Krawtchouk H-transfer correction with a smooth angular window | 8.97 | 9.41 | 10.53 | 10.31 | 10.76 |
| Original SO3LR | 13.90 | 11.85 | 13.65 | 13.29 | 14.02 |
| Tauri-M-2606 | 6.15 | 7.35 | 8.09 | 8.28 | 7.89 |
| Tauri-L-2606 | 4.47 | 5.52 | 6.03 | 5.99 | 6.06 |

Paired MAE changes for p89 relative to p49 use the same 589 reactions. The 95% bootstrap intervals resample whole transition-state formulas within each source.

| Observable | Change in MAE (kcal/mol) | 95% interval |
|---|---:|---:|
| Reaction | -0.162 | [-0.343, +0.019] |
| All barriers | -0.044 | [-0.212, +0.149] |
| Forward | -0.152 | [-0.343, +0.046] |
| Reverse | +0.064 | [-0.133, +0.313] |

Energy MAEs are in kcal/mol. The high-level surface scores RDB7 and BH9. The ωB97X-D3 surface scores Grambow. [Errors and reference energies by common reaction ID](../dataset_for_training/requested_public/so3lr-tauri-matched/p89-public-active-common.csv)

## Comparison by transition-state multiplicity

| Model | Spin | Reactions | Reaction | All barriers | Forward | Reverse |
|---|---|---:|---:|---:|---:|---:|
| Korny p49 | singlet | 286 | 8.84 | 10.45 | 9.62 | 11.28 |
| Korny p49 | doublet | 303 | 9.42 | 10.69 | 11.25 | 10.14 |
| Korny p53 | singlet | 286 | 8.61 | 10.36 | 9.54 | 11.17 |
| Korny p53 | doublet | 303 | 9.41 | 10.87 | 11.49 | 10.25 |
| Korny p89 | singlet | 286 | 8.73 | 10.39 | 9.40 | 11.38 |
| Korny p89 | doublet | 303 | 9.21 | 10.67 | 11.16 | 10.17 |
| Original SO3LR | singlet | 286 | 13.91 | 11.37 | 12.21 | 10.54 |
| Original SO3LR | doublet | 303 | 13.88 | 15.81 | 14.32 | 17.30 |
| Tauri-M-2606 | singlet | 286 | 7.00 | 9.83 | 10.38 | 9.27 |
| Tauri-M-2606 | doublet | 303 | 5.34 | 6.44 | 6.30 | 6.58 |
| Tauri-L-2606 | singlet | 286 | 5.51 | 7.20 | 7.23 | 7.17 |
| Tauri-L-2606 | doublet | 303 | 3.49 | 4.92 | 4.81 | 5.02 |

## Results by reaction dataset

| Dataset | Model | Reactions | Reaction | All barriers | Forward | Reverse |
|---|---|---:|---:|---:|---:|---:|
| RDB7 | Korny p49 | 259 | 7.84 | 9.68 | 9.07 | 10.29 |
| RDB7 | Korny p52 | 259 | 8.16 | 10.08 | 9.43 | 10.73 |
| RDB7 | Korny p62 | 259 | 7.85 | 9.73 | 9.08 | 10.37 |
| RDB7 | Korny p65 | 259 | 7.98 | 9.80 | 9.18 | 10.42 |
| RDB7 | Korny p89 | 259 | 7.76 | 9.62 | 8.86 | 10.37 |
| RDB7 | Original SO3LR | 259 | 13.59 | 11.38 | 12.12 | 10.64 |
| RDB7 | Tauri-M-2606 | 259 | 6.67 | 9.69 | 10.15 | 9.24 |
| RDB7 | Tauri-L-2606 | 259 | 5.41 | 6.90 | 6.93 | 6.88 |
| BH9 | Korny p49 | 47 | 16.52 | 15.39 | 14.53 | 16.25 |
| BH9 | Korny p52 | 47 | 17.71 | 16.19 | 16.00 | 16.38 |
| BH9 | Korny p62 | 47 | 17.34 | 16.00 | 15.65 | 16.34 |
| BH9 | Korny p65 | 47 | 17.29 | 15.93 | 15.60 | 16.26 |
| BH9 | Korny p89 | 47 | 16.96 | 15.37 | 14.78 | 15.97 |
| BH9 | Original SO3LR | 47 | 15.67 | 13.77 | 15.64 | 11.90 |
| BH9 | Tauri-M-2606 | 47 | 7.20 | 8.45 | 9.22 | 7.69 |
| BH9 | Tauri-L-2606 | 47 | 4.99 | 7.13 | 7.12 | 7.13 |
| Grambow | Korny p49 | 283 | 9.09 | 10.60 | 11.05 | 10.14 |
| Grambow | Korny p52 | 283 | 9.23 | 10.78 | 11.19 | 10.37 |
| Grambow | Korny p62 | 283 | 8.95 | 10.80 | 11.24 | 10.35 |
| Grambow | Korny p65 | 283 | 8.85 | 10.75 | 11.18 | 10.33 |
| Grambow | Korny p89 | 283 | 8.76 | 10.57 | 10.88 | 10.25 |
| Grambow | Original SO3LR | 283 | 13.88 | 15.71 | 13.97 | 17.46 |
| Grambow | Tauri-M-2606 | 283 | 5.49 | 6.55 | 6.42 | 6.69 |
| Grambow | Tauri-L-2606 | 283 | 3.53 | 5.04 | 4.94 | 5.14 |

## Frequency and saddle checks

| Model | Propane | Benzene | Ethanol | 1,3-butadiene | Signed unstable-direction MAE | Correct curvature sign |
|---|---:|---:|---:|---:|---:|---:|
| p37 | 107.3 | 200.3 | 105.2 | 163.2 | 796.6 | 212/278 |
| p41 | 89.4 | 151.2 | 104.1 | 156.3 | 795.6 | 212/278 |
| p42 | 108.7 | 203.8 | 99.5 | 170.2 | 794.8 | 212/278 |
| p43 | 103.2 | 220.4 | 117.0 | 177.4 | 796.5 | 213/278 |
| p44 | 89.4 | 151.2 | 104.1 | 156.3 | 795.5 | 213/278 |
| p45 | 89.4 | 151.2 | 104.1 | 156.3 | 795.1 | 213/278 |
| p46 | 92.5 | 154.8 | 102.8 | 155.1 | 796.9 | 213/278 |
| p47 | 94.7 | 154.0 | 97.7 | 149.7 | 796.3 | 213/278 |
| p48 | 89.4 | 151.2 | 104.1 | 156.3 | 796.7 | 210/278 |
| p49 | 89.4 | 151.2 | 104.1 | 156.3 | 795.3 | 212/278 |
| p50 | 137.3 | 142.4 | 153.2 | 123.6 | 803.8 | 210/278 |
| p51 | 90.6 | 154.3 | 90.1 | 155.5 | 795.9 | 212/278 |
| p52 | 139.1 | 139.2 | 154.7 | 125.1 | 804.0 | 210/278 |
| p53 | 112.7 | 263.1 | 154.7 | 157.5 | 795.3 | 212/278 |
| p54 | 137.8 | 143.2 | 152.2 | 123.3 | 795.3 | 212/278 |
| p55 | 89.4 | 151.2 | 104.1 | 156.3 | 795.9 | 212/278 |
| p56 | 89.4 | 151.2 | 104.1 | 156.3 | 796.1 | 212/278 |
| p57 | 85.0 | Relaxation failed | 117.5 | 192.6 | 796.1 | 212/278 |
| p58 | 104.9 | 173.2 | 125.4 | 163.5 | 796.1 | 213/278 |
| p59 | 89.4 | 151.2 | 104.1 | 156.3 | 796.6 | 212/278 |
| p60 | 89.4 | 151.2 | 104.1 | 156.3 | 796.4 | 212/278 |
| p61 | 89.4 | 151.2 | 104.1 | 156.3 | 794.5 | 212/278 |
| p62 | 87.8 | 148.8 | 129.5 | 135.6 | 798.3 | 211/278 |
| p65 | 89.4 | 132.3 | 124.3 | 126.4 | 797.1 | 211/278 |
| p68 | 89.4 | 151.2 | 104.1 | 156.3 | 794.6 | 212/278 |
| p69 | 89.9 | 159.2 | 112.2 | 161.1 | 793.8 | 213/278 |
| p70 | 89.9 | 159.2 | 112.2 | 161.1 | 796.8 | 211/278 |
| p71 | 92.0 | 156.1 | 111.2 | 159.7 | 794.3 | 212/278 |
| p72 | 90.0 | 148.5 | 101.9 | 153.7 | 796.4 | 211/278 |
| p73 | 93.0 | 170.7 | 110.1 | 155.6 | 795.7 | 212/278 |
| p74 | 89.1 | 158.6 | 111.7 | 159.5 | 794.2 | 212/278 |
| p75 | 94.2 | 166.5 | 121.4 | 150.4 | 795.1 | 212/278 |
| p76 | 89.1 | 160.8 | 119.7 | 159.0 | 795.3 | 212/278 |
| p77 | 93.7 | 169.2 | 112.6 | 155.5 | 795.6 | 212/278 |
| p78 | 94.8 | 161.9 | 125.2 | 159.2 | 795.0 | 213/278 |
| p79 | 88.7 | 166.1 | 117.0 | 148.5 | 794.5 | 213/278 |
| p80 | 94.5 | 158.5 | 119.9 | 149.9 | 794.0 | 213/278 |
| p81 | 93.5 | 151.6 | 121.1 | 152.1 | 793.8 | 213/278 |
| p82 | 93.1 | 157.4 | 107.6 | 148.4 | 794.3 | 213/278 |
| p83 | 93.0 | 169.7 | 115.2 | 153.5 | 795.2 | 212/278 |
| p84 | 90.3 | 178.9 | 113.6 | 169.6 | 797.9 | 210/278 |
| p85 | 94.7 | 184.6 | 120.8 | 158.2 | 797.3 | 211/278 |
| p86 | 93.8 | 188.5 | 115.5 | 167.8 | 799.8 | 212/278 |
| p87 | 94.6 | 177.3 | 111.5 | 161.9 | 806.6 | 212/278 |
| p88 | 93.0 | 170.7 | 110.1 | 155.6 | 794.4 | 211/278 |
| p89 | 93.1 | 157.4 | 107.6 | 148.4 | 795.9 | 213/278 |

All frequency MAEs are in cm⁻¹. The first four columns compare independently relaxed Korny minima with PBE0/cc-pVTZ. P44 inherits the p41 high-level surface. P48 and p49 inherit its singlet spectra because their spin corrections vanish at multiplicity one. The radical column measures signed curvature along one archived mode at each of 278 saddles.

| Relaxed geometry, p89 | Bond MAE (Å) | Angle MAE (°) | Aligned RMSD (Å) | Connectivity retained | Imaginary internal modes |
|---|---:|---:|---:|---|---:|
| propane | 0.0050 | 2.08 | 0.190 | yes | 0 |
| benzene | 0.0105 | 0.33 | 0.008 | yes | 0 |
| ethanol | 0.0250 | 2.48 | 0.097 | yes | 0 |
| 1,3-butadiene | 0.0121 | 2.92 | 0.320 | yes | 0 |

| Full Hessian at reference geometry | Reference unstable modes | p37 | p41 |
|---|---:|---:|---:|
| RDB7/rxn003164/ts003164 | 1 | 2 | 2 |
| RDB7/rxn008349/ts008349 | 1 | 6 | 6 |
| wb97xd3/rxn001126/ts001126.log | 1 | 4 | 3 |
| wb97xd3/rxn001137/ts001137.log | 1 | 4 | 3 |

The unstable-mode count uses a −50 cm⁻¹ threshold. All four p41 saddles retain excess negative curvature. The models also have residual forces at these archived geometries. Reactive MD and saddle optimization therefore require further validation.

## Training material

| Dataset | Training material | Reserved validation |
|---|---|---|
| Korny bare PBE0 backbone | 91,156 unique geometries across QM7-X 8000, DES15K, SPICE dipeptides and TorsionNet | 9,603 geometries |
| RDB7 / BH9 / Grambow | 11,667 / 400 / 1,084 training reactions | 259 / 47 / 283 common validation reactions |
| AIMNet2-NSE DFT supplement | 20,038 geometry/state observations, including 241 doublets and 8,531 triplets | 764 singlet/triplet benzopinacol frames |
| BASChem19 | 15 forward barriers | 4 forward barriers |
| AutoMeKin connected electronic channels | 538 DFT, 446 local CC and 51 canonical CC channels | 329 DFT and 265 local CC channels on the complete cBD-CN network |
| AutoMeKin separated fragments | 62 additional CC channels with independently identified charged/spin-labelled components | 8 additional CC channels on cBD-CN |
| RDB19-Rad p44 | 71 doublet reactions, 36,934 distinct path positions across both directions | 6 reactions, 2,840 positions |
| RDB19-Rad p45 | 4,974 doublet reactions, 2,810,057 distinct path positions across both directions | 626 reactions, 366,242 positions |
| AIMNet2 closed-shell pilot | 652,640 geometries | 45,926 formula-held geometries, 320,618 atoms |
| AIMNet2 closed-shell full | 18,882,262 geometries | 1,226,303 formula-held geometries, 27,746,317 atoms |
| p55 atom-resolved spin redistribution | 121,358 Ions-16 doublets and 656 neutral Lewis assignments | Formula-held ions, ChEMBL-20 ions and neutral radical centers |
| AutoMeKin DFT normal modes | 2,067 stationary points, 55,371 signed modes | 1,026 cBD-CN structures, 24,624 modes |

The AIMNet2-NSE paper describes 13 million additional open-shell configurations and 200,000 reaction profiles. Its public links provided the five-system DFT supplement, BASChem19 and trained weights. A public download for the complete open-shell training extension is still unlocated. The original AIMNet2 closed-shell ωB97M parent has 20,108,565 geometries and 474,624,017 atoms. [AIMNet2-NSE article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12851018/) · [AIMNet2 parent archive](https://doi.org/10.1184/R1/27629937.v2) · [BASChem19](https://doi.org/10.1184/R1/29710664.v1)

The RDB19-Rad import uses electronic energies and Cartesian forces from accepted Gaussian IRC points. Training uses path-relative energies and force components along short Cartesian displacements. The authors’ gas/solution Gibbs energies have separate fields. The two directions share a single transition state, counted once per retained reaction. The full deposit has 5,600 reaction IDs and 75 reaction-class labels. [RDB19-Rad data](https://zenodo.org/records/11493786)

A separated AutoMeKin product enters the CC fit only when all fragment ORCA outputs supply charge, multiplicity, coordinates and energy. The importer verifies each electronic energy against the CSV and checks atom and charge balance across the channel. MEA uses canonical CCSD(T)-F12 and the four other networks use DLPNO-CCSD(T)-F12. [AutoMeKin archive](https://zenodo.org/records/22808731) · [Imported electronic channels and coverage](../dataset_for_training/paper_references/fragment-channels.json)

## Radical transfer beyond the original panel

| External set and model | Reactions | Reaction | All barriers | Forward | Reverse |
|---|---:|---:|---:|---:|---:|
| RDB19-Rad 44 holdout, p43 | 6 | 12.10 | 12.34 | 14.47 | 10.20 |
| RDB19-Rad 44 holdout, p44 | 6 | 11.95 | 12.40 | 14.53 | 10.26 |
| RDB19-Rad 45 holdout, p43 | 626 | 19.21 | 20.23 | 21.12 | 19.35 |
| RDB19-Rad 45 holdout, p45 | 626 | 19.23 | 20.24 | 21.14 | 19.33 |
| RDB19-Rad 44 holdout, p49 | 6 | 11.94 | 13.02 | 15.45 | 10.58 |
| cBD-CN CC connected minima, p44 | 265 | 11.05 | 22.35 | 23.32 | 21.39 |
| cBD-CN CC connected minima, p48 | 265 | 10.77 | 12.92 | 13.06 | 12.78 |
| cBD-CN CC separated fragments, p44 | 8 | 10.51 | 17.64 | 21.38 | 13.91 |
| cBD-CN CC separated fragments, p48 | 8 | 11.32 | 16.61 | 19.34 | 13.89 |

RDB19-Rad assignments match both terminal IRC geometries to the authors’ educt and product complex XYZ files. Values describe electronic barriers measured from these complexes. The 265 connected and eight fragmented cBD-CN channels all belong to one held-out chemical network.

| Benzopinacol profile MAE, kcal/mol | Frames | p44 | p48 |
|---|---:|---:|---:|
| benzopinacol | 764 | 835.2 | 71.3 |
| benzopinacol within m=1 | 369 | 61.6 | 61.6 |
| benzopinacol within m=3 | 395 | 125.8 | 141.2 |

The complete scan uses the first singlet geometry as its energy origin. Each within-spin row uses its own first geometry. P48 normalizes three global spin contributions by the heavy-atom count and fits their amplitudes using training-only spin-state and reaction evidence. The cross-spin offset improved while the held-out triplet path remained inaccurate. [Spin fit and force checks](../experiments/full_public/trial_spin_extent.py)

| Held-out cBD-CN reference, p49 | Channels | Reaction | All barriers | Forward | Reverse |
|---|---:|---:|---:|---:|---:|
| separated fragments | 8 | 10.67 | 16.91 | 21.26 | 12.55 |
| connected minima | 265 | 9.91 | 19.17 | 19.61 | 18.73 |

Across both benzopinacol spin branches, the p49 MAE is 74.5 kcal/mol relative to the first singlet frame.

## Local chemical environments from vertical electronic energies

P51 uses environment scores learned from vertical ionization energies and electron affinities to modulate short-range pair interactions. The correction and its derivative go to zero between separated fragments. Both Korny energy surfaces retain their original reaction-gap, midpoint, force and curvature supervision.

| cBD-CN coupled-cluster channels | Model | Reactions | Reaction | All barriers | Forward | Reverse |
|---|---|---:|---:|---:|---:|---:|
| separated | p49 | 8 | 10.67 | 16.91 | 21.26 | 12.55 |
| separated | p51 | 8 | 10.60 | 17.03 | 21.51 | 12.56 |
| connected | p49 | 265 | 9.91 | 19.17 | 19.61 | 18.73 |
| connected | p51 | 265 | 9.84 | 19.15 | 19.60 | 18.70 |

The benzopinacol scan MAE across 764 frames is 73.8 kcal/mol for p51 and 74.5 for p49. [Kernel](../src/redox_environment.py) · [Reaction refit](../experiments/full_public/trial_redox_fit.py)

## Joint fit with the expanded archives

P52 calibrates four corrections per electronic reference. The native radical data supply 4,974 training reactions. AutoMeKin supplies 2,067 training stationary points. The AIMNet2 contribution transfers a pair-energy model estimated on the closed-shell parent archive. Ions-16 supplies vertical redox information.

| Target surface | Additional full-release radical IRC correction | AIMNet2 pair correction | AutoMeKin Hessian correction | Ions-16 pair correction |
|---|---:|---:|---:|---:|
| high | 0.000 | 0.226 | 0.984 | 0.272 |
| wb97xd3 | 0.461 | 0.089 | 1.000 | 0.394 |

The following p52 results use the held cBD-CN network. The high-level branch uses coupled-cluster reference channels. The second branch is compared with the DFT channels.

| Target | Channels | Reactions | Reaction | All barriers | Forward | Reverse |
|---|---|---:|---:|---:|---:|---:|
| high | separated | 8 | 10.73 | 16.44 | 20.17 | 12.72 |
| high | connected | 265 | 9.50 | 18.49 | 18.77 | 18.22 |
| wb97xd3 | connected | 329 | 13.06 | 11.43 | 13.47 | 9.40 |

## Transfer of the native force and curvature fits

P68 learns three coefficients on the Grambow training reactions and their stationary-point derivative constraints. The three inputs are scalar functions of one doublet geometry. They use the p61 native energy, the p64 minus p61 energy, and the p67 minus p64 energy. The last change includes both the measured saddle curvatures and the corresponding retuning of the Cartesian force weight. The high-level branch and relaxed singlet spectra come from p49. The target calculation includes explicit D3(BJ) and ATM once.

| RDB19-Rad contribution | Coefficient on the Grambow surface |
|---|---:|
| Native path energy | -0.00136 |
| Force-trained increment | -0.00898 |
| Curvature-trained increment | -0.00075 |

## AutoMeKin force and curvature fit

P50 takes two measured Hessian-vector products and a zero-force constraint per stationary point. One probe follows the softest mode. The second combines all remaining modes. The source DFT frequencies agree with the independent stationary-point table and the mass-weighted displacements are orthogonal. P50 uses the DFT forces and modes as auxiliary targets for its coupled-cluster energy surface. All cBD-CN channels and structures stayed outside model fitting and parameter selection.

| cBD-CN electronic reference | Model | Reactions | Reaction | All barriers | Forward | Reverse |
|---|---|---:|---:|---:|---:|---:|
| DLPNO-CCSD(T)-F12, connected | p49 | 265 | 9.91 | 19.17 | 19.61 | 18.73 |
| DLPNO-CCSD(T)-F12, connected | p50 | 265 | 9.50 | 18.49 | 18.75 | 18.22 |
| DLPNO-CCSD(T)-F12, separated | p49 | 8 | 10.67 | 16.91 | 21.26 | 12.55 |
| DLPNO-CCSD(T)-F12, separated | p50 | 8 | 10.82 | 16.42 | 20.13 | 12.71 |
| ωB97X-D3, connected | p49 | 329 | 13.86 | 11.79 | 14.16 | 9.42 |
| ωB97X-D3, connected | p50 | 329 | 13.05 | 11.44 | 13.47 | 9.41 |

| Korny surface | cBD-CN structures | Count | Model | Soft-mode MAE (cm⁻¹) | Correct sign | Cartesian force MAE (eV/Å) |
|---|---|---:|---|---:|---:|---:|
| high | minimum | 389 | p49 | 341.2 | 96.9% | 2.119 |
| high | minimum | 389 | p50 | 328.9 | 96.4% | 2.032 |
| high | saddle | 637 | p49 | 1732.7 | 35.3% | 2.720 |
| high | saddle | 637 | p50 | 1709.3 | 35.5% | 2.614 |
| wb97xd3 | minimum | 389 | p49 | 107.5 | 88.4% | 0.511 |
| wb97xd3 | minimum | 389 | p50 | 106.5 | 90.0% | 0.517 |
| wb97xd3 | saddle | 637 | p49 | 747.0 | 53.7% | 0.614 |
| wb97xd3 | saddle | 637 | p50 | 751.6 | 54.2% | 0.619 |

The soft-mode measurement projects the predicted Hessian along the lowest published DFT eigenvector at the archived DFT geometry. Checking the number of unstable eigenvectors requires the full model Hessian. [Mode importer](../experiments/full_public/automekin_modes.py) · [Fitting and evaluation](../experiments/full_public/trial_network_hessians.py)

## Atom-resolved spin supervision

Ions-16 supplied 55,885 training anion doublets and 65,473 cation doublets. The neutral training set supplied 656 radical-center labels from Korny’s existing mapped reactions.

| Validation | Charge | Geometries | Atom spin MAE (e) | Equal-share MAE (e) | Most populated atom correct |
|---|---:|---:|---:|---:|---:|
| ChEMBL-20 independent | -1 | 740 | 0.0358 | 0.0501 | 19.7% |
| ChEMBL-20 independent | +1 | 751 | 0.0369 | 0.0542 | 32.1% |
| Ions-16 formula holdout | -1 | 14,919 | 0.0374 | 0.0531 | 27.7% |
| Ions-16 formula holdout | +1 | 17,056 | 0.0362 | 0.0549 | 44.5% |
| neutral radical centers | +0 | 459 | 0.0977 | 0.1213 | 61.0% |

The 2021 AIMNet-NSE release supplies PBE0/ma-def2-SVP spin-resolved NBO populations for ions. The neutral target is a mapped Lewis radical center. The predictor enforces the input molecular spin and permits signed atomic polarization. The ChEMBL-20 set and all its molecular formulas were excluded from Ions-16 training. The original 2021 authors used Ions-16 for testing, so Korny’s Ions-16 training rows cannot benchmark Korny against the 2021 method. [Original AIMNet-NSE data](https://zenodo.org/records/5007980) · [Authors’ field definitions](https://github.com/isayevlab/aimnetnse/blob/main/README.md)

## Vertical ionization energies and electron affinities

The 2021 archive supplies charged doublets and neutral singlets at identical Cartesian coordinates. The auxiliary model learns the vertical ionization potential (cation minus neutral) and electron affinity (neutral minus anion) with independent coefficients. Both the analytical spin-weighted local environments and their molecular means enter the fit. Korny’s reaction and vibration tables report separate potential-energy-surface evaluations.

| Quantity | Paired training geometries | Validation set | Pairs | MAE (eV) | Training-mean baseline (eV) |
|---|---:|---|---:|---:|---:|
| EA | 27,667 | Ions-16, held formulas | 7,452 | 0.488 | 0.687 |
| EA | 27,667 | ChEMBL-20, independent | 698 | 0.452 | 0.722 |
| IP | 31,942 | Ions-16, held formulas | 8,515 | 0.376 | 0.529 |
| IP | 31,942 | ChEMBL-20, independent | 705 | 0.410 | 0.509 |

77,757 Ions-16 ion records and 88 ChEMBL-20 ion records lack the corresponding neutral reference. Their available atomic spin labels still contribute to the respective spin datasets. [Vertical-energy pairing and fit](../experiments/full_public/trial_vertical_gaps.py)

## Closed-shell auxiliary fits

| Atom-count groups | Held-out geometry count | Charge MAE (e) | Energy MAE (eV/atom) | Force-component MAE (eV/Å) |
|---|---:|---:|---:|---:|
| 2–8 atoms | 45,926 | 0.0674 | 0.2489 | 0.9667 |
| 2–76 atoms | 1,226,303 | 0.0412 | 0.1250 | 0.5322 |

The charge model conserves molecular charge through antisymmetric local transfers. The auxiliary radial energy fit uses source energies in eV per atom and two seeded force projections per training geometry. The force MAE uses every held-out Cartesian component. The auxiliary energy reference is dispersion-free ωB97M/def2-TZVPP. Its atom-energy offsets stay out of the reactive readouts. [Energy and force fit](../experiments/full_public/trial_parent_potential.py) · [Transfer to Korny](../experiments/full_public/trial_parent_transfer.py)

## Archive status

| Archive | Completed byte ranges (GB) | Total (GB) | Whole-file check |
|---|---:|---:|---|
| aimnet2_wb97m.h5 | 14.67 | 14.67 | Deposited MD5 matched |
| Reactions.tar.bz2 | 12.21 | 12.21 | Deposited MD5 matched |
| ions16.h5 | 0.19 | 0.19 | Deposited MD5 matched |

The five AutoMeKin archives total 236,400,640 bytes and each matches its deposited MD5. Korny jobs share an 8 GB hard memory limit, a 6 GB reclaim threshold and zero swap allocation. [Original source inventory](aimnet2-nse-and-automekin.md) · [Full benchmark history](barrier-and-frequency-validation.md)

## Learning radical path shapes at two resolutions

The expanded RDB19-Rad snapshot adds 2,433 training reactions. Each contributes seven sampled positions across the two IRC branches, including their common transition state. The fit uses the endpoint gap, midpoint height, two interior energies and two projected force labels. The signed errors use the archived educt/product geometries to orient every reaction. Kernel choice and ridge use three internal training-formula folds. The native evaluation uses the same 114 direction-assigned held-out reactions at both training sizes.

| Training reactions | Local kernel | Reaction MAE | Midpoint | All barriers | Forward | Reverse |
|---:|---|---:|---:|---:|---:|---:|
| 1,030 | cubic | 6.62 | 5.11 | 5.96 | 6.14 | 5.78 |
| 1,030 | sixth | 6.89 | 5.19 | 6.10 | 6.46 | 5.73 |
| 1,030 | both | 6.13 | 4.36 | 5.40 | 5.55 | 5.24 |
| 3,463 | cubic | 5.85 | 5.13 | 5.76 | 5.93 | 5.58 |
| 3,463 | sixth | 6.25 | 5.10 | 5.82 | 6.16 | 5.48 |
| 3,463 | both | 5.08 | 4.38 | 5.09 | 5.30 | 4.87 |

The native electronic errors are in kcal/mol. P56 transfers the shared and family-specific native energy readouts to the Grambow branch. The 589-reaction table above measures that transfer against Korny’s target references. [Path fitting code](../experiments/full_public/trial_path_scales.py)

The later-arriving validation panel contains 173 doublet reaction IDs across 17 held-out formulas. Those reaction IDs arrived after p56 was frozen. P59 used 4,974 training paths. The two rows score shared native RDB19-Rad energy readouts on identical IDs.

| Native readout | Reaction | Midpoint | All barriers | Forward | Reverse |
|---|---:|---:|---:|---:|---:|
| Frozen p56 native kernel | 5.83 | 4.41 | 5.40 | 5.59 | 5.21 |
| Full-release p59 native kernel | 4.81 | 3.63 | 4.38 | 4.52 | 4.25 |

## Inferring radical reaction families from one geometry

The same 453 held-out RDB19-Rad reactions test the shared surface, a global linear mixture, an atom-local mixture inferred from geometry, and a diagnostic supplied with the archived family at prediction. The first three can evaluate individual geometries during MD. The native training partition has 3,463 paths.

| RDB19-Rad model | Reaction | Midpoint | All barriers | Forward | Reverse |
|---|---:|---:|---:|---:|---:|
| shared | 5.60 | 4.66 | 5.47 | 5.55 | 5.39 |
| global mixture | 5.56 | 4.63 | 5.41 | 5.48 | 5.34 |
| local family mixture | 5.58 | 4.60 | 5.38 | 5.46 | 5.30 |
| given family | 4.24 | 3.08 | 3.71 | 3.75 | 3.66 |

## Full radical archive and kernel size

The center-count comparison uses identical training reactions, held formulas and electronic targets. The shared kernel takes one geometry at a time. The supplied-family diagnostic additionally receives the archived mechanism.

| Fit | Centers per resolution | Trained paths | Validation | Reactions | Reaction MAE | All barriers | Forward | Reverse |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| p59 native | 333 | 4,974 | All held, geometry only | 626 | 4.73 | 4.69 | 4.69 | 4.69 |
| p59 native | 333 | 4,974 | Later IDs, geometry only | 173 | 4.81 | 4.38 | 4.52 | 4.25 |
| p59 native | 333 | 4,974 | All held, supplied family | 626 | 4.08 | 3.86 | 3.84 | 3.88 |
| p61 native | 998 | 4,974 | All held, geometry only | 626 | 3.31 | 3.33 | 3.38 | 3.28 |
| p61 native | 998 | 4,974 | Later IDs, geometry only | 173 | 3.20 | 3.19 | 3.30 | 3.07 |
| p61 native | 998 | 4,974 | All held, supplied family | 626 | 2.95 | 2.92 | 2.95 | 2.88 |

## Native Cartesian forces and reaction energies

Every model in this table receives only a single neutral-doublet geometry. The energy test uses 626 held reactions. The force test uses all Cartesian components at the 626 transition states and 1,252 branch midpoints. Energy MAEs are in kcal/mol and force-component MAEs are in eV/Å. The standalone native checkpoints include the source dispersion and have an arbitrary energy zero. [Native checkpoint loader and force audit](../experiments/full_public/native_irc_pes.py)

| Native model | Reaction | All barriers | Forward | Reverse | Saddle force MAE | Midpath force MAE |
|---|---:|---:|---:|---:|---:|---:|
| p59 | 4.73 | 4.69 | 4.69 | 4.69 | 0.948 | 0.970 |
| p61 | 3.31 | 3.33 | 3.38 | 3.28 | 0.698 | 0.663 |
| p63 | 5.22 | 5.01 | 5.26 | 4.76 | 0.141 | 0.114 |
| p64 | 3.94 | 3.69 | 3.79 | 3.60 | 0.278 | 0.238 |
| p66 | 3.94 | 3.69 | 3.78 | 3.60 | 0.279 | 0.238 |
| p67 | 4.02 | 3.82 | 3.93 | 3.72 | 0.343 | 0.300 |

The mean absolute reference force components are 0.0016 eV/Å at the saddles and 0.0097 eV/Å at the branch midpoints. These full-vector checks probe directions outside the IRC tangent. A low reaction MAE alone cannot establish stationarity or MD accuracy.

P64 chooses the smallest training-fold directional force RMSE subject to a 5.0 kcal/mol MAE ceiling on each of the endpoint gap, forward barrier, and reverse barrier. Three internal folds group the 4,974 training reactions by formula. Each path contributes 20 Cartesian force projections. The chosen weight was 0.1. Training-fold MAEs were 4.99, 4.57, and 4.15 kcal/mol in the same order. The final table scores the separate 626 held-out reactions.

## Negative curvature at radical transition states

On 626 of 626 held paths, the first IRC steps point in opposing directions after mass-weighted rigid alignment. For each model and reference, the difference between their two Cartesian forces gives a directional curvature over the same chord. Conversion to a signed wavenumber gives the scale of the local error in cm⁻¹. This probes one finite direction at the archived saddle. The four molecular spectra above use full Hessians at relaxed minima. [Curvature audit](../experiments/full_public/audit_irc_curvature.py)

| RDB19-Rad model | Signed directional MAE (cm⁻¹) | Correct among reference downhill directions |
|---|---:|---:|
| p61 | 648.1 | 501/626 |
| p63 | 371.6 | 613/626 |
| p64 | 557.1 | 520/626 |
| p66 | 446.0 | 530/626 |
| p67 | 237.2 | 595/626 |

P66 and p67 add one observed saddle-curvature projection for each of 4,974 eligible training reactions. On three internal formula folds, selection minimizes signed directional wavenumber MAE subject to a 5 kcal/mol ceiling on each energy observable and a 0.60 eV/Å projected force RMSE ceiling. The 626 validation paths are excluded from coefficient and weight fitting.

| Model | Force weight | Curvature weight | Training-fold signed MAE (cm⁻¹) |
|---|---:|---:|---:|
| p66 | 0.1 | 0.0001 | 413.2 |
| p67 | 0.05 | 0.006 | 216.4 |

## Direct transfer across radical archives

Grambow evaluates a different electronic reference and reaction distribution. Each native energy readout below is frozen before evaluating Grambow. All energies use one geometry at a time and both directed barriers use the archived transition state.

| Grambow model | Reactions | Reaction | All barriers | Forward | Reverse |
|---|---:|---:|---:|---:|---:|
| Uncalibrated RDB19-Rad native p59 | 283 | 18.38 | 25.07 | 27.61 | 22.53 |
| Uncalibrated RDB19-Rad native p61 | 283 | 16.77 | 20.28 | 22.26 | 18.30 |
| Uncalibrated RDB19-Rad native p63 | 283 | 20.05 | 28.96 | 34.81 | 23.10 |
| Uncalibrated RDB19-Rad native p64 | 283 | 16.72 | 21.52 | 24.26 | 18.78 |
| Uncalibrated RDB19-Rad native p66 | 283 | 16.68 | 21.97 | 24.80 | 19.15 |
| Uncalibrated RDB19-Rad native p67 | 283 | 17.05 | 23.28 | 26.65 | 19.91 |
| Korny p49 | 283 | 9.09 | 10.60 | 11.05 | 10.14 |
| Original SO3LR | 283 | 13.88 | 15.71 | 13.97 | 17.46 |
| Tauri-M-2606 | 283 | 5.49 | 6.55 | 6.42 | 6.69 |
| Tauri-L-2606 | 283 | 3.53 | 5.04 | 4.94 | 5.14 |
