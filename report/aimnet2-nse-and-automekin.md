# AIMNet2-NSE and AutoMeKin data for Korny

## What the authors released

The AIMNet2-NSE paper describes 13 million additional open-shell configurations and 200,000 radical reaction profiles. Its data statement links BASChem19 and the pretrained model repository. I found the five-system DFT supplement and the original AIMNet2 closed-shell HDF5. I could not locate a public file containing the full open-shell training extension in the authors’ repository, model card, Figshare or DataCite records. The four downloadable ensemble members contain learned weights. [AIMNet2-NSE paper and supporting data](https://pmc.ncbi.nlm.nih.gov/articles/PMC12851018/) · [Authors’ model card](https://huggingface.co/isayevlab/aimnet2-nse)

| Download | Content verified | Relevance to Korny |
|---|---|---|
| AIMNet2-NSE supporting ZIP | Five reaction systems, including peroxide, spin-resolved bond dissociation and a radical ring-opening IRC | Fit relative energies along whole paths, including doublets and triplets |
| BASChem19 deposit | 19 pairs of reactant minima and transition states, with Cartesian coordinates, molecular state files and 19 forward barriers | Fit or validate forward barriers. Product energies are unavailable in these pairs |
| Original AIMNet2 ωB97M HDF5 | 20,108,565 geometries and 474,624,017 atom labels in 75 size groups (2–76 atoms), 14,669,307,950 bytes. The depositor documents dispersion-free energy and forces, Hirshfeld charges, dipoles and quadrupoles | Closed-shell auxiliary reference for the Korny backbone and charge model |
| AutoMeKin v2 | All five network archives, 236,400,640 bytes in total. All five match the deposited MD5 checksums | Learn from optimized minima and transition states, with paired DFT and coupled-cluster energies and same-level archived harmonic frequencies |
| RDB19-Rad, cited by AIMNet2-NSE for testing | Deposited reaction tables, molecular conformers and split definitions. The reaction table contains Gibbs free energies in gas and solution phases | Supplies reported reaction-class labels and kinetic observables. A temperature-dependent free energy needs a separate thermochemical model |

Sources are the [AIMNet2-NSE supporting archive](https://pmc-oa-opendata.s3.amazonaws.com/PMC12851018.1/ANIE-65-e16763-s001.zip), [BASChem19 deposit](https://doi.org/10.1184/R1/29710664.v1), [original AIMNet2 HDF5 deposit](https://doi.org/10.1184/R1/27629937), [AutoMeKin v2 deposit](https://zenodo.org/records/22808731) and [RDB19-Rad archive](https://zenodo.org/records/11493786).

## Fitting inventory

The table counts unique state/geometry/electronic-reference observations after removing 220 duplicated AIBN optimization frames. A geometry with two reference methods contributes one observation to each appropriate row. Validation reserves benzopinacol at both spin states, the complete cBD-CN network, and four BASChem19 reactions, including a radical hydrogen abstraction. No geometry with a matching fingerprint from Korny’s existing held-out reactions is eligible for new training.

| Source and energy labels | Training geometries | Validation geometries | Contents |
|---|---:|---:|---|
| AIMNet2-NSE, ωB97M-D3(BJ)/def2-TZVPP | 2,741 | 764 | Spin scans and a 241-frame doublet IRC |
| AIMNet2-NSE, ωB97M-D3(BJ) unrestricted singlet branch | 628 | 0 | Separate styrene electronic branch |
| AIMNet2-NSE, B3LYP-D3(BJ)/def2-TZVP | 16,669 | 0 | AIBN singlet and triplet scan/optimization trajectories |
| **AIMNet2-NSE supplement total** | **20,038** | **764** | **11,266 singlet, 241 doublet and 8,531 triplet training geometries** |
| BASChem19 archive | 30 | 8 | 15 training and 4 held-out minimum/TS pairs |
| AutoMeKin, ωB97X-D3/def2-TZVP | 2,067 | 1,026 | 538 training and 329 held-out complete minimum–TS–minimum channels |
| AutoMeKin, DLPNO-CCSD(T)-F12/cc-pVDZ-F12 | 1,343 | 810 | 446 training and 265 held-out complete channels |
| AutoMeKin, canonical CCSD(T)-F12/cc-pVDZ-F12 | 496 | 0 | MEA, with 51 complete training channels |

The importer reads actual energy-bearing ORCA trajectory records. The 20,802 unique frames include intermediate optimization steps and represent five chemical systems. The authors identify B3LYP as the method for the archived AIBN scan. BASChem19’s deposit attributes the energies embedded in its XYZ files to TPSS-D3(BJ)/def2-TZVP. All 19 barrier differences reproduce its CSV values to 0.002 kcal/mol. The published paper compares against separately calculated ωB97M-D3(BJ) single points. Archive gradient records carry different energies from the XYZ headers, so they have no force labels in the new fit. [AIMNet2-NSE supporting information](https://pmc-oa-opendata.s3.amazonaws.com/PMC12851018.1/ANIE-65-e16763-s002.pdf) · [BASChem19 data and README](https://doi.org/10.1184/R1/29710664.v1)

The AutoMeKin importer joins the authors’ high-level stationary-point IDs to successful coupled-cluster calculations. It checks the full-system charge and spin against ORCA inputs. It retains the DFT frequencies by electronic level and uses the reaction network to identify both endpoints. Channels ending in fragmented products require an explicit fragment-energy and geometry assignment before inclusion. The low-level PM7 searches, UMA-M, MACE-OMOL-0 and Δ-ML predictions remain available in the downloaded archives. [AutoMeKin study](https://doi.org/10.1021/acs.jctc.6c01565) · [Download and method correspondence](https://zenodo.org/records/22808731)

## What transfers to a kernel potential

AIMNet2-NSE supplies the total charge and spin multiplicity at inference, and its atomic α and β channels are constrained by the corresponding molecular totals. Training supervises local spin populations along with energies, forces and charges. For Korny, this suggests a test of learned spin redistribution between neighboring atoms during reaction, alongside the existing fixed-electron-count orbital features. The two singlet styrene paths illustrate a practical limitation of geometry, charge and multiplicity alone. Broken-symmetry and restricted solutions can require an additional electronic-state label. AIMNet2-NSE reports 2.34 kcal/mol mean absolute deviation on the 19 BASChem19 forward barriers at its reference level. A matched Korny comparison requires the same barriers, geometries, electronic levels and training exclusions. [AIMNet2-NSE study](https://doi.org/10.1002/anie.202516763)

The AutoMeKin study evaluates UMA and MACE as complete energy surfaces for finding stationary points and tracing reaction paths, then compares channel energies with coupled-cluster single points. For Korny, the corresponding tests are the number of unstable modes, the force at the reference stationary geometry and whether downhill displacement connects the two reported endpoints. Energy-error tables alone cannot establish whether a model supports reactive dynamics. [AutoMeKin study](https://doi.org/10.1021/acs.jctc.6c01565)

[Local fitting inventory](../dataset_for_training/paper_references/inventory.json) · [Archive importer](../experiments/full_public/paper_data.py) · [Matched Korny, SO3LR and Tauri results](barrier-and-frequency-validation.md)

## Validation after adding the published reaction data

P43 uses seven separate electronic-reference fits. Within each fit, reaction differences and energies relative to the system mean remove the arbitrary energy zero. Korny learns the contribution of each of the resulting corrections to each of its two deployed surfaces, under the existing energy, force and curvature constraints. The table uses the ordinary, deployable Korny readouts with D3(BJ)+ATM.

| Held-out benchmark | Model | Reactions | Reaction | All barriers | Forward | Reverse | Midpoint |
|---|---|---:|---:|---:|---:|---:|---:|
| cBD-CN, DLPNO-CCSD(T)-F12 | p42 | 265 | 11.11 | 22.47 | 23.80 | 21.14 | 21.55 |
| cBD-CN, DLPNO-CCSD(T)-F12 | p43 | 265 | 11.08 | 22.51 | 23.87 | 21.14 | 21.60 |
| cBD-CN, wB97X-D3 | p42 | 329 | 13.73 | 11.80 | 14.07 | 9.54 | 9.91 |
| cBD-CN, wB97X-D3 | p43 | 329 | 13.66 | 11.80 | 14.08 | 9.52 | 9.94 |
| BASChem19, TPSS-D3(BJ) deposit | p42 | 4 | — | — | 28.22 | — | — |
| BASChem19, TPSS-D3(BJ) deposit | p43 | 4 | — | — | 28.68 | — | — |

Errors are MAEs in kcal/mol. The two cBD-CN rows use the same held-out network at different electronic levels. The BASChem19 deposit supplies forward references for four withheld reactions. The bootstrap 95% interval for the p43 minus p42 change across the original 591 reactions is −0.061 to +0.098 kcal/mol for reaction energy and −0.073 to +0.026 for pooled barriers. [All external errors](../dataset_for_training/paper_references/p43-native-validation.json) · [Common Korny and SO3LR benchmarks](barrier-and-frequency-validation.md)

### Transfer to the held-out benzopinacol derivative

The archived molecule is C₃₂H₃₈O₂Si₂. Its singlet and triplet scans contain 369 and 395 frames. The large error comes primarily from the energy separation between the two spin branches. The table also checks the shape of each branch against its own first frame. All values are in kcal/mol.

| Quantity | DFT reference | p42 | p43 |
|---|---:|---:|---:|
| First triplet frame minus first singlet frame | 101.9 | 1681.3 | 1735.1 |
| Within-singlet profile MAE | — | 60.2 | 61.6 |
| Within-triplet profile MAE | — | 128.0 | 125.8 |
| Both scans relative to the first singlet frame, MAE | — | 805.0 | 835.2 |

The two initial scan frames have their own recorded coordinates. This comparison therefore measures a difference between archived states and geometries. A vertical spin gap would require both energies at identical coordinates.

[Expanded data, common-ID results and download status](expanded-reference-fits.md)
