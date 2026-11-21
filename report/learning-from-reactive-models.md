# What reactive models teach us about Korny's barrier errors

Measured on 20 November 2026. The target is a geometry-only energy surface, conservative forces, credible minima and saddle curvatures, and MAEs below 5 kcal/mol for reaction energies and both barrier directions. On the same 259 RDB7 reactions, a small Korny kernel reaches 2.23 kcal/mol pooled barrier MAE when the archived DFT energy is supplied at every stationary geometry. Replacing that input with our fitted energy raises the MAE to 9.56. This identifies the DFT surface as the larger error source in this experiment. [Per-reaction predictions](../dataset_for_training/requested_public/rdb7-baseline-audit/predictions.csv) · [Fitting and checks](../experiments/full_public/paired_baseline_audit.py)

## What the published numbers measure

| Model or study | Published evaluation | Difference relevant to Korny |
|---|---|---|
| AIMNet2-NSE | 4,887 supported RDB19-Rad reactions. Reaction MAE 1.76 and forward electronic barrier MAE 1.83 kcal/mol against ωB97M-D3(BJ)/def2-TZVPP | Uses one consistent electronic reference, samples radical reaction paths and trains on local charge and spin as well as energies and forces. RDB19-Rad was their test. Korny uses part of the original RDB19-Rad archive for training at its deposited level. [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12851018/) · [Supplement](https://pmc-oa-opendata.s3.amazonaws.com/PMC12851018.1/ANIE-65-e16763-s002.pdf) |
| Rodríguez-López et al., MACE correction | With an r²SCAN-3c baseline, barrier MAEs below 1 kcal/mol on selected GMTKN55 barrier sets, around 2.5 on the 129 accepted independent RGD1 reactions | The deployed method receives a quantum energy and adds a learned correction at each geometry. The training target is an atomization-energy difference including ZPE. The training script gives forces zero weight. [Study](https://doi.org/10.1002/sstr.70504) · [Exact training script](https://github.com/ComputationalChem-USC/AMK-ML-corrections-BH/blob/4f19444563d76db105524ef426c0e8fb7c61e8ec/Models/train_script/delta_train.sh) |
| AutoMeKin network benchmark | For monoethanolamine, the tighter channel match retains 130 UMA and 76 MACE-OMOL-0 reactions. Each matched group has reaction, forward and reverse MAEs below 2 kcal/mol against coupled cluster | Channel recovery and energy accuracy answer different questions. A method can score low energy errors on the subset of channels it recovers. [Network study](https://doi.org/10.1021/acs.jctc.6c01565) · [Deposited networks](https://zenodo.org/records/22808731) |
| Spiekermann et al., graph network | 1.16 kcal/mol test MAE for the **solution activation free energy**, using 5,600 elementary radical reactions with solvent and representation augmentation | Reactant and product atom-mapped graphs and solvent are inputs. This observable is valuable for reaction classification. An MD trajectory supplies one instantaneous geometry and requires its gradient. [Paper](https://doi.org/10.1021/acs.jpca.4c04121) · [Data and splits](https://zenodo.org/records/11493786) |

The AIMNet2-NSE source computes the two molecular spin-channel totals from charge and multiplicity. At successive passes it combines atomic features with the charges of neighboring atoms, restores the required totals and feeds the updated charges into the next pass. The paper supervises the final charges and spin populations with Hirshfeld labels from the same electronic reference. Korny p49 estimates atom-resolved spin from geometry and freezes the learned population map during the energy fit. P51 adds donor and acceptor scores learned from ionization energies and electron affinities. Neither experiment tests repeated redistribution of spin as neighboring atoms polarize one another. [AIMNet2 implementation](https://github.com/isayevlab/aimnetcentral/blob/ed6812bc20a84575d713d8b5ebb6d9f05129b5ed/aimnet/models/aimnet2.py#L153) · [Korny local spin kernel](../src/spin_local.py) · [Korny redox environments](../src/redox_environment.py)

The MACE correction uses two interaction layers, 192 channels, correlation order three, angular order up to two and a 4 Å neighborhood. Its many-body contractions combine directional environments. The workflow adds the predicted energy increment to each baseline energy and sums separate fragment energies for dissociated products. Korny's controlled analogue learns electronic CCSD(T)-F12a minus ωB97X-D3 on the same original RDB7 coordinates. Electronic energies preserve the distinction between the potential and the vibrational correction derived from it. [MACE contractions](https://github.com/ACEsuit/mace/blob/4f55e0b5b6b8a9fabe6521ef3fa48fe3171b7530/mace/modules/blocks.py#L459) · [AutoMeKin correction and fragment handling](https://github.com/ComputationalChem-USC/AutoMeKin-X/blob/1e4cf10c2421845fadbee9de9f3001a58c056b30/scripts/HLscripts/apply_delta_correction.py#L78)

Chemprop's reaction featurizer can encode changes in atom and bond features between the mapped endpoints. Its directed-bond update excludes the message traveling immediately back along the same bond. Korny can use endpoint changes to select training paths or supervise reaction-center environments while retaining geometry, charge and multiplicity as its calculator inputs. I inspected the canonical Chemprop v1.6.1 implementation. The Spiekermann paper describes a modified Chemprop model, so the canonical implementation documents the architectural idea only. [Reaction features](https://github.com/chemprop/chemprop/blob/f3d1bff19a6e1b03d28e9cfabdf4c80dd8c67382/chemprop/features/featurization.py#L427) · [Directed messages](https://github.com/chemprop/chemprop/blob/f3d1bff19a6e1b03d28e9cfabdf4c80dd8c67382/chemprop/models/mpn.py#L131)

## Where Korny's RDB7 error enters

Every row below scores the **same 259 reactions** against the same original electronic coupled-cluster references. All entries are MAEs in kcal/mol. Forward means TS minus reactants, reverse means TS minus products, and the pooled column includes all 518 directed barriers.

| Energy source at validation | Reaction | All barriers | Forward | Reverse |
|---|---:|---:|---:|---:|
| Korny p49, geometry only | 7.84 | 9.68 | 9.07 | 10.29 |
| Original SO3LR, geometry only | 13.59 | 11.38 | 12.12 | 10.64 |
| Tauri-M-2606, geometry only | 6.67 | 9.69 | 10.15 | 9.24 |
| Tauri-L-2606, geometry only | 5.41 | 6.90 | 6.93 | 6.88 |
| Archived ωB97X-D3 | 2.65 | 3.15 | 3.21 | 3.08 |
| Archived ωB97X-D3 + kernel fitted on energy differences | 1.54 | 2.22 | 2.15 | 2.29 |
| Archived ωB97X-D3 + kernel fitted on individual stationary geometries | **1.47** | 2.23 | **2.08** | 2.38 |
| P53 geometry-only DFT estimate, before CC correction | 7.91 | 9.94 | 9.24 | 10.63 |
| Korny p53, geometry-only DFT estimate + CC correction | **7.53** | **9.56** | **8.98** | **10.13** |

The CC correction uses 499 kernel centers. The geometry fit used 26,150 unique stationary geometries after excluding held or inconsistent duplicates. The difference control used 32,207 usable equations from 11,667 training reaction IDs. Training-only elemental offsets absorb the absolute energy gauge and cancel in balanced reactions. Two training formula folds chose the ridge. The two forms of supervision agree to 0.01 kcal/mol on pooled barriers with the archived baseline. The large gain comes from the baseline, so increasing the number of CC residual labels alone would address the smaller part of the error here. [Exact counts and MAEs](../dataset_for_training/requested_public/rdb7-baseline-audit/summary.json)

P53 starts with the high-level p49 surface and fits a separate set of 499 centers to reconstruct the DFT base. The resulting base itself misses the **DFT** reaction energies by 7.90 and pooled barriers by 10.06 kcal/mol. The existing p49 branch normally used for Grambow radicals does considerably worse on these singlet DFT barriers. None of these fits required fresh quantum calculations. [Decomposition](../dataset_for_training/requested_public/rdb7-baseline-audit/summary.json)

## An AIMNet2-NSE sampling idea tested with kernels

The AIMNet2-NSE authors selected configurations along reaction paths using model disagreement and recomputed reference labels at a common electronic level. Here we reused the archived electronic energies and projected forces. The native RDB19-Rad experiment freezes the kernel, ridge per reaction, training seed and held formulas. The endpoint direction must agree with the authors' educt and product XYZ assignments. The 114 held reactions are identical in every row. These values measure a radical-path auxiliary surface and do not enter the Korny–SO3LR comparison. [Author sampling procedure](https://pmc-oa-opendata.s3.amazonaws.com/PMC12851018.1/ANIE-65-e16763-s002.pdf) · [Korny experiment](../experiments/full_public/irc_committee.py)

| Training selection | Reactions | Training geometries | Reaction | All barriers | Forward | Reverse |
|---|---:|---:|---:|---:|---:|---:|
| Fixed random order | 71 | 41,228 | 13.89 | 13.55 | 14.56 | 12.54 |
| Fixed random order | 512 | 284,037 | 11.64 | 11.00 | 11.63 | 10.36 |
| Most committee disagreement | 512 | 283,160 | **11.58** | **10.61** | **11.23** | **9.99** |
| Disagreement within fixed reaction-class quotas | 512 | 292,242 | 11.85 | 10.78 | 11.30 | 10.25 |
| All available in the frozen snapshot | 1,030 | 571,758 | 11.70 | 10.78 | 11.38 | 10.18 |

Eight kernel models bootstrap whole formulas from the same 71 seed reactions. Their spread in reaction energy and midpoint height ranks additional paths using geometry alone. With a 512-reaction budget, disagreement reduces the pooled barrier MAE by 0.39 kcal/mol in this run. The fixed quota experiment preserves the per-family counts of the random control. This is one seed and a partial archive. Additional seeds and the completed archive are needed to establish how reliably the gain transfers. [Frozen learning curve](../dataset_for_training/RDB19_Rad/path-kernel45/snapshot-learning-curve.json) · [Acquired IDs and scores](../dataset_for_training/RDB19_Rad/path-kernel45/committee-selection.json)

Repeating the acquisition with five training seeds reduced the mean advantage of unrestricted disagreement to 0.07 kcal/mol on pooled barriers. Stratification by reaction class gave the best pooled MAE at a budget of 512. At full budget, half of the training weight went to all 1,030 paths equally and half to the selected 512. Pooled barrier MAE improved in all five full-budget fits, by 0.005–0.172 kcal/mol. The test reactions are shared across seeds. The table measures variability from training selection, with no additional independent test reactions. [Five-seed results](../dataset_for_training/RDB19_Rad/path-kernel45/committee-five-seeds.json)

| Five-seed mean | Training reactions | Reaction | All barriers | Forward | Reverse |
|---|---:|---:|---:|---:|---:|
| Random | 512 | 11.73 | 10.86 | 11.55 | 10.17 |
| Disagreement | 512 | 11.75 | 10.79 | 11.20 | 10.37 |
| Disagreement with matched class quotas | 512 | 11.62 | 10.62 | 10.93 | 10.32 |
| All paths equally weighted | 1,030 | 11.70 | 10.78 | 11.38 | 10.18 |
| All paths with disagreement weighting | 1,030 | 11.71 | 10.70 | 11.22 | 10.19 |

## Geometry-only calculators on the common panel

The active panel contains 259 RDB7, 47 BH9 and 283 Grambow radical reactions. D3(BJ)+ATM is enabled once in every Korny row. The high-level branch scores RDB7 and BH9. The ωB97X-D3 branch scores Grambow. The table rescores existing checkpoints trained under the earlier protocol using the same 589 reactions. [Exact common IDs and references](../dataset_for_training/requested_public/so3lr-tauri-matched/p54-public-active-common.csv)

| Model | Reactions | Reaction | All barriers | Forward | Reverse |
|---|---:|---:|---:|---:|---:|
| Korny p49 | 589 | 9.14 | 10.58 | 10.46 | 10.69 |
| Korny p50, network normal-mode constraints | 589 | 9.42 | 10.86 | 10.75 | 10.97 |
| Korny p51, redox environments | 589 | 9.18 | 10.70 | 10.57 | 10.83 |
| Korny p53, reconstructed DFT + CC correction | 589 | 9.03 | 10.62 | 10.54 | 10.70 |
| Korny p54, reoptimized structures | 589 | 9.37 | 10.79 | 10.70 | 10.88 |
| Original SO3LR | 589 | 13.90 | 13.65 | 13.29 | 14.02 |
| Tauri-M-2606 | 589 | 6.15 | 8.09 | 8.28 | 7.89 |
| Tauri-L-2606 | 589 | 4.47 | 6.03 | 5.99 | 6.06 |

| Relaxed frequency MAE against PBE0/cc-pVTZ, cm⁻¹ | Propane | Benzene | Ethanol | 1,3-butadiene |
|---|---:|---:|---:|---:|
| Korny p49 | 89.4 | 151.2 | 104.1 | 156.3 |
| Korny p50 | 137.3 | 142.4 | 153.2 | 123.6 |
| Korny p51 | 90.6 | 154.3 | 90.1 | 155.5 |
| Korny p53 | 112.7 | 263.1 | 154.7 | 157.5 |

P53's RDB7 gains accompany a change in BH9 pooled barrier MAE from 15.39 to 16.62 and a deterioration in all four molecular frequency MAEs. P50 used 2,067 training stationary points with 55,371 measured normal modes. Its constrained directions improved on some systems, while its common-panel barrier MAE increased. P49 remains the working checkpoint. [Full energies, geometries, frequencies and signed saddle checks](expanded-reference-fits.md)
