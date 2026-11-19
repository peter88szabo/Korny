# Reaction energies and barriers on matched geometries

The current 595-reaction Korny split has a separate table with forward and reverse barriers for each dataset. [Original SO3LR on the current Korny validation IDs](so3lr-matched-validation.md)

All errors below are MAEs in kcal/mol. Each reaction contributes one reaction energy and both directional barriers. Detached fragment energies are summed using the published stoichiometry. All four models received the same coordinates. Charge and multiplicity were supplied for each fragment and saddle. Korny used its D3(BJ) correction.

| Electronic reference panel | Reactions | Korny reaction | Korny barrier | Original SO3LR reaction | Original SO3LR barrier | SO3LR Tauri-M reaction | SO3LR Tauri-M barrier | Tauri-L reaction | Tauri-L barrier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RDB7, CCSD(T)-F12a/cc-pVDZ-F12 | 400 | 18.78 | 23.56 | 8.33 | 8.49 | 4.73 | 7.93 | 4.07 | 5.80 |
| BH9 and TRIP50, published electronic references | 65 | 17.12 | 15.78 | 17.22 | 17.00 | 9.18 | 10.00 | 7.98 | 8.49 |
| Neutral singlet subset of BH9 and TRIP50 | 16 | 35.50 | 26.52 | 14.35 | 17.63 | 4.38 | 13.72 | 4.01 | 12.98 |

The test sets were held out of Korny training by transition-state formula. Tauri-L achieved the lowest average error on each panel. Its RDB7 reaction MAE reached the 5 kcal/mol target. All four models exceeded that target for barriers. The installed original SO3LR checkpoint enables a spin embedding. Its aggregate BH9/TRIP50 score includes open-shell cases. Training-set overlap with the external pretrained models has not been established.

The fixed Korny checkpoint was `rdb7-b2b299a0aea2028a.npz`. The locally installed SO3LR checkpoints (`so3lr`, `so3lr-tauri-m-2606` and `so3lr-tauri-l-2606`, repository revision `900a0f0`) ran on CPU in float64 at a 12 Å long-range cutoff. The evaluator checked all 1,522 output geometries for atom and coordinate agreement before scoring. Individual models have different training data and electronic targets. These numbers measure predictive error against the named benchmarks, including any reference-method differences. [SO3LR implementation](https://github.com/general-molecular-simulations/so3lr)

## How many geometries entered training?

Across the Korny models compared in this report and the companion conformer report, the training sets contained **28,878 distinct nuclear geometries**. The table traces the data used by each deployed checkpoint, including its earlier training stages.

| Checkpoint or stage | Distinct geometries including earlier stages |
|---|---:|
| Bare PBE0 backbone (DES15K, QM7-X, SPICE) | 14,666 |
| Backbone after adding molecular geometry, curvature and water-cluster constraints | 14,870 |
| Public reaction model, including the reaction-center fit | **24,548** |
| Public reaction model with additional QM7-X geometry constraints | **24,794** |
| Private CC-calibrated isoprene–peroxy model | **27,062** |
| Final private H-shift model | **27,110** |

The backbone used 10,139 DES15K, 3,887 QM7-X and 640 SPICE geometries. Its stationary-point calibration added 55 previously unused QM7-X geometries, 140 water clusters and nine molecular minima. The 4,000 RDB7 and 159 BH9/TRIP50 training reactions then contributed **9,678** distinct reactant, saddle and product geometries. Detached fragments shared between reactions count once. Later fits reused these reaction geometries. The public geometry-constrained variant used 406 QM7-X anchor geometries, of which 246 were additional.

The private fit branched from the 24,548-geometry public checkpoint. Its 1,239 CC energy records covered 1,238 geometries and its DFT derivative constraints covered 2,438. The two sets shared 1,162 geometries, giving **2,514** distinct private geometries at that stage. The final H-shift calibration added **48** more from 16 paths with three geometries each. Across the public and private reaction-model variants together the union was **27,356**. The separate QM7-X pilot contributed 524 additional geometries from its 715 training records. The ethanol model contributed 998 distinct geometries from its 1,000 training records.

Counts use atomic composition and sorted, element-pair-specific interatomic distances rounded to 0.0001 Å. Rigid rotations, translations, atom renumberings, different electronic labels on the same geometry, and repeated use across folds do not add geometries. Only geometries used by a fit or an explicit geometry constraint enter the count. Numerical finite-difference probes, evaluation-only data and the 135,592 indexed ORCA trajectory frames add no training rows here. The historical backbone split included DES15K-2193567-0. A fresh split with the current multiplicity filter would exclude that one record because its spin label is missing.

## Korny after the comparison

The reaction-center kernel adds 590 atomic environments selected from training saddles. For each element, the largest change from the local environments in the endpoints identifies candidate atoms. Each class supplies kernel centers; the accompanying directional features point toward endpoint environments. KinBot family names come from its reaction finder, with published RMG, BH9 and TRIP50 labels retained where applicable. Structures with no assigned class still contribute training data. Inference uses Cartesian coordinates, charge and multiplicity. The fitted potential supplies analytic forces. [KinBot reaction finder](https://github.com/zadorlab/KinBot/blob/684e513c4379f28a3db67a7637671697f1df740b/kinbot/reaction_finder.py)

| Korny checkpoint | RDB7 reaction | RDB7 barrier | BH9/TRIP50 reaction | BH9/TRIP50 barrier |
|---|---:|---:|---:|---:|
| Organic-feature starting point | 18.78 | 23.56 | 17.12 | 15.78 |
| Reaction centers | **16.93** | **22.28** | **15.60** | **14.95** |
| Reaction centers with additional geometry constraints | 17.32 | 22.76 | 16.09 | 15.37 |

The reaction-center fit took 125 seconds and peaked at 3.25 GiB. Adding derivative constraints at optimized and displaced geometries from 203 QM7-X parents took 145 seconds and peaked at 4.51 GiB. The public test data did not set the coefficients or select the ridge. Repeated evaluation makes these panels regression benchmarks; an additional untouched test will be needed for an independent final accuracy estimate. The two models differ in both centers and training constraints from the original. The ablation compared uniform versus family-balanced barrier weights for the same new centers. Development errors favored uniform weights. A separate control pooled all training reactions without their class labels and selected exactly the same 590 centers by element. With the same source weights and ridge, the class-blind reaction/barrier MAEs were **17.37/22.54** on RDB7 and **16.48/15.74** on BH9/TRIP50. The class-aware figures are 16.93/22.28 and 15.60/14.95 respectively. This isolates the effect of center selection at fixed model capacity and fitting weights.

| Held-out family, reaction-center checkpoint | Reactions | Reaction MAE | Barrier MAE |
|---|---:|---:|---:|
| RDB7, unassigned | 299 | 17.89 | 24.18 |
| RDB7, intramolecular H migration | 52 | 13.01 | 17.53 |
| RDB7, keto–enol | 29 | 17.67 | 15.08 |
| BH9/TRIP50, radical rearrangement | 12 | 8.30 | 9.02 |
| BH9/TRIP50, nucleophilic addition | 11 | 17.93 | 13.52 |
| BH9/TRIP50, H-atom abstraction | 11 | 7.48 | 14.11 |
| BH9/TRIP50, electrocyclic | 7 | 42.91 | 33.91 |

### Open-shell performance

BH9/TRIP50 reactions are grouped below by the transition state's spin multiplicity. Individual detached fragments keep their own charges and multiplicities. Each entry gives reaction MAE / directional barrier MAE in kcal/mol.

| TS multiplicity | Reactions | Korny reaction centers | Korny with geometry constraints | Original SO3LR | Tauri-M | Tauri-L |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 30 | 23.09 / 19.66 | 24.19 / 20.09 | 12.03 / 13.15 | 7.21 / 10.74 | 5.73 / 9.53 |
| 2 | 26 | 9.19 / 11.16 | 8.63 / 11.51 | 12.20 / 18.07 | 5.73 / 7.39 | 3.90 / 4.79 |
| 3 | 9 | 9.10 / 10.21 | 10.61 / 10.76 | 49.07 / 26.72 | 25.73 / 15.05 | 27.28 / 15.69 |

### Relaxed geometries and harmonic frequencies

Each potential was relaxed from the same reference geometry. The harmonic frequency MAEs below compare sorted internal modes with the archived PBE0/cc-pVTZ references, in cm⁻¹. The bond and angle columns report the geometry-constrained model's MAEs in Å and degrees. The family-specific energy gains did not establish an improvement for MD.

| Molecule | Starting frequency MAE | Reaction centers | With geometry constraints | Bond MAE (Å) | Angle MAE (°) |
|---|---:|---:|---:|---:|---:|
| acetone | 92.0 | 116.0 | 94.3 | 0.044 | 11.71 |
| propane | 123.3 | 147.3 | 131.9 | 0.017 | 0.77 |
| benzene | 86.4 | 398.4 | 190.1 | 0.040 | 0.01 |
| n-butane | 172.9 | 156.3 | 182.6 | 0.022 | 1.33 |
| cyclopropane | 275.7 | 247.9 | 269.4 | 0.099 | 5.40 |
| propene | 90.1 | 143.6 | 104.1 | 0.007 | 2.17 |
| furan | 348.7 | 394.4 | 365.5 | 0.016 | 3.23 |

The checkpoint files are `experiments/reactivity/reaction-centers-8418b49d27745de7.npz` and `experiments/reactivity/reaction-centers-c95f15e99a957140.npz` respectively. These aggregate frequency errors alone do not determine the accuracy of individual bending modes or the stability of a reactive trajectory.

## Private Case II 1,6-H-shift paths

The three test conformers (15, 16 and 30) form **one independent geometry cluster**. The reference is the archive's ECC_VDZF12 electronic energy at the IRC start, optimized saddle and IRC end. The endpoint gap is end minus start. The second metric averages the absolute errors of saddle minus start and saddle minus end. The IRC calculation does not establish relaxed minimum-to-minimum thermochemistry.

| Model | Endpoint gap MAE | Saddle-to-endpoints MAE |
|---|---:|---:|
| Korny, transferable checkpoint | 27.30 | 13.65 |
| Korny, separately calibrated on the private training cluster | 18.33 | 12.00 |
| Stock sGDML, three fully labeled training saddles | 6.09 | 19.87 |
| sGDML kernel, all nine energy labels and three force vectors | 7.10 | 5.98 |
| Original SO3LR | 26.04 | 13.02 |
| SO3LR Tauri-M-2606 | 4.10 | 2.71 |
| SO3LR Tauri-L-2606 | 1.91 | 4.58 |

In conformer 15, H16 moves from C4 (1.090 Å) to O8 (0.967 Å). The IRC end has the connectivity of 44 optimized archive records. The closest is 0.52 Å all-atom RMSD away, and the IRC end still has a maximum gradient of 0.001712 hartree/bohr. The calculation identifies the bond change. The particular relaxed product conformer remains unconfirmed.

The calibrated Korny model gives an endpoint gap of −23.20 versus −4.86 kcal/mol at CC. ORCA M06-2X-D3 gives −5.73 at the same geometries. Moving from training conformer 11 to test conformer 15 shifts the CC start and end energies by −1.18 and −3.37 kcal/mol. Korny instead shifts them by +6.08 and −13.77. Both ends contribute to the failure to transfer between conformer groups.

The stock sGDML fit used its analytic solver on three training saddles. Six training endpoints have CC energies and lack matched Cartesian gradients. A second fit used the official sGDML kernel and predictor with the unavailable force rows omitted from the solve. It used the same three training paths as the specialized Korny model, with 12 symmetry permutations, fixed width 1 and ridge 1e-8. This is a molecule-specific comparison. The broad benchmarks deliberately hold out entire molecular formulas, so this sGDML model cannot supply a transferable MAE for those panels. [sGDML methodology and datasets](https://www.sgdml.org/)

## All 16 private CC-labeled H-shift paths

Four groups were formed from near-duplicate path geometries. The group sizes were 7, 3, 3 and 3. The table pools predictions for the omitted group in each of four folds. Korny selected its private calibration within the three training groups of each fold. The broader private CC parent excluded the entire C₅H₉O₃ doublet composition. The original three-conformer comparison above uses a different split; its ten reserved NEB-overlap paths entered this four-fold experiment. The two comparisons must not be counted as independent evidence.

| Model | IRC endpoint gap MAE | Saddle-to-endpoints MAE |
|---|---:|---:|
| Korny, private CC parent before path calibration | 4.79 | 5.59 |
| Korny, within- and between-conformer CC calibration | **1.35** | **1.00** |
| Stock sGDML, fully labeled training saddles | 6.13 | 20.63 |
| sGDML kernel, measured energies and all available force components | 2.47 | 3.00 |
| Original SO3LR | 24.25 | 12.12 |
| SO3LR Tauri-M-2606 | 1.81 | 1.89 |
| SO3LR Tauri-L-2606 | 0.62 | 3.73 |

Giving each group equal weight yields 1.63 kcal/mol for Korny’s endpoint gap and 1.18 for its saddle-to-endpoint gaps. The individual group endpoint MAEs ranged from 0.52 to 3.31. The final private checkpoint used all 16 paths after cross-validation. The Korny and sGDML path fits used the available energies and derivatives in their training groups; Korny also had a pretrained backbone and separate radical-archive calibration. The SO3LR checkpoints were used as distributed. These figures cover a single reaction family.

## Two Gamma channels with independently optimized end species

In the Case II Gamma archive, two channels passed an additional test. Each named species had the matching stereochemical suffix and balanced composition and state, an M06-2X-D3zero/aug-cc-pVTZ converged geometry and Hessian, and an ORCA CCSD(T)-F12/RI/cc-pVDZ-F12 single point at the same coordinates. The minima had no internal frequencies below −1 cm⁻¹. Each saddle had one imaginary mode below −20 cm⁻¹. We chose the lowest verified CC conformer of each named species. These are electronic differences between specified stationary points; the assignment uses the author's channel names, and no individual IRC–minimum link was assumed.

| Named channel | CC reaction | CC forward barrier | CC reverse barrier |
|---|---:|---:|---:|
| Gamma 2 → 3, ZZ-S | −2.61 | 23.03 | 25.64 |
| Gamma 4 → 9, ZZ-R | 14.52 | 33.95 | 19.43 |

| Model on these two reactions | Reaction MAE | Barrier MAE (4 directions) |
|---|---:|---:|
| Korny, public geometry-constrained reaction-center model | 17.76 | 22.84 |
| Original SO3LR | 4.32 | 20.95 |
| SO3LR Tauri-M-2606 | 1.54 | 8.33 |
| SO3LR Tauri-L-2606 | 1.54 | 6.62 |

The trained sGDML H-shift model uses 17 atoms, while these Gamma reactions use 19. All candidates for the Gamma 2 → 3 saddle failed an independent-conformer, same-connectivity training screen after exclusion of near duplicates. A matched sGDML stationary-barrier score for both channels remains unavailable. The private-data-tuned Korny checkpoint is omitted because these species can occur in its training archive.

## Reproduction

Run `python -m experiments.reactivity.compare_potentials export` in the Korny environment. Evaluate the exported `geometries.extxyz` using each named SO3LR checkpoint with `--targets energy --batch-size 8 --precision float64 --lr-cutoff 12`, writing `dataset_for_training/comparison/<model>.extxyz`. Then run `python -m experiments.reactivity.compare_potentials <model>`. Per-model JSON files contain MAE, RMSD, and counts. Private molecular coordinates and trained private checkpoints stay inside the ignored training directory.

For private sGDML runs, add the local sGDML source directory to `PYTHONPATH` and execute `python -m experiments.reactivity.sgdml_hshift`, or specify `--fold 0` through `--fold 3` for the complete four-group benchmark. This uses revision `a6ae5e8` of the authors’ implementation. The CPU-only import workaround operates in a temporary copy. Run `python -m experiments.reactivity.family_fit` with or without `--preserve-geometry` to reproduce the public kernel variants. [Conformer and force benchmarks](conformer-transfer.md) · [Private archive and model provenance](isoprene-peroxy.md)
