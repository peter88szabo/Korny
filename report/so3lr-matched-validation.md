# Original SO3LR and Korny on the 595-reaction Korny validation split

Every ID comes from the saved split for the joint Korny checkpoints. Original SO3LR covers 591 of the 595 reactions. Across those 591, SO3LR gives a reaction-energy MAE of 13.97 kcal/mol and a two-direction barrier MAE of 13.67 kcal/mol. Korny gives 20.82 and 23.91 kcal/mol on the same IDs. Both models exceed the 5 kcal/mol target on both pooled metrics.

## MAE by dataset

All energies are electronic and all errors are in kcal/mol. Each cell lists **original SO3LR / Korny**. The pooled row gives every reaction equal weight. The all-barriers column counts forward and reverse once each.

| Validation panel | Reactions scored / original | Directional barriers scored / original | Reaction energy | All barriers | Forward | Reverse |
|---|---:|---:|---:|---:|---:|---:|
| RDB7 | 259 / 259 | 518 / 518 | 13.59 / 25.54 | 11.38 / 30.03 | 12.12 / 32.30 | 10.64 / 27.77 |
| BH9 | 47 / 49 | 94 / 98 | 15.67 / 25.56 | 13.77 / 21.89 | 15.64 / 19.45 | 11.90 / 24.33 |
| TRIP50 | 2 / 4 | 4 / 8 | 34.31 / 55.35 | 17.15 / 38.93 | 10.87 / 57.50 | 23.44 / 20.37 |
| Grambow radicals | 283 / 283 | 566 / 566 | 13.88 / 15.47 | 15.71 / 18.53 | 13.97 / 20.19 | 17.46 / 16.86 |
| **Pooled** | **591 / 595** | **1,182 / 1,190** | **13.97 / 20.82** | **13.67 / 23.91** | **13.28 / 25.56** | **14.05 / 22.25** |

SO3LR's largest gain was on RDB7, with an 18.65 kcal/mol reduction in barrier MAE. Korny had the lower reverse-barrier MAE on Grambow and on the two evaluable TRIP50 reactions. Across the pooled panel, 151 of 591 SO3LR reaction energies and 307 of 1,182 SO3LR directional barriers had absolute errors within 5 kcal/mol. The corresponding Korny counts were 114 and 194.

## Coverage and references

| Panel | Reference | Original IDs outside SO3LR coverage |
|---|---|---|
| RDB7 | CCSD(T)-F12a/cc-pVDZ-F12 electronic energies | None |
| BH9 | Published high-level electronic differences | BH9/06_6, BH9/06_7 |
| TRIP50 | Published high-level electronic differences | TRIP50/32, TRIP50/33 |
| Grambow | wB97X-D3 electronic energies of the two minima and saddle | None |

All four omitted reactions contain silicon. The original checkpoint covers H, C, N, O, F, P, S and Cl. Its full 595-reaction MAE is therefore unavailable. Korny's full-panel MAEs on BH9 were 25.27 for reaction energy, 19.30 forward and 23.51 reverse. On all four TRIP50 reactions they were 29.47, 30.54 and 10.27, respectively. The table above recomputes Korny on the SO3LR-covered IDs. [Original SO3LR element coverage](https://github.com/general-molecular-simulations/so3lr/blob/900a0f0/README.md) · [Full Korny validation](full-public-fit.md)

## Calculation and checks

| Setting | Value |
|---|---|
| SO3LR | Original bundled checkpoint named `so3lr`, revision `900a0f0`, float64 on CPU |
| Geometry energies | 1,794 fresh evaluations and 92 checked cached predictions |
| Inputs | Korny's saved fragment and transition-state coordinates, individual charges and multiplicities |
| Korny | Joint high-level checkpoint on RDB7, BH9 and TRIP50. Joint radical checkpoint on Grambow. Both use PBE0-D3(BJ). |
| Comparability | 12 Å long-range cutoff throughout to match the existing SO3LR predictions |
| Forward barrier | Saddle energy minus the sum of reactant fragment energies |
| Reverse barrier | Saddle energy minus the sum of product fragment energies |
| Reaction energy | Sum of product fragment energies minus sum of reactant fragment energies |

The source keys, atomic numbers, coordinates and electronic states identify each reused geometry. The reader checks returned atom order, coordinates, charge and finite energies. Every paired prediction satisfies the forward-minus-reverse reaction-energy identity to within 0.000001 kcal/mol. Scoring the deployed Korny checkpoints on all original IDs also reproduces the saved per-panel MAEs to within 0.000001 kcal/mol.

The installed original SO3LR checkpoint enables both charge and spin embeddings. Its training-set overlap with these evaluation reactions is unverified. SO3LR recommends a 100 Å long-range cutoff for isolated molecules. Ten scored-domain BH9 geometries and one Grambow geometry span more than the 12 Å cutoff used here. [Checkpoint configuration](https://github.com/general-molecular-simulations/so3lr/blob/900a0f0/so3lr/models/so3lr/hyperparameters.yaml) · [SO3LR cutoff guidance](https://so3lr.github.io/advanced/llm-agents/)

The local ledgers retain every validation ID with the reference, both predictions, signed errors and any missing species. [RDB7](../dataset_for_training/requested_public/so3lr-original-matched/RDB7-equations.csv) · [BH9](../dataset_for_training/requested_public/so3lr-original-matched/BH9-equations.csv) · [TRIP50](../dataset_for_training/requested_public/so3lr-original-matched/TRIP50-equations.csv) · [Grambow](../dataset_for_training/requested_public/so3lr-original-matched/Grambow-equations.csv) · [MAEs and SO3LR RMSDs](../dataset_for_training/requested_public/so3lr-original-matched/summary.json) · [Evaluation script](../experiments/full_public/so3lr_validation.py)
