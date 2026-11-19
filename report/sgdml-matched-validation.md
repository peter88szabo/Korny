# Stock sGDML on Korny's saved reaction IDs

The same 595 reaction IDs and their equations were used for both methods. The available bare-PBE0 sGDML fits yielded complete energies for six RDB7 reactions, 29 forward barriers and six reverse barriers. A full-panel sGDML MAE is unavailable for each of the four datasets.

| RDB7 observable | Scored / original | sGDML + D3 MAE | Korny MAE on the same IDs |
|---|---:|---:|---:|
| Reaction energy | 6 / 259 | 37.52 | 36.48 |
| All directional barriers | 35 / 518 | 93.49 | 30.20 |
| Forward barrier | 29 / 259 | 95.68 | 30.51 |
| Reverse barrier | 6 / 259 | 82.90 | 28.70 |

| Other panels | Original reactions | Reaction energies scored | Forward barriers scored | Reverse barriers scored |
|---|---:|---:|---:|---:|
| BH9 | 49 | 0 | 0 | 0 |
| TRIP50 | 4 | 0 | 0 | 0 |
| Grambow radicals | 283 | 0 | 0 | 0 |

All MAEs are in kcal/mol. Each signed error uses the original stoichiometry and electronic reference. The saved split holds out transition-state formulas across the barrier datasets. Stock sGDML needs a fixed atom count and composition. This experiment used one model per composition, charge and multiplicity with matching labelled bare-PBE0 training structures. An equation with an unavailable term retains its ID and missing geometry names in the ledger.

The upstream CPU analytic solver used energies and Cartesian forces with a kernel width of 1.0 and regularization of 1e-8. Each dense solve selected up to 64 molecular geometries in training-fingerprint order and up to 1,800 energy-plus-Cartesian observations. Symmetries were discovered on up to eight training examples. Both methods used the same D3(BJ) correction for the high-level references. Korny's checkpoint also received high-level reaction supervision. These measurements describe this training and coverage. The sGDML run ended before molecular energy and force validation.

[Per-ID equation ledger](../dataset_for_training/requested_public/sgdml-matched/reaction-equations.csv) · [Validation scores](../dataset_for_training/requested_public/sgdml-matched/reaction-scores.json) · [sGDML evaluation script](../experiments/full_public/sgdml_validation.py)
