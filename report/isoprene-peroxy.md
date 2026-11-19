# Isoprene–peroxy reference data and model fits

The source is `/home/peter/Dropbox/Research_Leuven/LIM1_reevol/Ready_calc`. The inventory traversed each subdirectory and its compressed snapshots. Extracted coordinates, energies, derivatives and archive-only files are under Korny’s ignored `dataset_for_training/Leuven_isoprene/` directory. The experiments read existing calculations; no new quantum-chemistry jobs were submitted.

## What the archive contains

| Verified item | Count | Interpretation |
|---|---:|---|
| Archives inspected, including nested archives | 122 | All completed without a read error |
| Distinct outputs and logs checked | 7,030 | Identical file contents were parsed once |
| Electronic records extracted | 7,854 | Includes multiple electronic stages of a compound job |
| Converged ORCA geometry optimizations | 5,440 | Before geometric deduplication across methods and files |
| Optimizations with a complete printed frequency list | 5,404 | 3,462 with no internal mode below −20 cm⁻¹, 1,832 with exactly one, 110 with more than one |
| ORCA CCSD(T)-F12/RI single points | 2,110 | Confirmed against the explicitly printed F12 CC energy |
| Molpro UCCSD(T)-F12b/cc-pVDZ-F12 single points | 142 | Converged; the printed total and summary energy agree |
| Output stages with an independently matched XYZ / Hessian / gradient file | 5,544 / 3,509 / 3,469 | Atom identities and coordinates checked; gradients also require an energy match |
| Additional ORCA XYZ trajectory frames with explicit electronic energy | 135,592 | 7,603 distinct XYZ file contents linked to a verified job stage; frames include nonstationary geometries |

The TS count above is a **frequency-screened candidate count**. A single imaginary mode does not identify its connected end species. Compound DFT/CC jobs contribute distinct level-tagged records. Solvent calculations are indexed separately. The importer leaves unverified stages and unlabelled XYZs in the provenance inventory. In particular, 858 incomplete ORCA logs were excluded by the termination check. Individual trajectories may share frames with each other or with the final optimization records. The trajectory frame count is not a count of independent molecules or additional CC calculations.

## Gas-phase private fit

The energy fits were separate for CC and DFT. Each source level, composition, charge and multiplicity has its own training energy origin. Fits were assessed on held-out connectivity groups. The archived DFT gradient was used where matched; remaining converged training geometries supplied approximate stationarity constraints. The CC fit uses DFT geometry constraints as a mixed-level approximation.

| Doublet test quantity | Training energy records | Held-out structures | Held-out graphs | Parent MAE | Refit MAE |
|---|---:|---:|---:|---:|---:|
| CC relative electronic energy (kcal/mol) | 952 | 159 | 6 | 15.12 | 5.93 |
| DFT relative electronic energy (kcal/mol) | 2,034 | 609 | 20 | 17.44 | 7.88 |

The doublet derivative constraints used 2,034 training geometries, 1,148 with matched measured gradients. The CC fit also used 286 singlet energy records. The held-out singlet structures did not have a matching training reference origin, so this split supplies no singlet MAE. Only one usable triplet CC training energy was available, insufficient to constrain triplet energy differences. All 883 records of the reserved C₅H₉O₃ doublet composition were withheld at this stage.

Across all states, 1,239 CC and 2,438 DFT energy records entered the training split. The remaining groups were reserved for development and testing. The gas-phase calibration excluded 221 solution or unsupported records, 110 same-geometry/method/state repeats, and 22 non-CC, nonoptimized final geometries. The additional trajectory frames are available for future off-equilibrium fits; they were not treated as equilibrium force targets or included in the figures above. The present fit uses gradients and approximate stationary geometries; 3,509 indexed Hessian matches are available and were not all fitted.

## Private model artifacts

- The broader CC checkpoint is `dataset_for_training/Leuven_isoprene/leuven-cc-fd9b63b8606c.npz`.
- The specialized checkpoint after cross-validation on the 16 H-shift paths is `dataset_for_training/Leuven_isoprene/leuven-hshift-all-486181d051b0.npz`. It is an experimental neutral-doublet specialization.

Both load with `ForceField.load(path, dispersion="d3")`. Supply `charge=0, multiplicity=2` for the neutral doublet. The public Korny base still targets bare PBE0. These separately named private corrections target archived higher-level energy differences.

The private SQLite databases are `files.sqlite` for source provenance, `labels.sqlite` for coordinates, electronic energies and derivative matches, and `trajectories.sqlite` for electronic XYZ frames. Reproducible readers are `leuven_files.py`, `leuven_labels.py`, `leuven_derivatives.py` and `leuven_trajectories.py` under `experiments/reactivity/`. [Matched sGDML/SO3LR comparisons and Gamma barrier references](reaction-benchmarks.md)
