# Conformer energies and forces

These experiments measure conformational energy and force accuracy. Each uses its own electronic reference and split.

## Ethanol at coupled-cluster level

Korny’s molecule-specific inverse-distance Matérn model was fitted to the authors’ 1,000 training geometries, using CCSD(T)/cc-pVTZ energies and Cartesian forces. Width and ridge were selected on 120 structures drawn from the training file. The final fit used 320 centers, the 12 graph-preserving permutations of ethanol, and all 1,000 training structures. On the separate 1,000-geometry test file the **energy MAE was 0.75 kcal/mol** (RMSE 0.97) and the **force-component MAE was 3.25 kcal/mol/Å** (RMSE 4.39). The energy origin came from training. The complete run took 3.9 seconds and peaked at 458 MiB. The fixed-molecule checkpoint is `dataset_for_training/sGDML/korny_ethanol_cc.npz`. [Authors’ sGDML datasets](https://www.sgdml.org/#datasets)

## Gas-phase Aquamarine

The Korny model in this table was trained only on the QM7-X pilot shard. Dispersion was off, and the AQM references were the separate `ePBE0` and `pbe0FOR` fields. The screen required neutral, closed-shell H/C/N/O molecules with at least two conformers and agreement between the molecular formula of the published SMILES and the stored coordinates. From 726 eligible parents, 16 per size bin were selected by hashing their IDs. For each parent up to six conformers were selected independently of target values. The first selected geometry set the energy origin for that parent and was excluded from the energy metric. All force components were scored. [Aquamarine gas data and property definitions](https://zenodo.org/records/10208010)

| Atom range sampled | Parents | Geometries | Energy differences | Relative energy MAE (kcal/mol) | Force MAE (kcal/mol/Å) |
|---|---:|---:|---:|---:|---:|
| 11–20 | 16 | 62 | 46 | 5.59 | 11.06 |
| 21–38 | 16 | 91 | 75 | 10.71 | 12.71 |
| 41–79 | 16 | 96 | 80 | 17.06 | 11.80 |

## Recomputed MD22

The five SO3LR test-archive trajectories supply PBE0+MBD reference energies and forces. For each, 40 frames were selected by index hash across the trajectory. Each energy MAE uses 39 differences to its first selected frame. The same frozen QM7-X model supplies bare PBE0 for all columns. Adding TS uses the separately fitted Hirshfeld-volume predictor. [SO3LR model and dataset links](https://github.com/general-molecular-simulations/so3lr)

| Archive system | Atoms | Bare energy MAE | With TS | With D3(BJ) | D3(BJ) force MAE |
|---|---:|---:|---:|---:|---:|
| md22_ac | 42 | 20.82 | 20.02 | 19.89 | 14.97 |
| md22_dha | 56 | 16.85 | 16.03 | 16.03 | 13.62 |
| md22_at | 60 | 31.60 | 30.09 | 30.33 | 21.49 |
| md22_sta | 87 | 65.21 | 63.20 | 63.63 | 19.99 |
| md22_bb | 148 | 77.64 | 71.51 | 72.50 | 17.34 |

Energies are in kcal/mol and Cartesian force components are in kcal/mol/Å. Adding dispersion reduces the energy error on all five trajectories. The size of the remaining errors limits transfer of the small QM7-X pilot beyond its training distribution. The recomputed MD22 and AQM data were used exclusively for evaluation.

The reproducible experiments are `experiments.reactivity.ethanol_cc` and `experiments.reactivity.transfer_panels`. The original CC ethanol files, Aquamarine HDF5 and numerical outputs are inside the ignored `dataset_for_training` directory.
