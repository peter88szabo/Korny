# Korny

Kernel-based machine learning model with compression. The NumPy implementation fits molecular energies and conservative forces with species-specific sparse Gaussian or polynomial tensor-train (TT) readouts. Cartesian directional derivatives also support curvature labels and Hessian–vector products. See the [local model](src/local.py) and [derivations](notes/first_ideas.tex).

## Bare PBE0 force fields and switchable dispersion

The DES checkpoint fits `ePBE0` and `pbe0FOR` for 10,139 training geometries.
All 2,575 geometries from 890 reserved DES parent IDs stay together for evaluation.
A separate model fits the per-atom log Hirshfeld volume ratio on the same training
IDs. It predicts all eight elements from geometry. TS differentiates these ratios
with respect to all atomic positions when computing forces. D3 uses PBE0
Becke–Johnson damping and the ATM three-body term. The dispersion-inclusive
archive labels are reserved for evaluating the complete potential. [Dataset release](https://zenodo.org/records/14779793)

```python
import json
from pathlib import Path
from korny.forcefield import ForceField

directory = Path("experiments/des15k")
base = directory / json.loads((directory / "results.json").read_text())["checkpoint"]
volumes = directory / json.loads((directory / "volumes.json").read_text())["checkpoint"]
model = ForceField.load(base, dispersion="ts", volumes=volumes)
energy_eV, forces_eV_A = model.predict(atomic_numbers, positions_angstrom, charge=0, multiplicity=1)
```

Use `dispersion="off"` (default) or `dispersion="d3"` with the same base checkpoint.
Install the package with `python -m pip install -e '.[training,d3]'`. In Smite
select `qchem="PES"`, `pes_name="Korny"`, and supply absolute paths with
`korny_model` and, for TS, `korny_volumes`. Select the correction with
`korny_dispersion`. The adapter converts eV and Å to Hartree and bohr.

| DES validation against PBE0+MBD | Molecular energy MAE (eV) | Force RMSE (eV/Å) |
| --- | ---: | ---: |
| Bare PBE0 model | 0.4405 | 0.3091 |
| Same model + TS | 0.3257 | 0.3093 |
| Same model + D3 | 0.2819 | 0.3092 |

The bare model against its own PBE0 reference has 0.2749 eV energy MAE and
0.3093 eV/Å force RMSE. The TS volume ratio MAE is 0.0208. On the released
10,000-geometry QM7-X comparison the DES-only model plus TS gives 1.410 eV/Å
force RMSE and the original SO3LR checkpoint gives 0.06765 eV/Å. SO3LR had
QM7-X training coverage. [DES scores](experiments/des15k/des_validation.json) ·
[QM7-X scores](experiments/des15k/qm7x_validation.json) ·
[Neural comparison](experiments/des15k/so3lr_comparison.json)

### Charge and spin multiplicity

Pass the total molecular charge and multiplicity (2S+1) for each molecule or
complex. Korny checks electron-count parity and the states present in the
checkpoint. The state multiplies geometry features during fitting. Consequently
changing state at a fixed geometry can change forces as well as energy.

```python
directory = Path("experiments/reactivity")
report = json.loads((directory / "gmtkn_pbe0_states.json").read_text())
model = ForceField.load(directory / report["checkpoint"])
print(model.supported_states, model.prediction_target)
energy_eV, force_eV_A = model.predict(atomic_numbers, positions_angstrom,
                                     charge=0, multiplicity=2)
```

Smite takes the same `charge` and `multiplicity` keys in its Korny PES
configuration. One trajectory follows one fixed electronic surface. Legacy
checkpoints support singlets. The separate experimental BH9/TRIP50 checkpoints
fit high-level relative energies using D3(BJ) during calibration. The singlet
geometry/frequency checkpoint remains available for equilibrium spectroscopy.
The [bare PBE0 fit](experiments/reactivity/gmtkn_pbe0_states.json),
[high-level energy fit](experiments/reactivity/electronic_energy_only.json) and
[constrained fit](experiments/reactivity/electronic_fit.json) give the
checkpoint names, distinct targets and held-out errors.

To share an 8 GB RAM limit across experiment processes on this workstation,
use `experiments/run_limited.sh python -m experiments.reactivity.spin_fit`.
The user systemd slice starts reclaiming memory at 6 GB, counts cached training
files and child workers, and disables swap for those jobs.

### Functional groups on a reactive potential

Experimental checkpoints can add 82 smooth features for twelve oxygen-containing
environments. A carbon-bound aldehyde requires short C–O, C–H and C–C bonds
on one carbon. A hydroperoxide requires a connected C–O–O–H path. Each
environment contributes its occupancy, bond stretches and angle cosines with
linear and quadratic terms. Their fitted coefficients therefore enter energies,
Cartesian forces and frequency calculations. Smooth switching lets a molecule
change functional groups during MD. Total charge and spin multiplicity remain
inputs. The barrier fit learns a shared chemical contribution and separate
doublet and triplet deviations. [Functional-group definitions](src/chemical.py)

On 400 RDB7 test reactions, adding functional groups reduced reaction/barrier
MAEs from 19.04/23.91 to 18.39/23.51 kcal/mol. On 65 BH9/TRIP50 test reactions
the corresponding MAEs fell from 19.02/17.07 to 17.83/16.15 kcal/mol. The
controls use the same sparse kernel, training structures and loss weights.
Ethanol development frequency MAE increased from 118 to 145 cm⁻¹. Use the
recorded checkpoint with `dispersion="d3"` to reproduce these high-level
calibrations. [Fit, controls and chemical subgroup counts](experiments/reactivity/rdb7_balanced_centers128_chemical.json)

Peroxide transfer needs additional data. Among the 159 BH9/TRIP50 training
reactions, one doublet changed organic hydroperoxide occupancy by at least 0.5.
Four of 4,000 RDB7 singlets did so. In three independent archived geometries
of a single isoprene H shift, joint refitting increased forward-barrier MAE
from 4.42 to 30.49 kcal/mol. Freezing the previous potential and fitting only
the chemical correction gave 9.15 kcal/mol. The experimental shared-hydrogen
basis adds smooth donor–H–acceptor features, including C–H–O–O. Checkpoints
record the basis version. [Training coverage](experiments/reactivity/rdb7_chemical_residual.json)
[Peroxy validation and larger-molecule spectra](experiments/reactivity/chemical_validation.json)

The specialized isoprene experiments use the workstation's existing Leuven
archive. The cross-composition experiment fits 91 C₅H₉O₅ calculations at
UCCSD(T)-F12b/cc-pVDZ-F12 and reserves C₅H₉O₃. The within-reaction experiment
uses M06-2X-D3zero/aug-cc-pVTZ IRC energies, gradients and Cartesian Hessians
for existing C₅H₉O₃ conformers. It groups near duplicates and excludes groups
overlapping the external NEB validation. The JSON reports identify each model's
electronic reference and scope. [Composition transfer](experiments/reactivity/peroxy_transfer_bridges.json)
[Grouped conformer fit](experiments/reactivity/isoprene_irc_fit.json)

The third functional-group basis has 162 features. It adds directional O–H···O
hydrogen bonds with carbonyl or peroxy acceptors and four smooth torsional
features along oxygen-containing chains (two terms per torsion). The dihedral
response tends to zero when an adjacent bond angle straightens. The chemistry-only
IRC fit compares all 162 columns with the first 125 columns on identical
conformer groups. Adding hydrogen bonds and torsions reduced the equally weighted
group-validation score from 7.89 to 5.72 kcal/mol. Its three external peroxy
paths gave MAEs of 6.26 kcal/mol for reaction energies, 11.35 for forward
barriers and 8.48 for reverse barriers. [Conformer experiment](experiments/reactivity/isoprene_chemical_conformers.json)

On the broad held-out sets the 162-feature joint fit gave reaction/barrier
MAEs of 18.38/23.38 kcal/mol on RDB7 and 17.35/15.95 on BH9/TRIP50. In a
seven-molecule geometry check it changed benzene connectivity during relaxation.
The same checkpoint ran through Smite on a 17-atom neutral doublet with a
finite-difference force discrepancy of 2.3 × 10⁻⁸ eV/Å and a 3 fs energy
change of −0.0011 kcal/mol. Its reaction and geometry errors limit its use
for dynamical predictions. [Broad fit](experiments/reactivity/rdb7_balanced_centers128_chemical_v3.json)
[Relaxed structures and spectra](experiments/reactivity/chemical_v3_larger.json)
[Smite test](experiments/reactivity/smite_peroxy_chemical_v3.json)

The geometry-preserving 162-feature experiment freezes the previous potential
and fits only the chemical correction. It retained connectivity for all seven
larger-molecule minima and gave 18.93/16.74 kcal/mol on BH9/TRIP50 and
18.97/23.80 on RDB7 (reaction/barrier MAE). Frequency MAE was 90 cm⁻¹ for
acetone, 145 for benzene and 343 for furan. Its checkpoint is
`rdb7-ea561724508fdf4a.npz`. Use D3 for the reported calibration.
[Frozen fit](experiments/reactivity/rdb7_chemical_residual_v3.json)
[Seven spectra and geometries](experiments/reactivity/chemical_v3_frozen_larger.json)

To reproduce the comparison, place the original BH9, TRIP50, RDB7 and NIST
reference files under `dataset_for_training/` and run the commands below from
the repository root in the training environment. The paired peroxy validation
also reads the three archived ORCA saddle-point calculations on this workstation.

```sh
experiments/run_limited.sh python -m experiments.reactivity.matern_fit --compare-kernels
experiments/run_limited.sh python -m experiments.reactivity.barrier_curve --balanced --added-centers 128
experiments/run_limited.sh python -m experiments.reactivity.barrier_curve --balanced --added-centers 128 --chemical
experiments/run_limited.sh python -m experiments.reactivity.chemical_validation
experiments/run_limited.sh python -m experiments.reactivity.barrier_curve --balanced --chemical-residual
experiments/run_limited.sh python -m experiments.reactivity.peroxy_transfer --bridges
experiments/run_limited.sh python -m experiments.reactivity.isoprene_irc_fit
experiments/run_limited.sh python -m experiments.reactivity.isoprene_trajectory_fit
experiments/run_limited.sh python -m experiments.reactivity.isoprene_trajectory_fit --conformers
experiments/run_limited.sh python -m experiments.reactivity.barrier_curve --balanced --added-centers 128 --chemical --chemical-schema 3
experiments/run_limited.sh python -m experiments.reactivity.barrier_curve --balanced --chemical-residual --chemical-schema 3
```

The independent OSOS doublet thermochemistry experiment fits separate bare
PBE0 and W1-F12 energy functions and reserves entire heavy-element compositions.
`python -m experiments.reactivity.osos_fit --chemical` enables the same groups
for this comparison. [RDB7 reference data](https://doi.org/10.1038/s41597-022-01529-6)
[OSOS reference data](https://doi.org/10.1021/acs.jpca.6c03915)

### Reaction and frequency validation

The expanded checkpoint adds 3,887 bare PBE0 QM7-X geometries and 640 SPICE
dipeptide geometries. Its readout has 1,238 coefficients. Training took 189 s
on this workstation, compared with 116 s for the DES-only fit and 16.5 s for
the separate volume predictor. Timing includes basis construction and fitting
and excludes data loading and validation. The expanded model uses the same
volume checkpoint. [Fit and molecule IDs](experiments/reactivity/results.json)

| Bare model MAE in kcal/mol | DES only | Expanded |
| --- | ---: | ---: |
| 61 isomerizations across held-out QM7-X molecule IDs | 64.83 | 25.37 |
| 500 held-out TorsionNet scan energy ranges | 4.03 | 4.29 |
| 5 small-molecule GMTKN reaction energies | 49.99 | 48.15 |
| 5 directional GMTKN transition-state barriers | 47.96 | 49.51 |

Each isomerization compares the lowest available optimized conformer of two
distinct, stoichiometrically identical molecules. Each torsion range uses all
24 constrained conformations. The GMTKN barrier panel contains HCN/HNC
isomerization in both directions and linear inversion of water, hydrogen
sulfide and sulfur dioxide. Switching on either dispersion correction changes
the expanded model's GMTKN MAEs by less than 0.2 kcal/mol. Existing archived
bare PBE0/cc-pVTZ energies for the four G2RC reactions also give large errors
(46.49 kcal/mol MAE for the expanded model). [Fixed-geometry results](experiments/reactivity/augmented_gmtkn.json) ·
[Torsion ranges](experiments/reactivity/augmented_torsions.csv)

The current variants meet 10 kcal/mol MAE for torsion ranges. Both miss that
target for isomerizations, chemical reactions and transition-state barriers.
In the Smite water test the expanded model with TS gives harmonic frequencies
of 421, 3833 and 3870 cm⁻¹, compared with 1649, 3832 and 3943 cm⁻¹ from
spectroscopy. The bending potential is too soft. [Smite checks](experiments/reactivity/smite_water.json) ·
[NIST harmonic references](https://cccbdb.nist.gov/compvibs3x.asp?basis=13&casno=7732185&charge=0&method=6)

### Reproducing the fits

Run from the repository root in the shared Python environment. Two BLAS threads
sufficed for training. The benchmark programs read existing reference files.

```bash
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=1
python -m experiments.des15k.run
python -m experiments.des15k.volumes
python -m experiments.des15k.benchmark des
python -m experiments.des15k.benchmark qm7x
python -m experiments.reactivity.augmented
```

`experiments/reactivity/thermochemistry.py` lists the fixed reactions,
stoichiometry and paths to the existing Behemoth GMTKN manifests. The CLI takes
`--benchmark-root` for another checkout. `experiments/reactivity/torsions.py`
takes a model checkpoint and a JSON destination and also writes per-fragment
reference and predicted barriers. The raw release archives stay under
`dataset_for_training/`.

## First QM7-X experiment

On 30 held-out molecule IDs, a smooth element-pair potential with a sparse Gaussian correction achieved **0.639 eV/Å force MAE**. The pair potential alone achieved **0.661 eV/Å** with a 4.9 kB checkpoint. At the two ranks tested, the standalone TT took longer to fit and evaluate than the standalone sparse Gaussian. These results cover one sample and one molecule-disjoint split of the sulfur- and chlorine-containing shard.

### Data and target convention

The published MD5 of `8000.xz` is `c893ae88b8f5c32541c3f024fc1daa45`, matching the local archive. It is an XZ-compressed tar containing one file, `8000.hdf5` (826,407,136 bytes). The HDF5 file has 203 top-level molecule IDs, 215 stereoisomer IDs and 430 optimized conformer records. Each conformer has 100 displaced geometries, giving 43,430 records. Every molecule has seven heavy atoms and 8–17 atoms including hydrogens. Of the 203 IDs, 192 contain sulfur and 14 contain chlorine, with three containing both. All records have the same 42 property keys. [Original data release](https://zenodo.org/records/4288677) · [Property definitions and units](https://pmc.ncbi.nlm.nih.gov/articles/PMC7854709/)

| HDF5 fields | Role in a geometry-based force field |
|---|---|
| `atNUM`, `atXYZ` | Inference inputs. Nuclear species and positions in Å determine the pair distances and species-resolved radial and angular descriptors. |
| `eAT`, `pbe0FOR` | Pilot targets. Use **negative** stored atomization energy in eV with the stored PBE0 forces in eV/Å. Across geometries of each ID, `ePBE0 + eAT` is constant within about 0.000001 eV. |
| `ePBE0+MBD`, `totFOR` | Matched energy and forces for the dispersion-inclusive surface. `eMBD` and `vdwFOR` expose the dispersion contribution separately. |
| `hRAT`, `hVOL`, `atC6`, `atPOL`, `hCHG` | Auxiliary quantum targets for volume, dispersion or charge models. A geometry-only predictor must infer any such quantity it needs at deployment. |
| Orbital energies, gap, dipole, quadrupole, response tensors | Possible targets for separate property models. The pilot uses only geometry at prediction time. |

An audit of every geometry confirmed finite PBE0 fitting labels and matching coordinate and force shapes. The maximum absolute discrepancy in `ePBE0+MBD - ePBE0 - eMBD` was about 0.000001 eV. The corresponding discrepancy in `totFOR - pbe0FOR - vdwFOR` reached 0.000620 eV/Å for one Cartesian component. The experiment uses the directly supplied `pbe0FOR`.

### Protocol and results

Seed 7108 selects one conformer per molecule and four distinct displacements of that conformer, plus its optimized geometry. The split is stratified by the presence of chlorine. Training uses 143 molecules (715 geometries), validation 30 (150), and test 30 (150). All five geometries of any molecule stay together. Mean elemental energy offsets, feature scaling, center selection, pair coefficients and readout coefficients use training records only. The offsets are empirical composition coefficients. The archive's optimized geometries use DFTB3+MBD, so their PBE0 force labels are retained. [QM7-X methodology](https://pmc.ncbi.nlm.nih.gov/articles/PMC7854709/)

All kernel fits use H/C/N/O/S/Cl, three radial channels per species, angular degrees 1 and 2, a 3.6 Å cutoff, and up to 64 inducing centers per species. This gives 360 descriptor components in 48 blocks. TT fits use degree two and the stated rank cap, with the initial TT-SVD cores. The energy residual and all force components enter one ridge fit. Each force row is multiplied by 4 Å divided by the square root of the number of Cartesian components. The pair baseline fits a smooth radial energy for each unordered element pair; the combined fits learn the remaining energy and force residuals.

| Model | Energy MAE (eV/molecule) | Displacement energy MAE (eV) | Force MAE (eV/Å) | Fit (s) | Energy + force call (ms) |
|---|---:|---:|---:|---:|---:|
| Composition only | 1.932 | 2.974 | 1.186 | — | — |
| Sparse Gaussian | 1.501 | 1.828 | 1.056 | 4.89 | 4.55 |
| TT, rank cap 12 | 1.649 | 2.331 | 1.104 | 11.78 | 14.32 |
| TT, rank cap 32 | 1.599 | 2.087 | 1.062 | 20.43 | 26.12 |
| Pair potential | 1.329 | 0.772 | 0.661 | 0.48 | 0.14 |
| Pair + Gaussian | **1.070** | **0.697** | **0.639** | 5.53 | 4.84 |
| Pair + TT, rank cap 12 | 1.148 | 0.784 | 0.660 | 12.51 | 14.83 |

Errors refer to the test split. Displacement energy is the energy difference to the corresponding optimized geometry. Force MAE pools Cartesian components. The pair-plus-Gaussian force RMSE was 1.037 eV/Å. Timings use a single BLAS thread on an AMD Ryzen 9 7900. Fitting time includes center selection, TT construction or pair fitting as applicable, and coefficient regression. It excludes shared data loading and the elemental offset fit. Calls include descriptors, composition offsets and any pair term, measured over three warmed repetitions on 12 validation geometries. Dense TT cores required 14.2 MB and 99.9 MB at the two rank caps, compared with 1.13 MB for the Gaussian head. The tensor representation size depends on the basis width and adjacent bond ranks at each site.

### Reproducing the fits

Use Python with NumPy and h5py. From the repository root, extract the original archive if the HDF5 file is absent.

```sh
data=dataset_for_training/QM7_X/8000_starting_set
tar -xJkf "$data/8000.xz" -C "$data" 8000.hdf5
```

Run the consistency checks and experiment from the repository root. The experiment accepts `--data` and `--out` for other file and output locations.

```sh
python src/main.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.qm7x_pilot.run
```

The [results](experiments/qm7x_pilot/results.json) include all split metrics, checkpoint sizes, settings and elemental offsets. The [prediction table](experiments/qm7x_pilot/predictions.csv) lists each geometry ID, split, energy prediction and force error totals. The [runner](experiments/qm7x_pilot/run.py) saves local checkpoints alongside them.

To recover the complete pair-plus-Gaussian prediction, load both checkpoints and the composition offsets. Supply an integer array `Z` and an `(N, 3)` coordinate array `R` in Å.

```python
import json
import numpy as np
from experiments.qm7x_pilot.run import OUT, PairBaseline, predict
from src.local import Molecules

settings = json.loads((OUT / "results.json").read_text())["config"]
offsets = np.array(settings["composition_offsets_eV"])
local = Molecules.load_local(OUT / "pair_gaussian.npz")
pair = PairBaseline.load(OUT / "pair.npz")
energy_eV, force_eV_per_A = predict(Z, R, offsets, local, pair)
```

### Dispersion corrections

Each correction has its own module under `src/dispersion/`. The local environment model and directional derivatives live in `src/local.py` and `src/jets.py`. The original command-line entry point is `src/main.py`.

| Module | Inputs and output |
|---|---|
| [ts.py](src/dispersion/ts.py) | Native NumPy TS. `ts_energy` accepts positions in bohr and log volume ratios, including first and mixed derivatives. `ts_energy_forces` accepts positions in Å and a volume predictor and returns eV and eV/Å. |
| [volumes.py](src/dispersion/volumes.py) | Predicts atom-wise log Hirshfeld ratios and their Cartesian Jacobian. The fit uses scaled radial/angular features, a ridge of 0.01 and the 715 training geometries. An isolated atom has predicted ratio one. |
| [d3.py](src/dispersion/d3.py) | Grimme D3(BJ) energy and gradient through simple-dftd3, with explicit functional and ATM switch. Coordinates, energy and gradients use atomic units. |
| [mbd.py](src/dispersion/mbd.py) | MBD@rsSCS molecular energy through libMBD using volume ratios supplied as logs and H/C/N/O/S/Cl element identities. Coordinates and energy use atomic units. |

The PBE0 TS model uses Fermi damping with `sR=0.96` and `d=20`. PBE0 MBD@rsSCS uses `beta=0.85`. The six neutral free-atom triples come from upstream libMBD. [Functional parameters](https://github.com/libmbd/libmbd/blob/77d7ab424d21f43dea9f57734b355ce4ea4d1a42/src/mbd_damping.F90) · [Damping defaults](https://github.com/libmbd/libmbd/blob/77d7ab424d21f43dea9f57734b355ce4ea4d1a42/src/mbd_defaults.f90) · [Atomic coefficients](https://github.com/libmbd/libmbd/blob/77d7ab424d21f43dea9f57734b355ce4ea4d1a42/src/mbd_vdw_param.f90)

On the 150 held-out geometries, **the geometry-only TS correction had a 0.446 meV energy MAE** against the stored `eTS`. Its volume-ratio MAE was 0.0121. Its forces include the derivative of the learned volumes. For one random Cartesian direction per geometry, the largest discrepancy against central differences of its own energy was below 5 × 10⁻¹² eV/Å. QM7-X supplies MBD force labels; the TS force test measures energy–force consistency.

With the stored quantum `hRAT`, Korny and Behemoth both agreed with upstream libMBD TS energies to within 1 × 10⁻¹⁶ eV. Upstream TS with those same volumes differed from stored `eTS` by 0.316 meV on average, with a maximum of 1.486 meV. The cause of this residual is unverified. Upstream MBD@rsSCS reproduced `eMBD` within 0.00051 meV. Native Behemoth PBE0-D3(BJ)+ATM agreed with simple-dftd3 on energies and analytical gradients. Its dispersion energies differed from the dataset's MBD correction by 0.0956 eV on average. Adding this D3 correction to the existing pair-plus-Gaussian PBE0 model gave **1.060 eV molecular energy MAE and 0.640 eV/Å force MAE** against the atom-referenced PBE0+MBD targets. [Audit results](experiments/qm7x_pilot/dispersion.json)

For the geometry-only TS variant, add the following correction to the PBE0 prediction. Select one dispersion backend for the desired energy surface.

```python
from src.dispersion.ts import ts_energy_forces
from src.dispersion.volumes import HirshfeldVolumes

volume_model = HirshfeldVolumes.load(OUT / "volumes.npz")
dispersion_eV, dispersion_forces = ts_energy_forces(Z, R, volume_model)
```

To reproduce the audit with an existing matching Behemoth release build on this Linux workstation, run the following from the Korny repository. The probe calls the native Behemoth TS and D3 routines directly. The Python-side dependencies are NumPy, h5py, simple-dftd3's `dftd3` package and `pymbd` built against the matching native libMBD. The audit trains and exports `volumes.npz` and reads the pilot split and checkpoints produced by `run.py`.

```sh
behemoth=../Behemoth/target/release
rustc --edition=2021 -O -C panic=abort experiments/qm7x_pilot/behemoth_probe.rs \
  --extern "Behemoth=$behemoth/libBehemoth.rlib" -L "dependency=$behemoth/deps" \
  -L native=/usr/lib/x86_64-linux-gnu/openblas-pthread -o /tmp/korny-native-dispersion
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  python -m experiments.qm7x_pilot.dispersion /tmp/korny-native-dispersion
```

### Selecting an eigenfunction kernel

The public reaction experiment accepts an operator name. Its `--help` output lists all eleven families, including Chebyshev, Krawtchouk, Hahn, Hermite, Laguerre, Legendre, Jacobi, both interval-Laplacian bases, Bessel, and graph diffusion. The `tensor` step selects the extended Legendre basis. The conventional radial profiles retain their existing comparison entry point. Each saved force field restores its own basis when loaded.

```bash
python -m experiments.full_public.trial_operator_kernels --help
experiments/run_limited.sh python -m experiments.full_public.trial_operator_kernels fit-family --family chebyshev
experiments/run_limited.sh python -m experiments.full_public.trial_operator_kernels fit-family --family krawtchouk
```

With the archived input caches present, the expanded-source H-transfer experiments use the following sequence. The `rdb19` and `papers` steps evaluate the frozen parents once on the new data. Append `--angular` to each of the six commands to reproduce the angular-window controls after the unrestricted runs. The resulting general checkpoints use Korny's standard `ForceField.load(path, dispersion="d3")` interface.

```bash
experiments/run_limited.sh python -m experiments.full_public.trial_h_transfer prepare
experiments/run_limited.sh python -m experiments.full_public.h_transfer_sources rdb19
experiments/run_limited.sh python -m experiments.full_public.h_transfer_sources papers
experiments/run_limited.sh python -m experiments.full_public.h_transfer_reoptimized
experiments/run_limited.sh python -m experiments.full_public.trial_h_transfer chebyshev
experiments/run_limited.sh python -m experiments.full_public.trial_h_transfer krawtchouk
```

The dedicated neutral-doublet fit runs with `python -m experiments.full_public.trial_native_h_transfer` inside the same memory wrapper. It saves both native potentials alongside the validation results. The native loader is `NativeRadicalPotential.load(path)` in `experiments.full_public.native_irc_pes`. [Expanded-source benchmarks and training coverage](report/hydrogen-transfer-kernels.md)
