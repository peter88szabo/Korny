# Single-molecule CPU energy and force timings

The tested Korny p11 checkpoint was faster on the first call. Original SO3LR and all three Tauri sizes were faster on repeated calls. On benzene the original model took 2.72 ms per warmed call and Korny took 24.11 ms, an 8.9-fold difference. These measurements apply to the p11 readout. The newer accuracy experiments add kernel terms.

| Warm median per energy plus all Cartesian forces, ms | Methanol, 6 atoms | Ethanol, 9 atoms | Benzene, 12 atoms |
|---|---:|---:|---:|
| Korny p11 high + D3(BJ)-ATM | 10.41 | 17.03 | 24.11 |
| Original SO3LR | 1.60 | 2.34 | 2.72 |
| Tauri-S-2606 | 1.38 | 1.89 | 2.30 |
| Tauri-M-2606 | 1.68 | 2.39 | 2.94 |
| Tauri-L-2606 | 3.63 | 5.06 | 7.76 |

| One-time timings, ms | First model construction | Methanol first call | Ethanol first call | Benzene first call |
|---|---:|---:|---:|---:|
| Korny p11 high + D3(BJ)-ATM | 17 | 28 | 18 | 25 |
| Original SO3LR | 389 | 2,523 | 2,340 | 2,573 |
| Tauri-S-2606 | 318 | 2,457 | 2,143 | 2,423 |
| Tauri-M-2606 | 299 | 2,511 | 2,208 | 2,441 |
| Tauri-L-2606 | 314 | 2,776 | 2,478 | 2,653 |

The benchmark used an AMD Ryzen 9 7900, CPU affinity 9–11, one BLAS/OpenMP thread, batch size one, and float64. Each neutral singlet had the same starting coordinates and displacements across models. Each call returned one energy and an N × 3 array of finite forces; the changing forces confirmed that fresh positions reached the evaluator. The warm sample consists of five coordinate updates after one first call and one settling call. The SO3LR ASE calculator used a 12 Å long-range cutoff and a fresh neighbor workspace per molecule. Wall times include host-side forces and energy, JAX synchronization and ASE overhead. Package imports precede the model-construction timer. First-call times can depend on compilation caches. Five calls characterize latency on this CPU with these inputs. No GPU timings were collected. [Benchmark code](../experiments/full_public/force_timing.py) · [Individual timings and inputs](../dataset_for_training/requested_public/timings)

A separate profile of five p11 benzene force evaluations took 138 ms under instrumentation. Atomic descriptor construction took 56 ms across the model and the bond-environment design took 39 ms. These two measurements locate work that could be optimized after model selection.
