# H-transfer kernels on the expanded reaction data

On 265 coupled-cluster channels in the held-out AutoMeKin cBD-CN network, p87 reduced pooled barrier MAE from 19.24 to 17.17 kcal/mol. On the 172 held RDB19-Rad H transfers, the dedicated p67 + Chebyshev model reached 3.18 reaction and 2.93 pooled barrier MAE. The general checkpoints remain above 5 kcal/mol on the mixed panel.

The p86 Chebyshev and p87 Krawtchouk fits extend p73 and p82 respectively. The p88 and p89 controls refit the same parents on the same training examples with an additional angular window. The updated general checkpoints include D3(BJ) and ATM once. The dedicated radical checkpoints learn the archive's dispersion-inclusive electronic energy. Energy MAEs are in kcal/mol.

## Training coverage

| Source | Training | Held out | Quantity entering the fit |
|---|---:|---:|---|
| RDB7 / BH9 / Grambow | 11,667 / 400 / 1,084 | 259 / 47 / 283 | Endpoint gaps and midpoint heights |
| H transfers within RDB7 / BH9 / Grambow | 2,052 / 77 / 366 | 53 / 13 / 83 | Mapped exchanges or the published BH9 HAT family, included above |
| RDB19-Rad | 4,974 paths | 626 paths | Electronic gaps and midpoint heights on both endpoint complexes and their common saddle |
| RDB19-Rad H-transfer subset | 1,254 | 172 | H atom abstractions and intramolecular H migrations, included above |
| AutoMeKin connected DFT channels | 538 | 329 | Electronic gaps and midpoint heights |
| AutoMeKin connected CC channels | 51 canonical + 446 local | 265 local | Electronic gaps and midpoint heights |
| AutoMeKin fragmented CC channels | 40 canonical + 22 local | 8 local | Endpoint fragment energies summed by channel |
| AIMNet2-NSE supplement | 11,507 singlet/doublet geometries | 369 singlet geometries | Relative energy within each scan or IRC series and native Hamiltonian |
| BASChem19 | 15 reactions | 4 reactions | Forward barriers |
| Reoptimized RDB7 (PM6 / HF / r²SCAN-3c) | 9,584 / 10,438 / 11,310 reaction IDs | 207 / 231 / 252 | Three training-only energy directions enter the coefficient prior |

The reoptimized archives overlap RDB7 reaction IDs. AutoMeKin has multiple methods on many of the same channels. The RDB19 counts correspond to 16,800 stationary geometries over both splits, with three per path. Interior IRC steps were examined separately for the H-transfer audit. The two fits also retain the 91,156 bare-PBE0 anchor geometries and the NIST and Grambow derivative constraints. Only the locally deposited AIMNet2-NSE supplement entered this fit.

## Validation on the downloaded reaction data

These rows use the deployed coefficients. The native electronic level used to score each row is shown. The cBD-CN network is held out in its entirety. RDB19-Rad holds out complete molecular formulas.

| Dataset | Model | N | Reaction | Forward | Reverse | All barriers |
|---|---|---:|---:|---:|---:|---:|
| RDB19-Rad | p73 Chebyshev | 626 | 19.723 | 21.140 | 19.140 | 20.140 |
| RDB19-Rad | p86 Chebyshev + H bonds | 626 | 19.732 | 20.597 | 18.866 | 19.731 |
| RDB19-Rad | p88 Chebyshev + angular H bonds | 626 | 19.753 | 20.528 | 18.877 | 19.703 |
| RDB19-Rad | p82 Krawtchouk | 626 | 19.712 | 21.087 | 19.070 | 20.078 |
| RDB19-Rad | p87 Krawtchouk + H bonds | 626 | 19.732 | 20.262 | 18.698 | 19.480 |
| RDB19-Rad | p89 Krawtchouk + angular H bonds | 626 | 19.764 | 20.175 | 18.688 | 19.432 |
| RDB19-Rad H transfers | p73 Chebyshev | 172 | 10.520 | 13.689 | 13.293 | 13.491 |
| RDB19-Rad H transfers | p86 Chebyshev + H bonds | 172 | 10.579 | 13.221 | 12.884 | 13.053 |
| RDB19-Rad H transfers | p88 Chebyshev + angular H bonds | 172 | 10.592 | 13.388 | 12.958 | 13.173 |
| RDB19-Rad H transfers | p82 Krawtchouk | 172 | 10.557 | 13.695 | 13.349 | 13.522 |
| RDB19-Rad H transfers | p87 Krawtchouk + H bonds | 172 | 10.621 | 13.056 | 12.858 | 12.957 |
| RDB19-Rad H transfers | p89 Krawtchouk + angular H bonds | 172 | 10.637 | 13.136 | 12.827 | 12.981 |
| AutoMeKin · DLPNO-CCSD(T)-F12/cc-pVDZ-F12 | p73 Chebyshev | 265 | 9.786 | 19.648 | 18.683 | 19.166 |
| AutoMeKin · DLPNO-CCSD(T)-F12/cc-pVDZ-F12 | p86 Chebyshev + H bonds | 265 | 9.449 | 17.662 | 17.027 | 17.344 |
| AutoMeKin · DLPNO-CCSD(T)-F12/cc-pVDZ-F12 | p88 Chebyshev + angular H bonds | 265 | 9.786 | 19.615 | 18.653 | 19.134 |
| AutoMeKin · DLPNO-CCSD(T)-F12/cc-pVDZ-F12 | p82 Krawtchouk | 265 | 9.821 | 19.724 | 18.750 | 19.237 |
| AutoMeKin · DLPNO-CCSD(T)-F12/cc-pVDZ-F12 | p87 Krawtchouk + H bonds | 265 | 9.462 | 17.484 | 16.849 | 17.167 |
| AutoMeKin · DLPNO-CCSD(T)-F12/cc-pVDZ-F12 | p89 Krawtchouk + angular H bonds | 265 | 9.821 | 19.696 | 18.728 | 19.212 |
| AutoMeKin · wB97X-D3/def2-TZVP | p73 Chebyshev | 329 | 14.051 | 14.227 | 9.506 | 11.866 |
| AutoMeKin · wB97X-D3/def2-TZVP | p86 Chebyshev + H bonds | 329 | 14.041 | 14.015 | 9.620 | 11.818 |
| AutoMeKin · wB97X-D3/def2-TZVP | p88 Chebyshev + angular H bonds | 329 | 14.051 | 14.223 | 9.488 | 11.855 |
| AutoMeKin · wB97X-D3/def2-TZVP | p82 Krawtchouk | 329 | 14.078 | 14.227 | 9.515 | 11.871 |
| AutoMeKin · wB97X-D3/def2-TZVP | p87 Krawtchouk + H bonds | 329 | 14.062 | 13.918 | 9.661 | 11.790 |
| AutoMeKin · wB97X-D3/def2-TZVP | p89 Krawtchouk + angular H bonds | 329 | 14.078 | 14.222 | 9.490 | 11.856 |
| AutoMeKin fragments · DLPNO-CCSD(T)-F12/cc-pVDZ-F12 | p73 Chebyshev | 8 | 10.628 | 21.275 | 12.894 | 17.085 |
| AutoMeKin fragments · DLPNO-CCSD(T)-F12/cc-pVDZ-F12 | p86 Chebyshev + H bonds | 8 | 10.593 | 21.499 | 13.152 | 17.326 |
| AutoMeKin fragments · DLPNO-CCSD(T)-F12/cc-pVDZ-F12 | p88 Chebyshev + angular H bonds | 8 | 10.628 | 21.640 | 13.259 | 17.449 |
| AutoMeKin fragments · DLPNO-CCSD(T)-F12/cc-pVDZ-F12 | p82 Krawtchouk | 8 | 10.492 | 21.291 | 12.981 | 17.136 |
| AutoMeKin fragments · DLPNO-CCSD(T)-F12/cc-pVDZ-F12 | p87 Krawtchouk + H bonds | 8 | 10.449 | 21.538 | 13.271 | 17.404 |
| AutoMeKin fragments · DLPNO-CCSD(T)-F12/cc-pVDZ-F12 | p89 Krawtchouk + angular H bonds | 8 | 10.492 | 21.559 | 13.248 | 17.404 |

The four withheld BASChem19 reactions provide forward barriers. The archived benzopinacol singlet scan uses deviations from its mean energy over 369 frames.

| Dataset and observable | p73 | p86 | p88 | p82 | p87 | p89 |
|---|---:|---:|---:|---:|---:|---:|
| BASChem19 forward barrier | 26.915 | 24.065 | 25.646 | 25.089 | 24.458 | 25.358 |
| AIMNet2-NSE singlet relative scan energy | 17.939 | 17.987 | 17.948 | 18.039 | 18.085 | 18.046 |

## Comparison on identical public IDs

| Panel | Model | N | Reaction | Forward | Reverse | All barriers |
|---|---|---:|---:|---:|---:|---:|
| All public reactions | p73 Chebyshev | 589 | 9.060 | 10.228 | 10.718 | 10.473 |
| All public reactions | p82 Krawtchouk | 589 | 9.000 | 10.277 | 10.747 | 10.512 |
| All public reactions | p86 Chebyshev + H bonds | 589 | 9.057 | 10.429 | 10.980 | 10.704 |
| All public reactions | p87 Krawtchouk + H bonds | 589 | 9.003 | 10.521 | 11.062 | 10.791 |
| All public reactions | p88 Chebyshev + angular H bonds | 589 | 9.034 | 10.238 | 10.733 | 10.485 |
| All public reactions | p89 Krawtchouk + angular H bonds | 589 | 8.974 | 10.305 | 10.759 | 10.532 |
| All public reactions | Original SO3LR | 589 | 13.896 | 13.292 | 14.016 | 13.654 |
| All public reactions | Tauri-M-2606 | 589 | 6.146 | 8.282 | 7.888 | 8.085 |
| All public reactions | Tauri-L-2606 | 589 | 4.471 | 5.987 | 6.063 | 6.025 |
| H transfer | p73 Chebyshev | 149 | 8.445 | 9.591 | 9.036 | 9.314 |
| H transfer | p82 Krawtchouk | 149 | 8.397 | 9.520 | 9.164 | 9.342 |
| H transfer | p86 Chebyshev + H bonds | 149 | 8.418 | 9.874 | 9.538 | 9.706 |
| H transfer | p87 Krawtchouk + H bonds | 149 | 8.384 | 9.792 | 9.760 | 9.776 |
| H transfer | p88 Chebyshev + angular H bonds | 149 | 8.414 | 9.527 | 8.994 | 9.261 |
| H transfer | p89 Krawtchouk + angular H bonds | 149 | 8.362 | 9.426 | 9.108 | 9.267 |
| H transfer | Original SO3LR | 149 | 11.425 | 12.586 | 14.599 | 13.592 |
| H transfer | Tauri-M-2606 | 149 | 3.306 | 6.212 | 6.018 | 6.115 |
| H transfer | Tauri-L-2606 | 149 | 2.393 | 4.018 | 4.333 | 4.175 |
| RDB7 H transfer | p73 Chebyshev | 53 | 7.603 | 7.067 | 9.178 | 8.123 |
| RDB7 H transfer | p82 Krawtchouk | 53 | 7.638 | 7.230 | 9.377 | 8.303 |
| RDB7 H transfer | p86 Chebyshev + H bonds | 53 | 7.504 | 8.035 | 10.655 | 9.345 |
| RDB7 H transfer | p87 Krawtchouk + H bonds | 53 | 7.550 | 8.189 | 11.130 | 9.659 |
| RDB7 H transfer | p88 Chebyshev + angular H bonds | 53 | 7.585 | 6.960 | 9.153 | 8.057 |
| RDB7 H transfer | p89 Krawtchouk + angular H bonds | 53 | 7.613 | 7.092 | 9.348 | 8.220 |
| RDB7 H transfer | Original SO3LR | 53 | 11.020 | 11.617 | 11.141 | 11.379 |
| RDB7 H transfer | Tauri-M-2606 | 53 | 4.392 | 9.859 | 8.741 | 9.300 |
| RDB7 H transfer | Tauri-L-2606 | 53 | 3.387 | 5.513 | 6.556 | 6.034 |
| BH9 H transfer | p73 Chebyshev | 13 | 19.393 | 20.544 | 9.663 | 15.104 |
| BH9 H transfer | p82 Krawtchouk | 13 | 18.920 | 19.692 | 10.025 | 14.858 |
| BH9 H transfer | p86 Chebyshev + H bonds | 13 | 19.691 | 21.546 | 9.630 | 15.588 |
| BH9 H transfer | p87 Krawtchouk + H bonds | 13 | 19.331 | 20.669 | 9.703 | 15.186 |
| BH9 H transfer | p88 Chebyshev + angular H bonds | 13 | 19.355 | 21.196 | 9.169 | 15.182 |
| BH9 H transfer | p89 Krawtchouk + angular H bonds | 13 | 18.868 | 20.466 | 9.572 | 15.019 |
| BH9 H transfer | Original SO3LR | 13 | 12.634 | 18.720 | 14.523 | 16.622 |
| BH9 H transfer | Tauri-M-2606 | 13 | 2.747 | 5.581 | 5.133 | 5.357 |
| BH9 H transfer | Tauri-L-2606 | 13 | 2.850 | 2.938 | 2.710 | 2.824 |
| Grambow H transfer | p73 Chebyshev | 83 | 7.268 | 9.487 | 8.847 | 9.167 |
| Grambow H transfer | p82 Krawtchouk | 83 | 7.233 | 9.390 | 8.894 | 9.142 |
| Grambow H transfer | p86 Chebyshev + H bonds | 83 | 7.235 | 9.220 | 8.811 | 9.016 |
| Grambow H transfer | p87 Krawtchouk + H bonds | 83 | 7.201 | 9.112 | 8.894 | 9.003 |
| Grambow H transfer | p88 Chebyshev + angular H bonds | 83 | 7.230 | 9.339 | 8.864 | 9.102 |
| Grambow H transfer | p89 Krawtchouk + angular H bonds | 83 | 7.195 | 9.187 | 8.882 | 9.034 |
| Grambow H transfer | Original SO3LR | 83 | 11.494 | 12.243 | 16.819 | 14.531 |
| Grambow H transfer | Tauri-M-2606 | 83 | 2.700 | 3.981 | 4.418 | 4.200 |
| Grambow H transfer | Tauri-L-2606 | 83 | 1.686 | 3.233 | 3.167 | 3.200 |

## Forming and breaking bond lengths

D denotes the donor and A the acceptor in the reference reaction direction. Lengths (Å) and angles (degrees) are medians of individual measurements. RDB7 and Grambow require an explicitly mapped single H switch between two heavy atoms and a matching ordering of the archived stationary geometries. BH9 contributes its published HAT labels to the energy tables. RDB19 uses the author's atom-ordered educt and product complexes and verifies the changing heavy partner at both ends.

| Archive and split | N | D–H reactant | H–A reactant | D–H saddle | H–A saddle | D–H product | H–A product | D–H–A saddle angle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RDB7 validation | 53 | 1.091 | 3.112 | 1.526 | 1.274 | 2.720 | 1.086 | 90.446 |
| Grambow validation | 83 | 1.091 | 2.676 | 1.376 | 1.314 | 2.546 | 1.093 | 93.077 |
| RDB19-Rad train | 1254 | 1.087 | 3.014 | 1.396 | 1.292 | 2.879 | 1.092 | 175.730 |
| RDB19-Rad validation | 172 | 1.091 | 2.892 | 1.377 | 1.310 | 2.843 | 1.091 | 175.748 |

The two initial IRC steps moved the bond difference r(D–H) − r(H–A) in opposite directions on 172 of 172 RDB19-Rad validation H transfers. This is a property of the archived reference paths. It does not establish that a Korny trajectory reaches the two endpoints.

## Frequencies and relaxed geometry

| Model | Propane | Benzene | Ethanol | 1,3-butadiene | Equal-molecule mean (cm⁻¹) | Connectivity retained | Grambow saddle sign |
|---|---:|---:|---:|---:|---:|---:|---:|
| p73 Chebyshev | 93.0 | 170.7 | 110.1 | 155.6 | 132.3 | 4/4 | 212/278 |
| p82 Krawtchouk | 93.1 | 157.4 | 107.6 | 148.4 | 126.6 | 4/4 | 213/278 |
| p86 Chebyshev + H bonds | 93.8 | 188.5 | 115.5 | 167.8 | 141.4 | 4/4 | 212/278 |
| p87 Krawtchouk + H bonds | 94.6 | 177.3 | 111.5 | 161.9 | 136.3 | 4/4 | 212/278 |
| p88 Chebyshev + angular H bonds | 93.0 | 170.7 | 110.1 | 155.6 | 132.3 | 4/4 | 211/278 |
| p89 Krawtchouk + angular H bonds | 93.1 | 157.4 | 107.6 | 148.4 | 126.6 | 4/4 | 213/278 |

## Dedicated neutral-doublet potential

Here the parent is p67, trained at the RDB19-Rad electronic level. Both extensions use 4,974 training paths, two flank energy and projected-force measurements per path, and the change in force across the saddle where available. Each row uses a scalar potential with analytic forces. Curvature is a signed directional wavenumber from the two initial IRC steps. It is measured in cm⁻¹.

| Held-out RDB19-Rad panel | Model | N | Reaction | Forward | Reverse | All barriers | Saddle curvature MAE | Downhill sign |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | p67 | 626 | 4.019 | 3.927 | 3.723 | 3.825 | 237.2 | 595/626 |
| all | chebyshev | 626 | 4.034 | 3.906 | 3.699 | 3.802 | 236.4 | 595/626 |
| all | krawtchouk | 626 | 4.034 | 3.906 | 3.701 | 3.803 | 235.4 | 595/626 |
| H transfers | p67 | 172 | 3.187 | 3.161 | 2.851 | 3.006 | 129.3 | 170/172 |
| H transfers | chebyshev | 172 | 3.178 | 3.114 | 2.749 | 2.932 | 129.2 | 170/172 |
| H transfers | krawtchouk | 172 | 3.181 | 3.104 | 2.767 | 2.935 | 128.8 | 170/172 |

Changes on the 589 common IDs were paired by reaction. Intervals resample whole transition-state formulas within each source 4,000 times.

| Angular checkpoint and parent | Quantity | MAE change (kcal/mol) | 95% interval |
|---|---|---:|---:|
| p88 vs p73 | reaction | -0.025 | [-0.062, +0.006] |
| p88 vs p73 | forward | +0.010 | [-0.099, +0.144] |
| p88 vs p73 | reverse | +0.015 | [-0.111, +0.156] |
| p88 vs p73 | barriers | +0.013 | [-0.098, +0.145] |
| p89 vs p82 | reaction | -0.026 | [-0.068, +0.008] |
| p89 vs p82 | forward | +0.028 | [-0.098, +0.180] |
| p89 vs p82 | reverse | +0.011 | [-0.124, +0.174] |
| p89 vs p82 | barriers | +0.020 | [-0.103, +0.174] |

## Interpretation and fitting details

The earlier p67 radical-specific Korny potential scored 4.019 reaction, 3.927 forward, 3.723 reverse and 3.825 pooled barrier MAE on the same 626 RDB19-Rad IDs. Its reference and training objective specialize it to this neutral-doublet dataset.

Each new geometry feature measures the difference and sum of two smooth covalent contacts to one H and their included angle. Both contact distances are scaled by covalent radii. The basis is symmetric under exchange of equivalent heavy ends and has analytic Cartesian derivatives. A 336-coefficient correction is fitted for each target branch. All training channel records enter as electronic differences against the corresponding frozen parent, including the previously downloaded sources. A separate regularized coefficient offset for each native method is eliminated analytically during fitting. The deployed potential uses the shared coefficients. The AIMNet2 scan offsets are centered separately by series, state and Hamiltonian. The prior uses three directions learned on reoptimized training reactions. The new descriptor and expanded supervision changed together, so their contributions cannot be assigned individually from these four models.

The p88/p89 angular window switches the extra contribution smoothly from zero at 110° to full strength at 160°. Its derivative is included in the Cartesian forces. The angle choice was motivated by the structural audit, which included previously inspected validation geometries. The comparisons are development results.

[Earlier kernel comparisons](operator-kernel-benchmarks.md) · [Dataset inventories and primary source records](aimnet2-nse-and-automekin.md) · [H-transfer feature implementation](../src/hydrogen_transfer.py) · [Native data fitting and evaluation](../experiments/full_public/h_transfer_sources.py)
