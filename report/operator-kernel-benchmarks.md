# Eigenfunction kernels for reaction energies and barriers

On the same 589 reactions, Jacobi had the lowest reaction MAE (8.965 kcal/mol). Chebyshev had the lowest pooled barrier MAE (10.473 kcal/mol). The parent scored 9.133 and 10.573. These minima summarize a repeatedly inspected panel.

Of the 13 eigenfunction fits, 13 improved forward barriers and 0 improved reverse barriers relative to the parent. The 5 kcal/mol mixed-panel goal remains open. The graph operator ranked first on the development folds, while its mixed-panel reaction MAE rose to 9.218.

Korny was fitted on 11,667 RDB7, 400 BH9, and 1,084 Grambow training reactions. The fixed common panel contains 259 RDB7, 47 BH9, and 283 Grambow reactions, giving 589 endpoint gaps and 1,178 directed barriers. High-level references score RDB7 and BH9. The Grambow references are ωB97X-D3. All deployable Korny rows include D3(BJ) and ATM once.

## Formula-fold comparison

For each of 11 operator families, two coordinate scales and four ridge strengths were compared. There are 80 features per central element and 320 molecular features. Each score averages six MAEs equally across the three sources and two observables (reaction energy and pooled barrier). Two formula folds contain 1,909 RDB7, 102 BH9, and 445 Grambow reactions. The earlier p12 baseline has separately fitted predictions for each of these folds. The full-data corrections use the p68 parent. The scores of these two experiments have different baselines and populations.

| Family | Internal score (kcal/mol) | Scale multiplier | Ridge |
|---|---:|---:|---:|
| Parent alone | 15.799 | — | — |
| Graph diffusion | 15.652 | 0.80 | 0.3 |
| Chebyshev | 15.758 | 1.60 | 600 |
| Laguerre | 15.761 | 0.80 | 6 |
| Hermite oscillator | 15.764 | 1.60 | 600 |
| Legendre | 15.766 | 1.60 | 600 |
| Gaussian | 15.770 | 1.10 | 600 |
| Exponential | 15.777 | 1.10 | 600 |
| Hahn | 15.778 | 1.60 | 600 |
| Matérn 5/2 | 15.778 | 1.10 | 600 |
| Neumann cosine | 15.781 | 1.60 | 600 |
| Dirichlet sine | 15.781 | 1.60 | 60 |
| Rational quadratic | 15.786 | 1.10 | 600 |
| Krawtchouk | 15.801 | 0.80 | 600 |
| Bessel disk | 15.813 | 1.60 | 6 |
| Jacobi | 15.818 | 1.60 | 6 |

The ordinary radial profiles use a center-Gram ridge, equivalent to a regularized Nyström basis. Their scale multiplier sets a radial half-height distance. For the one-dimensional operator modes, scale controls the spread of the projection coordinates. The graph scale controls affinity bandwidth. All use training landmarks; the profile multipliers and operator multipliers parameterize different distances.

## Fixed mixed panel

Entries are MAEs in kcal/mol. Chebyshev and graph diffusion were chosen on the formula folds before scoring this panel. The other families are exploratory comparisons. The operator fits start from p68. They share the same 91,156 bare-PBE0 anchor geometries, directional force and curvature constraints, NIST training minima, and Grambow saddle constraints.

| Method | Reaction | Forward barrier | Reverse barrier | All barriers |
|---|---:|---:|---:|---:|
| Korny p68 · Parent | 9.133 | 10.454 | 10.693 | 10.573 |
| Korny p69 · Gaussian | 9.091 | 10.364 | 10.712 | 10.538 |
| Korny p70 · Gaussian plus singlet DFT supervision | 9.515 | 10.540 | 11.014 | 10.777 |
| Korny p71 · Gaussian plus reaction-family balancing | 9.087 | 10.359 | 10.690 | 10.525 |
| Korny p72 · Split-charge response | 9.190 | 10.534 | 10.687 | 10.610 |
| Korny p73 · Chebyshev | 9.060 | 10.228 | 10.718 | 10.473 |
| Korny p74 · Graph diffusion | 9.218 | 10.432 | 10.766 | 10.599 |
| Korny p75 · Hermite oscillator | 9.038 | 10.234 | 10.730 | 10.482 |
| Korny p76 · Laguerre | 9.081 | 10.225 | 10.871 | 10.548 |
| Korny p77 · Legendre | 9.052 | 10.231 | 10.716 | 10.473 |
| Korny p78 · Jacobi | 8.965 | 10.346 | 10.816 | 10.581 |
| Korny p79 · Dirichlet sine | 9.015 | 10.258 | 10.800 | 10.529 |
| Korny p80 · Neumann cosine | 9.018 | 10.254 | 10.738 | 10.496 |
| Korny p81 · Bessel disk | 9.013 | 10.276 | 10.796 | 10.536 |
| Korny p82 · Krawtchouk | 9.000 | 10.277 | 10.747 | 10.512 |
| Korny p83 · Hahn | 9.051 | 10.266 | 10.785 | 10.526 |
| Korny p84 · Morse stretches and Legendre bends | 9.061 | 10.389 | 10.734 | 10.561 |
| Korny p85 · Legendre with pairwise coordinate products | 9.200 | 10.361 | 10.843 | 10.602 |
| Original SO3LR | 13.896 | 13.292 | 14.016 | 13.654 |
| Tauri-M-2606 | 6.146 | 8.282 | 7.888 | 8.085 |
| Tauri-L-2606 | 4.471 | 5.987 | 6.063 | 6.025 |

## Dataset breakdown

| Dataset | Method | Reaction | Forward barrier | Reverse barrier | All barriers |
|---|---|---:|---:|---:|---:|
| RDB7 | Korny p68 · Parent | 7.841 | 9.067 | 10.290 | 9.678 |
| RDB7 | Korny p69 · Gaussian | 7.907 | 8.947 | 10.376 | 9.662 |
| RDB7 | Korny p70 · Gaussian plus singlet DFT supervision | 7.907 | 8.947 | 10.376 | 9.662 |
| RDB7 | Korny p71 · Gaussian plus reaction-family balancing | 7.882 | 8.947 | 10.318 | 9.632 |
| RDB7 | Korny p72 · Split-charge response | 7.853 | 9.109 | 10.246 | 9.678 |
| RDB7 | Korny p73 · Chebyshev | 7.730 | 8.701 | 10.387 | 9.544 |
| RDB7 | Korny p74 · Graph diffusion | 7.884 | 9.066 | 10.468 | 9.767 |
| RDB7 | Korny p75 · Hermite oscillator | 7.726 | 8.805 | 10.419 | 9.612 |
| RDB7 | Korny p76 · Laguerre | 7.791 | 8.704 | 10.418 | 9.561 |
| RDB7 | Korny p77 · Legendre | 7.729 | 8.715 | 10.388 | 9.551 |
| RDB7 | Korny p78 · Jacobi | 7.746 | 8.834 | 10.393 | 9.614 |
| RDB7 | Korny p79 · Dirichlet sine | 7.804 | 8.854 | 10.540 | 9.697 |
| RDB7 | Korny p80 · Neumann cosine | 7.722 | 8.865 | 10.419 | 9.642 |
| RDB7 | Korny p81 · Bessel disk | 7.781 | 8.837 | 10.411 | 9.624 |
| RDB7 | Korny p82 · Krawtchouk | 7.765 | 8.935 | 10.426 | 9.681 |
| RDB7 | Korny p83 · Hahn | 7.733 | 8.781 | 10.412 | 9.596 |
| RDB7 | Korny p84 · Morse stretches and Legendre bends | 7.860 | 9.037 | 10.363 | 9.700 |
| RDB7 | Korny p85 · Legendre with pairwise coordinate products | 7.771 | 9.026 | 10.539 | 9.783 |
| RDB7 | Original SO3LR | 13.594 | 12.121 | 10.642 | 11.381 |
| RDB7 | Tauri-M-2606 | 6.670 | 10.152 | 9.238 | 9.695 |
| RDB7 | Tauri-L-2606 | 5.410 | 6.930 | 6.877 | 6.904 |
| BH9 | Korny p68 · Parent | 16.520 | 14.527 | 16.245 | 15.386 |
| BH9 | Korny p69 · Gaussian | 17.113 | 14.518 | 16.368 | 15.443 |
| BH9 | Korny p70 · Gaussian plus singlet DFT supervision | 17.113 | 14.518 | 16.368 | 15.443 |
| BH9 | Korny p71 · Gaussian plus reaction-family balancing | 16.981 | 14.455 | 16.397 | 15.426 |
| BH9 | Korny p72 · Split-charge response | 17.058 | 14.912 | 16.313 | 15.613 |
| BH9 | Korny p73 · Chebyshev | 17.115 | 14.444 | 16.232 | 15.338 |
| BH9 | Korny p74 · Graph diffusion | 18.097 | 14.367 | 16.340 | 15.354 |
| BH9 | Korny p75 · Hermite oscillator | 17.034 | 14.254 | 16.185 | 15.220 |
| BH9 | Korny p76 · Laguerre | 17.044 | 15.059 | 17.685 | 16.372 |
| BH9 | Korny p77 · Legendre | 17.072 | 14.467 | 16.165 | 15.316 |
| BH9 | Korny p78 · Jacobi | 17.099 | 15.573 | 17.060 | 16.316 |
| BH9 | Korny p79 · Dirichlet sine | 16.716 | 14.588 | 16.336 | 15.462 |
| BH9 | Korny p80 · Neumann cosine | 16.979 | 14.419 | 16.189 | 15.304 |
| BH9 | Korny p81 · Bessel disk | 17.347 | 15.057 | 16.718 | 15.887 |
| BH9 | Korny p82 · Krawtchouk | 16.803 | 14.475 | 16.200 | 15.338 |
| BH9 | Korny p83 · Hahn | 17.261 | 14.782 | 16.797 | 15.789 |
| BH9 | Korny p84 · Morse stretches and Legendre bends | 16.459 | 14.885 | 15.994 | 15.440 |
| BH9 | Korny p85 · Legendre with pairwise coordinate products | 17.202 | 14.746 | 16.653 | 15.700 |
| BH9 | Original SO3LR | 15.668 | 15.639 | 11.898 | 13.768 |
| BH9 | Tauri-M-2606 | 7.196 | 9.220 | 7.688 | 8.454 |
| BH9 | Tauri-L-2606 | 4.990 | 7.124 | 7.134 | 7.129 |
| Grambow | Korny p68 · Parent | 9.088 | 11.047 | 10.140 | 10.593 |
| Grambow | Korny p69 · Gaussian | 8.842 | 10.970 | 10.080 | 10.525 |
| Grambow | Korny p70 · Gaussian plus singlet DFT supervision | 9.726 | 11.337 | 10.709 | 11.023 |
| Grambow | Korny p71 · Gaussian plus reaction-family balancing | 8.878 | 10.972 | 10.084 | 10.528 |
| Grambow | Korny p72 · Split-charge response | 9.107 | 11.112 | 10.156 | 10.634 |
| Grambow | Korny p73 · Chebyshev | 8.939 | 10.925 | 10.105 | 10.515 |
| Grambow | Korny p74 · Graph diffusion | 8.964 | 11.028 | 10.112 | 10.570 |
| Grambow | Korny p75 · Hermite oscillator | 8.912 | 10.875 | 10.109 | 10.492 |
| Grambow | Korny p76 · Laguerre | 8.939 | 10.815 | 10.154 | 10.484 |
| Grambow | Korny p77 · Legendre | 8.931 | 10.915 | 10.111 | 10.513 |
| Grambow | Korny p78 · Jacobi | 8.729 | 10.862 | 10.167 | 10.514 |
| Grambow | Korny p79 · Dirichlet sine | 8.845 | 10.824 | 10.118 | 10.471 |
| Grambow | Korny p80 · Neumann cosine | 8.882 | 10.835 | 10.124 | 10.479 |
| Grambow | Korny p81 · Bessel disk | 8.758 | 10.799 | 10.165 | 10.482 |
| Grambow | Korny p82 · Krawtchouk | 8.834 | 10.808 | 10.136 | 10.472 |
| Grambow | Korny p83 · Hahn | 8.894 | 10.875 | 10.128 | 10.502 |
| Grambow | Korny p84 · Morse stretches and Legendre bends | 8.931 | 10.880 | 10.199 | 10.539 |
| Grambow | Korny p85 · Legendre with pairwise coordinate products | 9.179 | 10.854 | 10.157 | 10.505 |
| Grambow | Original SO3LR | 13.878 | 13.974 | 17.455 | 15.714 |
| Grambow | Tauri-M-2606 | 5.493 | 6.415 | 6.686 | 6.551 |
| Grambow | Tauri-L-2606 | 3.525 | 4.935 | 5.140 | 5.038 |

Paired changes versus p68 use identical IDs. The intervals resample whole transition-state formulas within each source (4,000 resamples). Repeated inspection makes the common panel a development benchmark.

| Preselected model | Observable | MAE change | 95% interval, kcal/mol |
|---|---|---:|---:|
| p73 | reaction | -0.073 | [-0.269, +0.116] |
| p73 | forward | -0.226 | [-0.457, -0.049] |
| p73 | reverse | +0.025 | [-0.091, +0.153] |
| p73 | barriers | -0.101 | [-0.216, +0.001] |
| p74 | reaction | +0.085 | [-0.042, +0.223] |
| p74 | forward | -0.022 | [-0.168, +0.100] |
| p74 | reverse | +0.072 | [-0.042, +0.226] |
| p74 | barriers | +0.025 | [-0.065, +0.118] |

## Geometry and frequencies

The four full frequency MAEs are in cm⁻¹ against PBE0/cc-pVTZ at independently relaxed minima. The signed saddle error compares curvature along the reference unstable direction on 278 Grambow structures.

| Model | Propane | Benzene | Ethanol | 1,3-butadiene | Equal-molecule mean | Signed saddle MAE | Downhill sign |
|---|---:|---:|---:|---:|---:|---:|---:|
| p68 | 89.4 | 151.2 | 104.1 | 156.3 | 125.3 | 794.6 | 212/278 |
| p69 | 89.9 | 159.2 | 112.2 | 161.1 | 130.6 | 793.8 | 213/278 |
| p70 | 89.9 | 159.2 | 112.2 | 161.1 | 130.6 | 796.8 | 211/278 |
| p71 | 92.0 | 156.1 | 111.2 | 159.7 | 129.8 | 794.3 | 212/278 |
| p72 | 90.0 | 148.5 | 101.9 | 153.7 | 123.5 | 796.4 | 211/278 |
| p73 | 93.0 | 170.7 | 110.1 | 155.6 | 132.3 | 795.7 | 212/278 |
| p74 | 89.1 | 158.6 | 111.7 | 159.5 | 129.7 | 794.2 | 212/278 |
| p75 | 94.2 | 166.5 | 121.4 | 150.4 | 133.1 | 795.1 | 212/278 |
| p76 | 89.1 | 160.8 | 119.7 | 159.0 | 132.2 | 795.3 | 212/278 |
| p77 | 93.7 | 169.2 | 112.6 | 155.5 | 132.7 | 795.6 | 212/278 |
| p78 | 94.8 | 161.9 | 125.2 | 159.2 | 135.3 | 795.0 | 213/278 |
| p79 | 88.7 | 166.1 | 117.0 | 148.5 | 130.1 | 794.5 | 213/278 |
| p80 | 94.5 | 158.5 | 119.9 | 149.9 | 130.7 | 794.0 | 213/278 |
| p81 | 93.5 | 151.6 | 121.1 | 152.1 | 129.6 | 793.8 | 213/278 |
| p82 | 93.1 | 157.4 | 107.6 | 148.4 | 126.6 | 794.3 | 213/278 |
| p83 | 93.0 | 169.7 | 115.2 | 153.5 | 132.9 | 795.2 | 212/278 |
| p84 | 90.3 | 178.9 | 113.6 | 169.6 | 138.1 | 797.9 | 210/278 |
| p85 | 94.7 | 184.6 | 120.8 | 158.2 | 139.6 | 797.3 | 211/278 |

| Model | Largest bond-length MAE (Å) | Largest angle MAE (degrees) | Connectivity retained | Minima with no imaginary internal modes |
|---|---:|---:|---:|---:|
| p68 | 0.0255 | 2.91 | 4/4 | 4/4 |
| p69 | 0.0271 | 2.85 | 4/4 | 4/4 |
| p70 | 0.0271 | 2.85 | 4/4 | 4/4 |
| p71 | 0.0273 | 2.80 | 4/4 | 4/4 |
| p72 | 0.0253 | 2.92 | 4/4 | 4/4 |
| p73 | 0.0257 | 2.89 | 4/4 | 4/4 |
| p74 | 0.0269 | 2.88 | 4/4 | 4/4 |
| p75 | 0.0259 | 2.91 | 4/4 | 4/4 |
| p76 | 0.0262 | 2.88 | 4/4 | 4/4 |
| p77 | 0.0257 | 2.89 | 4/4 | 4/4 |
| p78 | 0.0259 | 2.95 | 4/4 | 4/4 |
| p79 | 0.0260 | 2.87 | 4/4 | 4/4 |
| p80 | 0.0253 | 2.92 | 4/4 | 4/4 |
| p81 | 0.0250 | 2.96 | 4/4 | 4/4 |
| p82 | 0.0250 | 2.92 | 4/4 | 4/4 |
| p83 | 0.0255 | 2.88 | 4/4 | 4/4 |
| p84 | 0.0248 | 2.80 | 4/4 | 4/4 |
| p85 | 0.0235 | 2.79 | 4/4 | 4/4 |

## What the basis functions represent

A finite kernel has the form $k(x,y)=\sum_j w_j\phi_j(x)\phi_j(y)$ with $w_j\geq0$. Korny evaluates molecular features by summing atom-centered mode amplitudes after subtracting the isolated-atom value. Permutation and rigid-motion invariance come from the atomic descriptors. The energy coefficients are real and the analytic Cartesian forces are derivatives of the same energy.

The continuous families use five modes on each of 16 fixed projections of the training environment descriptors. Hermite oscillator functions include a Gaussian envelope. Laguerre functions include an exponential envelope on the positive half-line. Legendre, Chebyshev, and Jacobi modes use a bounded coordinate. The interval Laplacian gives sine or cosine modes with Dirichlet or Neumann boundary conditions. The radial disk Laplacian gives Bessel J₀ modes with zeros at the boundary. The operator eigenfunctions are known on their native domains; projection, weighting, and summing over atoms induce the kernel on molecular geometry. [NIST orthogonal polynomials](https://dlmf.nist.gov/18.3) · [NIST Bessel zeros](https://dlmf.nist.gov/10.21) · [Solin and Särkkä](https://arxiv.org/abs/1401.5508)

Krawtchouk and Hahn modes come from two finite birth–death chains on 13 sites. Binomial averaging extends each discrete eigenfunction smoothly in the geometry descriptor. Graph diffusion instead uses the nonconstant eigenvectors of a reversible random walk on 256 training environments per element, with smooth affinity-weighted extension to new geometries. These features require no reaction label at prediction time. [NIST discrete orthogonal polynomials](https://dlmf.nist.gov/18.19) · [Coifman and Lafon, diffusion maps](https://doi.org/10.1016/j.acha.2006.04.006)

P84 changes the coordinate as well as the operator. Four Morse oscillator bound-state functions act on each bond length scaled by the sum of its covalent radii. A product of two radial ground-state functions multiplies the first three Legendre modes of the included angle. Element-group channels distinguish participating atoms. A smooth cutoff switches off each contribution at dissociation. The resulting 342 explicit features take part in the same fit and independent geometry checks. [Morse’s diatomic oscillator](https://doi.org/10.1103/PhysRev.34.57)

P70 adds 11,667 RDB7 singlet DFT training gaps and midpoint heights to the DFT branch at weight 0.25. P71 partially equalizes family frequencies within each training source. P72 uses the geometry-dependent energy response of variational atom-to-atom charge flow, with smooth bond conductance and softened electrostatics. Its induced flows sum to zero on each disconnected fragment. The baseline carries the total charge and spin. [Split-charge equilibration](https://doi.org/10.1063/1.2346671)

## Where the p68 parent incurs reaction-energy error

The three families contributing the largest total absolute endpoint error in each source are listed below. Unassigned reactions use atom-mapped changes in connectivity grouped by element pair and fragment count. These coarse groups can contain several mechanisms.

| Dataset | Family or edits | Training | Held | Reaction | Forward | Reverse |
|---|---|---:|---:|---:|---:|---:|
| RDB7 | KinBot/intra_H_migration | 1507 | 39 | 7.93 | 7.40 | 10.35 |
| RDB7 | 1↔1 fragments, C-C+C-H ↔ C-H | 1277 | 35 | 7.18 | 8.37 | 10.33 |
| RDB7 | 1↔1 fragments, C-N ↔ none | 64 | 6 | 30.22 | 22.70 | 12.36 |
| BH9 | BH9/Hydrogen atom transfer | 77 | 13 | 17.41 | 18.64 | 10.66 |
| BH9 | BH9/Nucleophilic addition | 16 | 10 | 19.46 | 12.76 | 19.29 |
| BH9 | KinBot/Diels_alder_addition | 40 | 9 | 17.02 | 19.25 | 23.87 |
| Grambow | 1↔1 fragments, C-H ↔ C-H | 207 | 52 | 5.90 | 8.12 | 7.21 |
| Grambow | 1↔1 fragments, C-H ↔ H-N | 103 | 20 | 11.08 | 10.35 | 10.06 |
| Grambow | 1↔1 fragments, C-C ↔ C-C | 60 | 20 | 8.18 | 10.56 | 11.42 |

## Does a larger function space improve energies?

At fixed input scaling the five Legendre, Chebyshev, Jacobi, Hahn and Krawtchouk modes used here all span the same centered polynomial space through degree five on each projected axis. The binomial lift of a degree-n discrete polynomial is a polynomial of degree at most n in its success probability. Projection of their centered columns onto the Legendre space gave relative residuals below 7 × 10⁻¹⁶ on 201 points. Their coefficient penalties, mode normalization and selected length scales can still give different predictions.

P85 retains all 320 p77 Legendre features and adds the 120 distinct products between the first-degree modes on its 16 coordinates for each of four central elements. It has 800 features. The coordinate scales, ridge, parent, training equations and force/frequency constraints are identical to p77. Both fits were scored on the same reaction IDs.

| Model | Reaction | Forward barrier | Reverse barrier | All barriers |
|---|---:|---:|---:|---:|
| P77 additive | 9.052 | 10.231 | 10.716 | 10.473 |
| P85 coupled | 9.200 | 10.361 | 10.843 | 10.602 |

Adding 480 cross-coordinate products changed the reaction MAE by +0.148 kcal/mol and the pooled barrier MAE by +0.129 kcal/mol at the same regularization strength.
