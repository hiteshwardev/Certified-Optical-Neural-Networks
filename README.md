# Certified Error and Output-Power Bounds for Deep Loss-Compensated Photonic Networks

**Author:** Hitesh Kumar Singh · [ORCID: 0009-0008-1583-4848](https://orcid.org/0009-0008-1583-4848)
*Department of Physics, Kurukshetra University, Kurukshetra, Haryana, India*

---

## Abstract

Deep loss-compensated photonic networks must control two competing risks: the accumulation of layer imperfections and fluctuations in output power. Because gain-compensated layers are generally non-normal, eigenvalue-based estimates can underestimate transient amplification. This work develops and numerically tests certified bounds for both quantities. For output error, a layer-resolved first-order certificate is derived from the norms of the partial network products, and each layer's contribution is shown to be attainable by a rank-one perturbation. Eigenvalue-based estimates are shown to underestimate certified sensitivity by a factor of 65.2 at depth 16, including in optimized physical layers with gain and loss. For random cascades with Haar-random meshes, a closed-form limiting eigenvalue deficit and a closed-form Lyapunov-spectrum gap are derived. For output power, a martingale-based certificate requires only mean-power gain compensation, independent of the mesh, input state, loss-distribution shape and inter-mode loss correlations under the stated assumptions. Tighter bounds, a floor for mesh-independent certificates and the accumulation of gain-calibration errors with depth are also derived.

## Core Objectives

- Provide error bounds for deep, non-unitary photonic cascades that hold for every realization, not only in empirical or unitary settings.
- Quantify how far eigenvalue-based reasoning underestimates the worst-case output error.
- Bound the upper tail of the output power and translate the bounds into gain-calibration requirements.

## Scientific Significance

Photonic processors with tens to over a hundred optical layers must restore loss with gain, so every imperfection is propagated and possibly amplified by all later layers. Non-normality makes eigenvalues unreliable for this. This project supplies the hardware counterpart of certified robustness: guarantees that a deep cascade stays within an error and power budget, with explicit design consequences such as calibration tolerance that shrinks inversely with depth.

## Methodological Approach

- **Certificates:** an exact telescoping identity gives a first-order certificate built from prefix and suffix norms. Attainability is shown by rank-one perturbations, and the joint worst case is computed by alternating maximization.
- **Non-normality analysis:** a chain of inequalities links the eigenvector condition number, Petermann factors and the Kreiss constant of a layer.
- **Random cascades:** the Lyapunov spectrum and the deep-cascade deficit are derived from Haar-frame statistics and the overlap statistics of rank-one partial products.
- **Output power:** a martingale argument and a hierarchy of tail bounds (Markov, Jensen-based moment bound, exact Haar moments), with an explicit mesh-independent floor.
- **Testing:** 40 benchmarks with pass criteria fixed in a written analysis plan before computation; three failed and are reported with their causes.

## Contributions

1. A layer-resolved perturbation certificate that is attained layer by layer and tracked against the joint worst case.
2. A quantification of the eigenvalue deficit, including an optimized physical layer (loss contrast 8.7 dB) with a deficit of about 100.
3. A closed-form deep-cascade limit of the deficit, R∞(n) ≈ 4n/π, and the Lyapunov gap Δ²/[6(n + 1)] that sets the approach to it.
4. A mean-power martingale certificate valid for any mesh, input, loss-law shape and inter-mode correlation, with a mesh-independent floor.
5. A gain-calibration tolerance that scales inversely with depth.

6. **Positioning relative to prior work.** Perturbation sums for matrix products and Kreiss-type transient bounds are standard tools (Trefethen and Embree), and pseudospectral bounds and Lipschitz certificates have recently been applied to learning dynamics and neural networks.

## Key Findings

- The certificate stays at 79% or more of the joint worst case in the random cascades examined (30 cascades per cell, n = 8).
- The eigenvalue deficit reaches 65.2 for a constructed non-normal family and about 100 for an optimized physical layer at depth 16.
- In deep Haar cascades the deficit saturates near 1.27 n (for example 9.87 at n = 8).
- The tier-0 power certificate Pr[G² > c] ≤ 1/c held in every tested configuration; the Haar tail is about 14 times below it at c = 2.
- Keeping the mean power within a factor of 2 requires a gain error of at most 0.54% at 64 layers and 0.034% at 1024; a 1% gain error invalidates the calibrated certificate at 64 layers.

## Scope and Limitations

The certificate is first order. The deep-cascade limit and exact power law use Haar meshes as a model of full mixing, and the loss law is idealized. All tests are numerical; no comparison with measured processor data is made. Three of 40 benchmarks failed and are discussed in the manuscript.

## Roles

Hitesh Kumar Singh is the sole author. Hitesh Kumar Singh conceived the study, derived the certificates, limits and tail bounds, wrote the code, designed and ran the benchmark plan, produced the figures and wrote the manuscript.

## Reproducibility and Availability

The repository contains the notebook, source code, numerical records and figures. Every random stream derives from a fixed master seed and a text label, and the notebook regenerates all numbers and figures in about 20 minutes on a two-core processor.
