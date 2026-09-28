# Certified error and output-power bounds for deep loss-compensated photonic networks

Hitesh Kumar Singh, Department of Physics, Kurukshetra University, Kurukshetra, Haryana, India

This repository contains the complete computational study behind the article of the same title. Deep photonic processors that restore optical loss with gain are built from non-unitary, and in general non-normal, layers. The study derives and tests

* a layer-resolved certificate for the output error caused by imperfect layers, its attainability, and its relation to the eigenvector condition number, the Petermann factors and the Kreiss constant of a layer;
* how far eigenvalue estimates underestimate that certificate, both for single programmed layers with a physical gain-loss contrast and for deep random cascades, where the underestimate saturates at a closed-form limit $R_\infty(n)$;
* the Lyapunov spectrum of loss-compensated Haar cascades and the exact law of the disorder-induced gain;
* a hierarchy of tail certificates for the output power, the floor below which no mesh-independent certificate can go, and the calibration tolerance of the amplifier gain.

## Contents

```
notebook.ipynb      the complete study, from verification checks to the final benchmark table
src/
  model.py          loss-compensated layers, mesh ensembles, loss laws, cascade propagation
  operators.py      non-normality measures, certificates, joint worst case, deep-cascade ensembles
  tails.py          tail certificates and the exact tail of the mode-diagonal cascade
  uncertainty.py    Clopper-Pearson limits and bootstrap intervals
  plotting.py       single-panel figures with automated layout checks
  records.py        writing the numerical records to results/
tests/test_src.py   regression tests of the modules in src/
figures/            every figure of the article, as PDF and 600 dpi PNG
results/            numerical records written by the notebook (JSON)
```

The notebook is self-contained: each stage states the physics, the mathematics and the algorithm before the code that carries it out, and every figure and number of the article is produced in it. The modules in `src/` hold only the functions that the notebook calls repeatedly.

| Stage | Content |
|---|---|
| 0 | Model, notation and certificates |
| V | Verification of the numerical tools against exact results |
| A | The certificate chain on a strongly non-normal family of layers |
| B | How much non-normality a physical layer can carry, with interferometer settings |
| C | Random cascades: eigenvalue deficit, Lyapunov spectrum, deep-cascade limit, comparators, joint and structured perturbations |
| D | Exact law of the disorder-induced gain |
| E | Tail certificates for the output power, their robustness, and gain calibration |
| F | Transient gain against certified sensitivity |
| S | Benchmark table and headline results |

## Reproducing the study

Python 3.11 and the package versions pinned in `requirements.txt` were used.

```
python -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                # regression tests of src/
jupyter lab notebook.ipynb         # then run all cells
```

With conda, `conda env create -f environment.yml` creates the same environment. The notebook can also be executed without a browser:

```
jupyter execute --inplace notebook.ipynb
```

A full run takes about 20 minutes on a two-core processor and rewrites `results/` and `figures/`. Every random stream is derived from the master seed 20260716 and a text label through SHA-256, so the records are reproduced exactly with the pinned versions, independently of the order in which the stages are run.

## Benchmarks

The analysis followed a written plan in which every benchmark and its pass criterion were fixed before the corresponding computation; later additions were recorded as numbered amendments before they were run. The notebook ends with the complete table. Of 40 benchmarks with a pass criterion, three fail and are reported in the article: B15, the pre-registered ten-fold eigenvalue deficit on random cascades, which cannot be reached because the deficit saturates at $R_\infty(8)=9.87$; B24, the agreement of every fitted asymptote with $R_\infty(n)$ within 5%, which fails for the one series that stops at a scaled depth below ten; and C1, a sample-size tolerance of 0.1% that lies below the Monte Carlo resolution of the production samples.

## License

The code is released under the MIT License (see `LICENSE`).
