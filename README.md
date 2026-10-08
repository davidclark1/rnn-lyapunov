# rnn-lyapunov

Code and data for **D. G. Clark, *Lyapunov spectrum of random neural networks***.

The model is a random recurrent network with time step δ,

    x(n+1) = (1 − δ) x(n) + δ J tanh(x(n)),        J_ij ~ N(0, g²/N) iid,

whose continuous-time limit δ → 0 is the network of Sompolinsky, Crisanti and Sommers. The paper derives an exact
expression for the cumulative distribution F(s) of the Lyapunov exponents, F(s) = (1/N) tr R^vv(k+1, k). It then
computes F(s) at large N from a self-consistent single-site (cavity) theory. This repository contains:

- **`rnn_lyapunov`**, a library with documented functions for every computation in the paper:
  - simulating the network (Euler map and RK4 flow) and computing its Lyapunov spectrum by the QR method;
  - the dynamic mean-field theory;
  - the single-site theory's F(s);
  - the Kaplan–Yorke dimension and entropy rate;
  - the maximum exponent (operator 𝒯_s, Schrödinger limit);
  - the exact trivial-fixed-point result;
  - participation-ratio dimensions (theory and simulation);
  - the exact finite-N objects of the derivation.
- **`paper/`**, one script per figure, built on the library and the bundled data.
- **`data/`**, every number plotted in the paper (theory F(s) per Sobol scramble, simulated spectra, participation
  ratios), with provenance.
- **`scripts/` and `reproduce/`**, command-line drivers and the exact job lists that regenerate all of `data/` from
  scratch.

## Install

Requires Python ≥ 3.10, PyTorch ≥ 2.1, NumPy, SciPy and Matplotlib. A CUDA GPU is needed for paper-resolution
theory and simulations, but everything also runs on a CPU (slowly).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
```

On the Flatiron cluster, use `source env.sh` (module Python plus the venv). For CPU runs of the single-site solver,
set `OPENBLAS_NUM_THREADS=1`; otherwise multithreaded BLAS makes its many small batched solves about 100× slower.

## Reproduce the figures (one minute, CPU)

```bash
python -m paper.make_figures                 # -> figures/{spectra,dimension_entropy,trivial_fixed_point,max_exponent_operator}.pdf
python -m paper.tables                       # summary of the plotted numbers (--json FILE: full tables)
```

## Use the library

```python
from rnn_lyapunov import network, observables, theory
from rnn_lyapunov.lyapunov import lyapunov_spectrum

# simulation: Lyapunov spectrum of one network
J = network.make_coupling(N=1024, g=3.0, seed=0)
sim = lyapunov_spectrum(J, delta=0.5, t_obs=1000)          # or rk4_dt=0.1 for continuous time
observables.spectrum_summary(sim.exponents)                 # max exponent, D_KY/N, h_KS/N, ...

# theory: F(s) of the single-site theory (paper settings; a few GPU minutes)
cfg = theory.TheoryConfig(g=3.0, delta=0.5)                 # delta=None: continuous time (then give m)
res = theory.cdf_curve(cfg, theory.thresholds_full(cfg))
res["s"], res["F"][:, -1]                                   # thresholds and F(s) at the smallest regulator
theory.dimension_entropy(res)                               # D_KY/N, h_KS/N
```

`examples/quickstart.py` runs both at reduced resolution on a CPU in about two minutes. Further pieces:

| Module | Computes |
|---|---|
| `dmft` | stationary single-site process: C^x(τ), C^φ(τ), gain autocorrelation C^d(τ) |
| `max_exponent` | upper edge s_* = λ₁, operator 𝒯_s, Schrödinger limit |
| `trivial_fixed_point` | exact F(s) for g < 1 (circular law), its single-site form, semicircle limit |
| `participation` | PR^x/N and PR^φ/N, two-site cavity theory and simulation estimators |
| `exact` | finite-N matrices K and 𝓑_η, minimum-norm and regularized responses |

[`PAPER_MAP.md`](PAPER_MAP.md) maps every equation and figure of the paper to the code.

## Recompute everything from scratch

`reproduce/jobs/*.txt` lists one command per data file, with the paper's exact settings and a GPU-hour estimate.
To run a list on the GPUs of one node, use:

```bash
reproduce/run_queue.sh reproduce/jobs/fig2_theory.txt 0 1 2 3
```

Results go to `results/`. The table builders accept `root="results"` in place of the bundled `data/`, so a
recomputation drops into the same figure scripts. [`reproduce/README.md`](reproduce/README.md) has the details and
[`docs/NUMERICS.md`](docs/NUMERICS.md) gives the settings and convergence evidence.

## Tests

```bash
python -m pytest -q            # fast suite (< 1 min, CPU)
python -m pytest -q -m slow    # slower scientific checks (minutes)
```

The tests check the library against exact results:
- the finite-N identity F(s) = (1/N) tr R^vv;
- Gauss's law at the trivial fixed point;
- Jacobians against finite differences;
- the δ = 1 and δ → 0 limits.

They also check it against simulations at reduced size.

## Repository layout

```
src/rnn_lyapunov/   library (see the module docstrings)
paper/              figure scripts, table builders, data loaders, figure style
data/               bundled results (JSON) + manifest.json
scripts/            command-line drivers: run_theory, run_network, run_participation, run_pr_theory, run_max_exponent
reproduce/          job lists for every data file, GPU queue runner
tests/              pytest suite
docs/               NUMERICS (settings, convergence), DATA (file formats), VALIDATION (port verification)
examples/           quickstart
tools/              dataset import and verification against the original research tree (owner only), figure
                    pixel comparison (tools/compare_figures.sh <reference dir>), job-list generator
```

## For AI agents

Read [`AGENTS.md`](AGENTS.md) first.
