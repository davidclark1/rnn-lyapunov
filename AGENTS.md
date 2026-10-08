# Guide for AI agents (Claude Code, Codex, ...)

This repository reproduces every result of the paper *Lyapunov spectrum of random neural networks*
(D. G. Clark). It is a library (`src/rnn_lyapunov/`) with figure scripts (`paper/`) on top,
bundled data (`data/`) and exact recompute recipes (`reproduce/`). Start with `README.md` and `PAPER_MAP.md`
(paper equation → function), then read the docstring of the module you work on. Every module docstring states the
equations it implements, with the paper's LaTeX labels in brackets.

## Orientation in five commands

```bash
source env.sh                                  # Flatiron; elsewhere: source .venv/bin/activate
python -m pytest -q                            # fast tests, < 1 min on CPU
python -m paper.make_figures                   # all figures from data/ -> figures/
python -m paper.tables                         # plotted numbers (summary; --json for the full tables)
python examples/quickstart.py                  # simulation vs theory at reduced resolution (CPU, 2 min)
```

## Conventions (keep them)

- **Time step.** The time step is `delta`. `delta=None` (or `0` on the command line) means continuous time, δ → 0.
  Simulations take `delta=` (Euler map) or `rk4_dt=` (continuous time, RK4), never both.
- **Units.** Exponents, thresholds `s`, entropy rates and times are per unit physical time, not per update, so
  results at different δ are directly comparable. Regulators are given as continuum `eps`, with η = δ·ε for the map.
- **Per neuron.** D_KY/N, h_KS/N and PR/N are always per neuron (keys `ky_dimension`, `entropy_rate`, `pr_x`, `pr_phi`).
- **Theory inputs.** The theory never uses simulated exponents. Its inputs are physical parameters and numerical
  resolution, and its thresholds come from DMFT quantities. Keep it that way. A diagnostic that feeds simulated
  quantities into the theory must be labelled semi-empirical.
- **Precision.** Everything runs in float64 (complex128 where needed). Gains are computed as `1/cosh(x)**2`, never
  `1 - tanh(x)**2`, which underflows for |x| > 19.
- **Shift.** The whole tangent step is shifted by `exp(-s*delta)`. Gains are evaluated at the old state.
- **Settings and provenance.** Numerical settings (window T, grid m, regulators, twists, scrambles, tolerance,
  burn-in, t_obs, QR interval) are explicit arguments and are stored in every result file, together with
  provenance (`rnn_lyapunov.io.save_result`).

## Changing numerics

The library is a verified port of the code that produced the paper. On the same GPU type, it reproduces the stored
results bitwise or to ~1e-11 (`docs/VALIDATION.md`). Do not alter a numerical path that the paper uses without:

1. running the relevant `tests/` (including `-m slow`), and
2. re-running the affected entries of `reproduce/jobs/*.txt` (or a representative subset) and comparing with
   `data/` through `paper.tables`.

Add new functionality as new functions or options with defaults that keep the old behavior.

## GPUs

- **Paper resolution needs a GPU.** The single-site solver inverts dense m×m matrices batched over 4096 gain
  trajectories. Network spectra use N = 4096–16384.
- **Pin one GPU per process.** Set `CUDA_VISIBLE_DEVICES=<id>` and use `--device cuda:0`. On nodes with GPUs in
  exclusive-process mode, a second process on the same GPU fails.
- **Keep MAGMA out.** The package selects the cuSOLVER backend, and `single_site._tri_inv` blocks triangular solves
  to ≤ 512. MAGMA's raw cudaMalloc calls stall for minutes when other processes share the node. Keep both.
- **Batch runs.** Use `reproduce/run_queue.sh <jobs.txt> <gpu ids...>`, which runs one worker per GPU on a shared
  queue.
- **DMFT cache.** DMFT solutions are cached in `$RNN_LYAPUNOV_CACHE/dmft` (default `~/.cache/rnn_lyapunov/dmft`).
- **CPU runs.** Set `OPENBLAS_NUM_THREADS=1` for the single-site solver on a CPU; with multithreaded BLAS its many
  small batched solves run about 100× slower. Large DMFT solves are faster multithreaded, so `env.sh` leaves it unset.

## Where things are

| Need | Look at |
|---|---|
| a paper equation | `PAPER_MAP.md` |
| how a figure is made | `paper/fig_*.py` (plotting only), `paper/tables.py` (numbers), `paper/data.py` (loading) |
| the settings of a plotted point | `docs/NUMERICS.md`, the file in `data/`, the line in `reproduce/jobs/` |
| file formats | `docs/DATA.md` |
| compute a new point | `scripts/run_*.py --help` |

## Do not

- **Do not hand-edit `data/`.** It is regenerated only by `tools/build_dataset.py` from the original research tree,
  or by scripts writing to `results/`.
- **Do not commit large arrays.** `results/`, `logs/`, `figures/` and `*.npz` are git-ignored.
