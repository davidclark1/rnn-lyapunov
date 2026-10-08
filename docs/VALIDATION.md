# Validation of this repository against the paper's original code and results

The paper's numbers were produced by the original research code (not public: package `scs_lyapunov` plus
archived modules and campaign drivers). This repository is a clean port of the parts that
the paper uses. It was validated in four steps (2026-10-06).

## 1. Same numerics: old code vs new code, same process, same GPU

Both packages were run side by side on the same inputs on an NVIDIA H100. The maximum absolute difference was
**0 for every quantity**:

| Quantity | Old | New |
|---|---|---|
| Couplings J | `network.make_coupling` | `network.make_coupling` |
| QR Lyapunov spectrum, Euler map and RK4 flow | `lyapunov_qr.lyapunov_spectrum` | `lyapunov.lyapunov_spectrum` |
| DMFT C^x, C^d at δ = 0.5, 0.25, δ → 0 | `dmft.solve_*` | `dmft.solve` |
| Upper edge s_* | `edge.*_edge` | `max_exponent.upper_edge` |
| PR theory | `participation.pr_theory` | `participation.pr_theory` |
| Theory F(s, ε), map and continuous time, with twists and warm starts | archived `convergence.cdf_curve` | `theory.cdf_curve` |
| Full threshold grid | `convergence.auto_s_grid` | `theory.thresholds_full` |
| PR simulation (block covariances, both estimators), map and flow | archived `participation.covariance_blocks` | `participation.covariance_blocks` |

## 2. Same plotted numbers: bundled data vs the original results

`tools/verify_dataset.py` rebuilds every plotted number with `paper.tables` from `data/`. It compares them with the
original code paths and result files, and finds a maximum difference of **0** for all of the following:

- **Fig. spectra:** the theory F (376 values), s, edges and scramble counts, against the original
  `theory_ensemble`. The 32768 simulated exponents are checked as well.
- **Fig. dimension_entropy:** all 45 points.
  - Theory D_KY/N and h_KS/N with their s.e.m., including the δ → 0 grid extrapolation.
  - Network medians of D_KY/N, h_KS/N and λ₁.
  - PR simulation means, PR theory at each point, and the dense PR theory curves.
  - The selected N, seed counts and scramble seeds.
- **Fig. max_exponent_operator:** the 57 simulated medians, and all 13410 numbers of the operator data.

Two further checks:

- Network summaries (D_KY/N, h_KS/N, split-half noise) recomputed from the stored exponents and QR block logs are
  bitwise equal to the stored ones.
- Every stored threshold grid is reproduced by the `theory.thresholds_*` rules, and every line of `reproduce/jobs/`
  parses to exactly the settings and file name of its data file.

## 3. Same figures

`python -m paper.make_figures`, followed by `tools/compare_figures.sh ~/overleaf/Lyapunov/figures`, renders the
figures at 150 dpi and compares them pixel by pixel:

| Figure | Differing pixels |
|---|---|
| spectra | 0 |
| dimension_entropy | 0 |
| trivial_fixed_point | 0 |
| max_exponent_operator | 0 against the original plotting script run on today's data |

The Overleaf copy of max_exponent_operator (rendered 2026-10-04) predates the δ → 0 simulations of 2026-10-06. It
therefore differs from today's rendering in the δ → 0 markers of panel (c), so regenerate it for the paper.

## 4. Recomputing from scratch with the new scripts

Paper points were rerun from scratch with `scripts/run_*.py` on H100 GPUs and compared with `data/`:

| Point | Max difference |
|---|---|
| Theory, δ = 0.5, g = 3, full spectrum (Fig. spectra) | F: 2·10⁻¹¹ |
| Theory, δ = 0.5, g = 1.25, near onset, 12 regulators down to ε = 1.6·10⁻⁶ | F: 9·10⁻⁹ |
| Theory, δ → 0, g = 20, m = 160 and m = 256 | F: 0 |
| Theory, δ = 0.25, g = 20, T = 64 | F: 2·10⁻¹¹ |
| Network, δ = 0.5, g = 3, N = 4096 full spectrum (originally on A100) | exponents: 8·10⁻¹³ |
| Network, RK4, g = 20, N = 4096, k = 1024 | exponents: 0 |
| PR theory, δ = 0.5 and δ → 0, 30 values of g each | 3·10⁻¹⁷ |
| PR simulation, RK4, g = 3, N = 16384 | 0 |
| PR simulation, δ = 0.5, g = 3, N = 16384 (originally on A100) | 1.5·10⁻⁴ = 0.7 standard errors |
| Max-exponent operator data (all panels and checks) | 0 |

The last PR simulation differs because the original ran on an A100. Different floating-point rounding decorrelates
the chaotic trajectory over 40000 steps, so the long-time average differs within its statistical error.

## 5. Complete recompute

Every data file was regenerated from scratch with the job lists (`reproduce/jobs/*.txt`), 470 jobs on 8 H100 GPUs
on 2026-10-06, and compared with `data/` by `python tools/compare_recompute.py`.

| Data | Files | Agreement |
|---|---|---|
| Theory F(s, ε) | 274 + 5 checks | max 2.6·10⁻⁸, median 5·10⁻¹² |
| Plotted theory D_KY/N, h_KS/N (45 points) | — | relative ≤ 4·10⁻⁷ |
| Fig. spectra theory F | 8 curves | ≤ 1.6·10⁻⁸ |
| PR theory | 3 | ≤ 3·10⁻¹⁷ |
| Max-exponent operator data | 1 | identical |
| Network spectra | 94 | 52 identical to round-off (≤ 10⁻¹¹); 42 differ (see below) |
| PR simulations | 90 | 30 identical; 60 differ (see below) |

**Theory** is deterministic and reproduces to round-off.

**Simulations** differ only where the original run used different hardware.
- **Which runs differ.** All differing networks and PR simulations were originally run on A100 GPUs, or on an
  H100 variant with different arithmetic. Every run whose hardware matched reproduces bitwise.
- **Why they differ.** A chaotic trajectory decorrelates after a few Lyapunov times, so these runs are
  independent samples of the same long-time averages.
- **Network spectra.** The largest differences are near onset (g = 1.25), where the positive exponents are
  ~0.015. Each plotted value is a median over five networks, so the plotted D_KY/N and h_KS/N change by at most
  2.5% and 2.9%; at g ≥ 2, the plotted values are unchanged.
- **PR simulations.** The difference divided by its expected standard deviation (√2 times the jackknife standard
  error) has a median of 0.72 and exceeds 2 in 10% of files, consistent with statistical noise. The plotted
  means change by ≤ 1.9%.
- **Max-exponent figure.** The simulated medians change by ≤ 2%.

**Time-grid check for δ → 0 at g = 3 and 5** (seed 0, `results/theory_gridcheck`).
- **Question.** The paper solves these points at spacing T/m = 0.4. They were rerun at spacings 0.2 and 0.125.
- **D_KY/N:**

  | g | spacing 0.4 | spacing 0.2 | spacing 0.125 |
  |---|---|---|---|
  | 3 | 0.06174 | 0.06014 | 0.06172 |
  | 5 | 0.08086 | 0.07915 | 0.08074 |

- **h_KS/N:** spacings 0.4 and 0.125 agree to 0.2%.
- **Noise level.** The scramble-to-scramble standard deviation of one run is 1–1.5% in D_KY/N and 2–3% in h_KS/N.
  The non-monotone 0.2 values lie within about 2 such deviations.
- **Conclusion.** Any time-grid bias at spacing 0.4 for g ≤ 5 is below the path-sampling noise of a single
  scramble. The plotted values average 8 scrambles.
