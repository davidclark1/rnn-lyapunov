# Numerical settings and convergence evidence

Every setting of every plotted number, and the evidence that it is converged. The exact command for every data file
is in `reproduce/jobs/*.txt`; the settings are also stored in each file (`config`, `params`) in `data/`.

The theory never uses simulated exponents: its inputs are the physical parameters (g, δ) and the numerical
resolution below. Thresholds are built from DMFT quantities only (the uncoupled exponent s_u = log(1−δ)/δ, or −1 for
δ → 0, and the upper edge s_*).

## Single-site theory (`rnn_lyapunov.theory`)

Common to all points:

- **Gain trajectories.** Scrambled-Sobol Gaussian paths in Fourier coordinates (`gain_paths`), 4096 per scramble
  (2048 and 1024 at δ = 0.1 and 0.05 in Fig. spectra). The seed is the Sobol scramble. The spread over scrambles
  gives the s.e.m. of each plotted value.
- **Regulators.** η = δ·ε with ε = 0.1, 0.03, 0.01, 0.003, 0.001, 4·10⁻⁴, 1.6·10⁻⁴, each solution warm-starting the
  next. F(s) is read out at the smallest ε. Near onset (g = 1.25), 12 regulators go down to ε = 1.6·10⁻⁶.
- **Fixed point.** Tolerance 10⁻⁷ on log(P, A−η), Anderson depth 6, warm start from the previous threshold. Every
  plotted file has `all_converged = true`.
- **Bloch twists.** Gauss–Legendre phases where |a_eff|·T < 12 (`theory.a_eff`). They remove the periodic-window
  artifact near s_u.
- **DMFT.** Continuous time uses the lag grid 0.025. Gaussian quadrature is the trapezoid rule with dz = 0.04 on
  [−9, 9]. The lag range ("tmax") depends on g:

  | Computation | g < 1.4 | 1.4 ≤ g ≤ 2 | g > 2 |
  |---|---|---|---|
  | Theory F(s) | 320 (g = 1.25) | 160 | 80 |
  | PR theory (`scripts/run_pr_theory.py`) | 320 | 160 | 80 |
  | s_*(g), Fig. max_exponent_operator (`scripts/run_max_exponent.py`) | 320 | 160 for g < 2.25 | 80 |

Settings of every theory file in `data/theory/`. Thresholds are given as `top` (s_low, then 12 uniform thresholds on
[s_low, 0) and 12 on [0, s_*)), `upper` (offsets above s_u, then 12 on [0, s_*)) or `full`. "m" is the number of time
points, so the time-grid spacing is T/m (equal to δ for the maps).

| Figure, δ | g | T | m (spacing) | twists | thresholds | scrambles |
|---|---|---|---|---|---|---|
| spectra, 0.5 | 3, 5 | 32 | 64 | 4 | `full` | 4 |
| spectra, 0.25 | 3, 5 | 32 | 128 | 4 | `full` | 4 |
| spectra, 0.1 | 3, 5 | 32 | 320 | 4 | `full` | 1 |
| spectra, 0.05 | 3, 5 | 32 | 640 | 2 | `full` | 1 |
| dimension_entropy, 0.5 | 1.25 | 64 | 128 | 0 | `top`, s_low = −0.1, 16 + 12, 12 regulators, tmax 320 | 4 |
| | 1.5, 2 | 64 | 128 | 0 | `top`, s_low = −0.8, tmax 160 | 8 |
| | 2.5–10 | 32 | 64 | 0 | `top`, s_low = −0.8 | 8 |
| | 12–20 | 64 | 128 | 4 | `upper` | 4 |
| dimension_entropy, 0.25 | 1.25 | 64 | 256 | 0 | `top`, s_low = −0.1, 16 + 12, 12 regulators, tmax 320 | 2 |
| | 1.5, 2 | 64 | 256 | 0 | `top`, s_low = −0.7, tmax 160 | 8 |
| | 2.5–6 | 32 | 128 | 0 | `top`, s_low = −0.7 | 8 |
| | 8–20 | 64 | 256 | 4 | `upper` | 2 |
| dimension_entropy, δ → 0 | 1.25 | 64 | 160 (0.4) | 0 | `top`, s_low = −0.1, 16 + 12, 12 regulators, tmax 320 | 2 |
| | 1.5, 2 | 64 | 160 (0.4) | 0 | `top`, s_low = −0.6, tmax 160 | 8 |
| | 2.5–5 | 32 | 80 (0.4) | 0 | `top`, s_low = −0.6 | 8 |
| | 6 | 32 | 160 (0.2) | 0 | `top`, s_low = −0.6 | 8 |
| | 8–20 | 32 | 160 (0.2) and 256 (0.125), extrapolated to spacing 0 | 4 | `upper` | 2 |

### Convergence evidence (Fig. dimension_entropy)

All of this evidence is stored in the original research tree. The δ → 0 grid study is bundled in
`data/theory_checks/`.

- **Window T** at g = 20, δ = 0.5:
  - D_KY/N = 0.1283 (T = 32), 0.1267 (T = 64), 0.1264 (T = 128).
  - T = 64 is converged within the scramble scatter.
- **Window T** at g = 20, δ = 0.25:
  - T = 32 is biased up by 1.6% (D_KY/N) and 3% (h_KS/N) relative to T = 64.
  - T = 64 is therefore used for g ≥ 8.
- **Other controls** at g = 20, δ = 0.5, T = 32 each change D_KY/N by ≤ 0.3%:
  - regulators down to ε = 2.6·10⁻⁵;
  - a 2× denser threshold grid;
  - quadrature dz = 0.02;
  - 8192 paths;
  - the `full` grid.
- **Near onset** (g = 1.25):
  - The coarse `top` grid overestimates D_KY/N (0.0059 against 0.0055), because F(s) is interpolated near the
    Kaplan–Yorke point.
  - The fine grid near s = 0 fixes this. T = 128 changes D_KY/N by < 1%.
- **δ → 0 at strong coupling.** The gains switch on a time ~ 1/g, so the time grid T/m matters, and the error is
  quadratic in the spacing.
  - At g = 20, seed 0, D_KY/N is:

    | spacing | D_KY/N |
    |---|---|
    | 0.25 | 0.0931 |
    | 0.2 | 0.0918 |
    | 0.125 | 0.0903 |
    | 0.1 | 0.0897 |

  - Extrapolating from 0.2 and 0.125 predicts 0.0899 at spacing 0.1.
  - The plotted values are extrapolated to zero spacing per scramble (`theory.extrapolate_to_zero_spacing`).
  - T = 64 changes D_KY/N by −0.2% at spacing 0.2.

- **δ → 0 at g ≤ 6** (spacing 0.4 for g = 1.5–5). Single-scramble reruns at g = 3 and 5 with spacings 0.2 and
  0.125 show no grid effect above the path-sampling noise; spacings 0.4 and 0.125 agree to 0.2% in D_KY/N
  (`docs/VALIDATION.md`, section 5).

## Simulations

- **Lyapunov spectra** (`scripts/run_network.py`, QR method):
  - Fig. spectra: N = 4096, full spectrum, one network (seed 0), t_burn = 300, t_tangent_burn = 200,
    t_obs = 2000, QR every 1/2/5/10 updates at δ = 0.5/0.25/0.1/0.05.
  - Fig. dimension_entropy, g ≥ 2: N = 4096, leading k = 1024 exponents, one network.
  - Near onset (g ≤ 1.5): N = 16384 (maps) or 8192 (δ → 0), median over 5 networks, k = 256–768, longer
    burn-in/observation. At N = 4096, only 3–5 exponents are positive at g = 1.25, and D_KY/N comes out 2.5–4×
    too low.
  - δ → 0: RK4 at step 0.1 (g ≤ 5) or 0.05 (g ≥ 6), QR every 0.5 time units. Step 0.25 biases D_KY/N by +2% and
    h_KS/N by +5% at g = 1.5.
- **Participation ratios** (`scripts/run_participation.py`): N = 16384, 2 networks, t_obs = 20000, 8 time blocks,
  cross-block estimator (unbiased for independent blocks), a sample every 0.5 time units.
  - The estimate is flat in t_obs from 20000 to 80000 within its s.e.
  - The finite-N excess decreases with N: g = 3 gives 0.0439 / 0.0429 / 0.0417 / 0.0410 at N = 2048 / 4096 / 8192 /
    16384.
- **PR theory** (`scripts/run_pr_theory.py`):
  - Frequency grid 16384: changing it to 65536 changes PR by ≤ 4·10⁻¹⁵.
  - DMFT lag range ×2 changes PR by ≤ 3·10⁻¹⁴.
  - Quadrature dz = 0.02: dz = 0.04 differs by up to 3.7·10⁻⁴ at g = 20.

## Reproducibility

- **Bitwise reproduction of the paper's code.** The library is a port of the code that produced the paper's
  results. On the same GPU type, it reproduces the stored results bitwise or to ~10⁻¹¹ (theory F(s), QR spectra,
  DMFT, edges, PR theory). `docs/VALIDATION.md` lists the checks.
- **Chaotic simulations across hardware.** On different hardware (or library versions), the floating-point
  rounding differs, so a chaotic trajectory decorrelates after a few Lyapunov times. Long-time averages (spectra,
  PR) then agree within their statistical error, not bitwise. For example, the PR of g = 3, δ = 0.5,
  N = 16384 on an H100 versus the original A100 differs by 0.7 standard errors.
- **What is stored with every result.** Each result file stores its argv, the package version and source SHA,
  library versions, host and GPU, and UTC time.
