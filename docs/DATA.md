# Data formats

## Layout

- **`data/`** holds every result that a figure of the paper uses, 18 MB of JSON in total.
- **Same formats as the scripts.** Every file has the format that the matching `scripts/run_*.py` writes, so a
  recomputation into `results/` is read by the same loaders (`paper.data`, `root="results"`).
- **Manifest.** `data/manifest.json` lists every file with its sha256, its size, its kind and the figure points that
  use it.

| Directory | Written by | One file per | Used by |
|---|---|---|---|
| `theory/` | `scripts/run_theory.py` | (g, δ, window, grid, Sobol scramble) | Fig. spectra, Fig. dimension_entropy |
| `theory_checks/` | `scripts/run_theory.py` | δ → 0 grid-spacing and window study at g = 20 | `docs/NUMERICS.md`, `paper.tables.grid_spacing_table` |
| `network/` | `scripts/run_network.py` | simulated network (g, δ or RK4 step, N, seed) | all simulation markers of Figs. spectra, dimension_entropy, max_exponent_operator |
| `participation/` | `scripts/run_participation.py` | simulated network | PR markers of Fig. dimension_entropy |
| `participation_theory/` | `scripts/run_pr_theory.py` | δ (30 values of g each) | PR lines of Fig. dimension_entropy |
| `max_exponent_operator.json` | `scripts/run_max_exponent.py` | — | Fig. max_exponent_operator |

## Common fields

- `kind` identifies the format (`theory_cdf`, `network_lyapunov`, `participation_simulation`,
  `participation_theory`, `max_exponent_operator`).
- `g` and `delta`. `delta` is `null` for continuous time; simulations of continuous time give `rk4_dt` instead.
- `provenance` holds argv, the source SHA, UTC time, Python/NumPy/SciPy/torch versions, and the host and GPU.
- **Imported files.** These are the files produced by the paper's original research code. Their `provenance` also
  records `imported_from` (the original path, relative to the research tree) and `imported_sha256` (that file's
  hash). For these files, `provenance.argv` is the original driver's command line; the equivalent command of
  this repository is the matching line of `reproduce/jobs/`.
  Their numbers are unchanged, and only field names were translated: `h` became `delta`, scheme `rk4` became
  `flow`, the old summaries `kaplan_yorke_fraction`/`positive_sum` became `ky_dimension`/`entropy_rate`, and
  DMFT `r_d` became `c_d`. `tools/build_dataset.py` performs the import, and `tools/verify_dataset.py` checks
  every plotted number against the original results (all identical).

## `theory_cdf`

| Field | Meaning |
|---|---|
| `config` | `rnn_lyapunov.theory.TheoryConfig` as a dict (T, m, n_paths, seed = Sobol scramble, eps, n_twist, tol, dmft_tmax, ...) |
| `grid` | threshold rule (`full`, `upper`, `top`) and its parameters; `s_u` the uncoupled exponent |
| `s` | thresholds |
| `eps` | regulators in continuum units (η = δ·ε for the map), decreasing |
| `F` | F(s) averaged over twists, shape (len(s), len(eps)); **the result is `F[:, -1]`** |
| `F_theta`, `thetas`, `theta_weights`, `twisted` | per-twist values and the Bloch-phase quadrature |
| `edge` | upper edge s_* of the Lyapunov spectrum (F = 1 above), from the DMFT |
| `iterations`, `residual`, `trace_identity_error`, `all_converged` | fixed-point diagnostics, same shape as `F_theta` |
| `readout_parts` | maps only: leak / contact / regulator terms of the readout |
| `dmft`, `gain_sample` | DMFT variance, mean squared gain and residual; gain-sample statistics |
| `dimension_entropy_by_eps` | D_KY/N, h_KS/N and s_KY at every regulator |
| `seconds` | GPU wall time |

## `network_lyapunov`

| Field | Meaning |
|---|---|
| `params` | N, k (number of exponents), scheme (`map`/`flow`), step, t_burn, t_tangent_burn, t_obs, qr_every, seed, seconds, device |
| `exponents` | the k leading Lyapunov exponents, per unit time, in QR order (descending up to noise) |
| `summaries` | `rnn_lyapunov.observables.spectrum_summary` plus `split_half_rms` (noise estimate) |
| `stats` | time and neuron averages of the gain, squared gain, log gain and variance |
| `zero_diagonal` | true only for two δ → 0 runs at g = 3 used in Fig. max_exponent_operator (J_ii = 0) |

## `participation_simulation`

| Field | Meaning |
|---|---|
| `x`, `phi` | `participation.pr_from_blocks`: `pr_cross` (plotted), `pr_cross_se` (jackknife), `pr_naive`, `variance` |
| `N`, `seed`, `t_burn`, `t_obs`, `blocks`, `sample_dt` | settings |

## `participation_theory`

`rows` is a list over g, each row with `pr_x`, `pr_phi`, `psi_x`, `psi_phi`, `mean_gain`, `q`, `dmft_tmax`.

## `max_exponent_operator`

Panel data (the keys predate the final panel order: key `a` is panel (b), key `b` is panel (a)):

- `a`: the Schrödinger problem at g = 3, δ → 0.
  - The potential `V(tau)`, the bound-state energies `E` and eigenfunctions `psi`.
  - The normalized `C^x'(tau)` (`cxp`).
  - Window and grid checks.
- `b`: the spectrum of α_s⁻¹𝒯_s against s at δ = 0.5, g = 3 (`levels`, continuum `band`, `s_star`).
- `c`: s_*(g) for δ = 0.25, 0.5, 1, 0. `edge_lib` is the theory pipeline's upper edge; it agrees with `s_star`.
- `c_checks`: DMFT lag-window and lag-grid checks.
