# Paper → code map

Paper: D. G. Clark, *Lyapunov spectrum of random neural networks*.
Equations are referred to by their LaTeX labels (e.g. `eq:Fsite`), which stay fixed when numbering changes; the code
docstrings use the same labels in brackets, e.g. `[eq:Fsite]`. Notation: time step δ = `delta`
(`None`, or `0` on the command line, means δ → 0), threshold `s`, regulator η = `eta` (= δ·ε for the map),
cumulative distribution F(s) = `F`.

## Model and simulations

| Paper | What | Code |
|---|---|---|
| `eq:x`, `eq:J` | Euler map x(n+1) = (1−δ)x(n) + δJφ(x(n)), J_ij ~ N(0, g²/N) | `network.map_step`, `network.make_coupling` |
| δ → 0 | continuous-time network, RK4 | `network.rk4_step`, `network.dynamics(rk4_dt=...)` |
| `eq:Ms` (s = 0) | Jacobian M(n) = (1−δ)I + δJD(n), applied to tangent vectors | `network.map_tangent_step`, `network.rk4_tangent_step` (exact RK4 tangent) |
| `eq:lyap`, Sec. `sec:numerics` | Lyapunov spectrum by the QR method | `lyapunov.lyapunov_spectrum` |
| `eq:F` | F(s) = fraction of exponents below s | `observables.empirical_cdf` |
| `eq:ky`, `eq:ruelle` | Kaplan–Yorke dimension, entropy rate (sum of positive exponents) | `observables.ky_dimension`, `observables.entropy_rate`, `observables.spectrum_summary` |
| `eq:PRdef` | participation ratio of simulated activity | `participation.covariance_blocks`, `participation.pr_from_blocks` (cross-block estimator), `participation.pr_simulation` |

## Derivation, Parts 1–2 (exact, finite N)

| Paper | What | Code |
|---|---|---|
| `eq:driven`, `eq:KV`, `eq:K`, `eq:Kaction` | driven shifted tangent dynamics, K𝒱 = ℐ^v | `exact.shifted`, `exact.residual_matrix` |
| `eq:bounded`, `eq:Rvvdef`, `eq:F` | minimum-norm solution; F(s) = (1/N) tr R^vv(k+1,k) | `exact.response_vv(K, 0)`, `exact.cdf_from_response` |
| `eq:Veta`, `eq:Feta` | regularized solution 𝒱_η | `exact.response_vv(K, eta)` |
| `eq:B`, `eq:Bdef` | forward–backward matrix 𝓑_η, ‖𝓑_η⁻¹‖_op ≤ 1/η | `exact.forward_backward_matrix` |

These are checked numerically for small networks in `tests/test_exact.py` (the window-center trace equals the
fraction of QR exponents below s).

## Part 3: single-site theory

| Paper | What | Code |
|---|---|---|
| `eq:x0lim`, `eq:dmft` | single-site process and DMFT self-consistency | `dmft.solve` (`solve_discrete`, `solve_continuous`), `dmft.cached` |
| gain trajectories d₀(n) = φ'(x₀(n)) | sampled from the DMFT Gaussian process | `gain_paths.sample_gain_paths` |
| `eq:L`, `eq:Bsite` | single-site forward–backward system | `single_site.discrete_problem`, `single_site.continuous_problem` (symbols of L, α_s, γ_s) |
| `eq:Rvu00`, `eq:Ruv00`, `eq:Rvv00` | single-site responses | `single_site.sweep` (`Q_d` = R^vu₀₀, `X_d` = R^uv₀₀) |
| `eq:closed`, `eq:A`, `eq:P` | self-consistent kernels A, P | `single_site.solve` (fixed point, Anderson), `single_site.continue_in_eta` |
| `eq:Fsite` | F(s) = lim_{η→0⁺} ⟨R^vv₀₀(k+1,k)⟩ | `single_site.readout`; whole pipeline `theory.cdf_curve` |
| Sec. `sec:numerics` | periodic window, η continuation, Bloch twists, grid extrapolation for δ → 0 | `theory.TheoryConfig`, `theory.twist_rule`, `theory.thresholds_*`, `theory.extrapolate_to_zero_spacing` |
| `eq:dky_theory`, `eq:hks_theory` | D_KY/N and h_KS/N from F(s) | `observables.dimension_entropy_from_cdf`, `theory.dimension_entropy` |

## Appendices

| Paper | What | Code |
|---|---|---|
| `app:tfp-exact`, `app:lam`, `app:Fexact` | trivial fixed point: exponents from eigenvalues; F(s) = overlap area / πg² | `trivial_fixed_point.exponents_from_eigenvalues`, `trivial_fixed_point.overlap_fraction` |
| `app:tfp-site`, `app:Ffp` | single-site readout at the trivial fixed point | `trivial_fixed_point.single_site_cdf` |
| `app:semicircle` | δ → 0: semicircle law | `trivial_fixed_point.semicircle_cdf` |
| `app:delta1` | δ = 1: concentric disks, F(s) = e^{2s}/g² for s < log g | `trivial_fixed_point.overlap_fraction` (δ = 1) |
| `app:replica`, `app:zeromode`, `app:T` | two copies of the network: operator 𝒯_s; λ₁ = s_*, the largest s with 0 in the spectrum of 𝒯_s | `max_exponent.scaled_operator` (α_s⁻¹𝒯_s), `max_exponent.lowest`, `max_exponent.s_star`, `max_exponent.discrete_edge` |
| `app:limits`, `app:H`, `app:schrodinger` | limiting cases (summarized in Table `tab:limits`): δ → 0, Schrödinger operator, s_* = −1 + √(1 − E₀); δ = 1, s_* = log(g√C^d(0)); zero mode C^x'(τ) | `max_exponent.s_star` (δ = 1), `max_exponent.schrodinger`, `max_exponent.s_star_from_energy`, `max_exponent.continuous_edge` |
| `app:edge` | upper edge of the single-site F(s) is s_* | `max_exponent.upper_edge` (used by `theory`) |
| `app:pr`, `eq:psiphi`, `eq:psix`, `eq:psi00`; PR^a/N limit in Sec. `sec:dimension` | two-site cavity PR theory at step δ | `participation.pr_theory`, `participation.spectra` |
| `app:numerics`, `app:num-site` | numerical solution of the single-site theory: periodic window, η continuation, twists, δ → 0 grid extrapolation | `theory.cdf_curve`, `theory.TheoryConfig`, `theory.extrapolate_to_zero_spacing`; settings per point in `docs/NUMERICS.md` |
| `app:num-sim` | simulations: transients, QR interval, observation times, network sizes, RK4 | `lyapunov.lyapunov_spectrum`, `participation.pr_simulation`; settings in `docs/NUMERICS.md`, commands in `reproduce/jobs/` |

## Figures

| Figure (label, file) | Script | Data | Recompute (job lists) |
|---|---|---|---|
| `fig:spectra`, `spectra.pdf` | `paper/fig_spectra.py` | `data/theory/theory_delta{0.05,0.1,0.25,0.5}_g{3,5}_*`, `data/network/map_g{3,5}_*_N4096_seed0.json` | `reproduce/jobs/fig1_*.txt` |
| `fig:dimension_entropy`, `dimension_entropy.pdf` | `paper/fig_dimension_entropy.py` | `data/theory/`, `data/network/`, `data/participation/`, `data/participation_theory/` (table: `paper.tables.dimension_entropy_table`) | `reproduce/jobs/fig2_*.txt` |
| `fig:tfp`, `trivial_fixed_point.pdf` | `paper/fig_trivial_fixed_point.py` | none (computed in seconds) | — |
| `fig:maxexp`, `max_exponent_operator.pdf` | `paper/fig_max_exponent.py` | `data/max_exponent_operator.json`, `data/network/` (max exponents) | `reproduce/jobs/fig4.txt` |

All four: `python -m paper.make_figures` (from the bundled data, about a minute on a CPU).

## Tests (what each checks)

| File | Paper claim or identity |
|---|---|
| `tests/test_exact.py` | Part 1: (1/N) tr R^vv(k+1,k) at the window center equals the fraction of QR exponents below s (chaotic N = 10 network); R^vv(k+1,k) is the stable projector; ‖𝓑_η⁻¹‖_op ≤ 1/η; regularized → minimum-norm solution as η → 0 |
| `tests/test_single_site.py` | the solver's blocks equal the dense inverse of `eq:Bsite`; constant gains reproduce the circular-law overlap fraction (`app:Fexact`) and the semicircle; Tr P = Tr V |
| `tests/test_trivial_fixed_point.py` | Gauss's law: `app:Ffp` = `app:Fexact` for all δ; finite-N eigenvalues; δ → 0 semicircle |
| `tests/test_max_exponent.py` | δ = 1 (Molgedey et al.) and δ → 0 (Sompolinsky et al.) limits of s_*; C^x'(τ) is the zero mode (E₁ = 0); s_* matches simulated λ₁ |
| `tests/test_dmft.py` | quadrature, Price's theorem, δ = 1 variance, continuous-time energy condition, δ → 0 convergence, rejection of non-covariance solutions |
| `tests/test_network.py`, `tests/test_lyapunov.py` | tangent steps equal finite-difference Jacobians (map and RK4); QR against NumPy; sum of exponents = ⟨log|det M|⟩ |
| `tests/test_observables.py` | D_KY and h_KS from exponents and from F(s) agree |
| `tests/test_theory.py` | threshold rules, extrapolation, a small `cdf_curve`; slow: theory F(s) vs a simulated network |
| `tests/test_participation.py` | δ = 1 closed form; cross-block estimator unbiased; slow: theory vs simulated networks |
| `tests/test_gain_paths.py` | ring periodization and sampled covariance of the gain trajectories (pseudo-random and Sobol) |
| `tests/test_io.py` | result files round-trip with provenance |
| `tests/test_paper_data.py` | integrity of `data/` (sha256), convergence flags, and the paper tables |
| `tests/test_gpu.py` | GPU = CPU to 1e-10 (runs only with CUDA; 4 tests, pass on H100) |
