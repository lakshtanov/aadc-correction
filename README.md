# aadc-correction

Unbiased Monte Carlo Greeks for discontinuous payoffs (barriers, digitals, autocallables) via **δ-function correction**.

No smoothing. No parameter tuning. Works with any model.

## Requirements

```bash
pip install "aadc>=2.22.1"   # switch registry required
```

## Quick start

```bash
python examples/example_digital.py
python examples/example_barrier.py
```

## The problem

Pathwise (AAD) Greeks fail at discontinuities — `δ(S - B)` cannot be sampled by MC. Standard workaround is smoothing (`Φ((S-B)/ε)`), which introduces bias and requires per-product tuning.

## The solution

The correction driver finds the boundary in normal space via Newton root-finding, evaluates the density and payoff jump there, and adds an exact correction term. No smoothing, no `ε`.

**Formula**: `correction = Σ_i jump_i · φ(u*_i) / |v·∇g_i| · (∂g_i/∂θ)`

## Switch registry (aadc >= 2.22.1)

Discontinuities are discovered **automatically** via the AADC switch registry. No manual indicator creation needed — the model code stays unchanged:

```python
fn = aadc.Functions()
fn.start_recording(register_switches=aadc.SWITCH_JUMP)

# ... model code with aadc.iif(S > B, ...) — unchanged ...

fn.stop_recording()

# Indicators discovered automatically
g_res = [sw.g for sw in fn.cmp_switches()]
```

Each switch provides `g` (signed distance to boundary) as a tape Result handle. The correction driver uses these for Newton root-finding and jump estimation.

**JUMP** = comparison (`>`, `<`, `>=`, `<=`): `g = A - B` (payoff discontinuity).
**KINK** = `max`/`min`/`abs`: `g = a - b` (slope discontinuity, tracked separately).

## Drivers

| Driver | Method | Bias | Speed |
|---|---|---|---|
| `correction_driver_vec2.py` | Fully vectorized Newton (AVX + MT) | None | 1× |
| `correction_driver_fries.py` | Fries (2018) discretized delta | O(w) | **~2-3×** faster |
| `rqmc.py` (`CorrectionDriverRQMC`) | Vec2 on scrambled Sobol points (see below) | None | 1× per path, far fewer paths |

Both drivers are fully vectorized — no scalar workspace loops. `jac_theta` computed in batch via `aadc.evaluate()`.

### Newton (exact, unbiased)

Root finding in normal space → exact boundary z*, exact density φ(u*). No tuning parameter. Uses `VectorFunctionSelectiveAD` with per-indicator reverse for Newton (the same pass gives ∂g_i/∂z and ∂g_i/∂θ, so nothing extra is computed at z*), forward-only for jump.

Robustness — nothing is dropped silently:
- **Root solver** (`root_solver="rtsafe"`, default): Newton with steps clipped to ±8 while it makes progress; once the root is bracketed (g < 0 and g > 0 seen) a step leaving the bracket, or no progress, switches to the Illinois method on the bracket (forward passes only, bisection if it stalls); no bracket and no progress (e.g. a valley of g that does not reach 0): search for a sign change. Plain Newton cycles across kinks such as a worst-of `min(S_i)` and lost roots that exist. `root_solver="newton"` keeps the previous plain Newton.
- **Linear indicators** (`linear_roots=True`, default with rtsafe): if the gradient of an indicator is parallel to its direction `v_i` on all of 64 pilot paths, the indicator is a function of `u = v_i · z` alone — a barrier on a GBM, a barrier or corridor on a Gaussian (Hull–White / LGM) rate. Its boundary is then `u = U*_i` for every path; `U*_i` is found once by bisection on the line `u v_i` and used as the solver's starting point, which then converges in one step. The solver still checks every root, so a wrong detection costs iterations, never accuracy. Benchmarks (same paths, results equal to 1e-11): GBM barrier 27.8 s → 15.6 s, GBM + Hull–White barrier 20.3 s → 13.3 s, Heston autocallable unchanged (no linear indicator).
- **No screening** by default (`skip_sigma=None`): skipping a pair when the *linearised* distance `|g| / |dg/du|` is large is biased where g is flat at the path point. A pair is dropped only where its contribution is exactly zero in double precision (`|u| > 40`, φ(u) = 0).
- **Bracketed jump**: `P(z* + εv) - P(z* - εv)` is accepted only if `g_i` and no other indicator changes sign between the two points; otherwise ε is adapted (`jump_eps=1e-7`, root located to `root_tol_u=1e-10` along v).
- **Counters** in the result: `newton_unconverged`, `newton_far` (dropped with φ = 0), `jump_unresolved` — 0 on every benchmark of this repository.

Speed options: `indicator_threads=3` (indicators processed concurrently; AADC releases the GIL), `num_threads` (AADC threads; best ≤ the cores of one socket), `vector_kind="best"`: the tape is **compiled** once (`aadc_extensions.compile_tape`: the `avx512` backend where the licence has that feature, else the free vector JIT `jit`) and replayed on the workspace of its kernels (`avx2` for `jit`). Without compiled kernels every replay runs in AADC's interpreter (an AVX-512 workspace on an unlicensed build, for example). The driver also asks AADC only for what it uses: a Newton step requests its own output and the adjoints along the nonzero entries of its direction. Results are bitwise identical; one driver run on the benchmarks of the paper became 1.2–2.3× faster (B1 3.34 → 1.65 s, B2 1.60 → 0.87 s, B3 1.92 → 0.97 s, credit autocallable with the coin 5.71 → 2.52 s, LGM range accrual 39.9 → 33.3 s). `indicator_workers` sets the AADC workers per concurrent indicator job (default `num_threads`; splitting them was slower).

### Fries (fast, biased)

Implements Fries (2018, arXiv:1811.05741) discretized delta: `δ(X) ≈ 1{|X|<w}/(2w)`. Supports **adaptive window selection** via split-half stability test.

## Benchmark

Down-and-out barrier (GBM, 10 steps), 50K paths, 500K bump reference (`benchmarks/bench_fries_vs_newton.py`):

| Method | Delta | Error | Time |
|---|---|---|---|
| Newton (vec, 4 threads) | +0.8134 | 0.2% | 1.2s |
| Fries (w=5%) | +0.8206 | 0.7% | 0.4s |
| Fries (w=10%) | +0.8288 | 1.7% | 0.6s |

Digital option (analytic reference): 0.0% error.

Times: Xeon Silver 4114, process pinned to 10 physical cores of one socket, `OPENBLAS_NUM_THREADS=1`. The Newton row was 8.5s before #1 (full reverse over all outputs after Newton) and 4.2s before #11 / #12 (indicators processed concurrently, AVX-512 replays) on the same machine. On this 2×10-core box the driver is fastest with `num_threads` ≤ cores of one socket: spreading over both sockets (16-20 threads) is slower than 10.

## Intensity default: the threshold integrated exactly (coins)

In an intensity (Cox) model the default is the first step with `A_k = sum lambda dt >= E`, `E = -log Phi(-z_E) ~ Exp(1)`
independent of everything else. The comparisons `g_k = A_k - E` are discontinuities; with the default directions every
(path, step) pair needs Newton. `z_E` is a **coin**: an input normal that enters the payoff only through comparisons,
each depending on it alone. Taking the correction of `g_k` along the direction of the coin (Theorem 1 allows any fixed
direction) gives, for every path and every step,

- root `z* = -Phi^{-1}(exp(-A_k))` in closed form (no Newton),
- weight `phi(z*) / |dg_k/dz_E| = exp(-A_k)`: the density of `E` at the boundary (nothing divides by the intensity, so
  no heavy tails when the CIR intensity violates the Feller condition),
- jump `F(E = A_k - h) - F(E = A_k + h)` (default in step `k` minus default later), `dg_k/dtheta = dA_k/dtheta`,

so the boundaries of all steps are integrated over `E` exactly: the derivative of the conditional expectation
`sum_k (exp(-A_{k-1}) - exp(-A_k)) D_k + exp(-A_N) L`, without writing the product in that survival form.

`CorrectionDriverVec2(..., coins="auto")` (default) finds coins on the tape on pilot paths: inputs with zero pathwise
derivative of the payoff whose comparison derivatives do not change when all other inputs are redrawn (separable) and
have one sign (monotone); all candidates are screened at once, survivors one by one; the law `-log Phi(-z)` is checked
to 1e-8 (exact term), any other monotone law gets the delta correction along the coin (Newton). `coins="off"`: the
default directions everywhere. Optional markup `coins.CoxDefault` (one call per step in the pricing code): the jumps
between non-date steps come from the recorded step values and all `dA_k/dtheta` from one reverse replay (fast path,
checked on pilot paths).

Credit-linked autocallable (2 equities, CIR intensity, 104 weekly steps; `benchmarks/bench_cox_coins.py`), 16 x 10K
paths, same paths for all modes, vs a CRN bump on 1.6e7 paths:

| sigma_lambda (2 kappa theta / sigma^2) | | dV/dlambda0 | dV/dsigma_lambda | time / 10K paths |
|---|---|---|---|---|
| 0.10 (4) | reference | -38.98 ± 0.13 | -0.775 ± 0.017 | |
| | `coins="off"` | -39.127 ± 0.038 | -0.7645 ± 0.010 | 47 s |
| | coin, found on the tape | -39.129 ± 0.044 | -0.7639 ± 0.011 | 20 s |
| | coin + `CoxDefault` markup | -39.129 ± 0.044 | -0.7639 ± 0.011 | **7 s** |
| 0.25 (0.64) | reference | -41.46 ± 0.14 | -0.309 ± 0.017 | |
| | `coins="off"` | -41.416 ± 0.052 | -0.2878 ± 0.013 | 48 s |
| | coin + markup | -41.433 ± 0.058 | -0.2861 ± 0.014 | **7 s** |

Scrambled Sobol (option B), 10 x 8192 paths; work = time x variance of one estimate:

| sigma_lambda | Greek | Sobol gain: `coins="off"` / coin + markup | work with Sobol: `coins="off"` / coin + markup |
|---|---|---|---|
| 0.10 | dV/dlambda0 | 4.2x / 3.1x | 0.134 / **0.034** |
| 0.10 | dV/dsigma_lambda | 4.4x / 3.9x | 0.0097 / **0.0019** |
| 0.25 | dV/dlambda0 | 2.0x / 2.1x | 0.77 / **0.145** |
| 0.25 | dV/dsigma_lambda | 3.9x / 3.7x | 0.013 / **0.0022** |

Per estimate: 40 s (`coins="off"`) vs 6 s (coin + markup). Spot Greeks are identical in all modes (no coin in them).
Not covered as is: a threshold used elsewhere in the payoff,
or several names whose thresholds are coupled by a copula.

## Daily monitoring: range accrual on a 3M rate (LGM) with a default

`benchmarks/bench_lgm_range_accrual.py`: LGM / Hull-White one factor (exact Ornstein-Uhlenbeck step, flat 3% curve,
a = 0.03, sigma_r = 0.01), index = 3M simple rate from the closed-form zero bond, corridor 2%..4% observed every day
for one year (252 steps, 504 comparisons), quarterly coupon 5% a year x days in the corridor / 63, CIR intensity
(correlation 0.3 with the rate) and a Cox default (252 steps, integrated over the threshold: coin with the
`CoxDefault` markup). The model, the bond formula and the payoff are written out in the module docstring.

Reference: CRN central bump on 8M paths. The bump of a discontinuous payoff is biased, O(h^2): for dV/dsigma_r the
steps h/2, h, 2h give -160.418, -159.884, -157.698 (differences 1 : 4), Richardson -160.60; with h = 0.001 (10% of
sigma_r) alone the reference would look 9 standard errors away from the driver. Driver, 8 x 4096 paths:

| Greek | reference | Sobol (option B) | Sobol gain over MC |
|---|---|---|---|
| dV/dR0 | -101.58 (Richardson) | -101.667 ± 0.018 | 5.2x |
| dV/dsigma_r | -160.60 (Richardson) | -160.518 ± 0.012 | 19.9x |
| dV/dlambda0 | -37.58 ± 0.18 | -37.4506 ± 0.0005 | 78x |
| dV/dsigma_lambda | +0.000 ± 0.022 | +0.0294 ± 0.0005 | 41x |

One estimate on 4096 paths takes about 100 s (10 threads): with daily monitoring the delta correction costs
(indicators x tape length); Sobol is the main lever here.

## Randomized QMC (Sobol)

`rqmc.py` — `CorrectionDriverRQMC`: the same correction (`CorrectionDriverVec2` inside) fed with scrambled Sobol
points; returns the Greeks and their standard error over independent scramblings.

```python
from rqmc import CorrectionDriverRQMC
drv = CorrectionDriverRQMC(fn, payoff_res, z_args, theta_args, num_threads=8)   # ordering="active_subspace"
drv.precompute_directions(theta_values)
res = drv.run(n_paths=2**14, n_scramble=16)
res["total"], res["stderr"]
```

Why it works: the correction integrates the payoff jump analytically along the Newton line, so the indicator
is gone from the per-path estimator; what is left is as smooth as the switch functions — QMC applies.
Smoothing cannot do this: a small width makes the integrand nearly discontinuous, a large one biases it.

- **Ordering** (which directions get the best Sobol coordinates; any orthogonal map `Q` keeps the estimator
  unbiased, it only decides how much Sobol helps). Three choices:
  - `"active_subspace"` (**default**, generic) — `Q` from the tape alone: `n_pilot` ordinary random paths
    (default 2000) go through the tape once, AADC returns the gradients of the payoff and of every indicator,
    and the eigenvectors of their averaged outer products become the Sobol coordinates, most important first.
    Needs nothing from the user. Cost: one AADC call on the pilot paths (a forward and a reverse per indicator),
    once per pricing setup — ~8 ms on the Heston digital (16 inputs; 6–23 ms across schemes and runs), ~0.2 s on the 3-asset autocallable
    (31 indicators); `drv.rotation_time` reports it, and the QMC benchmarks include it in every timing.
    Fewer pilot paths are cheaper but were measurably worse on a scheme with a (smoothed) floor
    (about 25% larger error with 256 paths), so the default stays at 2000.
  - `"bridge"` (**model-specific**) — Brownian bridge per factor. The driver cannot see which tape inputs are
    the time steps of which Brownian motion, so the caller gives the layout:
    ```python
    drv = CorrectionDriverRQMC(fn, payoff_res, z_args, theta_args, ordering="bridge",
                               bridge_factors=[[0, 2, 4, 6], [1, 3, 5, 7]])  # Heston: spot / variance normals
    ```
    `bridge_factors` lists, for each Brownian factor, the positions in `z_args` of its per-step normals in time
    order. Inputs not listed (e.g. a normal for a default time) take the first Sobol coordinates. Only distinct,
    in-range indices are checked: a wrong layout raises no error — the estimator stays unbiased, the Sobol gain
    silently drops. Use it when you wrote the model loop and the inputs really are Brownian increments in time
    order; no pilot paths, so it is free. Measured against the default: about equal on the smooth Heston scheme, worse with a
    smoothed floor (the bridge does not see the variance directions), mixed on GBM products (2-asset autocallable
    vega RMSE 0.5% vs 0.7%, down-and-out call standard error 4.9x vs 5.9x below MC) — see
    `benchmarks/bench_slides_qmc.py`, `benchmarks/bench_slides_fries.py`, `tests/test_rqmc.py`.
  - `"none"` — Sobol coordinates straight into the tape inputs (for comparison).
- **The model discretization must be smooth.** A kink such as a variance floor `max(v, v_min)` makes ∂g/∂θ jump
  on oblique surfaces and removes most of the gain. For Heston use the drift-implicit scheme for `sqrt(v)`
  (no floor; needs `4·kappa·theta > xi²`), see `examples/example_heston_rqmc.py`.

Heston digital, 2^13 paths x 16 (`examples/example_heston_rqmc.py`), standard-error reduction RQMC vs MC:

| variance scheme | delta | d/d(xi) | d/d(kappa) |
|---|---|---|---|
| Euler, `max(v, 0.1·theta)` | 2.0x | 1.3x | 1.4x |
| drift-implicit `sqrt(v)` (no floor) | 13.7x | 16.5x | 11.3x |

All numbers of the QMC slides (scheme comparison, equal time vs smoothing, paths needed for the same accuracy):
`benchmarks/bench_slides_qmc.py` (`QUICK=1` for a short run). Timings there are per estimate, warm, 10 threads;
they include the choice of `Q` (pilot paths) but **exclude generating the inputs** (Sobol points / pseudo-random
normals) for every method alike. This favours the methods that need many paths (smoothing): with
`rqmc.sobol_normals` the Sobol inputs for 2^12 paths x 16 take a few ms, for 2^22 seconds.
AADC has no random-number generators of its own; inputs are plain arrays.
Newton vs Fries with MC and Sobol (2-asset autocallable): `benchmarks/bench_slides_fries.py`.

GBM down-and-out call, 10 dates (`tests/test_rqmc.py`): standard error 5.9x smaller than MC with
`"active_subspace"`, 4.9x with `"bridge"`. On the Heston digital with the smooth scheme,
at equal wall time the δ-correction + Sobol is 6-14x more accurate (RMSE incl. bias) than smoothing + Sobol with
the best width chosen per budget.

`benchmarks/bench_slides_ladder.py`: delta and gamma ladders vs spot for Benchmark 2 (spot of asset 1 from 60 to 130,
the same 4000 paths on every point; gamma = central difference of two corrected deltas). The corrected estimator is
smooth in the spot for fixed paths, so the ladder is smooth at 4000 paths (RMSE vs an 8M-path bump: delta 0.006,
gamma 0.001; bump on the same paths: 0.024, 0.045). `PLOTS=1` writes `fig_ladder_b2.pdf`.

## Paths and time for a given accuracy (all benchmarks)

`benchmarks/bench_paths_for_accuracy.py` and `benchmarks/bench_smoothing_paths.py` give, for benchmarks 1–5, the paths and
the time per estimate (10 threads, compiled tape, generation of the random inputs excluded) needed to reach an RMSE of
1% and of 0.5% of each Greek:

* **correction** (unbiased: RMSE = standard deviation) with MC, Sobol option B (its one-off pilot of 512 paths reported
  separately) and option A where the layout is known; N doubled from 2^8 until every Greek reaches the target; 32
  repetitions (16 from 2^15). On the credit autocallable 8–10 scramblings underestimated the Sobol standard deviation
  about 2× (heavy tails), hence 32.
* **smoothing**: every indicator replaced by a logistic of width eps × a natural scale (`record(smooth=eps)` in the
  benchmark tapes; unsmoothed tapes unchanged bitwise), pathwise AAD, N up to 2^23; the RMSE includes the bias; for each
  target the best eps of the grid (an oracle in its favour) and the time of one estimate per eps of the grid at that N
  (the least a user spends to choose eps).

```
OPENBLAS_NUM_THREADS=1 PRODUCT=b1 R=32 R_HI=16 M_RSWITCH=15 STOP_TARGET=0.005 N_PILOT=512 M_LO=8 M_HI=20 \
    OUT=paths_b1.json python benchmarks/bench_paths_for_accuracy.py          # PRODUCT=b1|b2|b3|credit|lgm (credit: MODES=coin)
OPENBLAS_NUM_THREADS=1 PRODUCT=b1 REF_JSON=paths_b1.json EPS=0.01,0.03,0.1,0.3,1,3,10 M_LO=10 M_HI=23 R=16 \
    OUT=smooth_b1.json python benchmarks/bench_smoothing_paths.py
```

In short (with the eps search counted): the correction is faster on the vegas of the barrier options at 0.5% and on
every credit Greek (credit autocallable: 2K paths, 1.2 s for 1% on dV/dlambda0, where smoothing needs ~70M paths or
cannot reach the target at all); smoothing is faster on deltas and on the rate Greeks, and on the 3-asset Phoenix, where
Sobol helps the correction little (open problem). The full tables are in the paper draft.
## Comparison with the Kovalov–Naumann corrections

[docs/comparison_kovalov_naumann.md](docs/comparison_kovalov_naumann.md) compares this repository with two recent
corrections from RWTH Aachen (STCE): the event-time correction of A. Kovalov and U. Naumann (*Surrogate Greeks for
Regime-Switching Processes via Corrected AD*, first-order surrogate; second order by Stein's identity or quadrature) and
the two GVA variants of U. Naumann's QuantMinds 2026 talk (local mollification, integral co-area rule). Implementations:
`event_time.py` (`EventTimeDriver`, `AccumulatorEvent`), `gva_drivers.py`; benchmarks: `benchmarks/bench_event_time.py`,
`benchmarks/bench_gva.py`. In short: the first-order surrogate is biased whenever the payment at the event is curved in
the step's noise (wrong sign on a timer option); GVA-local carries the bias of a mollification; GVA-integral with a slice
along a coordinate axis loses the Greek where the axis does not cross the boundary (worst-of); with a fixed threshold and
many steps (timer) the second-order event-time correction needs less work than the correction of this repository, with
an exponential threshold (default) the exact integration of the threshold here is cheaper and less noisy.

## QuantLib compatibility

`tests/test_quantlib_compat.py` runs the driver on models taken from QuantLib's own processes, recorded on the AADC
tape with the AADC build of QuantLib (`pip install aadc-quantlib-tracing`; the test is skipped without it). Use a
separate environment: that package pins its own `aadc` version.

| QuantLib process | product | Greeks | driver vs CRN bump |
|---|---|---|---|
| `BlackScholesMertonProcess` | down-and-out call | delta, vega | +0.1 / -1.6 sd |
| `HestonProcess` (QE, QuantLib's default) | digital call | delta, d/d(xi) | -1.4 / -1.0 sd |
| `HullWhiteProcess` (short rate) + GBM spot | down-and-out call | delta, d/d(sigma_r) | +1.1 / -1.0 sd |

No lost root-finding pairs. QuantLib's internal comparisons appear in the switch registry too (e.g. the branches of
the QE scheme): the driver handles them as ordinary indicators. Known issue in aadc-quantlib-tracing 1.41.4, not in
the driver: `HestonProcess` with `FullTruncation` / `PartialTruncation` gives NaN states on the tape on paths where the
variance went negative (the same code without the tape is finite), so a digital on the tape prices 0.25 instead of
0.58; `QuadraticExponentialMartingale` and `Reflection` are identical to plain QuantLib. The test prints this check.

## API

```python
from correction_driver_vec2 import CorrectionDriverVec2

# Record with switch registry
fn = aadc.Functions()
fn.start_recording(register_switches=aadc.SWITCH_JUMP)
# ... model code ...
fn.stop_recording()

# Setup driver: indicators are taken from the switch registry when indicator_res is not given
# (or pass them explicitly: indicator_res=[sw.g for sw in fn.cmp_switches()])
driver = CorrectionDriverVec2(fn, payoff_res, z_args, theta_args, num_threads=4)
driver.precompute_directions({S_arg: S0, vol_arg: sigma})

# Run
result = driver.run(z_all)
# result['price'], result['pathwise'], result['correction'], result['total']
```

## How it works

1. **Record** the pricing kernel with `register_switches=SWITCH_JUMP`
2. **Discover** indicators automatically via `fn.cmp_switches()`
3. **Precompute** gradient directions `v_i = ∇_z g_i / |∇_z g_i|` at z=0
4. **Root** along direction `v_i`: `u*` where `g_i(z*) = 0` for every (path, indicator) pair — safeguarded Newton (one forward + one reverse of `g_i` per Newton step; the converged step also gives `∂g_i/∂θ`)
5. **Jump** evaluation: `P(z* + εv) - P(z* - εv)` via tape forward-only, accepted only if `g_i` and no other indicator changes sign in between
6. **Accumulate**: `correction_θ += jump · φ(u*) / |v·∇g| · (∂g/∂θ)`

All steps batch-vectorized via `aadc.evaluate()` with AVX + multithreading.

## References

- Lakshtanov, E. "Unbiased Monte Carlo Greeks for Discontinuous Payoffs" (2026)
- Fries, C.P., "Stochastic algorithmic differentiation of (expectations of) discontinuous functions", arXiv:1811.05741, 2018
- Joshi, M. and D. Kainth, "Rapid and accurate development of prices and Greeks for nth to default credit swaps", *Quantitative Finance* 4(3), 2004
- Chan, J.H. and M.S. Joshi, "Fast Monte Carlo Greeks for financial products with discontinuous pay-offs", *Mathematical Finance* 23(3), 2013
- `pip install aadc` — [matlogica.com](https://matlogica.com)
