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
- **No screening** by default (`skip_sigma=None`): skipping a pair when the *linearised* distance `|g| / |dg/du|` is large is biased where g is flat at the path point. A pair is dropped only where its contribution is exactly zero in double precision (`|u| > 40`, φ(u) = 0).
- **Bracketed jump**: `P(z* + εv) - P(z* - εv)` is accepted only if `g_i` and no other indicator changes sign between the two points; otherwise ε is adapted (`jump_eps=1e-7`, root located to `root_tol_u=1e-10` along v).
- **Counters** in the result: `newton_unconverged`, `newton_far` (dropped with φ = 0), `jump_unresolved` — 0 on every benchmark of this repository.

Speed options: `indicator_threads=3` (indicators processed concurrently; AADC releases the GIL), `vector_kind="best"` (AVX-512 replays where the CPU has them; results bitwise identical across kinds), `num_threads` (AADC threads; best ≤ the cores of one socket).

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

- **Ordering** (which directions get the best Sobol coordinates; any orthogonal map keeps the estimator
  unbiased): `"active_subspace"` (default) — automatic, from tape gradients on pilot paths, needs no knowledge of
  the inputs; `"bridge"` — Brownian bridge, pass `bridge_factors` (tape indices per Brownian factor); `"none"`.
- **The model discretization must be smooth.** A kink such as a variance floor `max(v, v_min)` makes ∂g/∂θ jump
  on oblique surfaces and removes most of the gain. For Heston use the drift-implicit scheme for `sqrt(v)`
  (no floor; needs `4·kappa·theta > xi²`), see `examples/example_heston_rqmc.py`.

Heston digital, 2^13 paths x 16 (`examples/example_heston_rqmc.py`), standard-error reduction RQMC vs MC:

| variance scheme | delta | d/d(xi) | d/d(kappa) |
|---|---|---|---|
| Euler, `max(v, 0.1·theta)` | 2.0x | 1.3x | 1.4x |
| drift-implicit `sqrt(v)` (no floor) | 13.7x | 16.5x | 11.3x |

All numbers of the QMC slides (scheme comparison, equal time vs smoothing, paths needed for the same accuracy):
`benchmarks/bench_slides_qmc.py` (`QUICK=1` for a short run).
Newton vs Fries with MC and Sobol (2-asset autocallable): `benchmarks/bench_slides_fries.py`.

GBM down-and-out call, 10 dates (`tests/test_rqmc.py`): standard error 5.9x smaller than MC with
`"active_subspace"`, 4.9x with `"bridge"`. On the Heston digital with the smooth scheme,
at equal wall time the δ-correction + Sobol is 6-14x more accurate (RMSE incl. bias) than smoothing + Sobol with
the best width chosen per budget.

`benchmarks/bench_slides_ladder.py`: delta and gamma ladders vs spot for Benchmark 2 (spot of asset 1 from 60 to 130,
the same 4000 paths on every point; gamma = central difference of two corrected deltas). The corrected estimator is
smooth in the spot for fixed paths, so the ladder is smooth at 4000 paths (RMSE vs an 8M-path bump: delta 0.006,
gamma 0.001; bump on the same paths: 0.024, 0.045). `PLOTS=1` writes `fig_ladder_b2.pdf`.

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
