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
| `correction_driver_fries.py` | Fries (2018) discretized delta | O(w) | **10-15×** faster |

Both drivers are fully vectorized — no scalar workspace loops. `jac_theta` computed in batch via `aadc.evaluate()`.

### Newton (exact, unbiased)

Newton root-finding in normal space → exact boundary z*, exact density φ(u*). No tuning parameter. Uses `VectorFunctionSelectiveAD` with per-indicator reverse for Newton, forward-only for jump.

### Fries (fast, biased)

Implements Fries (2018, arXiv:1811.05741) discretized delta: `δ(X) ≈ 1{|X|<w}/(2w)`. Supports **adaptive window selection** via split-half stability test.

## Benchmark

Down-and-out barrier (GBM, 10 steps), 50K paths, 500K bump reference:

| Method | Delta | Error | Time |
|---|---|---|---|
| Newton (vec, 4 threads) | +0.8134 | 0.2% | 10.5s |
| Fries (w=5%) | +0.8206 | 0.7% | 0.6s |
| Fries (w=10%) | +0.8288 | 1.7% | 0.7s |

Digital option (analytic reference): 0.0% error.

## API

```python
from correction_driver_vec2 import CorrectionDriverVec2

# Record with switch registry
fn = aadc.Functions()
fn.start_recording(register_switches=aadc.SWITCH_JUMP)
# ... model code ...
fn.stop_recording()

# Automatic indicators
g_res = [sw.g for sw in fn.cmp_switches()]

# Setup driver
driver = CorrectionDriverVec2(fn, payoff_res, g_res, z_args, theta_args,
                               num_threads=4)
driver.precompute_directions({S_arg: S0, vol_arg: sigma})

# Run
result = driver.run(z_all)
# result['price'], result['pathwise'], result['correction'], result['total']
```

## How it works

1. **Record** the pricing kernel with `register_switches=SWITCH_JUMP`
2. **Discover** indicators automatically via `fn.cmp_switches()`
3. **Precompute** gradient directions `v_i = ∇_z g_i / |∇_z g_i|` at z=0
4. **Screen** each indicator on each path: is the path near the boundary?
5. **Newton** root-finding along direction `v_i` to find `u*` where `g_i(z*) = 0`
6. **Jump** evaluation: `P(z* + εv) - P(z* - εv)` via tape forward-only
7. **Accumulate**: `correction_θ += jump · φ(u*) / |v·∇g| · (∂g/∂θ)`

All steps batch-vectorized via `aadc.evaluate()` with AVX + multithreading.

## References

- Lakshtanov, E. "Unbiased Monte Carlo Greeks for Discontinuous Payoffs" (2026)
- Fries, C.P., "Stochastic algorithmic differentiation of (expectations of) discontinuous functions", arXiv:1811.05741, 2018
- Joshi, M. and D. Kainth, "Rapid and accurate development of prices and Greeks for nth to default credit swaps", *Quantitative Finance* 4(3), 2004
- Chan, J.H. and M.S. Joshi, "Fast Monte Carlo Greeks for financial products with discontinuous pay-offs", *Mathematical Finance* 23(3), 2013
- `pip install aadc` — [matlogica.com](https://matlogica.com)
