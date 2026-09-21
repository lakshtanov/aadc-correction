# aadc-correction

Unbiased Monte Carlo Greeks for discontinuous payoffs (barriers, digitals, autocallables) via **δ-function correction**.

No smoothing. No parameter tuning. Works with any model.

## Installation

```bash
pip install aadc
```

Then clone this repo:
```bash
git clone https://github.com/lakshtanov/aadc-correction.git
cd aadc-correction
python examples/example_digital.py
```

## The problem

Pathwise (AAD) Greeks fail at discontinuities — `δ(S - B)` cannot be sampled by MC. Standard workaround is smoothing (`Φ((S-B)/ε)`), which introduces bias and requires per-product tuning.

## The solution

The correction driver finds the boundary in normal space via Newton root-finding, evaluates the density and payoff jump there, and adds an exact correction term. No smoothing, no `ε`.

**Formula**: `correction = Σ_i jump_i · φ(u*_i) / |v·∇g_i| · (∂g_i/∂θ)`

## Drivers

| Driver | Method | Bias | Speed |
|---|---|---|---|
| `correction_driver.py` | Scalar Newton | None | 1× (reference) |
| `correction_driver_vec.py` | Vectorized Newton (AVX + multithreading) | None | **7.9×** |
| `correction_driver_fries.py` | Fries (2018) discretized delta | O(w) | **4-6×** faster than Newton |

### Newton drivers (exact, unbiased)

Newton root-finding in normal space → exact boundary z*, exact density φ(u*). No tuning parameter.

Vectorized driver uses `aadc.VectorFunctionWithAD` with per-indicator single-output VFs for Newton — avoids computing full 51-output jacobian.

### Fries driver (fast, biased)

Implements Fries (2018, arXiv:1811.05741) discretized delta: `δ(X) ≈ 1{|X|<w}/(2w)`.
This is mathematically equivalent to smoothing with a rectangular kernel — same bias-variance trade-off as classic `Φ(X/ε)`.

Supports **adaptive window selection**: starts with w=2%,5%,10%,20%, then halves downward (1%→0.5%→0.2%→...) until unstable. Picks smallest stable w = optimal bias-variance trade-off.

```python
# Fixed window
driver = CorrectionDriverFries(..., window_fraction=0.05)

# Adaptive (recommended)
driver = CorrectionDriverFries(..., adaptive=True, adaptive_tol=0.10)
```

## Full benchmark results

50K paths, 500K bump-and-revalue reference. AVX + 4 threads.

### Barrier GBM (10 steps)

| Greek | Bump ref | Newton | N err | Fries 5% | F err | Correction % |
|---|---|---|---|---|---|---|
| Delta | +0.816 | +0.809 | 0.9% | +0.821 | 0.6% | 9% |
| Vega | +23.0 | +23.3 | 1.4% | +22.6 | 2.0% | **34%** |

### Barrier + Hull-White 1F (10 steps, 4 Greeks)

| Greek | Bump ref | Newton | N err | Fries 5% | F err | Correction % |
|---|---|---|---|---|---|---|
| Delta | +0.815 | +0.808 | 0.9% | +0.819 | 0.4% | 10% |
| Vega | +23.4 | +23.0 | 1.7% | +22.3 | 4.9% | **35%** |
| Rho | +59.1 | +58.7 | 0.7% | +59.0 | 0.2% | 6% |
| d/dσ_r | -3.5 | -3.7 | 7.2% | -3.7 | 7.6% | 7% |

### 2-asset autocallable (4 dates)

| Greek | Bump ref | Newton | N err | Fries 5% | F err | Correction % |
|---|---|---|---|---|---|---|
| dV/dS1 | -0.203 | -0.204 | 0.4% | -0.202 | 0.5% | **100%** |
| dV/dS2 | -0.132 | -0.137 | 4.1% | -0.125 | 4.7% | **100%** |

### Speed comparison

| Product | Scalar | Newton (AVX,4t) | Fries 5% |
|---|---|---|---|
| Barrier GBM | 12.1s | 1.5s (7.9×) | 1.5s |
| Barrier+HW | — | 14.9s | 2.3s (6.4×) |
| Autocallable | — | 1.4s | 0.4s (3.7×) |

### Key findings

- **Vega correction = 34%** of total — pathwise alone overestimates by one third
- **Autocallable: 100% correction** — pathwise = 0, entire Greek is the correction
- **Fries 4-6× faster** than Newton, comparable accuracy at optimal w
- **Newton unbiased** — error is MC noise only, converges to zero with more paths
- **AVX + 4 threads**: 7.9× speedup over scalar driver

## Examples

```bash
# Digital option
python examples/example_digital.py

# Down-and-out barrier (GBM)
python examples/example_barrier.py

# Barrier + Hull-White stochastic rate (QuantLib)
pip install aadc-quantlib-tracing
python examples/example_barrier_hw_ql.py
```

## Tests

```bash
# Vectorized vs scalar consistency (machine precision)
python tests/test_vectorized.py

# Autocallable reference
python tests/test_autocallable.py
```

## Benchmarks

```bash
# AVX + multithreading speedup
python benchmarks/bench_vec_vs_scalar.py

# Newton vs Fries comparison
python benchmarks/bench_fries_vs_newton.py

# Full comparison: all products, all Greeks
python benchmarks/bench_full_comparison.py
```

## API

```python
from correction_driver_vec import CorrectionDriverVec

# Record kernel with aadc
fn = aadc.Functions()
fn.start_recording()
# ... build payoff with aadc.iif ...
fn.stop_recording()

# Setup driver
driver = CorrectionDriverVec(fn, payoff_res, g_res, z_args, theta_args,
                              num_threads=4)
driver.precompute_directions({S_arg: S0, vol_arg: sigma})

# Run
z_all = np.random.randn(M, d)
result = driver.run(z_all)
# result['price'], result['pathwise'], result['correction'], result['total']
```

## How it works

1. **Record** the pricing kernel on AADC tape with `z` (normals) and `θ` (parameters) as inputs
2. **Precompute** gradient directions `v_i = ∇_z g_i / |∇_z g_i|` at z=0
3. **Screen** each indicator on each path: is the path near the boundary?
4. **Newton** root-finding along direction `v_i` to find `u*` where `g_i(z*) = 0`
5. **Jump** evaluation: `P(z* + εv) - P(z* - εv)` via central difference
6. **Accumulate**: `correction_θ += jump · φ(u*) / |v·∇g| · (∂g/∂θ)`

All steps are batch-vectorized via `aadc.VectorFunctionWithAD`.

## References

- Lakshtanov, E. "Unbiased Monte Carlo Greeks for Discontinuous Payoffs" (2026)
- Joshi, M. and D. Kainth, "Rapid and accurate development of prices and Greeks for nth to default credit swaps in the Li model", *Quantitative Finance* 4(3):266-275, 2004
- Chan, J.H. and M.S. Joshi, "Fast Monte Carlo Greeks for financial products with discontinuous pay-offs", *Mathematical Finance* 23(3):459-495, 2013
- Capriotti, L., S. Lee, and M. Peacock, "Real time counterparty credit risk management in Monte Carlo", *Risk*, June 2011
- Fries, C.P., "Stochastic algorithmic differentiation of (expectations of) discontinuous functions (indicator functions)", arXiv:1811.05741, 2018
- `pip install aadc` — [matlogica.com](https://matlogica.com)
