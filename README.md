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

- **`correction_driver.py`** — Scalar driver. Simple, correct, reference implementation.
- **`correction_driver_vec.py`** — Vectorized driver using `aadc.VectorFunctionWithAD` (AVX + multithreading). ~4× faster than scalar. Per-indicator single-output VFs for Newton — avoids full jacobian overhead.

## Examples

### Digital option (1 indicator)
```bash
python examples/example_digital.py
```
Analytic delta: 0.01876, correction delta: 0.01845, error: 1.7%.

### Down-and-out barrier (50 indicators, GBM)
```bash
python examples/example_barrier.py
```

### Down-and-out barrier + Hull-White stochastic rate (QuantLib)
```bash
pip install aadc-quantlib-tracing
python examples/example_barrier_hw_ql.py
```
GBM spot + HW 1F rate, correlated (`ρ=-0.3`). 50 monitoring dates, 50K paths.

All Greeks computed: delta, vega, rho, d/d(σ_r).

| Greek | Pathwise | Correction | Total | vs Bump | Agree |
|---|---|---|---|---|---|
| Delta (S₀) | +0.711 | +0.119 | **+0.830** | +0.830 | 0.0% |
| Vega (σ) | +31.56 | -12.17 | **+19.39** | +20.12 | 3.7% |
| Rho (r₀) | +53.75 | +4.85 | **+58.61** | +58.89 | 0.5% |

**Correction is essential for vega**: pathwise alone overestimates by 63%.

### AVX + multithreading benchmark

Down-and-out barrier, 10 steps, 20K paths:

| Driver | Time | Speedup |
|---|---|---|
| Scalar | 12.1s | 1.0x |
| Vectorized (1 thread, AVX) | 2.9s | **4.2x** |
| Vectorized (2 threads) | 2.4s | **5.0x** |
| Vectorized (4 threads) | 1.5s | **7.9x** |

Results identical to machine precision (diff < 1e-14).

```bash
python benchmarks/bench_vec_vs_scalar.py
```

### 2-asset autocallable (Heston, 8 indicators)
```bash
python tests/test_autocallable.py
```

## Tests

```bash
# Vectorized vs scalar consistency
python tests/test_vectorized.py

# Autocallable reference
python tests/test_autocallable.py
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

All steps except final dg/dθ are batch-vectorized via `aadc.VectorFunctionWithAD`.

## References

- Lakshtanov, E. "Unbiased Monte Carlo Greeks for Discontinuous Payoffs" (2026)
- `pip install aadc` — [matlogica.com](https://matlogica.com)
