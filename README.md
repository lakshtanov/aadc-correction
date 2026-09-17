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

Standard pathwise (AAD) differentiation gives **zero** or **biased** Greeks at discontinuities:

| Method | Price bias | Delta error | Needs tuning? |
|---|---|---|---|
| Pathwise AAD (no fix) | 0% | **−15%** | No |
| Smoothing ε=0.5 | +0.2% | −0.5% | Yes (ε) |
| Smoothing ε=5.0 | +1.4% | −2.6% | Yes (ε) |
| Bump-and-revalue | 0% | 0% | Yes (h) |
| **δ-correction** | **0%** | **< 1%** | **No** |

## The solution

The correction adds the missing δ-function contribution at each indicator surface `g(z) = 0`:

```
corrected_Greek = pathwise_Greek + Σ_i  Δf(z*_i) · φ(u*_i) / |∂g_i/∂u| · ∂g_i/∂θ
```

where `z*_i` is the boundary point found by Newton's method along a precomputed direction.

**Two-kernel pattern:**
- **Kernel A** (`z = NoDiff`): standard pathwise Greeks
- **Kernel B** (`z = Diff`): Newton root-finding → δ-correction

## Quick start

```python
import aadc
import numpy as np
from correction_driver import CorrectionDriver

# Record kernel with indicator g = S_T - K
fn = aadc.Functions()
fn.start_recording()
S = aadc.idouble(100.0); S_arg = S.mark_as_input()
z = aadc.idouble(0.0);   z_arg = z.mark_as_input()
S_T = S * np.exp(aadc.idouble(-0.02 * 1.0 + 0.2 * 1.0) * z)  # simplified GBM
g = S_T - aadc.idouble(100.0)
g_res = g.mark_as_output()
payoff = aadc.iif(S_T > aadc.idouble(100.0), aadc.idouble(1.0), aadc.idouble(0.0))
payoff_res = payoff.mark_as_output()
fn.stop_recording()

# Setup
driver = CorrectionDriver(fn, payoff_res, [g_res], [z_arg], [S_arg])
driver.precompute_directions({S_arg: 100.0})

# MC loop
ws = driver.ws
for _ in range(10000):
    zv = [np.random.randn()]
    ws.set_val(z_arg, zv[0]); ws.set_val(S_arg, 100.0)
    ws.forward()
    cr = driver.compute_correction(zv)
    # cr.correction[0] is the delta correction for this path
```

## Examples

| File | Product | Model | Key result |
|---|---|---|---|
| `examples/example_digital.py` | Digital call | GBM | Delta error 1.7% vs analytic |
| `examples/example_barrier.py` | Down-and-out call (50 steps) | GBM | Pathwise=0.71, corrected=0.83 |
| `tests/test_autocallable.py` | 2-asset Phoenix (ND=4) | GBM, ρ=0.5 | dS1, dS2 within 5% of C++ ref |

## Benchmarks

Results from [Goloubentsev, Lakshtanov, Piterbarg (2024)](http://dev.matlogica.com:8888/aad_v81.pdf).

### Accuracy: correction vs smoothing vs bump (Down-and-Out Call, 500K paths)

| Method | Price | Price bias | Delta | Delta error |
|---|---:|---:|---:|---:|
| Pathwise AAD | 17.444 | 0% | 0.7116 | **−15.0%** |
| **Pathwise + δ-correction** | **17.444** | **0%** | **0.8368** | **−0.09%** |
| Smoothing ε=0.5 | 17.48 | +0.21% | 0.8333 | −0.51% |
| Smoothing ε=1.0 | 17.51 | +0.40% | 0.8303 | −0.86% |
| Smoothing ε=5.0 | 17.68 | +1.37% | 0.8158 | −2.59% |
| Bump-and-revalue | 17.444 | 0% | 0.8375 | 0% |

### Variance reduction: paths needed for 2% accuracy

| Product | Greek | Correction paths | Bump paths | **Speedup** |
|---|---|---:|---:|---:|
| 2-asset autocall ND=4 | delta | 39K | 70M | **1,790×** |
| 2-asset autocall ND=4 | vega | 123K | 8.3B | **67,000×** |
| Down-and-out call | delta | 8K | 12.7M | **1,674×** |
| Down-and-out call | vega | 111K | 6.7B | **60,000×** |
| Digital call | delta | 1K | 40M | **47,000×** |
| Digital call | vega | 72K | 7.5B | **104,000×** |
| Up-and-out put | delta | 47K | 37M | **782×** |

### Standard error comparison (500K paths)

| Product | Greek | Correction SE | Bump SE | **SE ratio** |
|---|---|---:|---:|---:|
| Down-and-out call | delta | 0.0013 | 0.043 | Correction **34× tighter** |
| Down-and-out call | vega | 0.15 | 21.5 | Correction **143× tighter** |
| Digital call | delta | 0.00001 | 0.0021 | Correction **210× tighter** |

### Products validated

| Product | Model | Assets | Obs dates | Indicators | Delta error |
|---|---|---:|---:|---:|---:|
| Down-and-Out Call | GBM | 1 | 50 | 50 | −0.36% |
| Digital Cash-or-Nothing | GBM | 1 | 1 | 1 | −3.2% |
| Double Barrier KO | GBM | 1 | 50 | 100 | +4.3% |
| Up-and-Out Put | GBM | 1 | 50 | 50 | +1.04% |
| Down-and-In Call | GBM | 1 | 50 | 50 | −4.83% |
| Two-Asset Barrier | GBM | 2 | 50 | 50 | −1.3% |
| Phoenix Autocallable ND=4 | GBM | 2 | 4 | 16 | −0.7% |
| Phoenix Autocallable ND=8 | GBM | 2 | 8 | 32 | −4.0% |
| 3-Asset Phoenix ND=8 | Heston SV | 3 | 8 | 45 | < 2.3σ |

### Cost per path

Per unique indicator: 6–12 kernel replays (screening + Newton + jump).

For 3+ asset products, **deduplication** reduces cost by ~85% (same bottleneck asset across dates → identical boundary surfaces).

## Algorithm

For each MC path:

1. **Screen** — skip indicators far from boundary: `|g(z)| / ||∇g|| > σ`
2. **Newton** — find `z*` on the boundary `g(z*) = 0` along precomputed direction `v_i`
3. **Jump** — central difference: `Δf = P(z* + εv) - P(z* - εv)`
4. **Accumulate** — `correction_k += Δf · φ(u*) / |v·∇g(z*)| · ∂g/∂θ_k`

Precomputed directions `v_i = ∇g_i / ||∇g_i||` evaluated once at `z = 0`.

## API

```python
class CorrectionDriver:
    def __init__(self, funcs, payoff_res, indicator_res, z_args, theta_args,
                 skip_sigma=20.0, jump_eps=1e-4, max_newton=4)

    def precompute_directions(self, theta_values=None)
        # Call once before MC loop

    def compute_correction(self, z_values) -> CorrectionResult
        # Call per MC path after ws.forward()

class CorrectionResult:
    correction: list[float]    # per theta parameter
    num_active: int            # indicators processed
    num_newton_iters: int
    num_forwards: int
    num_reverses: int
```

## References

- D. Goloubentsev, E. Lakshtanov, V. Piterbarg. *Unbiased Monte Carlo Greeks for Discontinuous Payoffs via Delta-Function Correction.* [PDF](http://dev.matlogica.com:8888/aad_v81.pdf)
- QuantMinds presentation. [Slides](http://dev.matlogica.com:8888/qm_correction_v31.pdf)
- AADC: [matlogica.com](https://matlogica.com), `pip install aadc`

## License

MIT
