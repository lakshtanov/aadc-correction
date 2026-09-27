"""
example_digital.py — Correction for a digital (cash-or-nothing) option.

Uses switch registry (aadc >= 2.22.1) for automatic indicator discovery.

  P = N * 1{S_T > K}
  Pathwise dP/dS = N * delta(S_T - K) = 0 a.e.
  Correction gives the exact delta.

Reference: digital delta = N * phi(d2) / (S0 * sigma * sqrt(T))
"""
import math
import numpy as np
import aadc
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); from correction_driver_vec2 import CorrectionDriverVec2

# ── Parameters ──────────────────────────────────────────────────
S0, K, sigma, r, T = 100.0, 100.0, 0.2, 0.05, 1.0
NOTIONAL = 1.0

# ── Analytic reference ──────────────────────────────────────────
from scipy.stats import norm
d1 = (math.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
d2 = d1 - sigma*math.sqrt(T)
digital_price = math.exp(-r*T) * norm.cdf(d2)
digital_delta = math.exp(-r*T) * norm.pdf(d2) / (S0 * sigma * math.sqrt(T))
print(f"Analytic: price={digital_price:.6f}, delta={digital_delta:.6f}")

# ── Record with switch registry ─────────────────────────────────
fn = aadc.Functions()
fn.start_recording(register_switches=aadc.SWITCH_JUMP)

S = aadc.idouble(S0); S_arg = S.mark_as_input()
z = aadc.idouble(0.0); z_arg = z.mark_as_input()

drift = aadc.idouble((r - 0.5*sigma**2)*T)
diffusion = aadc.idouble(sigma * math.sqrt(T)) * z
S_T = S * aadc.math.exp(drift + diffusion)

df = math.exp(-r * T)
indicator = aadc.iif(S_T > aadc.idouble(K), aadc.idouble(1.0), aadc.idouble(0.0))
payoff = aadc.idouble(NOTIONAL * df) * indicator
payoff_res = payoff.mark_as_output()

fn.stop_recording()

# Indicators from switch registry
g_res = [sw.g for sw in fn.cmp_switches()]
print(f"Switch registry: {len(g_res)} indicators")

# ── Correction driver ──────────────────────────────────────────
M = 50000
rng = np.random.RandomState(42)
z_all = rng.randn(M, 1)

driver = CorrectionDriverVec2(fn, payoff_res, g_res, [z_arg], [S_arg],
                               skip_sigma=20.0, jump_eps=1e-4, num_threads=4)
driver.precompute_directions({S_arg: S0})

result = driver.run(z_all)

pathwise = result['pathwise'][0]
correction = result['correction'][0]
total = result['total'][0]

print(f"\nResults ({M} paths):")
print(f"  MC price:       {result['price']:.6f}  (analytic: {digital_price:.6f})")
print(f"  Pathwise delta: {pathwise:.6f}  (expected: ~0)")
print(f"  Correction:     {correction:.6f}")
print(f"  Total delta:    {total:.6f}  (analytic: {digital_delta:.6f})")
print(f"  Error:          {abs(total - digital_delta):.6f} ({100*abs(total-digital_delta)/abs(digital_delta):.1f}%)")
