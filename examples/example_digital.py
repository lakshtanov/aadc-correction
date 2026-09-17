"""
example_digital.py — Correction for a digital (cash-or-nothing) option.

The simplest case: one indicator, one time step.
  P = N * 1{S_T > K}
  Pathwise dP/dS = N * delta(S_T - K) = 0 a.e.
  Correction gives the exact delta.

Reference: digital delta = N * phi(d2) / (S0 * sigma * sqrt(T))
"""
import sys, os

import math
import numpy as np
import aadc
import sys; sys.path.insert(0, ".."); from correction_driver import CorrectionDriver

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

# ── Record kernel (z = Diff) ────────────────────────────────────
fn = aadc.Functions()
fn.start_recording()

S = aadc.idouble(S0)
S_arg = S.mark_as_input()
z = aadc.idouble(0.0)
z_arg = z.mark_as_input()  # z = Diff

# GBM: S_T = S0 * exp((r - 0.5*sigma^2)*T + sigma*sqrt(T)*z)
drift = aadc.idouble((r - 0.5*sigma**2)*T)
diffusion = aadc.idouble(sigma * math.sqrt(T)) * z
S_T = S * np.exp(drift + diffusion)

# Indicator g = S_T - K
g = S_T - aadc.idouble(K)
g_res = g.mark_as_output()

# Payoff = N * 1{S_T > K} * df
df = math.exp(-r * T)
indicator = aadc.iif(S_T > aadc.idouble(K), aadc.idouble(1.0), aadc.idouble(0.0))
payoff = aadc.idouble(NOTIONAL * df) * indicator
payoff_res = payoff.mark_as_output()

fn.stop_recording()
print("Kernel recorded: 1 z-input, 1 indicator")

# ── Correction driver ──────────────────────────────────────────
driver = CorrectionDriver(fn, payoff_res, [g_res], [z_arg], [S_arg],
                           skip_sigma=20.0, jump_eps=1e-4)
driver.precompute_directions({S_arg: S0})
print(f"Direction norm: {driver.directions[0][1]:.6f}")

# ── MC loop ────────────────────────────────────────────────────
M = 50000
rng = np.random.RandomState(42)
ws = driver.ws

sum_price = 0.0
sum_pathwise = 0.0
sum_correction = 0.0

for m in range(M):
    zv = [rng.randn()]
    ws.set_val(z_arg, zv[0])
    ws.set_val(S_arg, S0)
    ws.forward()

    price = ws.val(payoff_res)
    sum_price += price

    # Pathwise delta (will be 0 for digital)
    ws.reset_diff()
    ws.set_diff(payoff_res, 1.0)
    ws.reverse()
    sum_pathwise += ws.diff(S_arg)

    # Correction
    # Need to re-forward since reverse may have changed workspace state
    ws.set_val(z_arg, zv[0])
    ws.set_val(S_arg, S0)
    ws.forward()

    cr = driver.compute_correction(zv)
    sum_correction += cr.correction[0]

pathwise = sum_pathwise / M
correction = sum_correction / M
total = pathwise + correction
mc_price = sum_price / M

print(f"\nResults ({M} paths):")
print(f"  MC price:       {mc_price:.6f}  (analytic: {digital_price:.6f})")
print(f"  Pathwise delta: {pathwise:.6f}  (expected: ~0)")
print(f"  Correction:     {correction:.6f}")
print(f"  Total delta:    {total:.6f}  (analytic: {digital_delta:.6f})")
print(f"  Error:          {abs(total - digital_delta):.6f} ({100*abs(total-digital_delta)/abs(digital_delta):.1f}%)")
