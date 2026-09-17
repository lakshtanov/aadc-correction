"""
example_barrier.py — Correction for a down-and-out call option.

S0=100, K=90, Barrier=80, sigma=0.25, r=0.05, T=1, 50 monitoring steps.

Reference: Reiner-Rubinstein analytic formula.
Demonstrates correction with many indicators (50 barrier checks).
"""
import sys, os

import math
import numpy as np
import aadc
import sys; sys.path.insert(0, ".."); from correction_driver import CorrectionDriver

# ── Parameters ──────────────────────────────────────────────────
S0, K, B = 100.0, 90.0, 80.0
sigma, r, T = 0.25, 0.05, 1.0
N_STEPS = 50
dt = T / N_STEPS
sqrtDt = math.sqrt(dt)

# ── Record kernel (z = Diff) ────────────────────────────────────
fn = aadc.Functions()
fn.start_recording()

S = aadc.idouble(S0)
S_arg = S.mark_as_input()

z_list = []
z_args = []
for j in range(N_STEPS):
    zj = aadc.idouble(0.0)
    z_args.append(zj.mark_as_input())
    z_list.append(zj)

g_res = []  # barrier indicators
alive = aadc.idouble(1.0)

logS = np.log(S)
drift = aadc.idouble((r - 0.5 * sigma**2) * dt)

for j in range(N_STEPS):
    logS = logS + drift + aadc.idouble(sigma * sqrtDt) * z_list[j]
    S_j = np.exp(logS)

    # Indicator: g = S_j - B (positive = alive)
    g = S_j - aadc.idouble(B)
    g_res.append(g.mark_as_output())

    ko = aadc.iif(S_j > aadc.idouble(B), aadc.idouble(1.0), aadc.idouble(0.0))
    alive = alive * ko

# Payoff = alive * max(S_T - K, 0) * df
S_T = np.exp(logS)
call_payoff = aadc.iif(S_T > aadc.idouble(K),
                        S_T - aadc.idouble(K), aadc.idouble(0.0))
df = math.exp(-r * T)
payoff = alive * call_payoff * aadc.idouble(df)
payoff_res = payoff.mark_as_output()

fn.stop_recording()
print(f"Kernel recorded: {N_STEPS} z-inputs, {len(g_res)} barrier indicators")

# ── Correction driver ──────────────────────────────────────────
driver = CorrectionDriver(fn, payoff_res, g_res, z_args, [S_arg],
                           skip_sigma=20.0, jump_eps=1e-4)
driver.precompute_directions({S_arg: S0})
print(f"Directions computed: {sum(1 for _, n in driver.directions if n > 1e-10)} active")

# ── MC loop ────────────────────────────────────────────────────
M = 20000
rng = np.random.RandomState(42)
ws = driver.ws

sum_price = 0.0
sum_pathwise = 0.0
sum_correction = 0.0

for m in range(M):
    zv = rng.randn(N_STEPS)
    for j in range(N_STEPS):
        ws.set_val(z_args[j], float(zv[j]))
    ws.set_val(S_arg, S0)

    ws.forward()
    sum_price += ws.val(payoff_res)

    # Pathwise
    ws.reset_diff()
    ws.set_diff(payoff_res, 1.0)
    ws.reverse()
    sum_pathwise += ws.diff(S_arg)

    # Re-forward for correction
    for j in range(N_STEPS):
        ws.set_val(z_args[j], float(zv[j]))
    ws.set_val(S_arg, S0)
    ws.forward()

    cr = driver.compute_correction(zv)
    sum_correction += cr.correction[0]

    if (m + 1) % 5000 == 0:
        print(f"  path {m+1}/{M}: price={sum_price/(m+1):.4f}, "
              f"delta={sum_pathwise/(m+1) + sum_correction/(m+1):+.4f}")

price = sum_price / M
pathwise = sum_pathwise / M
correction = sum_correction / M
total = pathwise + correction

print(f"\nResults ({M} paths, {N_STEPS} steps):")
print(f"  Price:      {price:.4f}")
print(f"  Pathwise:   {pathwise:+.6f}")
print(f"  Correction: {correction:+.6f}")
print(f"  Total delta:{total:+.6f}")
