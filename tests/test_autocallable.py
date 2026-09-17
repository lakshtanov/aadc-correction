"""
test_correction_py.py — Test CorrectionDriver on 2-asset GBM autocallable.

Same product as correction_ng.cpp:
  S0=100, vol1=0.2, vol2=0.25, corr=0.5, r=0.03, T=1, ND=4
  Coupon barrier 70%, autocall barrier 100%

Reference (C++ correction_ng): dS1=-0.209, dS2=-0.134
"""

import sys, os
import math
import numpy as np

import aadc
import sys; sys.path.insert(0, ".."); from correction_driver import CorrectionDriver

# ── Parameters ──────────────────────────────────────────────────
ND = 4
S0_1, S0_2 = 100.0, 100.0
VOL1, VOL2, CORR, R, T_MAT = 0.2, 0.25, 0.5, 0.03, 1.0
COUPON_BARRIER, AUTOCALL_BARRIER = 0.70, 1.00
COUPON_RATE, NOTIONAL = 0.05, 100.0

nz = 2 * ND
dt = T_MAT / ND
sqrtDt = math.sqrt(dt)
rho2 = math.sqrt(1.0 - CORR * CORR)

# ── Record Kernel B (z = Diff) ──────────────────────────────────
fn = aadc.Functions()
fn.start_recording()

S1 = aadc.idouble(S0_1)
S2 = aadc.idouble(S0_2)
S1_arg = S1.mark_as_input()
S2_arg = S2.mark_as_input()

z_list = []
z_args = []
for j in range(nz):
    zj = aadc.idouble(0.0)
    z_args.append(zj.mark_as_input())  # z = Diff
    z_list.append(zj)

logS1 = np.log(S1)
logS2 = np.log(S2)
drift1 = aadc.idouble((R - 0.5 * VOL1 * VOL1) * dt)
drift2 = aadc.idouble((R - 0.5 * VOL2 * VOL2) * dt)

payoff = aadc.idouble(0.0)
alive = aadc.idouble(1.0)

g_res = []
alive_res = []

for d in range(ND):
    z1 = z_list[2 * d]
    z2 = aadc.idouble(CORR) * z_list[2 * d] + aadc.idouble(rho2) * z_list[2 * d + 1]
    logS1 = logS1 + drift1 + aadc.idouble(VOL1 * sqrtDt) * z1
    logS2 = logS2 + drift2 + aadc.idouble(VOL2 * sqrtDt) * z2

    price1 = np.exp(logS1)
    price2 = np.exp(logS2)
    perf1 = price1 / aadc.idouble(S0_1)
    perf2 = price2 / aadc.idouble(S0_2)
    worst = aadc.iif(perf1 < perf2, perf1, perf2)

    alive_res.append(alive.mark_as_output())
    g_res.append((worst - aadc.idouble(COUPON_BARRIER)).mark_as_output())
    g_res.append((worst - aadc.idouble(AUTOCALL_BARRIER)).mark_as_output())

    df = math.exp(-R * T_MAT * (d + 1.0) / ND)
    coupon_ind = aadc.iif(worst > aadc.idouble(COUPON_BARRIER),
                           aadc.idouble(1.0), aadc.idouble(0.0))
    payoff = payoff + alive * coupon_ind * aadc.idouble(COUPON_RATE * NOTIONAL * df)

    if d < ND - 1:
        autocall_ind = aadc.iif(worst > aadc.idouble(AUTOCALL_BARRIER),
                                 aadc.idouble(1.0), aadc.idouble(0.0))
        payoff = payoff + alive * autocall_ind * aadc.idouble(NOTIONAL * df)
        alive = alive * (aadc.idouble(1.0) - autocall_ind)

payoff = payoff + alive * aadc.idouble(NOTIONAL * math.exp(-R * T_MAT))
payoff_res = payoff.mark_as_output()

fn.stop_recording()
print(f"Kernel B recorded: {nz} z-inputs, {len(g_res)} indicators")

# ── Setup correction driver ─────────────────────────────────────
driver = CorrectionDriver(fn, payoff_res, g_res, z_args,
                           [S1_arg, S2_arg],
                           skip_sigma=20.0, jump_eps=1e-4)

# Set theta values for direction precomputation
driver.precompute_directions({S1_arg: S0_1, S2_arg: S0_2})
print(f"Directions: {len(driver.directions)} indicators")
for i, (entries, norm) in enumerate(driver.directions):
    print(f"  dir[{i}]: norm={norm:.4f}, nnz={len(entries)}")

# ── MC loop ─────────────────────────────────────────────────────
M = 10000
rng = np.random.RandomState(42)

sum_corr = np.zeros(2)
sum_price = 0.0
total_active = 0

ws = driver.ws

for m in range(M):
    zv = rng.randn(nz)

    # Set z and theta on workspace
    for j in range(nz):
        ws.set_val(z_args[j], float(zv[j]))
    ws.set_val(S1_arg, S0_1)
    ws.set_val(S2_arg, S0_2)

    # Forward
    ws.forward()
    price = ws.val(payoff_res)
    sum_price += price

    # Correction
    cr = driver.compute_correction(zv)
    for k in range(2):
        sum_corr[k] += cr.correction[k]
    total_active += cr.num_active

    if (m + 1) % 2000 == 0:
        print(f"  path {m+1}/{M}: price={sum_price/(m+1):.2f}, "
              f"dS1={sum_corr[0]/(m+1):+.4f}, dS2={sum_corr[1]/(m+1):+.4f}")

# ── Results ─────────────────────────────────────────────────────
print(f"\nResults ({M} paths, ND={ND}, seed=42)")
print(f"Price: {sum_price/M:.4f}")
print(f"  dS1 correction: {sum_corr[0]/M:+.6f}  (C++ ref: -0.209)")
print(f"  dS2 correction: {sum_corr[1]/M:+.6f}  (C++ ref: -0.134)")
print(f"  Active/path: {total_active/M:.1f} / {len(g_res)}")

# Check
ok = True
if abs(sum_corr[0]/M - (-0.209)) > 0.05:
    print("  WARNING: dS1 too far from reference")
    ok = False
if abs(sum_corr[1]/M - (-0.134)) > 0.05:
    print("  WARNING: dS2 too far from reference")
    ok = False
print(f"\n{'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
