"""
bench_slides.py — Reproduce ALL numerical data shown in slides.

Run this to verify slide numbers. Each section prints data for one slide.
Results must match what's in the .tex files.

Slide 41: Newton vs Fries barrier (value±std, bias, 5 seeds, bump 5M ref)
Slide 45: Convergence barrier (value±std, bias row, 5K-200K)
"""
import sys, os, math, time
import numpy as np
import aadc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from correction_driver_vec2 import CorrectionDriverVec2
from correction_driver_fries import CorrectionDriverFries

# ═══ Common: barrier tape ═══
S0, K, B = 100.0, 90.0, 80.0
sigma, r0, T = 0.25, 0.05, 1.0
N_STEPS = 10
dt = T / N_STEPS
sqrtDt = math.sqrt(dt)
N_SEEDS = 5

def build_barrier_tape():
    fn = aadc.Functions()
    fn.start_recording(register_switches=aadc.SWITCH_JUMP)
    S = aadc.idouble(S0); aS = S.mark_as_input()
    zl, za = [], []
    for j in range(N_STEPS):
        zj = aadc.idouble(0.0); za.append(zj.mark_as_input()); zl.append(zj)
    logS = aadc.math.log(S)
    alive = aadc.idouble(1.0)
    g_man = []
    for j in range(N_STEPS):
        logS = logS + (aadc.idouble(r0) - aadc.idouble(0.5) * aadc.idouble(sigma) * aadc.idouble(sigma)) * aadc.idouble(dt) + aadc.idouble(sigma * sqrtDt) * zl[j]
        Sj = aadc.math.exp(logS)
        g = Sj - aadc.idouble(B)
        g_man.append(g.mark_as_output())
        alive = alive * aadc.iif(Sj > aadc.idouble(B), aadc.idouble(1.0), aadc.idouble(0.0))
    ST = aadc.math.exp(logS)
    call = aadc.iif(ST > aadc.idouble(K), ST - aadc.idouble(K), aadc.idouble(0.0))
    payoff = alive * call * aadc.math.exp(-aadc.idouble(r0) * aadc.idouble(T))
    rP = payoff.mark_as_output()
    fn.stop_recording()
    return fn, rP, g_man, za, aS

def bump_ref(M=5000000):
    """Independent bump reference."""
    rng = np.random.RandomState(99)
    h = 0.5
    def mc(s0):
        logS = np.full(M, math.log(s0))
        alive = np.ones(M)
        for i in range(N_STEPS):
            logS += (r0 - 0.5 * sigma**2) * dt + sigma * sqrtDt * rng.randn(M)
            alive *= (np.exp(logS) > B).astype(float)
        return (alive * math.exp(-r0 * T) * np.maximum(np.exp(logS) - K, 0)).mean()
    rng2 = np.random.RandomState(99)
    p_up = mc(S0 + h)
    rng2 = np.random.RandomState(99)  # reset
    # Need same randoms for both — use same seed
    rng = np.random.RandomState(99)
    logS_up = np.full(M, math.log(S0 + h)); alive_up = np.ones(M)
    logS_dn = np.full(M, math.log(S0 - h)); alive_dn = np.ones(M)
    for i in range(N_STEPS):
        z = rng.randn(M)
        logS_up += (r0 - 0.5 * sigma**2) * dt + sigma * sqrtDt * z
        logS_dn += (r0 - 0.5 * sigma**2) * dt + sigma * sqrtDt * z
        alive_up *= (np.exp(logS_up) > B).astype(float)
        alive_dn *= (np.exp(logS_dn) > B).astype(float)
    p_up = (alive_up * math.exp(-r0 * T) * np.maximum(np.exp(logS_up) - K, 0)).mean()
    p_dn = (alive_dn * math.exp(-r0 * T) * np.maximum(np.exp(logS_dn) - K, 0)).mean()
    return (p_up - p_dn) / (2 * h)

print("Computing bump reference (5M paths)...")
ref = bump_ref()
print(f"Bump 5M ref: {ref:+.4f}")

fn, rP, g_man, za, aS = build_barrier_tape()
theta_args = [aS]
theta_vals = {aS: S0}

# ═══════════════════════════════════════════════════════════════
# SLIDE 41: Newton vs Fries (50K paths, 5 seeds)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("SLIDE 41: Newton vs Fries: Benchmark")
print(f"{'='*60}")
print(f"Barrier GBM, 10 steps, 50K paths, {N_SEEDS} seeds. Ref: bump 5M = {ref:+.4f}")

M = 50000
print(f"\n{'Method':>15} {'mean':>8} {'±std':>8} {'bias':>8} {'error':>8}")
print("-" * 52)

for method, w in [("Newton", None), ("Fries 1%", 0.01), ("Fries 5%", 0.05), ("Fries 10%", 0.10)]:
    vals = []
    for seed in range(N_SEEDS):
        z_all = np.random.RandomState(seed).randn(M, N_STEPS)
        if w is None:
            drv = CorrectionDriverVec2(fn, rP, za, theta_args, indicator_res=g_man, num_threads=8)
        else:
            drv = CorrectionDriverFries(fn, rP, g_man, za, theta_args, window_fraction=w, num_threads=8)
        drv.precompute_directions(theta_vals)
        r = drv.run(z_all)
        vals.append(r['total'][0])
    m, s = np.mean(vals), np.std(vals)
    bias = m - ref
    err = abs(bias) / abs(ref) * 100
    print(f"{method:>15} {m:+.4f} {s:.4f} {bias:+.4f} {err:.1f}%")

# ═══════════════════════════════════════════════════════════════
# SLIDE 45: Convergence (value±std, multiple M)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("SLIDE 45: Convergence: Down-and-Out Barrier (dV/dS)")
print(f"{'='*60}")
print(f"Ref: bump 5M = {ref:+.4f}. {N_SEEDS} seeds.")

print(f"\n{'M':>8} {'Newton':>14} {'Fries 1%':>14} {'Fries 5%':>14} {'Fries 10%':>14}")
print("-" * 70)

for M in [5000, 50000, 200000]:
    row = f"{M:>8}"
    for w in [None, 0.01, 0.05, 0.10]:
        vals = []
        for seed in range(N_SEEDS):
            z_all = np.random.RandomState(seed).randn(M, N_STEPS)
            if w is None:
                drv = CorrectionDriverVec2(fn, rP, za, theta_args, indicator_res=g_man, num_threads=8)
            else:
                drv = CorrectionDriverFries(fn, rP, g_man, za, theta_args, window_fraction=w, num_threads=8)
            drv.precompute_directions(theta_vals)
            r = drv.run(z_all)
            vals.append(r['total'][0])
        m, s = np.mean(vals), np.std(vals)
        row += f" {m:.3f}±{s:.3f}"
    print(row)

# Bias row
row = "    bias"
for w in [None, 0.01, 0.05, 0.10]:
    vals = []
    for seed in range(N_SEEDS):
        z_all = np.random.RandomState(seed).randn(200000, N_STEPS)
        if w is None:
            drv = CorrectionDriverVec2(fn, rP, za, theta_args, indicator_res=g_man, num_threads=8)
        else:
            drv = CorrectionDriverFries(fn, rP, g_man, za, theta_args, window_fraction=w, num_threads=8)
        drv.precompute_directions(theta_vals)
        r = drv.run(z_all)
        vals.append(r['total'][0])
    bias = np.mean(vals) - ref
    row += f" {bias:+.4f}       "
print(row)

print("\nDone. Compare with slides v49/v112.")
