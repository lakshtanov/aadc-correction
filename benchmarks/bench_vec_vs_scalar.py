"""
bench_vec_vs_scalar.py — Benchmark vectorized vs scalar correction driver.

Measures wall-clock time for both drivers on the same problem.
Reports speedup from AVX + multithreading.
"""
import sys, os, math, time
import numpy as np
import aadc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from correction_driver import CorrectionDriver
from correction_driver_vec import CorrectionDriverVec

# ── Build tape: down-and-out barrier, 10 steps ─────────────────
S0, K, B = 100.0, 90.0, 80.0
sigma, r0, T = 0.25, 0.05, 1.0
N_STEPS = 10
dt = T / N_STEPS
sqrtDt = math.sqrt(dt)

fn = aadc.Functions()
fn.start_recording()
S = aadc.idouble(S0); S_arg = S.mark_as_input()
z_list, z_args = [], []
for j in range(N_STEPS):
    zj = aadc.idouble(0.0); z_args.append(zj.mark_as_input()); z_list.append(zj)

logS = np.log(S)
drift = aadc.idouble((r0 - 0.5 * sigma**2) * dt)
alive = aadc.idouble(1.0)
g_res = []

for j in range(N_STEPS):
    logS = logS + drift + aadc.idouble(sigma * sqrtDt) * z_list[j]
    S_j = np.exp(logS)
    g = S_j - aadc.idouble(B)
    g_res.append(g.mark_as_output())
    ko = aadc.iif(S_j > aadc.idouble(B), aadc.idouble(1.0), aadc.idouble(0.0))
    alive = alive * ko

S_T = np.exp(logS)
call = aadc.iif(S_T > aadc.idouble(K), S_T - aadc.idouble(K), aadc.idouble(0.0))
payoff = alive * call * aadc.idouble(math.exp(-r0 * T))
payoff_res = payoff.mark_as_output()
fn.stop_recording()

print(f"Tape: {N_STEPS} steps, {len(z_args)} z-inputs, {len(g_res)} indicators")

# ── Generate paths ──────────────────────────────────────────────
M = 20000
rng = np.random.RandomState(42)
z_all = rng.randn(M, N_STEPS)

# ── Scalar driver ───────────────────────────────────────────────
print(f"\nScalar driver ({M} paths)...")
drv_s = CorrectionDriver(fn, payoff_res, g_res, z_args, [S_arg],
                          skip_sigma=20.0, jump_eps=1e-4)
drv_s.precompute_directions({S_arg: S0})

ws = drv_s.ws
t0 = time.time()
sum_price_s = 0.0
sum_pw_s = 0.0
sum_corr_s = 0.0
for m in range(M):
    for j in range(N_STEPS):
        ws.set_val(z_args[j], float(z_all[m, j]))
    ws.set_val(S_arg, S0)
    ws.forward()
    sum_price_s += ws.val(payoff_res)

    ws.reset_diff()
    ws.set_diff(payoff_res, 1.0)
    ws.reverse()
    sum_pw_s += ws.diff(S_arg)

    # Re-forward for correction
    for j in range(N_STEPS):
        ws.set_val(z_args[j], float(z_all[m, j]))
    ws.set_val(S_arg, S0)
    ws.forward()
    cr = drv_s.compute_correction(z_all[m])
    sum_corr_s += cr.correction[0]

t_scalar = time.time() - t0
price_s = sum_price_s / M
pw_s = sum_pw_s / M
corr_s = sum_corr_s / M
print(f"  Time: {t_scalar:.1f}s")
print(f"  Price: {price_s:.4f}, pathwise: {pw_s:+.6f}, correction: {corr_s:+.6f}, total: {pw_s+corr_s:+.6f}")

# ── Vectorized driver (1 thread) ────────────────────────────────
print(f"\nVectorized driver, 1 thread ({M} paths)...")
drv_v1 = CorrectionDriverVec(fn, payoff_res, g_res, z_args, [S_arg],
                               skip_sigma=20.0, jump_eps=1e-4, num_threads=1)
drv_v1.precompute_directions({S_arg: S0})
t0 = time.time()
res_v1 = drv_v1.run(z_all)
t_vec1 = time.time() - t0
print(f"  Time: {t_vec1:.1f}s")
print(f"  Price: {res_v1['price']:.4f}, pathwise: {res_v1['pathwise'][0]:+.6f}, "
      f"correction: {res_v1['correction'][0]:+.6f}, total: {res_v1['total'][0]:+.6f}")

# ── Vectorized driver (2 threads) ───────────────────────────────
print(f"\nVectorized driver, 2 threads ({M} paths)...")
drv_v2 = CorrectionDriverVec(fn, payoff_res, g_res, z_args, [S_arg],
                               skip_sigma=20.0, jump_eps=1e-4, num_threads=2)
drv_v2.precompute_directions({S_arg: S0})
t0 = time.time()
res_v2 = drv_v2.run(z_all)
t_vec2 = time.time() - t0
print(f"  Time: {t_vec2:.1f}s")

# ── Vectorized driver (4 threads) ───────────────────────────────
print(f"\nVectorized driver, 4 threads ({M} paths)...")
drv_v4 = CorrectionDriverVec(fn, payoff_res, g_res, z_args, [S_arg],
                               skip_sigma=20.0, jump_eps=1e-4, num_threads=4)
drv_v4.precompute_directions({S_arg: S0})
t0 = time.time()
res_v4 = drv_v4.run(z_all)
t_vec4 = time.time() - t0
print(f"  Time: {t_vec4:.1f}s")

# ── Summary ─────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Down-and-Out Barrier, {N_STEPS} steps, {M} paths")
print(f"")
print(f"{'Driver':<25} {'Time':>8} {'Speedup':>8} {'Delta':>10}")
print(f"{'-'*55}")
print(f"{'Scalar':<25} {t_scalar:>7.1f}s {'1.0x':>8} {pw_s+corr_s:>+10.6f}")
print(f"{'Vectorized (1 thread)':<25} {t_vec1:>7.1f}s {t_scalar/t_vec1:>7.1f}x {res_v1['total'][0]:>+10.6f}")
print(f"{'Vectorized (2 threads)':<25} {t_vec2:>7.1f}s {t_scalar/t_vec2:>7.1f}x {res_v2['total'][0]:>+10.6f}")
print(f"{'Vectorized (4 threads)':<25} {t_vec4:>7.1f}s {t_scalar/t_vec4:>7.1f}x {res_v4['total'][0]:>+10.6f}")
print(f"")

# Verify results match
diff = abs((pw_s + corr_s) - res_v4['total'][0])
print(f"Scalar vs Vec(4) delta difference: {diff:.2e}")
print(f"{'PASS' if diff < 0.001 else 'FAIL'}")
