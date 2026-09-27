"""
bench_fries_vs_newton.py — Compare Fries regression vs Newton correction.

Same product (down-and-out barrier, 10 steps), same paths.
Uses switch registry (aadc >= 2.22.1) for automatic indicator discovery.
"""
import sys, os, math, time
import numpy as np
import aadc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from correction_driver_vec2 import CorrectionDriverVec2
from correction_driver_fries import CorrectionDriverFries

# ── Build tape with switch registry ──────────────────────────
S0, K, B = 100.0, 90.0, 80.0
sigma, r0, T = 0.25, 0.05, 1.0
N_STEPS = 10
dt = T / N_STEPS
sqrtDt = math.sqrt(dt)

fn = aadc.Functions()
fn.start_recording(register_switches=aadc.SWITCH_JUMP)
S = aadc.idouble(S0); S_arg = S.mark_as_input()
z_list, z_args = [], []
for j in range(N_STEPS):
    zj = aadc.idouble(0.0); z_args.append(zj.mark_as_input()); z_list.append(zj)

logS = aadc.math.log(S)
drift = aadc.idouble((r0 - 0.5 * sigma**2) * dt)
alive = aadc.idouble(1.0)
for j in range(N_STEPS):
    logS = logS + drift + aadc.idouble(sigma * sqrtDt) * z_list[j]
    S_j = aadc.math.exp(logS)
    ko = aadc.iif(S_j > aadc.idouble(B), aadc.idouble(1.0), aadc.idouble(0.0))
    alive = alive * ko

S_T = aadc.math.exp(logS)
call = aadc.iif(S_T > aadc.idouble(K), S_T - aadc.idouble(K), aadc.idouble(0.0))
payoff = alive * call * aadc.idouble(math.exp(-r0 * T))
payoff_res = payoff.mark_as_output()
fn.stop_recording()

# Indicators from switch registry — no manual g creation needed
g_res = [sw.g for sw in fn.cmp_switches()]
print(f"Switch registry: {len(g_res)} indicators (automatic)")

M = 50000
rng = np.random.RandomState(42)
z_all = rng.randn(M, N_STEPS)

theta_vals = {S_arg: S0}

# ── Newton (vectorized) ────────────────────────────────────────
print("Newton correction (vectorized, 4 threads)...")
drv_newton = CorrectionDriverVec2(fn, payoff_res, g_res, z_args, [S_arg],
                                   skip_sigma=20.0, jump_eps=1e-4, num_threads=4)
drv_newton.precompute_directions(theta_vals)
t0 = time.time()
res_newton = drv_newton.run(z_all)
t_newton = time.time() - t0

# ── Fries (various window sizes) ───────────────────────────────
for wf in [0.05, 0.10, 0.20, 0.30]:
    print(f"\nFries correction (window={wf*100:.0f}%)...")
    drv_fries = CorrectionDriverFries(fn, payoff_res, g_res, z_args, [S_arg],
                                       window_fraction=wf, num_threads=4)
    drv_fries.precompute_directions(theta_vals)
    t0 = time.time()
    res_fries = drv_fries.run(z_all)
    t_fries = time.time() - t0

    print(f"  Delta: {res_fries['total'][0]:+.6f} (Newton: {res_newton['total'][0]:+.6f})")

# ── Bump-and-revalue reference ──────────────────────────────────
def mc_ref(s0_val, n_paths=200000):
    rng2 = np.random.RandomState(42)
    logS_a = np.full(n_paths, math.log(s0_val))
    alive_a = np.ones(n_paths)
    for i in range(N_STEPS):
        z1 = rng2.randn(n_paths)
        logS_a += (r0 - 0.5*sigma**2)*dt + sigma*sqrtDt*z1
        alive_a *= (np.exp(logS_a) > B).astype(float)
    return (alive_a * math.exp(-r0*T) * np.maximum(np.exp(logS_a) - K, 0)).mean()

h = 0.5
p_up = mc_ref(S0 + h)
p_dn = mc_ref(S0 - h)
bump_delta = (p_up - p_dn) / (2 * h)

# ── Summary ─────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"Down-and-Out Barrier, {N_STEPS} steps, {M} paths")
print(f"Bump-and-revalue reference delta: {bump_delta:+.6f}")
print(f"\n{'Method':<36} {'Delta':>10} {'Error':>8} {'Time':>8}")
print("-" * 65)

methods = [("Newton (vec, 4 threads)", res_newton['total'][0], t_newton)]
# Re-run Fries for timing
for wf in [0.05, 0.10, 0.20, 0.30]:
    drv_fries = CorrectionDriverFries(fn, payoff_res, g_res, z_args, [S_arg],
                                       window_fraction=wf, num_threads=4)
    drv_fries.precompute_directions(theta_vals)
    t0 = time.time()
    res_fries = drv_fries.run(z_all)
    t_fries = time.time() - t0
    methods.append((f"Fries (w={wf*100:.0f}%)", res_fries['total'][0], t_fries))

for name, delta, t in methods:
    err = abs(delta - bump_delta) / abs(bump_delta) * 100
    print(f"{name:<36} {delta:+10.6f} {err:7.1f}% {t:7.1f}s")
