"""
example_barrier.py — Down-and-out barrier call (GBM).

Uses switch registry (aadc >= 2.22.1) for automatic indicator discovery.
No manual indicator creation needed — model code unchanged.
"""
import sys, os, math, time
import numpy as np
import aadc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from correction_driver_vec2 import CorrectionDriverVec2

S0, K, B = 100.0, 90.0, 80.0
sigma, r0, T = 0.25, 0.05, 1.0
N_STEPS = 10
dt = T / N_STEPS
sqrtDt = math.sqrt(dt)

# ── Record with switch registry ──────────────────────────────
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

g_res = [sw.g for sw in fn.cmp_switches()]
print(f"Switch registry: {len(g_res)} indicators")

# ── Run ──────────────────────────────────────────────────────
M = 50000
z_all = np.random.RandomState(42).randn(M, N_STEPS)

driver = CorrectionDriverVec2(fn, payoff_res, g_res, z_args, [S_arg],
                               skip_sigma=20.0, jump_eps=1e-4, num_threads=4)
driver.precompute_directions({S_arg: S0})
result = driver.run(z_all)

# ── Bump reference ───────────────────────────────────────────
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
bump_delta = (mc_ref(S0 + h) - mc_ref(S0 - h)) / (2 * h)

print(f"\nPrice: {result['price']:.4f}")
print(f"Pathwise:   {result['pathwise'][0]:+.6f}")
print(f"Correction: {result['correction'][0]:+.6f}")
print(f"Total:      {result['total'][0]:+.6f}")
print(f"Bump ref:   {bump_delta:+.6f}")
print(f"Error:      {abs(result['total'][0] - bump_delta)/abs(bump_delta)*100:.1f}%")
