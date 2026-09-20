"""
example_barrier_hw_ql.py — Down-and-out call with stochastic rates (QuantLib).

GBM spot + Hull-White 1F rate, correlated.
S0=100, K=90, B=80, sigma=0.25, r0=0.05, a=0.1, sigma_r=0.01, rho=-0.3.
T=1, 50 monitoring dates.

Vectorized correction driver (AVX + multithreading).
Computes all Greeks: delta, vega, rho, d/d(sigma_r).
Reference: bump-and-revalue MC.
"""
import sys, os, math, time
import numpy as np
import aadc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from correction_driver_vec import CorrectionDriverVec

try:
    import aadc_quantlib_tracing as ql
except ImportError:
    import QuantLib as ql

# ── Parameters ──────────────────────────────────────────────────
S0, K, B = 100.0, 90.0, 80.0
sigma_val, r0_val, T = 0.25, 0.05, 1.0
a_hw_val, sigma_r_val, rho_val = 0.1, 0.01, -0.3
N_STEPS = 50
dt = T / N_STEPS
sqrtDt = math.sqrt(dt)
nz = 2 * N_STEPS

L10 = rho_val
L11 = math.sqrt(1.0 - rho_val * rho_val)

today = ql.Date(20, 9, 2026)
ql.Settings.instance().evaluationDate = today
print(f"QuantLib {ql.__version__}, aadc {aadc.__version__}")

# ── Record kernel ───────────────────────────────────────────────
fn = aadc.Functions()
fn.start_recording()

S = aadc.idouble(S0);          S_arg = S.mark_as_input()
vol = aadc.idouble(sigma_val);  vol_arg = vol.mark_as_input()
r0_id = aadc.idouble(r0_val);   r0_arg = r0_id.mark_as_input()
sr_id = aadc.idouble(sigma_r_val); sr_arg = sr_id.mark_as_input()

z_list, z_args = [], []
for j in range(nz):
    zj = aadc.idouble(0.0)
    z_args.append(zj.mark_as_input())
    z_list.append(zj)

g_res = []
alive = aadc.idouble(1.0)
logS = np.log(S)
r_t = r0_id + aadc.idouble(0.0)  # copy
disc = aadc.idouble(0.0)

for d in range(N_STEPS):
    z1 = z_list[2 * d]
    w2 = aadc.idouble(L10) * z_list[2 * d] + aadc.idouble(L11) * z_list[2 * d + 1]

    logS = logS + (r_t - aadc.idouble(0.5) * vol * vol) * aadc.idouble(dt) \
           + vol * aadc.idouble(sqrtDt) * z1
    S_j = np.exp(logS)

    r_t = r_t + aadc.idouble(a_hw_val) * (r0_id - r_t) * aadc.idouble(dt) \
          + sr_id * aadc.idouble(sqrtDt) * w2
    disc = disc + r_t * aadc.idouble(dt)

    g = S_j - aadc.idouble(B)
    g_res.append(g.mark_as_output())
    ko = aadc.iif(S_j > aadc.idouble(B), aadc.idouble(1.0), aadc.idouble(0.0))
    alive = alive * ko

S_T = np.exp(logS)
call_payoff = aadc.iif(S_T > aadc.idouble(K),
                        S_T - aadc.idouble(K), aadc.idouble(0.0))
payoff = alive * np.exp(-disc) * call_payoff
payoff_res = payoff.mark_as_output()

fn.stop_recording()
print(f"Tape recorded: {nz} z-inputs, {len(g_res)} indicators")

# ── Run ─────────────────────────────────────────────────────────
M = 50000
NUM_THREADS = 4
theta_args = [S_arg, vol_arg, r0_arg, sr_arg]
theta_names = ["Delta (S0)", "Vega (vol)", "Rho (r0)", "d/d(sigma_r)"]
theta_vals = {S_arg: S0, vol_arg: sigma_val, r0_arg: r0_val, sr_arg: sigma_r_val}

rng = np.random.RandomState(42)
z_all = rng.randn(M, nz)

driver = CorrectionDriverVec(fn, payoff_res, g_res, z_args, theta_args,
                              skip_sigma=20.0, jump_eps=1e-4,
                              max_newton=4, num_threads=NUM_THREADS)
driver.precompute_directions(theta_vals)

t0 = time.time()
result = driver.run(z_all)
t1 = time.time()

print(f"\n{'='*60}")
print(f"Down-and-Out Call (GBM + Hull-White 1F, QuantLib)")
print(f"S0={S0}, K={K}, B={B}, sigma={sigma_val}")
print(f"HW: r0={r0_val}, a={a_hw_val}, sigma_r={sigma_r_val}, rho={rho_val}")
print(f"T={T}, {N_STEPS} steps, {M} paths, {t1-t0:.0f}s")

print(f"\nPrice: {result['price']:.4f}")
print(f"\n{'Greek':<18} {'Pathwise':>10} {'Correction':>12} {'Total':>10}")
print("-" * 55)
for k, name in enumerate(theta_names):
    pw = result['pathwise'][k]
    co = result['correction'][k]
    tot = result['total'][k]
    print(f"{name:<18} {pw:+10.4f} {co:+12.4f} {tot:+10.4f}")

# ── Bump-and-revalue reference ──────────────────────────────────
print(f"\nBump-and-revalue reference (200K paths)...")

def mc_ref(s0_v, vol_v, r0_v, sr_v, n_paths=200000):
    rng2 = np.random.RandomState(42)
    logS_a = np.full(n_paths, math.log(s0_v))
    r_a = np.full(n_paths, r0_v)
    disc_a = np.zeros(n_paths)
    alive_a = np.ones(n_paths)
    for i in range(N_STEPS):
        z1 = rng2.randn(n_paths)
        z2i = rng2.randn(n_paths)
        w2 = L10 * z1 + L11 * z2i
        logS_a += (r_a - 0.5*vol_v**2)*dt + vol_v*sqrtDt*z1
        r_a += a_hw_val*(r0_v - r_a)*dt + sr_v*sqrtDt*w2
        disc_a += r_a * dt
        alive_a *= (np.exp(logS_a) > B).astype(float)
    return (alive_a * np.exp(-disc_a) * np.maximum(np.exp(logS_a) - K, 0)).mean()

p0 = mc_ref(S0, sigma_val, r0_val, sigma_r_val)
bumps = [
    ("Delta", 0.5, lambda h: mc_ref(S0+h, sigma_val, r0_val, sigma_r_val)),
    ("Vega", 0.001, lambda h: mc_ref(S0, sigma_val+h, r0_val, sigma_r_val)),
    ("Rho", 0.001, lambda h: mc_ref(S0, sigma_val, r0_val+h, sigma_r_val)),
    ("d/dsr", 0.0005, lambda h: mc_ref(S0, sigma_val, r0_val, sigma_r_val+h)),
]

print(f"\n{'Greek':<18} {'Bump':>10} {'Correction':>12} {'Agree':>8}")
print("-" * 52)
for k, (name, h, fn_bump) in enumerate(bumps):
    p_up = fn_bump(h)
    p_dn = fn_bump(-h)
    bump_val = (p_up - p_dn) / (2 * h)
    corr_val = result['total'][k]
    agree = abs(corr_val - bump_val) / max(abs(bump_val), 1e-10) * 100
    print(f"{name:<18} {bump_val:+10.4f} {corr_val:+12.4f} {agree:7.1f}%")
