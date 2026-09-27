"""
example_autocallable.py — Phoenix autocallable with correction driver.

Uses switch registry (aadc >= 2.22.1) for automatic indicator discovery.
N observation dates, early redemption if S > autocall barrier.

This is the Python version; a C++ version with AADCNG_IF (BranchManager)
is proposed for matlogica/aadc-example-pybind11.
"""
import sys, os, math, time
import numpy as np
import aadc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from correction_driver_vec2 import CorrectionDriverVec2
from correction_driver_fries import CorrectionDriverFries

# ── Parameters ──────────────────────────────────────────────────
S0 = 100.0
autocall_barrier = 100.0   # early redemption if S > 100
put_barrier = 70.0          # capital protection barrier
coupon = 0.05               # 5% annual coupon
sigma = 0.25
r0 = 0.03
T = 2.0
N_OBS = 4                   # quarterly observations over 2 years
dt = T / N_OBS
sqrtDt = math.sqrt(dt)

print(f"Phoenix Autocallable: S0={S0}, autocall={autocall_barrier}, "
      f"put={put_barrier}, coupon={coupon}, vol={sigma}, r={r0}, T={T}, obs={N_OBS}")

# ── Record with switch registry ─────────────────────────────────
fn = aadc.Functions()
fn.start_recording(register_switches=aadc.SWITCH_JUMP)

S = aadc.idouble(S0); aS = S.mark_as_input()
vol_id = aadc.idouble(sigma); a_vol = vol_id.mark_as_input()

z_list, z_args = [], []
for j in range(N_OBS):
    zj = aadc.idouble(0.0); z_args.append(zj.mark_as_input()); z_list.append(zj)

logS = aadc.math.log(S)
drift = (aadc.idouble(r0) - aadc.idouble(0.5) * vol_id * vol_id) * aadc.idouble(dt)
disc = aadc.idouble(0.0)
payoff = aadc.idouble(0.0)
alive = aadc.idouble(1.0)

for i in range(N_OBS):
    logS = logS + drift + vol_id * aadc.idouble(sqrtDt) * z_list[i]
    disc = disc + aadc.idouble(r0 * dt)
    S_i = aadc.math.exp(logS)

    # Autocall: ONE comparison per date (avoids double-counted switches)
    autocalled = aadc.iif(S_i > aadc.idouble(autocall_barrier),
                          aadc.idouble(1.0), aadc.idouble(0.0))
    redemption = alive * (aadc.idouble(1.0) + aadc.idouble(coupon * (i + 1) * dt)) \
                 * aadc.math.exp(-disc)
    payoff = payoff + autocalled * redemption
    alive = alive * (aadc.idouble(1.0) - autocalled)

# Final payoff
S_T = aadc.math.exp(logS)
df = aadc.math.exp(-disc)
final_above = aadc.idouble(1.0) * df
final_below = (S_T / S) * df
final_pay = aadc.iif(S_T > aadc.idouble(put_barrier), final_above, final_below)
payoff = payoff + alive * final_pay

rP = payoff.mark_as_output()
fn.stop_recording()

switches = fn.cmp_switches()
print(f"Switch registry: {len(switches)} indicators (expect {N_OBS * 2 + 1})")
for sw in switches[:5]:
    print(f"  op={sw.op} g_record={sw.g_record:.2f}")

# ── Run correction ──────────────────────────────────────────────
M = 50000
z_all = np.random.RandomState(42).randn(M, N_OBS)
theta_args = [aS, a_vol]
theta_names = ["dV/dS (Delta)", "dV/dvol (Vega)"]

print(f"\nNewton correction ({M} paths)...")
t0 = time.time()
driver = CorrectionDriverVec2(fn, rP, z_args, theta_args, num_threads=4)
driver.precompute_directions({aS: S0, a_vol: sigma})
result = driver.run(z_all)
t_newton = time.time() - t0

print(f"\nFries correction (w=5%)...")
t0 = time.time()
drv_fries = CorrectionDriverFries(fn, rP,
    [sw.g for sw in fn.cmp_switches()], z_args, theta_args,
    window_fraction=0.05, num_threads=4)
drv_fries.precompute_directions({aS: S0, a_vol: sigma})
res_fries = drv_fries.run(z_all)
t_fries = time.time() - t0

# ── Bump reference ──────────────────────────────────────────────
def mc_ref(s0_val, vol_val, n_paths=200000):
    rng = np.random.RandomState(42)
    logS_a = np.full(n_paths, math.log(s0_val))
    alive_a = np.ones(n_paths)
    payoff_a = np.zeros(n_paths)
    disc_a = np.zeros(n_paths)
    for i in range(N_OBS):
        logS_a += (r0 - 0.5*vol_val**2)*dt + vol_val*sqrtDt*rng.randn(n_paths)
        disc_a += r0 * dt
        S_i = np.exp(logS_a)
        redeemed = alive_a * (S_i > autocall_barrier) * \
                   (1.0 + coupon*(i+1)*dt) * np.exp(-disc_a)
        payoff_a += redeemed
        alive_a *= (S_i <= autocall_barrier)
    S_T = np.exp(logS_a)
    df = np.exp(-disc_a)
    final = np.where(S_T > put_barrier, 1.0*df, (S_T/s0_val)*df)
    payoff_a += alive_a * final
    return payoff_a.mean()

h_s = 0.5
bump_delta = (mc_ref(S0+h_s, sigma) - mc_ref(S0-h_s, sigma)) / (2*h_s)
h_v = 0.001
bump_vega = (mc_ref(S0, sigma+h_v) - mc_ref(S0, sigma-h_v)) / (2*h_v)

# ── Results ─────────────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"Phoenix Autocallable, {N_OBS} obs dates, {M} paths")
print(f"Price: {result['price']:.4f}")
print(f"\n{'Greek':<20} {'Bump':>10} {'Newton':>10} {'N err':>7} {'Fries':>10} {'F err':>7}")
print("-" * 68)
for k, name in enumerate(theta_names):
    b = [bump_delta, bump_vega][k]
    n = result['total'][k]
    f = res_fries['total'][k]
    ne = abs(n-b)/max(abs(b),1e-10)*100
    fe = abs(f-b)/max(abs(b),1e-10)*100
    print(f"{name:<20} {b:+10.4f} {n:+10.4f} {ne:6.1f}% {f:+10.4f} {fe:6.1f}%")

print(f"\nNewton: {t_newton:.1f}s, Fries: {t_fries:.1f}s")

print(f"\n{'Greek':<20} {'Pathwise':>10} {'Correction':>12} {'Corr %':>8}")
print("-" * 55)
for k, name in enumerate(theta_names):
    pw = result['pathwise'][k]
    co = result['correction'][k]
    pct = abs(co)/(abs(pw)+abs(co))*100 if (abs(pw)+abs(co))>1e-15 else 0
    print(f"{name:<20} {pw:+10.4f} {co:+12.4f} {pct:7.1f}%")
