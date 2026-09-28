"""
bench_heston_autocallable.py — 3-Asset Phoenix Autocallable (Heston).

Benchmark 2 from WBS/QuantMinds slides. Compares Newton vs Fries with
honest bias analysis: run at multiple M to separate bias from noise.

3 assets, Heston SV, 8 observation dates, coupon + autocall + put.
"""
import sys, os, math, time
import numpy as np
import aadc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from correction_driver_vec2 import CorrectionDriverVec2
from correction_driver_fries import CorrectionDriverFries

# ── Parameters ──────────────────────────────────────────────────
N_ASSETS = 3
S0 = [100.0, 100.0, 100.0]
VOL = [0.20, 0.25, 0.22]          # initial vols
V0  = [v**2 for v in VOL]          # initial variance
KAPPA = [2.0, 2.0, 2.0]            # mean reversion
THETA = [v**2 for v in VOL]        # long-run variance = V0
XI    = [0.3, 0.35, 0.32]          # vol-of-vol
RHO_SV = [-0.7, -0.65, -0.7]       # spot-vol correlation per asset
CORR_ASSETS = 0.5                   # pairwise spot correlation
R = 0.03
T = 2.0
N_OBS = 8                          # quarterly over 2 years
COUPON_BARRIER = 0.70
AUTOCALL_BARRIER = 1.00
COUPON_RATE = 0.05
NOTIONAL = 100.0

dt = T / N_OBS
sqrtDt = math.sqrt(dt)
nz = 2 * N_ASSETS * N_OBS  # 2 normals per asset per step (spot + vol)

# Cholesky for asset correlation (simplified: pairwise = CORR_ASSETS)
# For 3 assets with equal pairwise corr ρ:
rho = CORR_ASSETS
L = np.linalg.cholesky(np.array([
    [1.0, rho, rho],
    [rho, 1.0, rho],
    [rho, rho, 1.0]
]))

print(f"3-Asset Heston Phoenix Autocallable")
print(f"  Assets: {N_ASSETS}, Obs: {N_OBS}, z-dim: {nz}")
print(f"  S0={S0}, vol={VOL}, xi={XI}, kappa={KAPPA}")

# ── Record tape with switch registry ────────────────────────────
fn = aadc.Functions()
fn.start_recording(register_switches=aadc.SWITCH_JUMP)

S_ids = []; S_args = []
for a in range(N_ASSETS):
    s = aadc.idouble(S0[a]); S_args.append(s.mark_as_input()); S_ids.append(s)

xi_id = aadc.idouble(XI[0]); xi_arg = xi_id.mark_as_input()  # vol-of-vol asset 1
kappa_id = aadc.idouble(KAPPA[0]); kappa_arg = kappa_id.mark_as_input()  # mean-rev asset 1

z_list = []; z_args = []
for j in range(nz):
    zj = aadc.idouble(0.0); z_args.append(zj.mark_as_input()); z_list.append(zj)

# State: logS[a], v[a]
logS = [aadc.math.log(S_ids[a]) for a in range(N_ASSETS)]
v = [aadc.idouble(V0[a]) for a in range(N_ASSETS)]

payoff = aadc.idouble(0.0)
alive = aadc.idouble(1.0)

for d in range(N_OBS):
    # Independent normals for this step
    z_spot_indep = [z_list[2*N_ASSETS*d + 2*a] for a in range(N_ASSETS)]
    z_vol = [z_list[2*N_ASSETS*d + 2*a + 1] for a in range(N_ASSETS)]

    # Correlate spot normals via Cholesky
    z_spot = []
    for a in range(N_ASSETS):
        zs = aadc.idouble(0.0)
        for b in range(a + 1):
            zs = zs + aadc.idouble(L[a, b]) * z_spot_indep[b]
        z_spot.append(zs)

    for a in range(N_ASSETS):
        # Heston: correlated spot-vol
        rho_sv = RHO_SV[a]
        rho_sv2 = math.sqrt(1.0 - rho_sv**2)
        w_v = aadc.idouble(rho_sv) * z_spot[a] + aadc.idouble(rho_sv2) * z_vol[a]

        # Variance process (truncated Euler)
        v_pos = aadc.iif(v[a] > aadc.idouble(0.0), v[a], aadc.idouble(0.0))
        sqrt_v = aadc.math.sqrt(v_pos + aadc.idouble(1e-10))

        if a == 0:
            dv = kappa_id * (aadc.idouble(THETA[a]) - v_pos) * aadc.idouble(dt) \
                 + xi_id * sqrt_v * aadc.idouble(sqrtDt) * w_v
        else:
            dv = aadc.idouble(KAPPA[a]) * (aadc.idouble(THETA[a]) - v_pos) * aadc.idouble(dt) \
                 + aadc.idouble(XI[a]) * sqrt_v * aadc.idouble(sqrtDt) * w_v
        v[a] = v[a] + dv

        # Spot process
        logS[a] = logS[a] + (aadc.idouble(R) - aadc.idouble(0.5) * v_pos) * aadc.idouble(dt) \
                  + sqrt_v * aadc.idouble(sqrtDt) * z_spot[a]

    # Worst-of performance
    perfs = [aadc.math.exp(logS[a]) / S_ids[a] for a in range(N_ASSETS)]
    worst = perfs[0]
    for a in range(1, N_ASSETS):
        worst = aadc.iif(perfs[a] < worst, perfs[a], worst)

    df = math.exp(-R * (d + 1) * dt)

    # Coupon: if worst > coupon_barrier
    coupon_ind = aadc.iif(worst > aadc.idouble(COUPON_BARRIER),
                          aadc.idouble(1.0), aadc.idouble(0.0))
    payoff = payoff + alive * coupon_ind * aadc.idouble(COUPON_RATE * NOTIONAL * df)

    # Autocall: if worst > autocall_barrier (not on last date)
    if d < N_OBS - 1:
        autocall_ind = aadc.iif(worst > aadc.idouble(AUTOCALL_BARRIER),
                                aadc.idouble(1.0), aadc.idouble(0.0))
        payoff = payoff + alive * autocall_ind * aadc.idouble(NOTIONAL * df)
        alive = alive * (aadc.idouble(1.0) - autocall_ind)

# Final: return notional if alive
payoff = payoff + alive * aadc.idouble(NOTIONAL * math.exp(-R * T))
rP = payoff.mark_as_output()

fn.stop_recording()

switches = fn.cmp_switches()
g_res = [sw.g for sw in switches]
types = [sw.origin for sw in switches]
print(f"Switch registry: {len(switches)} indicators")

# ── Run ─────────────────────────────────────────────────────────
theta_args = [S_args[0], S_args[2], xi_arg, kappa_arg]
theta_names = ["dV/dS1", "dV/dS3", "dV/dXi (vol-of-vol)", "dV/dKappa (mean-rev)"]
theta_vals = {S_args[0]: S0[0], S_args[1]: S0[1], S_args[2]: S0[2],
              xi_arg: XI[0], kappa_arg: KAPPA[0]}

# Multiple M for bias analysis
for M in [10000, 50000]:
    z_all = np.random.RandomState(42).randn(M, nz)
    print(f"\n{'='*70}")
    print(f"  M = {M}")
    print(f"{'='*70}")

    # Newton
    t0 = time.time()
    drv_n = CorrectionDriverVec2(fn, rP, z_args, theta_args, num_threads=4)
    drv_n.precompute_directions(theta_vals)
    rn = drv_n.run(z_all)
    t_n = time.time() - t0

    # Fries 5%
    t0 = time.time()
    drv_f = CorrectionDriverFries(fn, rP, g_res, z_args, theta_args,
                                   window_fraction=0.05, num_threads=4)
    drv_f.precompute_directions(theta_vals)
    rf = drv_f.run(z_all)
    t_f = time.time() - t0

    print(f"\nPrice: {rn['price']:.2f}")
    print(f"\n{'Greek':<25} {'Newton':>10} {'Fries 5%':>10} {'Diff':>8}")
    print("-" * 58)
    for k, name in enumerate(theta_names):
        n_val = rn['total'][k]
        f_val = rf['total'][k]
        diff = abs(n_val - f_val) / max(abs(n_val), 1e-10) * 100
        print(f"{name:<25} {n_val:+10.4f} {f_val:+10.4f} {diff:7.1f}%")

    print(f"\n{'Greek':<25} {'Pathwise':>10} {'Correction':>12} {'Corr%':>7}")
    print("-" * 58)
    for k, name in enumerate(theta_names):
        pw = rn['pathwise'][k]
        co = rn['correction'][k]
        pct = abs(co)/(abs(pw)+abs(co))*100 if (abs(pw)+abs(co))>1e-15 else 0
        print(f"{name:<25} {pw:+10.4f} {co:+12.4f} {pct:6.1f}%")

    print(f"\nNewton: {t_n:.1f}s, Fries: {t_f:.1f}s")

# ── Fries bias analysis: vary w at fixed large M ────────────────
print(f"\n{'='*70}")
print(f"  Fries bias analysis (M=50000, varying w)")
print(f"{'='*70}")
M = 50000
z_all = np.random.RandomState(42).randn(M, nz)

drv_n = CorrectionDriverVec2(fn, rP, z_args, theta_args, num_threads=4)
drv_n.precompute_directions(theta_vals)
rn = drv_n.run(z_all)

print(f"\n{'w':>6} {'dS1 Newton':>12} {'dS1 Fries':>12} {'diff':>8}  "
      f"{'Xi Newton':>12} {'Xi Fries':>12} {'diff':>8}")
print("-" * 80)
for w in [0.01, 0.02, 0.05, 0.10, 0.20]:
    drv_f = CorrectionDriverFries(fn, rP, g_res, z_args, theta_args,
                                   window_fraction=w, num_threads=4)
    drv_f.precompute_directions(theta_vals)
    rf = drv_f.run(z_all)
    d_s1_n = rn['total'][0]; d_s1_f = rf['total'][0]
    d_xi_n = rn['total'][2]; d_xi_f = rf['total'][2]
    print(f"{w:6.0%} {d_s1_n:+12.4f} {d_s1_f:+12.4f} {abs(d_s1_f-d_s1_n)/max(abs(d_s1_n),1e-10)*100:7.1f}%  "
          f"{d_xi_n:+12.4f} {d_xi_f:+12.4f} {abs(d_xi_f-d_xi_n)/max(abs(d_xi_n),1e-10)*100:7.1f}%")
