"""
bench_full_comparison.py — Full comparison: Newton vs Fries on all benchmarks, all Greeks.
"""
import sys, os, math, time
import numpy as np
import aadc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from correction_driver_vec import CorrectionDriverVec
from correction_driver_fries import CorrectionDriverFries

try:
    import aadc_quantlib_tracing as ql
except ImportError:
    ql = None

def run_comparison(name, fn, payoff_res, g_res, z_args, theta_args, theta_vals,
                   theta_names, bump_fn, M=20000, seed=42):
    """Run Newton, Fries, bump on same paths, print comparison."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    z_all = np.random.RandomState(seed).randn(M, len(z_args))
    n_theta = len(theta_args)

    # Newton
    drv_n = CorrectionDriverVec(fn, payoff_res, g_res, z_args, theta_args, num_threads=4)
    drv_n.precompute_directions(theta_vals)
    t0 = time.time()
    res_n = drv_n.run(z_all)
    t_newton = time.time() - t0

    # Fries 5%
    drv_f = CorrectionDriverFries(fn, payoff_res, g_res, z_args, theta_args,
                                   window_fraction=0.05)
    drv_f.precompute_directions(theta_vals)
    t0 = time.time()
    res_f = drv_f.run(z_all)
    t_fries = time.time() - t0

    # Bump reference
    bump_vals = bump_fn()

    # Print
    print(f"\n{'Greek':<18} {'Bump':>10} {'Newton':>10} {'N err':>7} {'Fries':>10} {'F err':>7}")
    print("-" * 68)
    for k in range(n_theta):
        b = bump_vals[k]
        n_tot = res_n['total'][k]
        f_tot = res_f['total'][k]
        n_err = abs(n_tot - b) / max(abs(b), 1e-10) * 100
        f_err = abs(f_tot - b) / max(abs(b), 1e-10) * 100
        print(f"{theta_names[k]:<18} {b:>+10.4f} {n_tot:>+10.4f} {n_err:>6.1f}% {f_tot:>+10.4f} {f_err:>6.1f}%")

    print(f"\nNewton: {t_newton:.1f}s, Fries: {t_fries:.1f}s, speedup: {t_newton/t_fries:.1f}x")
    print(f"Price: {res_n['price']:.4f}")

    # Pathwise vs correction breakdown for Newton
    print(f"\n{'Greek':<18} {'Pathwise':>10} {'Correction':>12} {'Corr %':>8}")
    print("-" * 52)
    for k in range(n_theta):
        pw = res_n['pathwise'][k]
        co = res_n['correction'][k]
        tot = res_n['total'][k]
        pct = abs(co) / max(abs(tot), 1e-10) * 100
        print(f"{theta_names[k]:<18} {pw:>+10.4f} {co:>+12.4f} {pct:>7.1f}%")


# ═══════════════════════════════════════════════════════════════
# Benchmark 1: Digital option
# ═══════════════════════════════════════════════════════════════
def build_digital():
    S0, sigma, r, T = 100.0, 0.25, 0.05, 1.0
    fn = aadc.Functions(); fn.start_recording()
    S = aadc.idouble(S0); S_arg = S.mark_as_input()
    z = aadc.idouble(0.0); z_arg = z.mark_as_input()
    logS = np.log(S) + aadc.idouble((r-0.5*sigma**2)*T) + aadc.idouble(sigma*math.sqrt(T))*z
    S_T = np.exp(logS)
    g = S_T - aadc.idouble(S0); g_res = g.mark_as_output()
    payoff = aadc.iif(S_T > aadc.idouble(S0), aadc.idouble(1.0), aadc.idouble(0.0))
    payoff = payoff * aadc.idouble(math.exp(-r*T))
    payoff_res = payoff.mark_as_output()
    fn.stop_recording()

    from scipy.stats import norm
    d2 = (math.log(1)+(r-0.5*sigma**2)*T)/(sigma*math.sqrt(T))
    analytic = math.exp(-r*T)*norm.pdf(d2)/(S0*sigma*math.sqrt(T))

    def bump():
        return [analytic]

    return ("Digital Option (analytic ref)", fn, payoff_res, [g_res], [z_arg],
            [S_arg], {S_arg: S0}, ["Delta"], bump, 50000)


# ═══════════════════════════════════════════════════════════════
# Benchmark 2: Down-and-out barrier (GBM, 10 steps)
# ═══════════════════════════════════════════════════════════════
def build_barrier():
    S0, K, B, sigma, r0, T = 100.0, 90.0, 80.0, 0.25, 0.05, 1.0
    N = 10; dt = T/N; sqrtDt = math.sqrt(dt)

    fn = aadc.Functions(); fn.start_recording()
    S = aadc.idouble(S0); S_arg = S.mark_as_input()
    vol = aadc.idouble(sigma); vol_arg = vol.mark_as_input()
    z_list, z_args = [], []
    for j in range(N):
        zj = aadc.idouble(0.0); z_args.append(zj.mark_as_input()); z_list.append(zj)
    logS = np.log(S); alive = aadc.idouble(1.0); g_res = []
    for j in range(N):
        logS = logS + (aadc.idouble(r0) - aadc.idouble(0.5)*vol*vol)*aadc.idouble(dt) + vol*aadc.idouble(sqrtDt)*z_list[j]
        S_j = np.exp(logS)
        g = S_j - aadc.idouble(B); g_res.append(g.mark_as_output())
        alive = alive * aadc.iif(S_j > aadc.idouble(B), aadc.idouble(1.0), aadc.idouble(0.0))
    S_T = np.exp(logS)
    call = aadc.iif(S_T > aadc.idouble(K), S_T - aadc.idouble(K), aadc.idouble(0.0))
    payoff = alive * call * aadc.idouble(math.exp(-r0*T))
    payoff_res = payoff.mark_as_output()
    fn.stop_recording()

    def mc_ref(s0_v, vol_v, n_paths=200000):
        rng2 = np.random.RandomState(42)
        logS_a = np.full(n_paths, math.log(s0_v)); alive_a = np.ones(n_paths)
        for i in range(N):
            z1 = rng2.randn(n_paths)
            logS_a += (r0-0.5*vol_v**2)*dt + vol_v*sqrtDt*z1
            alive_a *= (np.exp(logS_a) > B).astype(float)
        return (alive_a*math.exp(-r0*T)*np.maximum(np.exp(logS_a)-K, 0)).mean()

    def bump():
        h_s, h_v = 0.5, 0.001
        delta = (mc_ref(S0+h_s, sigma) - mc_ref(S0-h_s, sigma))/(2*h_s)
        vega = (mc_ref(S0, sigma+h_v) - mc_ref(S0, sigma-h_v))/(2*h_v)
        return [delta, vega]

    return ("Down-and-Out Barrier (GBM, 10 steps)", fn, payoff_res, g_res, z_args,
            [S_arg, vol_arg], {S_arg: S0, vol_arg: sigma},
            ["Delta (S0)", "Vega (vol)"], bump, 20000)


# ═══════════════════════════════════════════════════════════════
# Benchmark 3: Down-and-out barrier + HW (10 steps)
# ═══════════════════════════════════════════════════════════════
def build_barrier_hw():
    S0, K, B, sigma, r0, T = 100.0, 90.0, 80.0, 0.25, 0.05, 1.0
    a_hw, sigma_r, rho = 0.1, 0.01, -0.3
    N = 10; dt = T/N; sqrtDt = math.sqrt(dt); nz = 2*N
    L10 = rho; L11 = math.sqrt(1-rho*rho)

    fn = aadc.Functions(); fn.start_recording()
    S = aadc.idouble(S0); S_arg = S.mark_as_input()
    vol = aadc.idouble(sigma); vol_arg = vol.mark_as_input()
    r0_id = aadc.idouble(r0); r0_arg = r0_id.mark_as_input()
    z_list, z_args = [], []
    for j in range(nz):
        zj = aadc.idouble(0.0); z_args.append(zj.mark_as_input()); z_list.append(zj)
    logS = np.log(S); r_t = r0_id + aadc.idouble(0.0)
    disc = aadc.idouble(0.0); alive = aadc.idouble(1.0); g_res = []
    for d in range(N):
        z1 = z_list[2*d]; w2 = aadc.idouble(L10)*z_list[2*d]+aadc.idouble(L11)*z_list[2*d+1]
        logS = logS + (r_t - aadc.idouble(0.5)*vol*vol)*aadc.idouble(dt) + vol*aadc.idouble(sqrtDt)*z1
        S_j = np.exp(logS)
        r_t = r_t + aadc.idouble(a_hw)*(r0_id - r_t)*aadc.idouble(dt) + aadc.idouble(sigma_r*sqrtDt)*w2
        disc = disc + r_t*aadc.idouble(dt)
        g = S_j - aadc.idouble(B); g_res.append(g.mark_as_output())
        alive = alive * aadc.iif(S_j > aadc.idouble(B), aadc.idouble(1.0), aadc.idouble(0.0))
    S_T = np.exp(logS)
    call = aadc.iif(S_T > aadc.idouble(K), S_T - aadc.idouble(K), aadc.idouble(0.0))
    payoff = alive * np.exp(-disc) * call
    payoff_res = payoff.mark_as_output()
    fn.stop_recording()

    def mc_ref(s0_v, vol_v, r0_v, n_paths=200000):
        rng2 = np.random.RandomState(42)
        logS_a = np.full(n_paths, math.log(s0_v))
        r_a = np.full(n_paths, r0_v); disc_a = np.zeros(n_paths); alive_a = np.ones(n_paths)
        for i in range(N):
            z1 = rng2.randn(n_paths); z2i = rng2.randn(n_paths)
            w2 = L10*z1 + L11*z2i
            logS_a += (r_a-0.5*vol_v**2)*dt + vol_v*sqrtDt*z1
            r_a += a_hw*(r0_v-r_a)*dt + sigma_r*sqrtDt*w2
            disc_a += r_a*dt; alive_a *= (np.exp(logS_a)>B).astype(float)
        return (alive_a*np.exp(-disc_a)*np.maximum(np.exp(logS_a)-K, 0)).mean()

    def bump():
        h_s, h_v, h_r = 0.5, 0.001, 0.001
        delta = (mc_ref(S0+h_s, sigma, r0) - mc_ref(S0-h_s, sigma, r0))/(2*h_s)
        vega = (mc_ref(S0, sigma+h_v, r0) - mc_ref(S0, sigma-h_v, r0))/(2*h_v)
        rho_g = (mc_ref(S0, sigma, r0+h_r) - mc_ref(S0, sigma, r0-h_r))/(2*h_r)
        return [delta, vega, rho_g]

    return ("Down-and-Out Barrier + HW 1F (10 steps)", fn, payoff_res, g_res, z_args,
            [S_arg, vol_arg, r0_arg], {S_arg: S0, vol_arg: sigma, r0_arg: r0},
            ["Delta (S0)", "Vega (vol)", "Rho (r0)"], bump, 20000)


# ═══════════════════════════════════════════════════════════════
# Benchmark 4: 2-asset autocallable
# ═══════════════════════════════════════════════════════════════
def build_autocallable():
    ND = 4; S0_1, S0_2 = 100.0, 100.0
    VOL1, VOL2, CORR, R, T_MAT = 0.2, 0.25, 0.5, 0.03, 1.0
    nz = 2*ND; dt = T_MAT/ND; sqrtDt = math.sqrt(dt)
    rho2 = math.sqrt(1-CORR*CORR)

    fn = aadc.Functions(); fn.start_recording()
    S1 = aadc.idouble(S0_1); S1_arg = S1.mark_as_input()
    S2 = aadc.idouble(S0_2); S2_arg = S2.mark_as_input()
    z_list, z_args = [], []
    for j in range(nz):
        zj = aadc.idouble(0.0); z_args.append(zj.mark_as_input()); z_list.append(zj)
    logS1 = np.log(S1); logS2 = np.log(S2)
    d1 = aadc.idouble((R-0.5*VOL1**2)*dt); d2 = aadc.idouble((R-0.5*VOL2**2)*dt)
    payoff = aadc.idouble(0.0); alive = aadc.idouble(1.0); g_res = []
    for d in range(ND):
        z1 = z_list[2*d]; z2 = aadc.idouble(CORR)*z_list[2*d]+aadc.idouble(rho2)*z_list[2*d+1]
        logS1 = logS1+d1+aadc.idouble(VOL1*sqrtDt)*z1
        logS2 = logS2+d2+aadc.idouble(VOL2*sqrtDt)*z2
        p1 = np.exp(logS1); p2 = np.exp(logS2)
        pf1 = p1/aadc.idouble(S0_1); pf2 = p2/aadc.idouble(S0_2)
        worst = aadc.iif(pf1<pf2, pf1, pf2)
        g_res.append((worst-aadc.idouble(0.70)).mark_as_output())
        g_res.append((worst-aadc.idouble(1.00)).mark_as_output())
        df = math.exp(-R*T_MAT*(d+1.0)/ND)
        ci = aadc.iif(worst>aadc.idouble(0.70), aadc.idouble(1.0), aadc.idouble(0.0))
        payoff = payoff+alive*ci*aadc.idouble(0.05*100.0*df)
        if d < ND-1:
            ai = aadc.iif(worst>aadc.idouble(1.00), aadc.idouble(1.0), aadc.idouble(0.0))
            payoff = payoff+alive*ai*aadc.idouble(100.0*df)
            alive = alive*(aadc.idouble(1.0)-ai)
    payoff = payoff+alive*aadc.idouble(100.0*math.exp(-R*T_MAT))
    payoff_res = payoff.mark_as_output()
    fn.stop_recording()

    def bump():
        return [-0.209, -0.134]  # C++ reference

    return ("2-Asset Autocallable (4 dates)", fn, payoff_res, g_res, z_args,
            [S1_arg, S2_arg], {S1_arg: S0_1, S2_arg: S0_2},
            ["dV/dS1", "dV/dS2"], bump, 10000)


# ═══════════════════════════════════════════════════════════════
# Run all
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    benchmarks = [build_digital(), build_barrier(), build_barrier_hw(), build_autocallable()]
    for args in benchmarks:
        name, fn, payoff_res, g_res, z_args, theta_args, theta_vals, theta_names, bump_fn, M = args
        run_comparison(name, fn, payoff_res, g_res, z_args, theta_args,
                       theta_vals, theta_names, bump_fn, M=M)
