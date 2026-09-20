"""
test_vectorized.py — Verify CorrectionDriverVec matches scalar CorrectionDriver.

Tests on digital option (simple, 1 indicator) and 2-asset autocallable (complex, 8 indicators).
"""
import sys, os, math
import numpy as np
import aadc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from correction_driver import CorrectionDriver
from correction_driver_vec import CorrectionDriverVec

# ═══════════════════════════════════════════════════════════════
# Test 1: Digital option
# ═══════════════════════════════════════════════════════════════
def test_digital():
    print("Test 1: Digital option")
    S0 = 100.0; sigma = 0.25; r = 0.05; T = 1.0

    fn = aadc.Functions()
    fn.start_recording()
    S = aadc.idouble(S0); S_arg = S.mark_as_input()
    z = aadc.idouble(0.0); z_arg = z.mark_as_input()

    logS = np.log(S) + aadc.idouble((r - 0.5*sigma**2)*T) + aadc.idouble(sigma*math.sqrt(T)) * z
    S_T = np.exp(logS)
    g = S_T - aadc.idouble(S0)
    g_res = g.mark_as_output()
    payoff = aadc.iif(S_T > aadc.idouble(S0), aadc.idouble(1.0), aadc.idouble(0.0))
    payoff = payoff * aadc.idouble(math.exp(-r * T))
    payoff_res = payoff.mark_as_output()
    fn.stop_recording()

    M = 20000
    rng = np.random.RandomState(42)
    z_all = rng.randn(M, 1)

    # Scalar driver
    drv_s = CorrectionDriver(fn, payoff_res, [g_res], [z_arg], [S_arg],
                              skip_sigma=20.0, jump_eps=1e-4)
    drv_s.precompute_directions({S_arg: S0})
    ws = drv_s.ws
    sum_corr_s = 0.0
    for m in range(M):
        ws.set_val(z_arg, float(z_all[m, 0]))
        ws.set_val(S_arg, S0)
        ws.forward()
        for j in range(1):
            ws.set_val(z_arg, float(z_all[m, 0]))
        ws.set_val(S_arg, S0)
        ws.forward()
        cr = drv_s.compute_correction(z_all[m])
        sum_corr_s += cr.correction[0]
    scalar_corr = sum_corr_s / M

    # Vectorized driver
    drv_v = CorrectionDriverVec(fn, payoff_res, [g_res], [z_arg], [S_arg],
                                 skip_sigma=20.0, jump_eps=1e-4, num_threads=4)
    drv_v.precompute_directions({S_arg: S0})
    result_v = drv_v.run(z_all)
    vec_corr = result_v['correction'][0]

    print(f"  Scalar correction: {scalar_corr:+.6f}")
    print(f"  Vector correction: {vec_corr:+.6f}")
    diff = abs(scalar_corr - vec_corr)
    print(f"  Difference: {diff:.2e}")

    # Analytic reference
    from scipy.stats import norm
    d2 = (math.log(1) + (r - 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    analytic = math.exp(-r*T) * norm.pdf(d2) / (S0 * sigma * math.sqrt(T))
    print(f"  Analytic delta: {analytic:+.6f}")
    print(f"  Vec total delta: {result_v['total'][0]:+.6f}")

    ok = diff < 0.002
    print(f"  {'PASS' if ok else 'FAIL'}\n")
    return ok


# ═══════════════════════════════════════════════════════════════
# Test 2: 2-asset autocallable (from test_autocallable.py)
# ═══════════════════════════════════════════════════════════════
def test_autocallable():
    print("Test 2: 2-asset autocallable")
    ND = 4
    S0_1, S0_2 = 100.0, 100.0
    VOL1, VOL2, CORR, R, T_MAT = 0.2, 0.25, 0.5, 0.03, 1.0
    COUPON_BARRIER, AUTOCALL_BARRIER = 0.70, 1.00
    COUPON_RATE, NOTIONAL = 0.05, 100.0
    nz = 2 * ND; dt = T_MAT / ND; sqrtDt = math.sqrt(dt)
    rho2 = math.sqrt(1.0 - CORR * CORR)

    fn = aadc.Functions()
    fn.start_recording()
    S1 = aadc.idouble(S0_1); S1_arg = S1.mark_as_input()
    S2 = aadc.idouble(S0_2); S2_arg = S2.mark_as_input()
    z_list, z_args = [], []
    for j in range(nz):
        zj = aadc.idouble(0.0); z_args.append(zj.mark_as_input()); z_list.append(zj)

    logS1 = np.log(S1); logS2 = np.log(S2)
    drift1 = aadc.idouble((R - 0.5*VOL1**2)*dt)
    drift2 = aadc.idouble((R - 0.5*VOL2**2)*dt)
    payoff = aadc.idouble(0.0); alive = aadc.idouble(1.0)
    g_res = []

    for d in range(ND):
        z1 = z_list[2*d]
        z2 = aadc.idouble(CORR)*z_list[2*d] + aadc.idouble(rho2)*z_list[2*d+1]
        logS1 = logS1 + drift1 + aadc.idouble(VOL1*sqrtDt)*z1
        logS2 = logS2 + drift2 + aadc.idouble(VOL2*sqrtDt)*z2
        price1 = np.exp(logS1); price2 = np.exp(logS2)
        perf1 = price1/aadc.idouble(S0_1); perf2 = price2/aadc.idouble(S0_2)
        worst = aadc.iif(perf1 < perf2, perf1, perf2)

        g_res.append((worst - aadc.idouble(COUPON_BARRIER)).mark_as_output())
        g_res.append((worst - aadc.idouble(AUTOCALL_BARRIER)).mark_as_output())

        df = math.exp(-R*T_MAT*(d+1.0)/ND)
        coupon_ind = aadc.iif(worst > aadc.idouble(COUPON_BARRIER),
                               aadc.idouble(1.0), aadc.idouble(0.0))
        payoff = payoff + alive*coupon_ind*aadc.idouble(COUPON_RATE*NOTIONAL*df)
        if d < ND-1:
            autocall_ind = aadc.iif(worst > aadc.idouble(AUTOCALL_BARRIER),
                                     aadc.idouble(1.0), aadc.idouble(0.0))
            payoff = payoff + alive*autocall_ind*aadc.idouble(NOTIONAL*df)
            alive = alive*(aadc.idouble(1.0) - autocall_ind)
    payoff = payoff + alive*aadc.idouble(NOTIONAL*math.exp(-R*T_MAT))
    payoff_res = payoff.mark_as_output()
    fn.stop_recording()

    M = 10000
    rng = np.random.RandomState(42)
    z_all = rng.randn(M, nz)

    # Scalar
    drv_s = CorrectionDriver(fn, payoff_res, g_res, z_args, [S1_arg, S2_arg],
                              skip_sigma=20.0, jump_eps=1e-4)
    drv_s.precompute_directions({S1_arg: S0_1, S2_arg: S0_2})
    ws = drv_s.ws
    sum_corr_s = np.zeros(2)
    for m in range(M):
        for j in range(nz):
            ws.set_val(z_args[j], float(z_all[m, j]))
        ws.set_val(S1_arg, S0_1); ws.set_val(S2_arg, S0_2)
        ws.forward()
        for j in range(nz):
            ws.set_val(z_args[j], float(z_all[m, j]))
        ws.set_val(S1_arg, S0_1); ws.set_val(S2_arg, S0_2)
        ws.forward()
        cr = drv_s.compute_correction(z_all[m])
        for k in range(2):
            sum_corr_s[k] += cr.correction[k]
    scalar_corr = sum_corr_s / M

    # Vectorized
    drv_v = CorrectionDriverVec(fn, payoff_res, g_res, z_args, [S1_arg, S2_arg],
                                 skip_sigma=20.0, jump_eps=1e-4, num_threads=4)
    drv_v.precompute_directions({S1_arg: S0_1, S2_arg: S0_2})
    result_v = drv_v.run(z_all)
    vec_corr = result_v['correction']

    print(f"  Scalar dS1: {scalar_corr[0]:+.6f}, dS2: {scalar_corr[1]:+.6f}")
    print(f"  Vector dS1: {vec_corr[0]:+.6f}, dS2: {vec_corr[1]:+.6f}")
    diff1 = abs(scalar_corr[0] - vec_corr[0])
    diff2 = abs(scalar_corr[1] - vec_corr[1])
    print(f"  Diff dS1: {diff1:.2e}, dS2: {diff2:.2e}")

    ok = diff1 < 0.01 and diff2 < 0.01
    print(f"  C++ ref: dS1=-0.209, dS2=-0.134")
    print(f"  {'PASS' if ok else 'FAIL'}\n")
    return ok


if __name__ == "__main__":
    ok1 = test_digital()
    ok2 = test_autocallable()
    print(f"\nAll tests: {'PASS' if ok1 and ok2 else 'FAIL'}")
    sys.exit(0 if ok1 and ok2 else 1)
