"""
bench_convergence.py — Multi-seed convergence study: Newton vs Fries vs Adaptive.

Runs 10 independent seeds at each path count. Reports mean error and std.
Demonstrates: Newton converges, Fries bias plateaus, Adaptive has higher variance.
"""
import sys, os, math, numpy as np, aadc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from correction_driver_vec import CorrectionDriverVec
from correction_driver_fries import CorrectionDriverFries

# ── Digital (analytic reference) ────────────────────────────────
def build_digital():
    S0, sigma, r, T = 100.0, 0.25, 0.05, 1.0
    fn = aadc.Functions(); fn.start_recording()
    S = aadc.idouble(S0); S_arg = S.mark_as_input()
    z = aadc.idouble(0.0); z_arg = z.mark_as_input()
    logS = np.log(S) + aadc.idouble((r-0.5*sigma**2)*T) + aadc.idouble(sigma*math.sqrt(T))*z
    S_T = np.exp(logS); g = S_T - aadc.idouble(S0); g_res = g.mark_as_output()
    payoff = aadc.iif(S_T > aadc.idouble(S0), aadc.idouble(1.0), aadc.idouble(0.0))
    payoff = payoff * aadc.idouble(math.exp(-r*T)); payoff_res = payoff.mark_as_output()
    fn.stop_recording()
    from scipy.stats import norm
    d2 = ((r-0.5*sigma**2)*T)/(sigma*math.sqrt(T))
    ref = math.exp(-r*T)*norm.pdf(d2)/(S0*sigma*math.sqrt(T))
    return "Digital (analytic)", fn, payoff_res, [g_res], [z_arg], [S_arg], {S_arg: S0}, ref

# ── Autocallable ────────────────────────────────────────────────
def build_autocallable():
    ND=4; S0_1,S0_2=100.0,100.0; VOL1,VOL2,CORR,R,T_MAT=0.2,0.25,0.5,0.03,1.0
    nz=2*ND; dt=T_MAT/ND; sqrtDt=math.sqrt(dt); rho2=math.sqrt(1-CORR*CORR)
    fn=aadc.Functions(); fn.start_recording()
    S1=aadc.idouble(S0_1); S1_arg=S1.mark_as_input()
    S2=aadc.idouble(S0_2); S2_arg=S2.mark_as_input()
    z_list,z_args=[],[]
    for j in range(nz):
        zj=aadc.idouble(0.0); z_args.append(zj.mark_as_input()); z_list.append(zj)
    logS1=np.log(S1); logS2=np.log(S2)
    d1=aadc.idouble((R-0.5*VOL1**2)*dt); d2=aadc.idouble((R-0.5*VOL2**2)*dt)
    payoff=aadc.idouble(0.0); alive=aadc.idouble(1.0); g_res=[]
    for d in range(ND):
        z1=z_list[2*d]; z2=aadc.idouble(CORR)*z_list[2*d]+aadc.idouble(rho2)*z_list[2*d+1]
        logS1=logS1+d1+aadc.idouble(VOL1*sqrtDt)*z1; logS2=logS2+d2+aadc.idouble(VOL2*sqrtDt)*z2
        p1=np.exp(logS1); p2=np.exp(logS2)
        pf1=p1/aadc.idouble(S0_1); pf2=p2/aadc.idouble(S0_2)
        worst=aadc.iif(pf1<pf2,pf1,pf2)
        g_res.append((worst-aadc.idouble(0.70)).mark_as_output())
        g_res.append((worst-aadc.idouble(1.00)).mark_as_output())
        df=math.exp(-R*T_MAT*(d+1.0)/ND)
        ci=aadc.iif(worst>aadc.idouble(0.70),aadc.idouble(1.0),aadc.idouble(0.0))
        payoff=payoff+alive*ci*aadc.idouble(0.05*100.0*df)
        if d<ND-1:
            ai=aadc.iif(worst>aadc.idouble(1.00),aadc.idouble(1.0),aadc.idouble(0.0))
            payoff=payoff+alive*ai*aadc.idouble(100.0*df)
            alive=alive*(aadc.idouble(1.0)-ai)
    payoff=payoff+alive*aadc.idouble(100.0*math.exp(-R*T_MAT))
    payoff_res=payoff.mark_as_output()
    fn.stop_recording()
    # Bump ref for dS2
    def mc_auto(s1,s2,n=500000):
        rng2=np.random.RandomState(42)
        logS1_a=np.full(n,math.log(s1)); logS2_a=np.full(n,math.log(s2))
        payoff_a=np.zeros(n); alive_a=np.ones(n)
        for d in range(ND):
            z1=rng2.randn(n); z2i=rng2.randn(n); z2c=CORR*z1+rho2*z2i
            logS1_a+=(R-0.5*VOL1**2)*dt+VOL1*sqrtDt*z1
            logS2_a+=(R-0.5*VOL2**2)*dt+VOL2*sqrtDt*z2c
            pf1=np.exp(logS1_a)/S0_1; pf2=np.exp(logS2_a)/S0_2; worst=np.minimum(pf1,pf2)
            df_d=math.exp(-R*T_MAT*(d+1.0)/ND)
            payoff_a+=alive_a*(worst>0.70).astype(float)*0.05*100.0*df_d
            if d<ND-1:
                ac=(worst>1.00).astype(float); payoff_a+=alive_a*ac*100.0*df_d; alive_a*=(1.0-ac)
        payoff_a+=alive_a*100.0*math.exp(-R*T_MAT)
        return payoff_a.mean()
    h=0.5
    ref = (mc_auto(S0_1,S0_2+h)-mc_auto(S0_1,S0_2-h))/(2*h)  # dS2
    return "Autocallable dS2", fn, payoff_res, g_res, z_args, [S1_arg,S2_arg], {S1_arg:S0_1,S2_arg:S0_2}, ref

# ── Run convergence study ───────────────────────────────────────
N_SEEDS = 10

for builder in [build_digital, build_autocallable]:
    name, fn, payoff_res, g_res, z_args, theta_args, theta_vals, ref = builder()
    # For autocallable, we look at theta index 1 (dS2); for digital, index 0
    k = 1 if len(theta_args) > 1 else 0

    print(f"\n{'='*70}")
    print(f"  {name}, ref = {ref:+.6f}, {N_SEEDS} seeds")
    print(f"{'='*70}")
    print(f"{'M':>8} {'Newton mean':>12} {'N std':>8} {'N err':>6} "
          f"{'Fries5% mean':>13} {'F std':>8} {'F err':>6} "
          f"{'Adapt mean':>12} {'A std':>8} {'A err':>6}")
    print("-" * 100)

    for M in [5000, 10000, 50000, 100000]:
        nv, fv, av = [], [], []
        for seed in range(N_SEEDS):
            z_all = np.random.RandomState(seed*100+7).randn(M, len(z_args))

            dn = CorrectionDriverVec(fn, payoff_res, g_res, z_args, theta_args,
                                      num_threads=4, max_newton=8)
            dn.precompute_directions(theta_vals)
            rn = dn.run(z_all)
            nv.append(rn['total'][k])

            df5 = CorrectionDriverFries(fn, payoff_res, g_res, z_args, theta_args,
                                         window_fraction=0.05)
            df5.precompute_directions(theta_vals)
            rf5 = df5.run(z_all)
            fv.append(rf5['total'][k])

            dfa = CorrectionDriverFries(fn, payoff_res, g_res, z_args, theta_args,
                                         adaptive=True, adaptive_tol=0.15)
            dfa.precompute_directions(theta_vals)
            rfa = dfa.run(z_all)
            av.append(rfa['total'][k])

        nm, ns = np.mean(nv), np.std(nv)
        fm, fs = np.mean(fv), np.std(fv)
        am, as_ = np.mean(av), np.std(av)
        ne = abs(nm-ref)/abs(ref)*100
        fe = abs(fm-ref)/abs(ref)*100
        ae = abs(am-ref)/abs(ref)*100
        print(f"{M:>8} {nm:>+12.6f} {ns:>8.4f} {ne:>5.1f}% "
              f"{fm:>+13.6f} {fs:>8.4f} {fe:>5.1f}% "
              f"{am:>+12.6f} {as_:>8.4f} {ae:>5.1f}%")
