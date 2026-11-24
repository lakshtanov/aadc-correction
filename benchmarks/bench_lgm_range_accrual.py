"""
bench_lgm_range_accrual.py — credit-linked range accrual on a 3M rate in LGM (Hull-White one factor), daily monitoring
for one year, with an intensity default: 504 corridor comparisons and 252 default steps per path.

Rates. LGM in the Hull-White parametrisation: r_t = f(0,t) + sigma_r^2/(2a^2) (1 - e^{-a t})^2 + x_t,
dx = -a x dt + sigma_r dW, stepped exactly: x_{k} = x_{k-1} e^{-a dt} + sigma_r sqrt((1 - e^{-2a dt})/(2a)) z_k.
Flat initial forward curve f(0,t) = R0. Zero bond
    P(t, t+tau) = P0(t+tau)/P0(t) exp(-B x_t - B^2 V(t)/2 - B sigma_r^2/(2a^2) (1 - e^{-a t})^2),
    B = (1 - e^{-a tau})/a,   V(t) = sigma_r^2 (1 - e^{-2 a t})/(2a)
(the G1++ formula of Brigo-Mercurio written in the state x). Index: 3M simple rate L_t = (1/P(t, t+0.25) - 1)/0.25.
Discount on the path: D_k = exp(-sum_{j<k} r_{t_j} dt).
Credit. CIR intensity lambda (lambda0 = theta = 0.02, kappa = 1, vol SIG_L), drift-implicit scheme for y = sqrt(lambda):
    y_k = (a_k + sqrt(a_k^2 + 4 c dt (kappa theta/2 - sigma^2/8))) / (2c),  a_k = y_{k-1} + sigma sqrt(dt)/2 w_k,
    c = 1 + kappa dt/2,  w_k = rho z_k + sqrt(1 - rho^2) z'_k  (rho = 0.3 with the rate normal);
cumulative intensity A_k = sum_{j<k} lambda_j dt (left point); threshold E = -log Phi(-z_E) ~ Exp(1); default in the
first step k with A_k >= E.
Payoff (notional 100, T = 1, N = 252 daily steps, quarters of 63 days, coupon 5% a year, corridor 2%..4% on L):
each day the note is alive in_i = 1{0.02 < L_i < 0.04}; at a quarter end (no default up to it) coupon
0.05/4 x 100 x (days in the corridor in the quarter)/63 x D; at T: 100 D_T; default in step k: recovery 40 D_k.
Greeks: dV/dR0 (parallel level of the curve), dV/dsigma_r, dV/dlambda0, dV/dsigma_lambda.

Usage:  OPENBLAS_NUM_THREADS=1 python benchmarks/bench_lgm_range_accrual.py           (MC, SEEDS x M paths)
        SOBOL=1 R=8 N=4096 ...      MC vs scrambled Sobol (option B)
        REF_M=8 ...                 CRN central-bump reference first: rate Greeks with steps h/2, h, 2h and the
                                    Richardson value f(h/2) - (f(h) - f(h/2))/3 (the bump of a discontinuous payoff
                                    is biased: O(h^2), visible at h = 10% of sigma_r); credit Greeks with one step
        SIG_L=0.25 ...              2 kappa theta / sigma^2 = 0.64 < 1 (Feller condition fails)
The default threshold is integrated exactly (coins="auto", CoxDefault markup: fast path); the corridor by the delta
correction. Daily monitoring makes the delta correction cost grow with (indicators x tape length): a 10K-path
estimate takes minutes, Sobol is the main lever (gains 20-78x on the Greeks).
"""
import os, sys, io, time, math, contextlib
import numpy as np
import aadc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from correction_driver_vec2 import CorrectionDriverVec2
from coins import CoxDefault

c = aadc.idouble
THREADS = int(os.environ.get("THREADS", "10"))
N, T = 252, 1.0
dt = T / N
QUARTER = 63
DATE_STEPS = [QUARTER * (q + 1) for q in range(N // QUARTER)]
R0, A_MR, SIG_R, TAU = 0.03, 0.03, 0.01, 0.25
LAM0, KAP_L, THE_L, RHO_RL = 0.02, 1.0, 0.02, 0.3
SIG_L = float(os.environ.get("SIG_L", "0.10"))
assert 4 * KAP_L * THE_L > SIG_L ** 2
REC, NOTIONAL, CPN_Q, LO, HI = 0.4, 100.0, 0.05 / 4, 0.02, 0.04
THETA = [R0, SIG_R, LAM0, SIG_L]
NAMES = ["dV/dR0", "dV/dsigma_r", "dV/dlambda0", "dV/dsigma_lambda"]
H_REF = [0.001, 0.001, 0.004, 0.02]


def record(markup=True):
    """Tape; inputs: R0, sigma_r, lambda0, sigma_lambda, then normals z_r (N), z_lambda (N), z_E."""
    fn = aadc.Functions(); fn.start_recording(register_switches=aadc.SWITCH_JUMP)
    r0 = c(R0); a_r0 = r0.mark_as_input()
    sr = c(SIG_R); a_sr = sr.mark_as_input()
    l0 = c(LAM0); a_l0 = l0.mark_as_input()
    sl = c(SIG_L); a_sl = sl.mark_as_input()
    z = [c(0.0) for _ in range(2 * N + 1)]; za = [q.mark_as_input() for q in z]
    zr, zl, zE = z[:N], z[N:2 * N], z[2 * N]
    one, zero = c(1.0), c(0.0)
    a = A_MR
    E = -aadc.math.log(aadc.math.cdf_normal(-zE))
    ea, B = math.exp(-a * dt), (1 - math.exp(-a * TAU)) / a
    ou_sd = math.sqrt((1 - math.exp(-2 * a * dt)) / (2 * a))
    x, logD, y, Lam, alive, P, acc = zero, zero, aadc.math.sqrt(l0), zero, one, zero, zero
    rl, cA = math.sqrt(1 - RHO_RL ** 2), 1.0 + 0.5 * KAP_L * dt
    dflt = CoxDefault() if markup else None
    for k in range(1, N + 1):
        t_prev, t = (k - 1) * dt, k * dt
        r_prev = r0 + sr * sr / c(2 * a * a) * c((1 - math.exp(-a * t_prev)) ** 2) + x
        logD = logD - r_prev * c(dt)
        lam = y * y
        w = c(0.5 * math.sqrt(dt)) * sl * (c(RHO_RL) * zr[k - 1] + c(rl) * zl[k - 1])
        yy = y + w
        y = (yy + aadc.math.sqrt(yy * yy + c(4.0 * cA * dt) * (c(0.5 * KAP_L * THE_L) - c(0.125) * sl * sl))) / c(2.0 * cA)
        x = x * c(ea) + sr * c(ou_sd) * zr[k - 1]
        D = aadc.math.exp(logD)
        pay_k = c(REC * NOTIONAL) * D
        if markup:
            hit = dflt.step(k, A_prev=Lam, rate=lam, dt=dt, threshold=E, alive=alive, pay=pay_k, on_date=(k in DATE_STEPS))
        else:
            hit = aadc.iif(Lam + lam * c(dt) >= E, one, zero)
        P = P + alive * hit * pay_k
        alive = alive * (one - hit)
        Lam = Lam + lam * c(dt)
        V = sr * sr * c((1 - math.exp(-2 * a * t)) / (2 * a))
        logP = -r0 * c(TAU) - c(B) * x - c(0.5 * B * B) * V - c(B / (2 * a * a) * (1 - math.exp(-a * t)) ** 2) * sr * sr
        L = (aadc.math.exp(-logP) - one) / c(TAU)
        acc = acc + aadc.iif(L > c(LO), one, zero) * aadc.iif(L < c(HI), one, zero)
        if k in DATE_STEPS:
            P = P + alive * c(CPN_Q * NOTIONAL / QUARTER) * acc * D
            acc = zero
    P = P + alive * c(NOTIONAL) * aadc.math.exp(logD)
    if markup:
        dflt.finalize()
    rP = P.mark_as_output(); fn.stop_recording()
    th = [a_r0, a_sr, a_l0, a_sl]
    thv = dict(zip(th, THETA))
    if markup:
        thv.update(dflt.zero_inputs())
    return dict(fn=fn, rP=rP, za=za, th=th, thv=thv, defaults=[dflt] if markup else [])


def driver(t, **kw):
    d = CorrectionDriverVec2(t["fn"], t["rP"], t["za"], t["th"], num_threads=THREADS, indicator_threads=8,
                             defaults=t["defaults"], **kw)
    d.precompute_directions(t["thv"])
    return d


def run(d, Z):
    with contextlib.redirect_stdout(io.StringIO()):
        return d.run(Z)


def reference(t, millions, chunk=250_000):
    """CRN central bumps on the same paths: rate Greeks with h/2, h, 2h, credit Greeks with h."""
    steps = {0: [H_REF[0] / 2, H_REF[0], 2 * H_REF[0]], 1: [H_REF[1] / 2, H_REF[1], 2 * H_REF[1]], 2: [H_REF[2]], 3: [H_REF[3]]}
    acc = {(k, h): [0.0, 0.0] for k in steps for h in steps[k]}; n = 0; rng = np.random.default_rng(20261125)
    while n < millions * 1_000_000:
        Z = rng.standard_normal((chunk, len(t["za"]))); base = {a: np.ascontiguousarray(Z[:, j]) for j, a in enumerate(t["za"])}
        for k in steps:
            a = t["th"][k]
            for h in steps[k]:
                r = []
                for sg in (1, -1):
                    inp = dict(base); inp.update(t["thv"]); inp[a] = t["thv"][a] + sg * h
                    r.append(np.asarray(aadc.evaluate(funcs=t["fn"], request={t["rP"]: []}, inputs=inp, workers=THREADS)[0][t["rP"]]))
                dd = (r[0] - r[1]) / (2 * h); acc[(k, h)][0] += dd.sum(); acc[(k, h)][1] += (dd * dd).sum()
        n += chunk
    out = {}
    for (k, h), (s1, s2) in acc.items():
        m = s1 / n; out[(k, h)] = (m, math.sqrt(max(s2 / n - m * m, 0.0) / n))
    return steps, out, n


if __name__ == "__main__":
    t = record(True)
    if int(os.environ.get("REF_M", "0")):
        t0 = time.perf_counter(); steps, out, n = reference(t, int(os.environ["REF_M"]))
        print(f"SIG_L={SIG_L}: CRN bump reference, {n/1e6:.0f}M paths ({time.perf_counter()-t0:.0f}s)", flush=True)
        for k in steps:
            line = f"  {NAMES[k]:17s}" + "".join(f" | h={h:g}: {out[(k, h)][0]:+.4f}±{out[(k, h)][1]:.4f}" for h in steps[k])
            if len(steps[k]) == 3:
                f1, f2 = out[(k, steps[k][0])][0], out[(k, steps[k][1])][0]
                line += f" | Richardson {f1 - (f2 - f1) / 3:+.4f}"
            print(line, flush=True)
    d = driver(t)
    print(f"SIG_L={SIG_L} (2 kappa theta / sigma^2 = {2*KAP_L*THE_L/SIG_L**2:.2f}): coins {d.coins.diagnostics}; "
          f"delta indicators {d.n_ind - len(d.coin_skip)}", flush=True)
    if os.environ.get("SOBOL") == "1":
        from rqmc import CorrectionDriverRQMC
        R, NP = int(os.environ.get("R", "8")), int(os.environ.get("N", "4096"))
        q = CorrectionDriverRQMC(t["fn"], t["rP"], t["za"], t["th"], num_threads=THREADS, pilot_seed=999)
        with contextlib.redirect_stdout(io.StringIO()):
            q.precompute_directions(t["thv"])
        est = {s: [] for s in ("MC", "Sobol")}; tm = {s: 0.0 for s in est}
        for r in range(R):
            for s in ("MC", "Sobol"):
                Z = np.random.default_rng(700 + r).standard_normal((NP, len(t["za"]))) if s == "MC" else q.sample(NP, r)
                t0 = time.perf_counter(); o = run(d, Z); tm[s] += time.perf_counter() - t0
                est[s].append(np.asarray(o["total"]))
                print(f"    rep {r} {s:5s} {time.perf_counter()-t0:6.1f}s " + " ".join(f"{x:+.4f}" for x in o["total"]), flush=True)
        print(f"  {R} x {NP} paths, MC vs Sobol (option B); work = time x variance of one estimate")
        for k, nm in enumerate(NAMES):
            a, b = np.array(est["MC"])[:, k], np.array(est["Sobol"])[:, k]
            print(f"  {nm:17s} MC {a.mean():+10.4f} std {a.std(ddof=1):.4f} | Sobol {b.mean():+10.4f} std {b.std(ddof=1):.4f} "
                  f"(gain {a.std(ddof=1)/b.std(ddof=1):5.1f}x) | {tm['Sobol']/R:5.1f}s/est | work MC {tm['MC']/R*a.var(ddof=1):.3g} "
                  f"Sobol {tm['Sobol']/R*b.var(ddof=1):.3g}", flush=True)
    else:
        M, S = int(os.environ.get("M", "10000")), int(os.environ.get("SEEDS", "4"))
        res, tm = [], 0.0
        for s in range(S):
            Z = np.random.default_rng(300 + s).standard_normal((M, len(t["za"])))
            t0 = time.perf_counter(); o = run(d, Z); tm += time.perf_counter() - t0; res.append(np.asarray(o["total"]))
            print(f"    seed {s} {time.perf_counter()-t0:6.1f}s lost {o['newton_unconverged']}/{o['jump_unresolved']} "
                  + " ".join(f"{x:+.4f}" for x in o["total"]), flush=True)
        res = np.array(res)
        print(f"  {S} x {M} paths: " + "  ".join(f"{nm} {res[:, k].mean():+.4f}±{res[:, k].std(ddof=1)/math.sqrt(S):.4f}"
                                                  for k, nm in enumerate(NAMES)) + f"; {tm/S:.0f}s per run")
