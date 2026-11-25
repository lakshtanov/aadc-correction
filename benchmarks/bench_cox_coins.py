"""
bench_cox_coins.py — credit-linked autocallable with an intensity default: the coin integration of the default
threshold (CorrectionDriverVec2(coins="auto")) vs the delta correction with the default directions (coins="off"),
MC and scrambled Sobol.

Product (T = 2, N = 104 weekly steps, r = 3%): two equities, GBM (vol 0.20 / 0.25, corr 0.5); default intensity
lambda: CIR (lambda0 = theta = 0.02, kappa = 1, vol SIG_L env, default 0.10), drift-implicit scheme for sqrt(lambda),
correlation -0.5 with asset 1 (wrong way); Lambda_k = Lambda_{k-1} + lambda_{k-1} dt; threshold E = -log Phi(-z_E) ~
Exp(1); default in the first step with Lambda_k >= E. While alive: default in step k pays 40% x 100 x min(S1_k/S1_0, 1)
at t_k; quarterly dates (every 13 steps): coupon 5% x 100 if worst > 70%, autocall (not last date) 100 if worst >= 100%;
maturity: 100 if worst >= 60%, else 100 x worst. Greeks: dV/dS1, dV/dS2, dV/dlambda0, dV/dsigma_lambda.

Usage:  OPENBLAS_NUM_THREADS=1 python benchmarks/bench_cox_coins.py            (MC, 16 x 10K paths, three modes)
        SOBOL=1 R=10 N=8192 ...      MC vs Sobol (option B), coin vs default directions
        REF_M=16 ...                 CRN central-bump reference (millions of paths) first
        SIG_L=0.25 ...               2 kappa theta / sigma^2 = 0.64 < 1 (Feller condition fails)
Modes: "coins off" (delta with the default directions on every comparison), "coin" (coins="auto", found on the tape,
two forward passes per step), "coin + markup" (coins="auto" with the CoxDefault markup: fast path).
"""
import os, sys, io, time, math, contextlib
import numpy as np
import aadc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from correction_driver_vec2 import CorrectionDriverVec2
from coins import CoxDefault

c = aadc.idouble
THREADS = int(os.environ.get("THREADS", "10"))
N, T, R_RATE = 104, 2.0, 0.03
dt = T / N
DATE_STEPS = [N // 8 * (d + 1) for d in range(8)]
S0, VOL, CORR = (100.0, 100.0), (0.20, 0.25), 0.5
LAM0, KAP_L, THE_L, RHO_L = 0.02, 1.0, 0.02, -0.5
SIG_L = float(os.environ.get("SIG_L", "0.10"))
assert 4 * KAP_L * THE_L > SIG_L ** 2
REC, NOTIONAL, CPN, CB, AB, KI = 0.4, 100.0, 0.05, 0.70, 1.00, 0.60
THETA = [S0[0], S0[1], LAM0, SIG_L]
NAMES = ["dV/dS1", "dV/dS2", "dV/dlambda0", "dV/dsigma_lambda"]
H_REF = [0.5, 0.5, 0.004, 0.02]


SMOOTH_SCALE = {"default": THE_L * dt, "equity": VOL[0] * math.sqrt(13 * dt)}  # per-step hazard; sd of S/S0 between dates


def step_fn(smooth, kind):
    """1{x >= 0}, or with smoothing (benchmarks/bench_smoothing_paths.py) a logistic of width smooth x SMOOTH_SCALE[kind]."""
    if smooth is None:
        return lambda x: aadc.iif(x >= c(0.0), c(1.0), c(0.0))
    w = 2.0 * smooth * SMOOTH_SCALE[kind]
    return lambda x: c(0.5) * (c(1.0) + aadc.math.tanh(x / c(w)))


def record(markup=False, smooth=None, conditional=False):
    """Tape; inputs: S1, S2, lambda0, sigma_lambda, then normals z1 (N), z2 (N), z_lambda (N), z_E.
    smooth: None (indicators) or eps: every indicator replaced by a logistic of width eps x SMOOTH_SCALE.
    conditional: the default threshold integrated out exactly -- the indicator of a default in step k replaced by its
    conditional probability given survival, 1 - exp(-lambda dt) (the payoff is linear in the alive flag), so alive is
    the survival probability exp(-Lambda_k) and z_E drops out of the payoff."""
    H_def, H_eq = step_fn(smooth, "default"), step_fn(smooth, "equity")
    fn = aadc.Functions(); fn.start_recording(register_switches=aadc.SWITCH_JUMP)
    s1 = c(S0[0]); a_s1 = s1.mark_as_input()
    s2 = c(S0[1]); a_s2 = s2.mark_as_input()
    l0 = c(LAM0); a_l0 = l0.mark_as_input()
    sl = c(SIG_L); a_sl = sl.mark_as_input()
    z = [c(0.0) for _ in range(3 * N + 1)]; za = [x.mark_as_input() for x in z]
    z1, z2, zl, zE = z[:N], z[N:2 * N], z[2 * N:3 * N], z[3 * N]
    one, zero = c(1.0), c(0.0)
    E = -aadc.math.log(aadc.math.cdf_normal(-zE))
    S1, S2, y, Lam, alive, P = s1, s2, aadc.math.sqrt(l0), zero, one, zero
    cA = 1.0 + 0.5 * KAP_L * dt; sq = math.sqrt(dt)
    rc, rl = math.sqrt(1 - CORR ** 2), math.sqrt(1 - RHO_L ** 2)
    dflt = CoxDefault() if markup else None
    for k in range(1, N + 1):
        lam = y * y
        a = y + c(0.5 * sq) * sl * (c(RHO_L) * z1[k - 1] + c(rl) * zl[k - 1])
        y = (a + aadc.math.sqrt(a * a + c(4.0 * cA * dt) * (c(0.5 * KAP_L * THE_L) - c(0.125) * sl * sl))) / c(2.0 * cA)
        S1 = S1 * aadc.math.exp(c((R_RATE - 0.5 * VOL[0] ** 2) * dt) + c(VOL[0] * sq) * z1[k - 1])
        S2 = S2 * aadc.math.exp(c((R_RATE - 0.5 * VOL[1] ** 2) * dt) + c(VOL[1] * sq) * (c(CORR) * z1[k - 1] + c(rc) * z2[k - 1]))
        df = math.exp(-R_RATE * k * dt)
        pay_k = c(REC * NOTIONAL * df) * aadc.math.min(S1 / c(S0[0]), one)
        if conditional:
            hit = one - aadc.math.exp(-lam * c(dt))
        elif markup:
            hit = dflt.step(k, A_prev=Lam, rate=lam, dt=dt, threshold=E, alive=alive, pay=pay_k, on_date=(k in DATE_STEPS))
        else:
            hit = H_def(Lam + lam * c(dt) - E)
        P = P + alive * hit * pay_k
        alive = alive * (one - hit)
        Lam = Lam + lam * c(dt)
        if k in DATE_STEPS:
            worst = aadc.math.min(S1 / c(S0[0]), S2 / c(S0[1]))
            P = P + alive * H_eq(worst - c(CB)) * c(CPN * NOTIONAL * df)
            if k < N:
                ac = H_eq(worst - c(AB))
                P = P + alive * ac * c(NOTIONAL * df)
                alive = alive * (one - ac)
            else:
                above = H_eq(worst - c(KI))
                P = P + alive * (above * c(NOTIONAL) + (one - above) * c(NOTIONAL) * worst) * c(df)
    if markup:
        dflt.finalize()
    rP = P.mark_as_output(); fn.stop_recording()
    th = [a_s1, a_s2, a_l0, a_sl]
    thv = dict(zip(th, THETA))
    if markup:
        thv.update(dflt.zero_inputs())
    return dict(fn=fn, rP=rP, za=za, th=th, thv=thv, defaults=[dflt] if markup else [])


def driver(t, coins):
    d = CorrectionDriverVec2(t["fn"], t["rP"], t["za"], t["th"], num_threads=THREADS, coins=coins, defaults=t["defaults"])
    d.precompute_directions(t["thv"])
    return d


def run(d, Z):
    with contextlib.redirect_stdout(io.StringIO()):
        return d.run(Z)


def reference(t, millions):
    rng = np.random.default_rng(20261123); s1 = np.zeros(4); s2 = np.zeros(4); n = 0
    while n < millions * 1_000_000:
        Z = rng.standard_normal((500_000, len(t["za"]))); base = {a: np.ascontiguousarray(Z[:, j]) for j, a in enumerate(t["za"])}
        for k, (a, h) in enumerate(zip(t["th"], H_REF)):
            r = []
            for sg in (1, -1):
                inp = dict(base); inp.update(t["thv"]); inp[a] = t["thv"][a] + sg * h
                r.append(aadc.evaluate(funcs=t["fn"], request={t["rP"]: []}, inputs=inp, workers=THREADS)[0][t["rP"]])
            dd = (r[0] - r[1]) / (2 * h); s1[k] += dd.sum(); s2[k] += (dd * dd).sum()
        n += len(Z)
    m = s1 / n
    return m, np.sqrt((s2 / n - m * m) / n), n


if __name__ == "__main__":
    plain, marked = record(False), record(True)
    if int(os.environ.get("REF_M", "0")):
        t0 = time.perf_counter(); ref, se, n = reference(plain, int(os.environ["REF_M"]))
        print(f"SIG_L={SIG_L}: CRN bump {n/1e6:.0f}M paths ({time.perf_counter()-t0:.0f}s): "
              + "  ".join(f"{nm} {a:+.4f}±{b:.4f}" for nm, a, b in zip(NAMES, ref, se)), flush=True)
    modes = {"coins off": driver(plain, "off"), "coin": driver(plain, "auto"), "coin + markup": driver(marked, "auto")}
    print(f"SIG_L={SIG_L} (2 kappa theta / sigma^2 = {2*KAP_L*THE_L/SIG_L**2:.2f}); coins found: "
          f"{modes['coin'].coins.diagnostics}; with markup: {modes['coin + markup'].coins.diagnostics}", flush=True)
    if os.environ.get("SOBOL") == "1":
        from rqmc import CorrectionDriverRQMC
        R, NP = int(os.environ.get("R", "10")), int(os.environ.get("N", "8192"))
        q = CorrectionDriverRQMC(plain["fn"], plain["rP"], plain["za"], plain["th"], num_threads=THREADS, pilot_seed=999)
        with contextlib.redirect_stdout(io.StringIO()):
            q.precompute_directions(plain["thv"])
        use = {"coins off": modes["coins off"], "coin + markup": modes["coin + markup"]}
        est = {(m, s): [] for m in use for s in ("MC", "Sobol")}; tm = {k: 0.0 for k in est}
        for r in range(R):
            for s in ("MC", "Sobol"):
                Z = np.random.default_rng(700 + r).standard_normal((NP, len(plain["za"]))) if s == "MC" else q.sample(NP, r)
                for m, d in use.items():
                    t0 = time.perf_counter(); o = run(d, Z); tm[(m, s)] += time.perf_counter() - t0
                    est[(m, s)].append(np.asarray(o["total"]))
        print(f"  {R} x {NP} paths, MC vs Sobol (option B); work = time x variance of one estimate")
        for k, nm in enumerate(NAMES):
            for m in use:
                a, b = np.array(est[(m, "MC")])[:, k], np.array(est[(m, "Sobol")])[:, k]
                ta, tb = tm[(m, "MC")] / R, tm[(m, "Sobol")] / R
                print(f"  {nm:17s} {m:13s} MC {a.mean():+9.4f} std {a.std(ddof=1):.4f} | Sobol {b.mean():+9.4f} std "
                      f"{b.std(ddof=1):.4f} (gain {a.std(ddof=1)/b.std(ddof=1):4.1f}x) | {ta:5.1f}s/est | work Sobol {tb*b.var(ddof=1):.3g}", flush=True)
    else:
        M, S = int(os.environ.get("M", "10000")), int(os.environ.get("SEEDS", "16"))
        res = {m: [] for m in modes}; tm = {m: 0.0 for m in modes}; lost = {m: 0 for m in modes}
        for s in range(S):
            Z = np.random.default_rng(100 + s).standard_normal((M, len(plain["za"])))
            for m, d in modes.items():
                t0 = time.perf_counter(); o = run(d, Z); tm[m] += time.perf_counter() - t0
                res[m].append(np.asarray(o["total"])); lost[m] += o["newton_unconverged"] + o["jump_unresolved"]
        print(f"  {S} x {M} paths, lost pairs {lost}")
        for k, nm in enumerate(NAMES):
            print(f"  {nm:17s} " + " | ".join(f"{m} {np.mean(np.array(res[m])[:, k]):+9.4f}±"
                                            f"{np.std(np.array(res[m])[:, k], ddof=1)/math.sqrt(S):.4f}" for m in modes))
        print("  time per run: " + ", ".join(f"{m} {t/S:.1f}s" for m, t in tm.items()))
