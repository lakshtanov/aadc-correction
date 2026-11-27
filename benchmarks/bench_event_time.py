"""
bench_event_time.py — the event-time correction of Kovalov & Naumann (module event_time.py) against the delta correction
of this repository, on the products with an accumulator event: benchmark 4 (credit-linked autocallable, default when
int lambda dt exceeds an exponential threshold), benchmark 5 (range accrual on a 3M LGM rate with the same default) and a
timer option (Heston; the event is the realised variance budget, a fixed threshold -- no coin).

Methods (the same paths for all; MC and scrambled Sobol, option B with 512 pilot paths):
  correction            this repository: benchmarks 4/5 CorrectionDriverVec2(coins="auto") (the threshold integrated
                        exactly, derivative of the grid model); timer: the delta correction on every step (grid model)
  event-time, 1st       EventTimeDriver(monitoring="continuous", generator="drift"): the Kovalov-Naumann surrogate
  event-time, 2nd Stein generator="stein": + the Ito term by Stein's identity
  event-time, 2nd quad  generator="quadrature1d": the Ito term exactly (1-D Gauss-Hermite, 9 nodes)
All event-time variants use the delta correction on the other discontinuities (dates, coupons, corridor).
Output per Greek: mean +- s.e. over R repetitions, std of one estimate, time per estimate, the paired difference to the
correction (same paths). The event-time methods target the continuous event time; the correction the grid model.

Usage: OPENBLAS_NUM_THREADS=1 PRODUCT=credit|lgm|timer R=16 NPATH=8192 python benchmarks/bench_event_time.py
       (timer: XI=0.3|0.5, NSTEP=52|208; NPATH must be a power of two for Sobol; OUT=<json>)
"""
import os, sys, io, math, time, json, contextlib, warnings
import numpy as np
import aadc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from event_time import AccumulatorEvent, EventTimeDriver
from correction_driver_vec2 import CorrectionDriverVec2
from rqmc import CorrectionDriverRQMC

c = aadc.idouble


# ── benchmark 4: the credit-linked autocallable of bench_cox_coins.py, default marked as an accumulator event ──
import bench_cox_coins as CA


def record_credit():
    N, dt = CA.N, CA.dt
    fn = aadc.Functions(); fn.start_recording(register_switches=aadc.SWITCH_JUMP)
    s1 = c(CA.S0[0]); a_s1 = s1.mark_as_input()
    s2 = c(CA.S0[1]); a_s2 = s2.mark_as_input()
    l0 = c(CA.LAM0); a_l0 = l0.mark_as_input()
    sl = c(CA.SIG_L); a_sl = sl.mark_as_input()
    z = [c(0.0) for _ in range(3 * N + 1)]; za = [x.mark_as_input() for x in z]
    z1, z2, zl, zE = z[:N], z[N:2 * N], z[2 * N:3 * N], z[3 * N]
    one, zero = c(1.0), c(0.0)
    # THRESH=lognormal: a threshold of another law (a test of the generic coin path), E = exp(z_E - 3)
    E = -aadc.math.log(aadc.math.cdf_normal(-zE))
    S1, S2, y, Lam, alive, P = s1, s2, aadc.math.sqrt(l0), zero, one, zero
    cA = 1.0 + 0.5 * CA.KAP_L * dt; sq = math.sqrt(dt)
    rc = math.sqrt(1 - CA.CORR ** 2); rl = math.sqrt(1 - CA.RHO_L ** 2)
    ev = AccumulatorEvent("default")
    for k in range(1, N + 1):
        lam = y * y
        a = y + c(0.5 * sq) * sl * (c(CA.RHO_L) * z1[k - 1] + c(rl) * zl[k - 1])
        y = (a + aadc.math.sqrt(a * a + c(4.0 * cA * dt) * (c(0.5 * CA.KAP_L * CA.THE_L) - c(0.125) * sl * sl))) / c(2.0 * cA)
        S1 = S1 * aadc.math.exp(c((CA.R_RATE - 0.5 * CA.VOL[0] ** 2) * dt) + c(CA.VOL[0] * sq) * z1[k - 1])
        S2 = S2 * aadc.math.exp(c((CA.R_RATE - 0.5 * CA.VOL[1] ** 2) * dt) + c(CA.VOL[1] * sq) * (c(CA.CORR) * z1[k - 1] + c(rc) * z2[k - 1]))
        df = math.exp(-CA.R_RATE * k * dt)
        pay_k = c(CA.REC * CA.NOTIONAL * df) * aadc.math.min(S1 / c(CA.S0[0]), one)
        hit, first = ev.step(k, A_prev=Lam, rate=lam, dt=dt, threshold=E, alive=alive, pay=pay_k,
                             normals=[za[k - 1], za[N + k - 1], za[2 * N + k - 1]], on_date=(k in CA.DATE_STEPS),
                             pay_prev=c(CA.REC * CA.NOTIONAL) * aadc.math.min(s1 / c(CA.S0[0]), one) if k == 1 else None)
        P = P + first * pay_k
        alive = alive * (one - hit)
        Lam = Lam + lam * c(dt)
        if k in CA.DATE_STEPS:
            worst = aadc.math.min(S1 / c(CA.S0[0]), S2 / c(CA.S0[1]))
            P = P + alive * aadc.iif(worst > c(CA.CB), one, zero) * c(CA.CPN * CA.NOTIONAL * df)
            if k < N:
                ac = aadc.iif(worst >= c(CA.AB), one, zero)
                P = P + alive * ac * c(CA.NOTIONAL * df)
                alive = alive * (one - ac)
            else:
                above = aadc.iif(worst >= c(CA.KI), one, zero)
                P = P + alive * (above * c(CA.NOTIONAL) + (one - above) * c(CA.NOTIONAL) * worst) * c(df)
    ev.finalize()
    rP = P.mark_as_output(); fn.stop_recording()
    th = [a_s1, a_s2, a_l0, a_sl]
    return fn, rP, za, th, dict(zip(th, CA.THETA)), ev


# ── benchmark 5: the LGM range accrual of bench_lgm_range_accrual.py, default marked as an accumulator event ──
import bench_lgm_range_accrual as LGM


def record_lgm(declare=False):
    """Tape; inputs: theta, then normals z_r (N), z_lam (N), z_E (1). Returns (fn, rP, za, th, thv, ev)."""
    N, dt, DATE_STEPS, QUARTER = LGM.N, LGM.dt, LGM.DATE_STEPS, LGM.QUARTER
    R0, A_MR, SIG_R, TAU, LAM0, KAP_L, THE_L, RHO_RL, SIG_L = LGM.R0, LGM.A_MR, LGM.SIG_R, LGM.TAU, LGM.LAM0, LGM.KAP_L, LGM.THE_L, LGM.RHO_RL, LGM.SIG_L
    REC, NOTIONAL, CPN_Q, LO, HI, THETA = LGM.REC, LGM.NOTIONAL, LGM.CPN_Q, LGM.LO, LGM.HI, LGM.THETA
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
    ev = AccumulatorEvent("default", threshold_normal=za[2 * N] if declare else None)
    for k in range(1, N + 1):
        t_prev, t = (k - 1) * dt, k * dt
        # short rate over the step (left point) and the path discount factor
        shift_prev = sr * sr / c(2 * a * a) * c((1 - math.exp(-a * t_prev)) ** 2)
        r_prev = r0 + shift_prev + x
        logD = logD - r_prev * c(dt)
        # credit: CIR intensity (left point accumulator), correlated with the rate normal of the step
        lam = y * y
        w = c(0.5 * math.sqrt(dt)) * sl * (c(RHO_RL) * zr[k - 1] + c(rl) * zl[k - 1])
        yy = y + w
        y = (yy + aadc.math.sqrt(yy * yy + c(4.0 * cA * dt) * (c(0.5 * KAP_L * THE_L) - c(0.125) * sl * sl))) / c(2.0 * cA)
        # rates: exact OU step of the LGM state
        x = x * c(ea) + sr * c(ou_sd) * zr[k - 1]
        D = aadc.math.exp(logD)
        pay_k = c(REC * NOTIONAL) * D
        hit, first = ev.step(k, A_prev=Lam, rate=lam, dt=dt, threshold=E, alive=alive, pay=pay_k,
                             normals=[za[k - 1], za[N + k - 1]], on_date=(k in DATE_STEPS),
                             pay_prev=c(REC * NOTIONAL) if k == 1 else None)
        P = P + first * pay_k
        alive = alive * (one - hit)
        Lam = Lam + lam * c(dt)
        # 3M rate at t from the LGM zero bond, flat initial curve: P0(t+tau)/P0(t) = exp(-R0 tau)
        # P(t,t+tau) = P0(t+tau)/P0(t) exp(-B x - B^2 V(t)/2 - B sigma^2/(2a^2) (1-e^{-at})^2)
        # (= 1/2 [V(t,T) - V(0,T) + V(0,t)] - B x of Brigo-Mercurio G1++; checked to 1e-15)
        V = sr * sr * c((1 - math.exp(-2 * a * t)) / (2 * a))
        logP = -r0 * c(TAU) - c(B) * x - c(0.5 * B * B) * V - c(B / (2 * a * a) * (1 - math.exp(-a * t)) ** 2) * sr * sr
        L = (aadc.math.exp(-logP) - one) / c(TAU)
        inside = aadc.iif(L > c(LO), one, zero) * aadc.iif(L < c(HI), one, zero)
        acc = acc + inside
        if k in DATE_STEPS:
            P = P + alive * c(CPN_Q * NOTIONAL / QUARTER) * acc * D
            acc = zero
    P = P + alive * c(NOTIONAL) * aadc.math.exp(logD)
    ev.finalize()
    rP = P.mark_as_output(); fn.stop_recording()
    th = [a_r0, a_sr, a_l0, a_sl]
    return fn, rP, za, th, dict(zip(th, THETA)), ev


# ── timer option (Heston, drift-implicit sqrt(v) scheme): pays (S_tau - K)^+ when the realised variance reaches V* ──
class TB:
    S0, K, R = 100.0, 100.0, 0.03
    V0, THV, KAPV, RHO = 0.04, 0.04, 2.0, -0.7
    XI = float(os.environ.get("XI", "0.5"))
    RHOC = math.sqrt(1 - RHO ** 2)
    VSTAR, T, N = 0.04, 2.0, int(os.environ.get("NSTEP", "52"))
    dt = T / N
    THETA = [S0, V0, THV]
    NAMES = ["dV/dS0", "dV/dv0", "dV/dtheta"]


def record_timer():
    N, dt = TB.N, TB.dt
    fn = aadc.Functions(); fn.start_recording(register_switches=aadc.SWITCH_JUMP)
    s0 = c(TB.S0); a_s = s0.mark_as_input()
    v0 = c(TB.V0); a_v = v0.mark_as_input()
    thv = c(TB.THV); a_t = thv.mark_as_input()
    zs = [c(0.0) for _ in range(2 * N)]; za = [z.mark_as_input() for z in zs]
    zv, zx = zs[:N], zs[N:]
    one, zero = c(1.0), c(0.0)
    S, A, alive, P = s0, zero, one, zero
    y = aadc.math.sqrt(v0); cA = 1.0 + 0.5 * TB.KAPV * dt
    ev = AccumulatorEvent("variance budget")
    for i in range(N):
        vp = y * y; sv = y * c(math.sqrt(dt))
        a = y + c(0.5 * TB.XI * math.sqrt(dt)) * zv[i]
        y = (a + aadc.math.sqrt(a * a + c(4.0 * cA * dt) * (c(0.5 * TB.KAPV) * thv - c(0.125 * TB.XI * TB.XI)))) / c(2.0 * cA)
        S = S * aadc.math.exp((c(TB.R) - c(0.5) * vp) * c(dt) + sv * (c(TB.RHO) * zv[i] + c(TB.RHOC) * zx[i]))
        pay = c(math.exp(-TB.R * (i + 1) * dt)) * aadc.math.max(S - c(TB.K), zero)
        hit, first = ev.step(i + 1, A_prev=A, rate=vp, dt=dt, threshold=c(TB.VSTAR), alive=alive, pay=pay,
                             normals=[za[i], za[N + i]], on_date=False)
        P = P + first * pay
        alive = alive * (one - hit)
        A = A + vp * c(dt)
    P = P + alive * c(math.exp(-TB.R * TB.T)) * aadc.math.max(S - c(TB.K), zero)
    ev.finalize()
    rP = P.mark_as_output(); fn.stop_recording()
    th = [a_s, a_v, a_t]
    return fn, rP, za, th, dict(zip(th, TB.THETA)), ev


def methods(prod):
    if prod == "credit":
        t = CA.record(True); names = CA.NAMES
        ref = CorrectionDriverVec2(t["fn"], t["rP"], t["za"], t["th"], num_threads=10, coins="auto", defaults=t["defaults"])
        with contextlib.redirect_stdout(io.StringIO()):
            ref.precompute_directions(t["thv"])
        fn, rP, za, th, thv, ev = record_credit()
    elif prod == "lgm":
        t = LGM.record(True); names = LGM.NAMES
        ref = CorrectionDriverVec2(t["fn"], t["rP"], t["za"], t["th"], num_threads=10, coins="auto", defaults=t["defaults"], indicator_threads=8)
        with contextlib.redirect_stdout(io.StringIO()):
            ref.precompute_directions(t["thv"])
        fn, rP, za, th, thv, ev = record_lgm()
    else:
        fn, rP, za, th, thv, ev = record_timer(); names = TB.NAMES
        ref = EventTimeDriver(fn, rP, za, th, events=[ev], monitoring="discrete", coins="off")
        with contextlib.redirect_stdout(io.StringIO()):
            ref.precompute_directions(thv)
        t = dict(fn=fn, rP=rP, za=za, th=th, thv=thv)
    out = {"correction": ref}
    for nm, gen in (("event-time, 1st (drift)", "drift"), ("event-time, 2nd (Stein)", "stein"), ("event-time, 2nd (quadrature)", "quadrature1d")):
        h = EventTimeDriver(fn, rP, za, th, events=[ev], monitoring="continuous", generator=gen, indicator_threads=8)
        with contextlib.redirect_stdout(io.StringIO()):
            h.precompute_directions(thv)
        out[nm] = h
    return t, (fn, rP, za, th, thv), names, out


if __name__ == "__main__":
    PROD = os.environ.get("PRODUCT", "credit"); R, N = int(os.environ.get("R", "16")), int(os.environ.get("NPATH", "8192"))
    t, (fn, rP, za, th, thv), names, M = methods(PROD)
    # Sobol rotation (option B, 512 pilot paths) from the event tape (same inputs as the reference tape)
    q = CorrectionDriverRQMC(fn, rP, za, th, num_threads=10, pilot_seed=999, n_pilot=512)
    with contextlib.redirect_stdout(io.StringIO()):
        q.precompute_directions(dict(M["event-time, 1st (drift)"].theta_values))
    est = {(m, s): [] for m in M for s in ("MC", "Sobol")}; tm = {k: 0.0 for k in est}
    for r in range(R):
        for s in ("MC", "Sobol"):
            Z = np.random.default_rng(700 + r).standard_normal((N, len(za))) if s == "MC" else q.sample(N, r)
            for m, d in M.items():
                t0 = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                    warnings.simplefilter("ignore"); o = d.run(Z)
                tm[(m, s)] += time.perf_counter() - t0; est[(m, s)].append(np.asarray(o["total"]))
    print(f"{PROD}: {R} x {N} paths, MC and Sobol (option B, 512 pilot paths)" + (f", XI={TB.XI}, N={TB.N}" if PROD == "timer" else ""))
    res = dict(product=PROD, R=R, N=N, names=names, rows={})
    for k, nm in enumerate(names):
        for m in M:
            line = f"  {nm:17s} {m:30s}"
            for s in ("MC", "Sobol"):
                x = np.array(est[(m, s)])[:, k]; tt = tm[(m, s)] / R
                dd = x - np.array(est[("correction", s)])[:, k]
                line += f" | {s} {x.mean():+.5g}±{x.std(ddof=1)/math.sqrt(R):.2g} sd {x.std(ddof=1):.3g} {tt:.2f}s, minus corr. {dd.mean():+.3g}±{dd.std(ddof=1)/math.sqrt(R):.2g}"
                res["rows"].setdefault(m, {}).setdefault(s, {})[nm] = dict(mean=float(x.mean()), std=float(x.std(ddof=1)), time=tt,
                                                                        diff=float(dd.mean()), diff_se=float(dd.std(ddof=1) / math.sqrt(R)))
            print(line, flush=True)
    if os.environ.get("OUT"):
        json.dump(res, open(os.environ["OUT"], "w"), indent=1)
