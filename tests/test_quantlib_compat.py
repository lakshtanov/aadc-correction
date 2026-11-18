"""
test_quantlib_compat.py — the correction driver on models taken from QuantLib, recorded on the AADC tape.

Needs the AADC build of QuantLib:  pip install aadc-quantlib-tracing   (skipped if it is not installed).
QuantLib's own processes generate the paths (process.evolve on active normals); indicators come from the switch
registry; parameters enter through QuantLib quotes / constructor arguments as tape inputs. Each case compares
CorrectionDriverVec2 (MC, seeds) with a central bump with common random numbers on the same tape and requires no
lost root-finding pairs.

  1. Down-and-out call, BlackScholesMertonProcess (spot quote, vol quote).
  2. Digital call, HestonProcess with QuantLib's default QE discretisation (spot quote, vol of vol). QE branches
     on the state inside each step: those switches are discontinuities of the grid model itself and are corrected
     too.
  3. Down-and-out call, GBM spot driven by a HullWhiteProcess short rate (spot, Hull-White sigma).
  4. Known issue (not a driver problem; aadc-quantlib-tracing 1.41.4): on the AADC tape HestonProcess FullTruncation /
     PartialTruncation give NaN states on paths where the variance went negative, while the same QuantLib code
     without the tape is finite; Reflection and QE are identical to plain QuantLib. A payoff that compares the
     state (e.g. a digital) then silently becomes wrong. The test prints the NaN share and the digital
     on the tape vs plain QuantLib, and requires QE / Reflection to be finite.

Usage: python tests/test_quantlib_compat.py   (about 5-10 minutes on 10 cores)
"""
import sys, os, io, math, contextlib
import numpy as np
import aadc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from correction_driver_vec2 import CorrectionDriverVec2
try:
    import aadc_quantlib_tracing as ql
except ImportError:
    print("aadc-quantlib-tracing not installed: skipped"); sys.exit(0)

THREADS = int(os.environ.get("THREADS", "10"))
TODAY = ql.Date(17, 11, 2026); ql.Settings.instance().evaluationDate = TODAY
DC = ql.Actual365Fixed()
c = aadc.idouble


def flat(r):
    return ql.YieldTermStructureHandle(ql.FlatForward(TODAY, r, DC))


def evaluate(fn, rP, za, thv, Z, request=None):
    inp = {a: np.ascontiguousarray(Z[:, j]) for j, a in enumerate(za)}; inp.update(thv)
    return aadc.evaluate(funcs=fn, request=request or {rP: []}, inputs=inp, workers=THREADS)


def bump_reference(fn, rP, za, th, thv, h, paths=2_000_000, chunk=250_000, seed=20261117):
    rng = np.random.default_rng(seed); s1 = np.zeros(len(th)); s2 = np.zeros(len(th)); n = 0
    while n < paths:
        Z = rng.standard_normal((chunk, len(za)))
        for k, a in enumerate(th):
            r = []
            for sg in (1, -1):
                v = dict(thv); v[a] = thv[a] + sg * h[k]
                r.append(np.asarray(evaluate(fn, rP, za, v, Z)[0][rP]))
            g = (r[0] - r[1]) / (2 * h[k]); s1[k] += g.sum(); s2[k] += (g * g).sum()
        n += chunk
    m = s1 / n
    return m, np.sqrt((s2 / n - m * m) / n)


def check(name, fn, rP, za, th, thv, h, names, M=50_000, seeds=4):
    est = []; cnt = {"newton_unconverged": 0, "newton_far": 0, "jump_unresolved": 0}
    for s in range(seeds):
        Z = np.random.default_rng(100 + s).standard_normal((M, len(za)))
        d = CorrectionDriverVec2(fn, rP, za, th, num_threads=THREADS)
        with contextlib.redirect_stdout(io.StringIO()):
            d.precompute_directions(thv); o = d.run(Z)
        est.append(o["total"]); [cnt.__setitem__(k, cnt[k] + o.get(k, 0)) for k in cnt]
    est = np.array(est); m, se = est.mean(0), est.std(0, ddof=1) / math.sqrt(seeds)
    ref, rse = bump_reference(fn, rP, za, th, thv, h)
    print(f"\n{name}: {len(fn.cmp_switches())} switches in the registry; counters {cnt}")
    ok = cnt["newton_unconverged"] == 0 and cnt["jump_unresolved"] == 0
    for k, nm in enumerate(names):
        z = (m[k] - ref[k]) / math.hypot(se[k], rse[k])
        print(f"  {nm:14s} driver {m[k]:+.5f} ± {se[k]:.5f}   bump {ref[k]:+.5f} ± {rse[k]:.5f}   {z:+.1f} sd")
        ok &= abs(z) < 4.0
    assert ok, f"{name}: driver and bump disagree or root-finding pairs were lost"
    print("  OK")


def case_bsm_barrier(N=20, T=1.0, B=80.0, K=90.0):
    fn = aadc.Functions(); fn.start_recording(register_switches=aadc.SWITCH_JUMP)
    s0 = c(100.0); a_s = s0.mark_as_input(); vol = c(0.25); a_v = vol.mark_as_input()
    proc = ql.BlackScholesMertonProcess(ql.QuoteHandle(ql.SimpleQuote(s0)), flat(0.0), flat(0.05),
                                        ql.BlackVolTermStructureHandle(ql.BlackConstantVol(TODAY, ql.NullCalendar(), ql.QuoteHandle(ql.SimpleQuote(vol)), DC)))
    dt = T / N; z = [c(0.0) for _ in range(N)]; za = [q.mark_as_input() for q in z]
    x, alive = proc.x0(), c(1.0)
    for k in range(N):
        x = proc.evolve(k * dt, x, dt, z[k])
        alive = alive * aadc.iif(x > c(B), c(1.0), c(0.0))
    rP = (alive * aadc.math.max(x - c(K), c(0.0)) * c(math.exp(-0.05 * T))).mark_as_output(); fn.stop_recording()
    check("Down-and-out call, BlackScholesMertonProcess", fn, rP, za, [a_s, a_v], {a_s: 100.0, a_v: 0.25},
          [0.5, 0.0025], ["dV/dS0", "dV/dsigma"])


def heston_tape(scheme, payoff="digital", N=8, T=2.0):
    fn = aadc.Functions(); fn.start_recording(register_switches=aadc.SWITCH_JUMP)
    s0 = c(100.0); a_s = s0.mark_as_input(); xi = c(0.3); a_x = xi.mark_as_input()
    proc = ql.HestonProcess(flat(0.03), flat(0.0), ql.QuoteHandle(ql.SimpleQuote(s0)), 0.04, 2.0, 0.04, xi, -0.7,
                            getattr(ql.HestonProcess, scheme))
    dt = T / N; z = [c(0.0) for _ in range(2 * N)]; za = [q.mark_as_input() for q in z]
    x = proc.initialValues()
    for k in range(N):
        dw = ql.Array(2); dw[0] = z[2 * k]; dw[1] = z[2 * k + 1]
        x = proc.evolve(k * dt, x, dt, dw)
    rP = (aadc.iif(x[0] > c(100.0), c(1.0), c(0.0)) * c(math.exp(-0.03 * T))).mark_as_output(); fn.stop_recording()
    return fn, rP, za, [a_s, a_x], {a_s: 100.0, a_x: 0.3}


def case_heston_qe():
    fn, rP, za, th, thv = heston_tape("QuadraticExponentialMartingale")
    check("Digital call, HestonProcess (QE)", fn, rP, za, th, thv, [0.5, 0.005], ["dV/dS0", "dV/dxi"])


def case_hw_barrier(N=20, T=1.0, B=80.0, K=90.0, rho=-0.3):
    fn = aadc.Functions(); fn.start_recording(register_switches=aadc.SWITCH_JUMP)
    s0 = c(100.0); a_s = s0.mark_as_input(); sr = c(0.01); a_r = sr.mark_as_input()
    hw = ql.HullWhiteProcess(flat(0.05), 0.1, sr)
    dt = T / N; z = [c(0.0) for _ in range(2 * N)]; za = [q.mark_as_input() for q in z]
    logS, r, D, alive = aadc.math.log(s0), hw.x0(), c(0.0), c(1.0)
    for k in range(N):
        zs = z[2 * k]; zr = c(rho) * z[2 * k] + c(math.sqrt(1 - rho * rho)) * z[2 * k + 1]
        logS = logS + (r - c(0.5 * 0.25 ** 2)) * c(dt) + c(0.25 * math.sqrt(dt)) * zs
        r = hw.evolve(k * dt, r, dt, zr); D = D + r * c(dt)
        alive = alive * aadc.iif(aadc.math.exp(logS) > c(B), c(1.0), c(0.0))
    rP = (alive * aadc.math.exp(-D) * aadc.math.max(aadc.math.exp(logS) - c(K), c(0.0))).mark_as_output(); fn.stop_recording()
    check("Down-and-out call, GBM + HullWhiteProcess", fn, rP, za, [a_s, a_r], {a_s: 100.0, a_r: 0.01},
          [0.5, 0.0005], ["dV/dS0", "dV/dsigma_r"])


def case_truncation_known_issue(M=4000, N=8, T=2.0):
    """HestonProcess on the tape vs the same QuantLib code without the tape, same normals: state S_T and the digital."""
    print("\nKnown issue: HestonProcess truncation schemes on the AADC tape (tape vs plain QuantLib, same normals)")
    Z = np.random.default_rng(1).standard_normal((M, 2 * N)); dt = T / N
    res = {}
    for scheme in ("FullTruncation", "PartialTruncation", "Reflection", "QuadraticExponentialMartingale"):
        proc = ql.HestonProcess(flat(0.03), flat(0.0), ql.QuoteHandle(ql.SimpleQuote(100.0)), 0.04, 2.0, 0.04, 0.3, -0.7,
                                getattr(ql.HestonProcess, scheme))
        plain = np.empty(M)
        for p in range(M):
            x = proc.initialValues()
            for k in range(N):
                dw = ql.Array(2); dw[0] = float(Z[p, 2 * k]); dw[1] = float(Z[p, 2 * k + 1])
                x = proc.evolve(k * dt, x, dt, dw)
            plain[p] = float(x[0])
        fn = aadc.Functions(); fn.start_recording()
        z = [c(0.0) for _ in range(2 * N)]; za = [q.mark_as_input() for q in z]
        x = proc.initialValues()
        for k in range(N):
            dw = ql.Array(2); dw[0] = z[2 * k]; dw[1] = z[2 * k + 1]
            x = proc.evolve(k * dt, x, dt, dw)
        rS = x[0].mark_as_output(); fn.stop_recording()
        tape = np.asarray(evaluate(fn, rS, za, {}, Z)[0][rS])
        nan = float(np.mean(np.isnan(tape)))
        dig_plain, dig_tape = np.mean(plain > 100.0), np.mean(np.nan_to_num(tape, nan=-1.0) > 100.0)
        res[scheme] = nan
        print(f"  {scheme:32s} NaN S_T on the tape {nan:6.1%}   digital: tape {dig_tape:.4f} vs plain QuantLib {dig_plain:.4f}")
    assert res["Reflection"] == 0 and res["QuadraticExponentialMartingale"] == 0, "QE / Reflection must be finite on the tape"
    if res["FullTruncation"] or res["PartialTruncation"]:
        print("  -> with this aadc-quantlib-tracing version use QE (QuantLib's default) or Reflection: the truncation schemes"
              " give NaN states on the tape (a payoff that compares them, e.g. a digital, silently becomes wrong)")
    else:
        print("  truncation schemes finite on the tape: fixed in this aadc-quantlib-tracing version")

if __name__ == "__main__":
    print(f"aadc {aadc.__version__}, aadc-quantlib-tracing (QuantLib {ql.__version__})")
    case_bsm_barrier()
    case_heston_qe()
    case_hw_barrier()
    case_truncation_known_issue()
    print("\nAll QuantLib compatibility tests passed.")
