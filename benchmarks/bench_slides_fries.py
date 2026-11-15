"""
bench_slides_fries.py — reproduce the slide "Newton vs Fries: MC and Sobol".

Product: 2-asset worst-of autocallable, GBM, 4 quarterly dates, T = 1 (S0 = 100, 100; vol 0.20, 0.25; corr 0.5;
r = 0.03; coupon 5% if worst > 70%, autocall if worst > 100%). Greeks dV/dS1, dV/dS2, dV/dsigma1.

Methods, each with MC and with scrambled Sobol (automatic ordering, rqmc.CorrectionDriverRQMC's rotation):
  Newton     CorrectionDriverVec2 (δ-correction, unbiased)
  Fries w    CorrectionDriverFries with window fraction w = 1%, 5%, 10% (discretised delta, biased O(w))
N = 2^16 paths per estimate, R = 16 independent repetitions (seeds / scramblings) of every method.
Reported per Greek: |bias| vs the reference, std of one estimate, RMSE = sqrt(bias^2 + std^2), in % of |Greek|.
Reference: central bump with common random numbers on AADC, 6.4e7 paths (h and 2h as a check).

Usage: OPENBLAS_NUM_THREADS=1 python benchmarks/bench_slides_fries.py      (QUICK=1: 2^14 paths, 8M reference)
Raw results -> bench_slides_fries.json (OUT=...).
"""
import sys, os, io, json, math, contextlib
import numpy as np
import aadc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from correction_driver_vec2 import CorrectionDriverVec2
from correction_driver_fries import CorrectionDriverFries
from rqmc import CorrectionDriverRQMC

QUICK = os.environ.get("QUICK", "0") == "1"
THREADS = int(os.environ.get("THREADS", "10"))
N = 2 ** (14 if QUICK else 16)
R = 16
REF_M = 8 if QUICK else 64
WINDOWS = [0.01, 0.05, 0.10]
NAMES = ["dV/dS1", "dV/dS2", "dV/dsigma1"]
c = aadc.idouble


def tape():
    ND, S0, VOL, CORR, RATE, T = 4, (100.0, 100.0), (0.20, 0.25), 0.5, 0.03, 1.0
    dt = T / ND
    fn = aadc.Functions(); fn.start_recording()
    S1 = c(S0[0]); a1 = S1.mark_as_input()
    S2 = c(S0[1]); a2 = S2.mark_as_input()
    v1 = c(VOL[0]); av = v1.mark_as_input()
    Z = [c(0.0) for _ in range(2 * ND)]; za = [z.mark_as_input() for z in Z]
    l1, l2 = aadc.math.log(S1), aadc.math.log(S2)
    payoff, alive, g = c(0.0), c(1.0), []
    for d in range(ND):
        z1 = Z[2 * d]; z2 = c(CORR) * Z[2 * d] + c(math.sqrt(1 - CORR * CORR)) * Z[2 * d + 1]
        l1 = l1 + (c(RATE) - c(0.5) * v1 * v1) * c(dt) + v1 * c(math.sqrt(dt)) * z1
        l2 = l2 + c((RATE - 0.5 * VOL[1] ** 2) * dt) + c(VOL[1] * math.sqrt(dt)) * z2
        p1, p2 = aadc.math.exp(l1) / c(S0[0]), aadc.math.exp(l2) / c(S0[1])
        worst = aadc.iif(p1 < p2, p1, p2)
        g.append((worst - c(0.70)).mark_as_output()); g.append((worst - c(1.00)).mark_as_output())
        df = math.exp(-RATE * T * (d + 1) / ND)
        payoff = payoff + alive * aadc.iif(worst > c(0.70), c(1.0), c(0.0)) * c(5.0 * df)
        if d < ND - 1:
            ac = aadc.iif(worst > c(1.00), c(1.0), c(0.0))
            payoff = payoff + alive * ac * c(100.0 * df); alive = alive * (c(1.0) - ac)
    payoff = payoff + alive * c(100.0 * math.exp(-RATE * T))
    rP = payoff.mark_as_output(); fn.stop_recording()
    return fn, rP, za, [a1, a2, av], {a1: S0[0], a2: S0[1], av: VOL[0]}, g


def reference(fn, rP, za, th, thv):
    out = []
    for H in ((0.5, 0.5, 0.0025), (1.0, 1.0, 0.005)):
        rng = np.random.default_rng(20261115); s1 = np.zeros(3); s2 = np.zeros(3); n = 0
        while n < REF_M * 1_000_000:
            Z = rng.standard_normal((1_000_000, len(za))); base = {a: np.ascontiguousarray(Z[:, j]) for j, a in enumerate(za)}
            for k, (a, h) in enumerate(zip(th, H)):
                r = []
                for sg in (1, -1):
                    inp = dict(base); inp.update(thv); inp[a] = thv[a] + sg * h
                    r.append(aadc.evaluate(funcs=fn, request={rP: []}, inputs=inp, workers=THREADS)[0][rP])
                d = (r[0] - r[1]) / (2 * h); s1[k] += d.sum(); s2[k] += (d * d).sum()
            n += 1_000_000
        m = s1 / n; out.append((m, np.sqrt((s2 / n - m * m) / n)))
    return out


if __name__ == "__main__":
    fn, rP, za, th, thv, g = tape()
    (ref, se), (ref2, se2) = reference(fn, rP, za, th, thv)
    print(f"2-asset GBM autocallable, 4 dates; N = {N:,} paths per estimate, R = {R} repetitions; reference {REF_M}M paths")
    print("reference: " + ", ".join(f"{n} {a:+.5f}±{b:.5f}" for n, a, b in zip(NAMES, ref, se))
          + "   (h vs 2h: " + " ".join(f"{(a - b) / math.hypot(c_, d):+.1f}σ" for a, b, c_, d in zip(ref, ref2, se, se2)) + ")")
    q = CorrectionDriverRQMC(fn, rP, za, th, indicator_res=g, indicator_types=["jump"] * len(g), num_threads=THREADS, pilot_seed=999)
    q.precompute_directions(thv)
    points = {"MC": lambda r: np.random.default_rng(10_000 + r).standard_normal((N, len(za))), "Sobol": lambda r: q.sample(N, r)}
    methods = [("Newton", None)] + [(f"Fries {int(w * 100)}%", w) for w in WINDOWS]
    rows = []
    for setup, gen in points.items():
        for name, w in methods:
            est = []
            for r in range(R):
                Z = gen(r)
                d = (CorrectionDriverVec2(fn, rP, za, th, indicator_res=g, indicator_types=["jump"] * len(g), num_threads=THREADS)
                     if w is None else CorrectionDriverFries(fn, rP, g, za, th, window_fraction=w, num_threads=THREADS))
                d.precompute_directions(thv)
                with contextlib.redirect_stdout(io.StringIO()):
                    est.append(d.run(Z)["total"])
            est = np.array(est); bias = np.abs(est.mean(0) - ref); sd = est.std(0, ddof=1)
            rows.append((f"{name} + {setup}", bias, sd, np.sqrt(bias ** 2 + sd ** 2)))
    print(f"\nSLIDE: Newton vs Fries: MC and Sobol   (|bias| + std = RMSE, in % of |Greek|; one estimate on N paths)")
    print(f"{'method':18s}" + "".join(f"{n:>26s}" for n in NAMES))
    for name, bias, sd, rm in rows:
        print(f"{name:18s}" + "".join(f"{100 * b / abs(r_):6.1f}+{100 * s / abs(r_):5.1f}={100 * m / abs(r_):5.1f}%{'':5s}"
                                       for b, s, m, r_ in zip(bias, sd, rm, ref)))
    print("\n(|bias| below about 2 s.e. of the mean, i.e. 2·std/4, is not resolved by 16 repetitions)")
    json.dump(dict(N=N, R=R, ref_paths_millions=REF_M, names=NAMES, ref=ref.tolist(), ref_se=se.tolist(),
                   rows=[dict(method=n, bias=b.tolist(), std=s_.tolist(), rmse=m.tolist()) for n, b, s_, m in rows]),
              open(os.environ.get("OUT", "bench_slides_fries.json"), "w"), indent=1)
