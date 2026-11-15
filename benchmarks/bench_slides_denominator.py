"""
bench_slides_denominator.py — reproduce the slide "Denominator: Numerical Verification".

Question (P. Jäckel): the correction divides by |dg/du| ~ S sigma sqrt(t), which goes to 0 for small maturities.
Does it blow up? Down-and-out call, GBM, n = max(2, min(50, round(252 T))) monitoring dates (daily for short T),
K = 70 < B = 80 (in the money at the barrier, so the payoff jump at the barrier is not small), sigma = 0.25,
r = 0.05. Spot S0 from far (100) to very near (80.1) the barrier, T from 1 to 0.01. Greek: dV/dS0.

Driver: CorrectionDriverVec2, pathwise / correction / total, mean ± s.e. over 10 seeds of M paths.
Reference: central bump with common random numbers on AADC (forward passes only), independent seed,
h = min(0.5, (S0 - B) / 5) and 2h as a check of the finite-difference bias.

Usage:  OPENBLAS_NUM_THREADS=1 python benchmarks/bench_slides_denominator.py      (QUICK=1: fewer paths)
Raw results -> bench_slides_denominator.json (OUT=...).
"""
import sys, os, io, json, math, time, contextlib
import numpy as np
import aadc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from correction_driver_vec2 import CorrectionDriverVec2

QUICK = os.environ.get("QUICK", "0") == "1"
THREADS = int(os.environ.get("THREADS", "10"))
M, SEEDS = (5_000, 4) if QUICK else (20_000, 10)
REF_M = 2 if QUICK else 20                       # millions of reference paths
K, B, SIG, R = 70.0, 80.0, 0.25, 0.05
CASES = [(100.0, 1.0), (100.0, 0.01), (81.0, 1.0), (81.0, 0.05), (81.0, 0.01), (80.1, 0.05), (80.1, 0.01)]
c = aadc.idouble


def tape(S0, T):
    n = max(2, min(50, round(252 * T))); dt = T / n
    fn = aadc.Functions(); fn.start_recording(register_switches=aadc.SWITCH_JUMP)
    S = c(S0); aS = S.mark_as_input()
    Z = [c(0.0) for _ in range(n)]; za = [z.mark_as_input() for z in Z]
    logS, alive = aadc.math.log(S), c(1.0)
    for j in range(n):
        logS = logS + c((R - 0.5 * SIG * SIG) * dt) + c(SIG * math.sqrt(dt)) * Z[j]
        alive = alive * aadc.iif(aadc.math.exp(logS) > c(B), c(1.0), c(0.0))
    P = alive * aadc.math.max(aadc.math.exp(logS) - c(K), c(0.0)) * c(math.exp(-R * T))
    rP = P.mark_as_output(); fn.stop_recording()
    return fn, rP, za, aS, n


def driver(fn, rP, za, aS, S0):
    pw, co, tot = [], [], []
    for s in range(SEEDS):
        Z = np.random.RandomState(42 + s).randn(M, len(za))
        d = CorrectionDriverVec2(fn, rP, za, [aS], num_threads=THREADS); d.precompute_directions({aS: S0})
        with contextlib.redirect_stdout(io.StringIO()):
            r = d.run(Z)
        pw.append(r["pathwise"][0]); co.append(r["correction"][0]); tot.append(r["total"][0])
        assert np.isfinite(r["total"][0])
    f = lambda a: (float(np.mean(a)), float(np.std(a, ddof=1) / math.sqrt(SEEDS)))
    return f(pw), f(co), f(tot)


def reference(fn, rP, za, aS, S0, h):
    rng = np.random.default_rng(20261115); s1 = s2 = 0.0; n = 0
    chunk = 1_000_000 if len(za) <= 20 else 250_000
    while n < REF_M * 1_000_000:
        Z = rng.standard_normal((chunk, len(za))); inp = {a: np.ascontiguousarray(Z[:, j]) for j, a in enumerate(za)}
        r = []
        for sg in (1, -1):
            inp[aS] = S0 + sg * h
            r.append(aadc.evaluate(funcs=fn, request={rP: []}, inputs=inp, workers=THREADS)[0][rP])
        d = (r[0] - r[1]) / (2 * h); s1 += d.sum(); s2 += (d * d).sum(); n += chunk
    m = s1 / n
    return float(m), float(math.sqrt((s2 / n - m * m) / n))


if __name__ == "__main__":
    print(f"Down-and-out call, K={K:g} < B={B:g}, sigma={SIG}, n = min(50, daily) dates; driver {SEEDS} x {M:,} paths, "
          f"reference {REF_M}M paths (CRN central bump)", flush=True)
    print(f"{'S0':>6} {'T':>5} {'n':>4} {'pathwise':>15} {'correction':>15} {'total':>15} {'bump h':>15} {'bump 2h':>15} {'dev':>6}")
    rows = []
    for S0, T in CASES:
        t0 = time.time()
        fn, rP, za, aS, n = tape(S0, T)
        pw, co, tot = driver(fn, rP, za, aS, S0)
        h = min(0.5, (S0 - B) / 5)
        ref, ref2 = reference(fn, rP, za, aS, S0, h), reference(fn, rP, za, aS, S0, 2 * h)
        dev = (tot[0] - ref[0]) / math.hypot(tot[1], ref[1])
        pm = lambda x: f"{x[0]:+.3f}±{x[1]:.3f}"
        print(f"{S0:6g} {T:5g} {n:4d} {pm(pw):>15} {pm(co):>15} {pm(tot):>15} {pm(ref):>15} {pm(ref2):>15} {dev:+5.1f}σ"
              f"  ({time.time() - t0:.0f}s)", flush=True)
        rows.append(dict(S0=S0, T=T, n=n, pathwise=pw, correction=co, total=tot, h=h, ref=ref, ref_2h=ref2))
    json.dump(dict(K=K, B=B, sigma=SIG, r=R, M=M, seeds=SEEDS, ref_paths_millions=REF_M, rows=rows),
              open(os.environ.get("OUT", "bench_slides_denominator.json"), "w"), indent=1)
