"""
bench_slides_benchmarks.py — reproduce the slides "Benchmark 1-3".

For every Greek: pathwise (AAD), correction, total = pathwise + correction (mean ± s.e. over seeds) with the
δ-correction driver (CorrectionDriverVec2), and an independent reference: central bump with common random
numbers on AADC (forward passes only), with two step sizes h and 2h as a check of the finite-difference bias.

Benchmark 1  Down-and-out call, GBM, 50 monitoring dates.
             S0 = 100, K = 90, B = 80, sigma = 0.25, r = 0.05, T = 1.  Greeks: dV/dS0, dV/dsigma.
Benchmark 2  3-asset worst-of Phoenix autocallable, Heston, 8 quarterly dates, T = 2.
             Coupon 5% if worst > 70%, autocall if worst > 100% (not on the last date), notional at maturity.
             Initial fixing = S0 (a contract constant). Variance: drift-implicit scheme for sqrt(v) (Alfonsi),
             no truncation (4 kappa theta > xi^2 for every asset). Greeks: dV/dS1, dV/dS3, dV/dxi1, dV/dkappa1.
Benchmark 3  Down-and-out call, GBM spot + Hull-White 1F short rate, 50 dates (same model as
             examples/example_barrier_hw_ql.py). Greeks: dV/dS0, dV/dsigma, dV/dr0, dV/dsigma_r.

Usage:  OPENBLAS_NUM_THREADS=1 python benchmarks/bench_slides_benchmarks.py         (~1 h)
        QUICK=1 ...  (a few minutes: fewer paths)       ONLY=1,3 ...  (subset)      THREADS=10 (default)
        TIMING=1 ... cost only: wall time of pathwise AAD (forward + one reverse of the payoff) vs one run of the
                     correction driver on the same paths and threads, and their ratio (the "overhead").
        STATS=1 ...  root-finder statistics on the paths of seed 0 (numbers of the slides "Cost Analysis",
                     "Root Finding: Safeguarded Newton", "No Screening"): tape replays per path and per
                     (path, indicator) pair for the default solver; the same paths with plain Newton
                     (root_solver="newton": lost roots, replays, change of the Greeks) and with the old
                     screening skip_sigma=20 (change of the Greeks in %, same paths).
Raw results -> bench_slides_benchmarks.json (OUT=...).
"""
import sys, os, io, json, math, time, contextlib
import numpy as np
import aadc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from correction_driver_vec2 import CorrectionDriverVec2

QUICK = os.environ.get("QUICK", "0") == "1"
THREADS = int(os.environ.get("THREADS", "10"))
ONLY = [int(x) for x in os.environ.get("ONLY", "1,2,3").split(",")]
OUT = os.environ.get("OUT", "bench_slides_benchmarks.json")
c = aadc.idouble


# ── Benchmark 1: GBM down-and-out call, 50 dates ───────────────────────────────
def tape_b1():
    S0, K, B, sig, r, T, N = 100.0, 90.0, 80.0, 0.25, 0.05, 1.0, 50
    dt = T / N
    fn = aadc.Functions(); fn.start_recording(register_switches=aadc.SWITCH_JUMP)
    S = c(S0); aS = S.mark_as_input()
    vol = c(sig); av = vol.mark_as_input()
    Z = [c(0.0) for _ in range(N)]; za = [z.mark_as_input() for z in Z]
    logS, alive = aadc.math.log(S), c(1.0)
    for j in range(N):
        logS = logS + (c(r) - c(0.5) * vol * vol) * c(dt) + vol * c(math.sqrt(dt)) * Z[j]
        alive = alive * aadc.iif(aadc.math.exp(logS) > c(B), c(1.0), c(0.0))
    P = alive * aadc.math.max(aadc.math.exp(logS) - c(K), c(0.0)) * c(math.exp(-r * T))
    rP = P.mark_as_output(); fn.stop_recording()
    return dict(fn=fn, rP=rP, za=za, th=[aS, av], thv={aS: S0, av: sig}, names=["dV/dS0", "dV/dsigma"],
                h=[(0.5, 0.0025), (1.0, 0.005)])


# ── Benchmark 2: 3-asset Heston Phoenix, Alfonsi scheme ────────────────────────
def tape_b2():
    NA, S0, VOL = 3, [100.0] * 3, [0.20, 0.25, 0.22]
    V0 = [v * v for v in VOL]; THETA = list(V0); KAPPA = [2.0] * 3; XI = [0.3, 0.35, 0.32]
    RHO_SV = [-0.7, -0.65, -0.7]; CORR, R, T, NOBS = 0.5, 0.03, 2.0, 8
    CB, AB, CR, NOT = 0.70, 1.00, 0.05, 100.0
    for a in range(NA):
        assert 4 * KAPPA[a] * THETA[a] > XI[a] ** 2, "Alfonsi scheme needs 4 kappa theta > xi^2"
    dt = T / NOBS
    L = np.linalg.cholesky(np.full((NA, NA), CORR) + (1 - CORR) * np.eye(NA))
    fn = aadc.Functions(); fn.start_recording(register_switches=aadc.SWITCH_JUMP)
    S = [c(s) for s in S0]; aS = [s.mark_as_input() for s in S]
    xi1 = c(XI[0]); axi = xi1.mark_as_input()
    ka1 = c(KAPPA[0]); aka = ka1.mark_as_input()
    Z = [c(0.0) for _ in range(2 * NA * NOBS)]; za = [z.mark_as_input() for z in Z]
    logS = [aadc.math.log(s) for s in S]
    y = [aadc.math.sqrt(c(v)) for v in V0]                       # y = sqrt(v)
    payoff, alive = c(0.0), c(1.0)
    for d in range(NOBS):
        zs_ind = [Z[2 * NA * d + 2 * a] for a in range(NA)]
        zv = [Z[2 * NA * d + 2 * a + 1] for a in range(NA)]
        zs = []
        for a in range(NA):
            q = c(0.0)
            for b in range(a + 1):
                q = q + c(L[a, b]) * zs_ind[b]
            zs.append(q)
        for a in range(NA):
            ka = ka1 if a == 0 else c(KAPPA[a]); xi = xi1 if a == 0 else c(XI[a])
            w = c(RHO_SV[a]) * zs[a] + c(math.sqrt(1 - RHO_SV[a] ** 2)) * zv[a]
            sv, vp = y[a], y[a] * y[a]
            aa = y[a] + c(0.5 * math.sqrt(dt)) * xi * w
            cc = c(1.0) + c(0.5 * dt) * ka
            y[a] = (aa + aadc.math.sqrt(aa * aa + c(4.0 * dt) * cc * (c(0.5 * THETA[a]) * ka - c(0.125) * xi * xi))) / (c(2.0) * cc)
            logS[a] = logS[a] + (c(R) - c(0.5) * vp) * c(dt) + sv * c(math.sqrt(dt)) * zs[a]
        perf = [aadc.math.exp(logS[a]) / c(S0[a]) for a in range(NA)]       # fixing = S0, a contract constant
        worst = perf[0]
        for a in range(1, NA):
            worst = aadc.iif(perf[a] < worst, perf[a], worst)
        df = math.exp(-R * (d + 1) * dt)
        payoff = payoff + alive * aadc.iif(worst > c(CB), c(1.0), c(0.0)) * c(CR * NOT * df)
        if d < NOBS - 1:
            ac = aadc.iif(worst > c(AB), c(1.0), c(0.0))
            payoff = payoff + alive * ac * c(NOT * df)
            alive = alive * (c(1.0) - ac)
    payoff = payoff + alive * c(NOT * math.exp(-R * T))
    rP = payoff.mark_as_output(); fn.stop_recording()
    th = [aS[0], aS[2], axi, aka]
    thv = {aS[0]: S0[0], aS[1]: S0[1], aS[2]: S0[2], axi: XI[0], aka: KAPPA[0]}   # S2 has a value, not a Greek
    return dict(fn=fn, rP=rP, za=za, th=th, thv=thv, names=["dV/dS1", "dV/dS3", "dV/dxi1", "dV/dkappa1"],
                h=[(0.5, 0.5, 0.005, 0.025), (1.0, 1.0, 0.01, 0.05)])


# ── Benchmark 3: GBM + Hull-White 1F, 50 dates ─────────────────────────────────
def tape_b3():
    S0, K, B, sig, r0, T, N = 100.0, 90.0, 80.0, 0.25, 0.05, 1.0, 50
    a_hw, sr0, rho = 0.1, 0.01, -0.3
    dt = T / N
    fn = aadc.Functions(); fn.start_recording(register_switches=aadc.SWITCH_JUMP)
    S = c(S0); aS = S.mark_as_input()
    vol = c(sig); av = vol.mark_as_input()
    r0i = c(r0); ar = r0i.mark_as_input()
    sr = c(sr0); asr = sr.mark_as_input()
    Z = [c(0.0) for _ in range(2 * N)]; za = [z.mark_as_input() for z in Z]
    logS, r_t, disc, alive = aadc.math.log(S), r0i + c(0.0), c(0.0), c(1.0)
    for d in range(N):
        z1 = Z[2 * d]
        w2 = c(rho) * Z[2 * d] + c(math.sqrt(1 - rho * rho)) * Z[2 * d + 1]
        logS = logS + (r_t - c(0.5) * vol * vol) * c(dt) + vol * c(math.sqrt(dt)) * z1
        r_t = r_t + c(a_hw) * (r0i - r_t) * c(dt) + sr * c(math.sqrt(dt)) * w2
        disc = disc + r_t * c(dt)
        alive = alive * aadc.iif(aadc.math.exp(logS) > c(B), c(1.0), c(0.0))
    P = alive * aadc.math.exp(-disc) * aadc.math.max(aadc.math.exp(logS) - c(K), c(0.0))
    rP = P.mark_as_output(); fn.stop_recording()
    return dict(fn=fn, rP=rP, za=za, th=[aS, av, ar, asr], thv={aS: S0, av: sig, ar: r0, asr: sr0},
                names=["dV/dS0", "dV/dsigma", "dV/dr0", "dV/dsigma_r"],
                h=[(0.5, 0.0025, 0.0005, 0.0005), (1.0, 0.005, 0.001, 0.001)])


BENCH = {1: ("Benchmark 1: down-and-out call, GBM, 50 dates", tape_b1, (100_000, 10), (8 if QUICK else 64)),
         2: ("Benchmark 2: 3-asset Heston Phoenix (Alfonsi scheme)", tape_b2, (10_000, 10), (4 if QUICK else 64)),
         3: ("Benchmark 3: down-and-out call, GBM + Hull-White, 50 dates", tape_b3, (50_000, 10), (4 if QUICK else 32))}


def driver(t, M, seeds):
    pw, co, tot, tm = [], [], [], []
    for s in range(seeds):
        Z = np.random.RandomState(42 + s).randn(M, len(t["za"]))
        d = CorrectionDriverVec2(t["fn"], t["rP"], t["za"], t["th"], num_threads=THREADS)
        d.precompute_directions(t["thv"])
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            r = d.run(Z)
        tm.append(time.perf_counter() - t0)
        pw.append(r["pathwise"]); co.append(r["correction"]); tot.append(r["total"])
    f = lambda a: (np.mean(a, 0), np.std(a, 0, ddof=1) / math.sqrt(seeds))
    return f(pw), f(co), f(tot), float(np.mean(tm))


def reference(t, millions):
    out = []
    chunk = 1_000_000 if len(t["za"]) <= 20 else 250_000
    for H in t["h"]:
        rng = np.random.default_rng(20261115); k_ = len(t["th"]); s1 = np.zeros(k_); s2 = np.zeros(k_); n = 0
        while n < millions * 1_000_000:
            Z = rng.standard_normal((chunk, len(t["za"]))); base = {a: np.ascontiguousarray(Z[:, j]) for j, a in enumerate(t["za"])}
            for k, (a, h) in enumerate(zip(t["th"], H)):
                r = []
                for sg in (1, -1):
                    inp = dict(base); inp.update(t["thv"]); inp[a] = t["thv"][a] + sg * h
                    r.append(aadc.evaluate(funcs=t["fn"], request={t["rP"]: []}, inputs=inp, workers=THREADS)[0][t["rP"]])
                d = (r[0] - r[1]) / (2 * h); s1[k] += d.sum(); s2[k] += (d * d).sum()
            n += chunk
        m = s1 / n; out.append((m, np.sqrt((s2 / n - m * m) / n)))
    return out


def timing(t, M):
    """(pathwise AAD time, driver time) for one estimate on M paths, same paths and threads."""
    Z = np.random.RandomState(7).randn(M, len(t["za"]))
    inp = {a: np.ascontiguousarray(Z[:, j]) for j, a in enumerate(t["za"])}; inp.update(t["thv"])
    aadc.evaluate(funcs=t["fn"], request={t["rP"]: t["th"]}, inputs=inp, workers=THREADS)          # warm-up
    t0 = time.perf_counter(); aadc.evaluate(funcs=t["fn"], request={t["rP"]: t["th"]}, inputs=inp, workers=THREADS)
    t_pw = time.perf_counter() - t0
    d = CorrectionDriverVec2(t["fn"], t["rP"], t["za"], t["th"], num_threads=THREADS); d.precompute_directions(t["thv"])
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        d.run(Z)
    return t_pw, time.perf_counter() - t0


def solver_stats(t, M, **kw):
    """One driver run on the paths of seed 0 with replay counters: (result, n_pairs, fwd+rev rows, fwd-only rows, time).
    A row = one path in one tape replay; a Newton step is a forward + reverse, bracket/search steps and the
    jump are forward only."""
    import threading
    Z = np.random.RandomState(42).randn(M, len(t["za"]))
    d = CorrectionDriverVec2(t["fn"], t["rP"], t["za"], t["th"], num_threads=THREADS, **kw)
    d.precompute_directions(t["thv"])
    cnt, lock = {"fr": 0, "f": 0}, threading.Lock()

    def counted(name, key):
        fn = getattr(d.vf, name)
        def wrap(cols, *a, **k):
            with lock:
                cnt[key] += len(cols[0])
            return fn(cols, *a, **k)
        setattr(d.vf, name, wrap)
    counted("newton_step_columns", "fr"); counted("forward_columns", "f"); counted("forward_columns_all", "f")
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        r = d.run(Z)
    n_pairs = M * sum(1 for _, nrm in d.directions if nrm >= 1e-15)
    return r, n_pairs, cnt["fr"], cnt["f"], time.perf_counter() - t0


if __name__ == "__main__" and os.environ.get("STATS") == "1":
    for b in ONLY:
        title, build, (M, seeds), refM = BENCH[b]
        t = build()
        r0, P, fr0, f0, t_0 = solver_stats(t, M)
        print(f"\n{title}: {M:,} paths (seed 0), {P // M} indicators, {P:,} (path, indicator) pairs, all solved")
        print(f"  default solver: {fr0 / P:.2f} Newton steps (forward + reverse) per pair; per path {fr0 / M:.0f} "
              f"forward + reverse and {f0 / M:.0f} forward-only replays; {t_0:.1f}s")
        print(f"                  counters: newton_unconverged {r0['newton_unconverged']}, newton_far {r0['newton_far']}, "
              f"jump_unresolved {r0['jump_unresolved']}")
        r1, _, fr1, f1, t_1 = solver_stats(t, M, root_solver="newton")
        dg = [(a - b_) / abs(b_) * 100 if b_ else float("nan") for a, b_ in zip(r1["total"], r0["total"])]
        print(f"  plain Newton:   lost roots (newton_unconverged) {r1['newton_unconverged']}; replays per path "
              f"{fr1 / M:.0f} + {f1 / M:.0f} (work {(fr1 + f1) / (fr0 + f0) - 1:+.1%} in rows, time {t_1 / t_0 - 1:+.0%}); "
              f"Greeks vs default: " + ", ".join(f"{n} {x:+.2f}%" for n, x in zip(t["names"], dg)))
        r2, *_ = solver_stats(t, M, skip_sigma=20.0)
        dg = [(a - b_) / abs(b_) * 100 if b_ else float("nan") for a, b_ in zip(r2["total"], r0["total"])]
        print(f"  screening skip_sigma=20: Greeks vs no screening (same paths): "
              + ", ".join(f"{n} {x:+.2f}%" for n, x in zip(t["names"], dg)), flush=True)
    sys.exit(0)

if __name__ == "__main__" and os.environ.get("TIMING") == "1":
    print(f"{'benchmark':58s} {'paths':>8s} {'pathwise AAD':>13s} {'driver':>9s} {'overhead':>9s}")
    for b in ONLY:
        title, build, (M, seeds), refM = BENCH[b]
        t = build(); tp, td = timing(t, M)
        print(f"{title:58s} {M:8,d} {tp:12.3f}s {td:8.1f}s {td / tp:8.0f}x", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    res = {}
    for b in ONLY:
        title, build, (M, seeds), refM = BENCH[b]
        if QUICK:
            M, seeds = M // 4, 4
        t = build()
        print(f"\n{'=' * 92}\n{title}\n  {len(t['za'])} normals/path, {len(t['fn'].cmp_switches())} switches; "
              f"driver {seeds} seeds x {M:,} paths; reference {refM}M paths\n{'=' * 92}", flush=True)
        (pw, pws), (co, cos), (tot, tots), tm = driver(t, M, seeds)
        (g, se), (g2, se2) = reference(t, refM)
        print(f"{'Greek':12s} {'pathwise':>10s} {'correction':>11s} {'total':>20s} {'reference (h)':>20s} {'diff/σ':>7s} {'ref h vs 2h':>11s}")
        for k, nm in enumerate(t["names"]):
            print(f"{nm:12s} {pw[k]:+10.4f} {co[k]:+11.4f} {tot[k]:+12.4f} ± {tots[k]:.4f} {g[k]:+12.4f} ± {se[k]:.4f} "
                  f"{(tot[k] - g[k]) / math.hypot(tots[k], se[k]):+7.1f} {(g[k] - g2[k]) / math.hypot(se[k], se2[k]):+10.1f}σ")
        print(f"driver time per run: {tm:.1f} s ({THREADS} threads)", flush=True)
        res[b] = dict(title=title, names=t["names"], paths=M, seeds=seeds, ref_paths_millions=refM,
                      pathwise=[pw.tolist(), pws.tolist()], correction=[co.tolist(), cos.tolist()], total=[tot.tolist(), tots.tolist()],
                      reference=[g.tolist(), se.tolist()], reference_2h=[g2.tolist(), se2.tolist()], driver_time=tm)
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"\nraw results -> {OUT}")
