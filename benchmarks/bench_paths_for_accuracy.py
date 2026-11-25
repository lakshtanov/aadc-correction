"""
bench_paths_for_accuracy.py — paths and time per estimate to reach a relative accuracy (1% and 0.5% of |Greek|) on
the event products: the credit-linked autocallable (PRODUCT=credit, benchmarks/bench_cox_coins.py) and the daily
range accrual on a 3M LGM rate (PRODUCT=lgm, benchmarks/bench_lgm_range_accrual.py).

Rows:
  bump (CRN, MC)             central bump with common random numbers; its per-path standard deviation is measured on
                             BUMP_M paths and N = (sd / target)^2; time = N x (time of the two replays per Greek)
  correction, default directions, MC / Sobol   CorrectionDriverVec2(coins="off")      (credit only)
  correction, coin, MC / Sobol                  CorrectionDriverVec2(coins="auto") with the CoxDefault markup
The corrections are unbiased (checked against bump references, benchmarks/bench_cox_coins.py and
bench_lgm_range_accrual.py), so their RMSE is the standard deviation of one estimate, measured over R repetitions
(independent MC seeds / independent scramblings, option B) on N = 2^M_LO .. 2^M_HI; the smallest N whose standard
deviation is <= target |Greek| is reported, with its time (10 threads; generation of the random inputs excluded).
|Greek| is taken from the most accurate estimate (coin + Sobol at the largest N).
PRODUCT=b1|b2|b3: benchmarks 1-3 of bench_slides_benchmarks.py (correction only; plus Sobol option A, Brownian
bridge on the tape layout). The one-off pilot time of option B (rotation_time_option_B) is saved in the JSON; the
row times do not include it (the table adds it to every Sobol option-B estimate).
Usage: OPENBLAS_NUM_THREADS=1 PRODUCT=credit python benchmarks/bench_paths_for_accuracy.py   (OUT=<json> saves the raw data)
"""
import os, sys, io, time, json, math, contextlib
import numpy as np
import aadc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from correction_driver_vec2 import CorrectionDriverVec2
from rqmc import CorrectionDriverRQMC

PROD = os.environ.get("PRODUCT", "credit")
THREADS = int(os.environ.get("THREADS", "10"))
R = int(os.environ.get("R", "10"))
if PROD == "credit":
    import bench_cox_coins as P
    t = P.record(True); GREEKS = [2, 3, 0]                     # dV/dlambda0, dV/dsigma_lambda, dV/dS1
    M_LO, M_HI = int(os.environ.get("M_LO", "10")), int(os.environ.get("M_HI", "17"))
    MODES = {"correction, default directions": dict(coins="off", m_hi=int(os.environ.get("M_HI_OFF", "14"))),
             "correction, coin": dict(coins="auto")}
    H = P.H_REF
elif PROD in ("b1", "b2", "b3"):                                 # benchmarks 1-3 (bench_slides_benchmarks.py)
    import bench_slides_benchmarks as P
    t = getattr(P, f"tape_{PROD}")(); t["defaults"] = []; GREEKS = list(range(len(t["names"])))
    P.NAMES = t["names"]
    M_LO, M_HI = int(os.environ.get("M_LO", "8")), int(os.environ.get("M_HI", "16"))
    MODES = {"correction": dict()}
    H = list(t["h"][0])
else:
    import bench_lgm_range_accrual as P
    t = P.record(True); GREEKS = [0, 1, 2]                     # dV/dR0, dV/dsigma_r, dV/dlambda0
    M_LO, M_HI = int(os.environ.get("M_LO", "10")), int(os.environ.get("M_HI", "13"))
    MODES = {"correction, coin": dict(coins="auto", indicator_threads=8)}
    H = [0.0005, 0.0005, 0.004, 0.02]                          # rate steps h/2 of the bias study (Richardson-close)
NAMES = [P.NAMES[k] for k in GREEKS]
d_args = len(t["za"])


def make(kw):
    d = CorrectionDriverVec2(t["fn"], t["rP"], t["za"], t["th"], num_threads=THREADS, defaults=t["defaults"], **kw)
    d.precompute_directions(t["thv"])
    return d


q = CorrectionDriverRQMC(t["fn"], t["rP"], t["za"], t["th"], num_threads=THREADS, pilot_seed=999)
with contextlib.redirect_stdout(io.StringIO()):
    q.precompute_directions(t["thv"])
SOURCES = {"MC": None, "Sobol": q}                                 # Sobol = option B (pilot paths: rotation_time)
if t.get("bridge_factors"):                                        # option A: Brownian bridge on the known layout
    qa = CorrectionDriverRQMC(t["fn"], t["rP"], t["za"], t["th"], num_threads=THREADS, ordering="bridge",
                              bridge_factors=t["bridge_factors"])
    with contextlib.redirect_stdout(io.StringIO()):
        qa.precompute_directions(t["thv"])
    SOURCES["Sobol, option A"] = qa
ROTATION_TIME = q.rotation_time                                    # option B: one-off pilot cost, added per estimate
rows = {}
for name, kw in MODES.items():
    kw = dict(kw); m_hi = kw.pop("m_hi", M_HI)
    d = make(kw)
    for src, qq in SOURCES.items():
        res = []
        for m in range(M_LO, m_hi + 1):
            N = 2 ** m; est, tm = [], 0.0
            for r in range(R):
                Z = np.random.default_rng(10_000 * m + r).standard_normal((N, d_args)) if qq is None else qq.sample(N, r)
                t0 = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    o = d.run(Z)
                tm += time.perf_counter() - t0; est.append(np.asarray(o["total"])[GREEKS])
            est = np.array(est)
            res.append(dict(N=N, mean=est.mean(0).tolist(), std=est.std(0, ddof=1).tolist(), time=tm / R))
            print(f"  {name} + {src} N=2^{m}: std " + " ".join(f"{x:.3g}" for x in est.std(0, ddof=1)) + f"  {tm/R:.1f}s", flush=True)
        rows[f"{name} + {src}"] = res
# bump: per-path standard deviation of the CRN central difference, and the time of its replays per path
BUMP_M = int(os.environ.get("BUMP_M", "200000"))
Z = np.random.default_rng(20261126).standard_normal((BUMP_M, d_args)); base = {a: np.ascontiguousarray(Z[:, j]) for j, a in enumerate(t["za"])}
sd, tpath = [], []
for k in GREEKS:
    a = t["th"][k]; h = H[k]; r2 = []; t0 = time.perf_counter()
    for sg in (1, -1):
        inp = dict(base); inp.update(t["thv"]); inp[a] = t["thv"][a] + sg * h
        r2.append(np.asarray(aadc.evaluate(funcs=t["fn"], request={t["rP"]: []}, inputs=inp, workers=THREADS)[0][t["rP"]]))
    tpath.append((time.perf_counter() - t0) / BUMP_M)
    sd.append(float(np.std((r2[0] - r2[1]) / (2 * h), ddof=1)))
best = rows[[k for k in rows if ("coin" in k or PROD.startswith("b")) and k.endswith("+ Sobol")][0]][-1]["mean"]
out = dict(product=PROD, names=NAMES, greek=best, rows=rows, bump=dict(sd=sd, time_per_path=tpath, h=[H[k] for k in GREEKS]), R=R,
           rotation_time_option_B=ROTATION_TIME, n_pilot=q.n_pilot)
if os.environ.get("OUT"):
    json.dump(out, open(os.environ["OUT"], "w"), indent=1)
print(f"\n{PROD}: |Greek| " + "  ".join(f"{n} {g:+.4f}" for n, g in zip(NAMES, best)))
for target in (0.01, 0.005):
    print(f"\nRMSE <= {target:.1%} of |Greek|  (paths (time per estimate)):")
    for k, n in enumerate(NAMES):
        tgt = target * abs(best[k])
        line = f"  {n:17s} bump (CRN, MC): {math.ceil((sd[k]/tgt)**2):>12,d} ({(sd[k]/tgt)**2*tpath[k]:.2f}s)"
        for name, res in rows.items():
            hit = next((x for x in res if x["std"][k] <= tgt), None)
            line += f" | {name}: " + (f"{hit['N']:,} ({hit['time']:.2f}s)" if hit else f">{res[-1]['N']:,}")
        print(line, flush=True)
