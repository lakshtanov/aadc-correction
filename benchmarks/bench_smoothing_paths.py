"""
bench_smoothing_paths.py — the smoothing baseline for the event products, as in the Heston-digital table of
benchmarks/bench_slides_qmc.py: paths and time per estimate to reach an RMSE (incl. bias) of 1% and 0.5% of |Greek|.

Smoothing: every indicator of the product (the default comparison, the coupon / autocall / knock-in barriers of the
credit autocallable; the default and the corridor of the LGM range accrual) replaced by a logistic of width
eps x SMOOTH_SCALE (record(smooth=eps) in bench_cox_coins.py / bench_lgm_range_accrual.py: the per-step hazard for the
default, the sd of S/S0 between dates, the daily sd of the rate); pathwise AAD. For every target the best eps of the
grid is taken (an oracle in favour of smoothing). MC, and scrambled Sobol with the same rotation as the correction
(option B, CorrectionDriverRQMC; its pilot time not charged to smoothing). R repetitions per (eps, N).
With a known tape layout (benchmarks 1-3, PRODUCT=b1|b2|b3) smoothing + Sobol uses the Brownian bridge (option A,
no pilot), as in the Heston-digital table.
RMSE = sqrt(std^2 (R-1)/R + (mean - ref)^2), ref = |Greek| of benchmarks/bench_paths_for_accuracy.py (REF_JSON: its
OUT file; the correction is unbiased). Time: 10 threads, generation of the random inputs excluded.
Usage: OPENBLAS_NUM_THREADS=1 PRODUCT=credit REF_JSON=paths_credit.json python benchmarks/bench_smoothing_paths.py
       (EPS=0.25,0.5,1,2,4  M_LO=10 M_HI=21  R=6  OUT=<json>)
"""
import os, sys, io, time, json, math, contextlib
import numpy as np
import aadc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
from rqmc import CorrectionDriverRQMC
from aadc_extensions import compile_tape

PROD = os.environ.get("PRODUCT", "credit")
THREADS = int(os.environ.get("THREADS", "10"))
R = int(os.environ.get("R", "6"))
EPS = [float(x) for x in os.environ.get("EPS", "0.25,0.5,1,2,4").split(",")]
M_LO, M_HI = int(os.environ.get("M_LO", "10")), int(os.environ.get("M_HI", "21"))
CHUNK = 1 << 17
if PROD == "credit":
    import bench_cox_coins as P
    GREEKS = [2, 3, 0]                                          # dV/dlambda0, dV/dsigma_lambda, dV/dS1
elif PROD in ("b1", "b2", "b3"):                                # benchmarks 1-3 (bench_slides_benchmarks.py)
    import bench_slides_benchmarks as SB
    _tape = getattr(SB, f"tape_{PROD}")

    class P:                                                    # the same interface as the event products
        NAMES = _tape()["names"]

        @staticmethod
        def record(markup=False, smooth=None):
            return _tape(smooth=smooth)
    GREEKS = list(range(len(P.NAMES)))
else:
    import bench_lgm_range_accrual as P
    GREEKS = [0, 1, 2]                                          # dV/dR0, dV/dsigma_r, dV/dlambda0
NAMES = [P.NAMES[k] for k in GREEKS]
ref_json = json.load(open(os.environ["REF_JSON"]))
assert ref_json["names"] == NAMES, (ref_json["names"], NAMES)
REF = np.array(ref_json["greek"])

t0 = P.record(False)
BRIDGE = t0.get("bridge_factors")                               # known layout: smoothing + Sobol with a Brownian
q = CorrectionDriverRQMC(t0["fn"], t0["rP"], t0["za"], t0["th"], num_threads=THREADS, pilot_seed=999,   # bridge (as
                         **(dict(ordering="bridge", bridge_factors=BRIDGE) if BRIDGE else {}))           # Table 3)
with contextlib.redirect_stdout(io.StringIO()):
    q.precompute_directions(t0["thv"])
d_args = len(t0["za"])


def chunks(src, N, r):
    """(N, d) inputs in row chunks: MC seed 10000 m + r; Sobol scrambling r (the same points as q.sample(N, r))."""
    if src == "MC":
        rng = np.random.default_rng(10_000 * int(math.log2(N)) + r)
        for a in range(0, N, CHUNK):
            yield rng.standard_normal((min(CHUNK, N - a), d_args))
    else:
        from scipy.stats import qmc
        from scipy.special import ndtri
        eng = qmc.Sobol(d=d_args, scramble=True, seed=r)
        for a in range(0, N, CHUNK):
            u = eng.random(min(CHUNK, N - a)); np.clip(u, 1e-15, 1.0 - 1e-15, out=u); ndtri(u, out=u)
            yield u @ q.Q.T


rows = {"smoothing + MC": [], "smoothing + Sobol": []}
for eps in EPS:
    t = P.record(False, smooth=eps)
    compile_tape(t["fn"])                                        # compiled kernels, as for the correction
    for src in ("MC", "Sobol"):
        for m in range(M_LO, M_HI + 1):
            N = 2 ** m; est, tm = [], 0.0
            for r in range(R):
                s = np.zeros(len(GREEKS))
                for Z in chunks(src, N, r):
                    inp = {a: np.ascontiguousarray(Z[:, j]) for j, a in enumerate(t["za"])}; inp.update(t["thv"])
                    t1 = time.perf_counter()
                    _, dd, _ = aadc.evaluate(funcs=t["fn"], request={t["rP"]: [t["th"][k] for k in GREEKS]}, inputs=inp, workers=THREADS)
                    tm += time.perf_counter() - t1
                    s += [np.sum(dd[t["rP"]][t["th"][k]]) for k in GREEKS]
                est.append(s / N)
            est = np.array(est); mean, std = est.mean(0), est.std(0, ddof=1)
            rmse = np.sqrt(std ** 2 * (R - 1) / R + (mean - REF) ** 2)
            rows[f"smoothing + {src}"].append(dict(eps=eps, N=N, mean=mean.tolist(), std=std.tolist(), rmse=rmse.tolist(), time=tm / R))
            print(f"  eps={eps:g} {src} N=2^{m}: bias " + " ".join(f"{x:+.3g}" for x in mean - REF)
                  + "  std " + " ".join(f"{x:.3g}" for x in std) + f"  {tm/R:.2f}s", flush=True)

print(f"\n{PROD}: |Greek| " + "  ".join(f"{n} {g:+.4f}" for n, g in zip(NAMES, REF)))
for tgt in (0.01, 0.005):
    print(f"\nRMSE <= {tgt*100:g}% of |Greek|, best eps per target  (paths (time per estimate), eps):")
    for i, nm in enumerate(NAMES):
        line = f"  {nm:17s}"
        for setup, rr in rows.items():
            ok = [x for x in rr if x["rmse"][i] <= tgt * abs(REF[i])]
            if ok:
                b = min(ok, key=lambda x: (x["N"], x["time"]))
                line += f" | {setup}: {b['N']:,} ({b['time']:.2f}s, eps {b['eps']:g})"
            else:
                b = min(rr, key=lambda x: x["rmse"][i])
                line += f" | {setup}: >{2**M_HI:,} (best RMSE {100*b['rmse'][i]/abs(REF[i]):.2f}% at eps {b['eps']:g}, N {b['N']:,})"
        print(line, flush=True)
if os.environ.get("OUT"):
    json.dump(dict(product=PROD, names=NAMES, ref=REF.tolist(), R=R, eps=EPS, rows=rows), open(os.environ["OUT"], "w"), indent=1)
