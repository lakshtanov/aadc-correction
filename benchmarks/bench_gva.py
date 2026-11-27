"""
bench_gva.py -- both GVA variants (gva_drivers.py: Naumann, QuantMinds 2026) on benchmarks 1-5, MC and Sobol (option B,
512 pilot paths), R repetitions at a fixed N per product. Reference: the corrected estimate of this repository at the
largest N of benchmarks/bench_paths_for_accuracy.py (REF_DIR: folder with its OUT files paths_<product>.json).
GVA-local: s_frac grid (S_FRACS), the best per Greek by RMSE (an oracle in its favour). Output: OUT (JSON) and the log.
Usage: OPENBLAS_NUM_THREADS=1 REF_DIR=<folder> python benchmarks/bench_gva.py      (CASES=b1,b2,b3,credit,lgm)
"""
import os, sys, io, json, time, math, traceback, contextlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from rqmc import CorrectionDriverRQMC
import bench_slides_benchmarks as SB, bench_cox_coins as C, bench_lgm_range_accrual as L
from gva_drivers import GVALocal, gva_integral
REF_DIR = os.environ.get("REF_DIR", ".")
S_FRACS = [float(x) for x in os.environ.get("S_FRACS", "0.01,0.03,0.1,0.3").split(",")]
CASES = [("b1", SB.tape_b1, 8192, 16), ("b2", SB.tape_b2, 8192, 16), ("b3", SB.tape_b3, 8192, 16),
         ("credit", lambda: dict(C.record(False), names=C.NAMES), 8192, 16), ("lgm", lambda: dict(L.record(False), names=L.NAMES), 1024, 8)]
only = os.environ.get("CASES"); out = {}
for prod, mk, N, R in CASES:
    if only and prod not in only.split(","):
        continue
    try:
        ref = json.load(open(os.path.join(REF_DIR, f"paths_{prod}.json"))); t = mk(); names = t["names"]
        G = {nm: g for nm, g in zip(ref["names"], ref["greek"])}
        idx = [names.index(nm) for nm in ref["names"]]
        gs = [sw.g for sw in t["fn"].cmp_switches()]
        q = CorrectionDriverRQMC(t["fn"], t["rP"], t["za"], t["th"], num_threads=10, pilot_seed=999, n_pilot=512, coins="off")
        with contextlib.redirect_stdout(io.StringIO()):
            q.precompute_directions(t["thv"])
        meths = {}
        for sf in S_FRACS:
            gl = GVALocal(t["fn"], t["rP"], gs, t["za"], t["th"], s_frac=sf, num_threads=10)
            gl.precompute_directions(t["thv"]); meths[f"GVA-local s={sf}"] = gl
        with contextlib.redirect_stdout(io.StringIO()):
            meths["GVA-integral (axis)"] = gva_integral(t["fn"], t["rP"], t["za"], t["th"], t["thv"], num_threads=10)
        res = {}
        for m, d in meths.items():
            for src in ("MC", "Sobol"):
                est, tt = [], 0.0
                for r in range(R):
                    Z = np.random.default_rng(10_000 + r).standard_normal((N, len(t["za"]))) if src == "MC" else q.sample(N, r)
                    t0 = time.perf_counter()
                    with contextlib.redirect_stdout(io.StringIO()):
                        o = d.run(Z)
                    tt += time.perf_counter() - t0; est.append(np.asarray(o["total"])[idx])
                est = np.array(est)
                res[f"{m} + {src}"] = dict(mean=est.mean(0).tolist(), std=est.std(0, ddof=1).tolist(), time=tt / R)
        out[prod] = dict(N=N, R=R, names=ref["names"], ref=ref["greek"], rows=res)
        print(f"\n=== {prod}: N={N}, R={R}; reference = the correction of this repository", flush=True)
        for k, nm in enumerate(ref["names"]):
            g = ref["greek"][k]; line = f"  {nm:17s} ref {g:+.5g}"
            for key, r_ in res.items():
                rm = math.sqrt(r_["std"][k] ** 2 * (R - 1) / R + (r_["mean"][k] - g) ** 2)
                line += f"\n      {key:32s} mean {r_['mean'][k]:+.5g}  bias {100*(r_['mean'][k]-g)/abs(g):+6.2f}%  sd {100*r_['std'][k]/abs(g):6.2f}%  RMSE {100*rm/abs(g):6.2f}%  {r_['time']:.2f}s"
            print(line, flush=True)
    except Exception:
        print(f"\n=== {prod}: FAILED\n" + traceback.format_exc(), flush=True)
    json.dump(out, open(os.environ.get("OUT", "bench_gva.json"), "w"), indent=1)
