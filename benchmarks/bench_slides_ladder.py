"""
bench_slides_ladder.py — delta and gamma ladders vs spot for Benchmark 2 (3-asset Heston Phoenix autocallable,
Alfonsi scheme; product and model as in bench_slides_benchmarks.py). Spot S1 of asset 1 moves from 60 to 130,
the initial fixing stays at 100 (a contract constant).

On every ladder point, with the SAME N = 4000 paths for all points (common random numbers):
  correction   dV/dS1 = pathwise + correction (CorrectionDriverVec2);
               gamma  = central difference of two corrected deltas at S1 ± h, h = 1
  bump         central bump of the price on the same 4000 paths, h = 1 (delta and second difference for gamma)
Reference: central bump with common random numbers on AADC, 8M paths (independent seed), h = 2.

The corrected estimator is a smooth function of S1 for fixed paths (the payoff jumps are integrated analytically
along the Newton line), so its ladder is smooth already at 4000 paths and its finite difference gives a usable
gamma; the bump of the discontinuous payoff is not.

Usage:  OPENBLAS_NUM_THREADS=1 python benchmarks/bench_slides_ladder.py      (QUICK=1: coarse ladder, 1M reference)
        PLOTS=1 also writes fig_ladder_b2.pdf / .png (needs matplotlib).  FROM_JSON=<file>: plots only.
Raw results -> bench_slides_ladder.json (OUT=...).
"""
import sys, os, io, json, math, contextlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

QUICK = os.environ.get("QUICK", "0") == "1"
THREADS = int(os.environ.get("THREADS", "10"))
N = 4000
LADDER = np.arange(60.0, 130.01, 10.0 if QUICK else 2.5)
H, H_REF = 1.0, 2.0
REF_M = 1 if QUICK else 8
OUT = os.environ.get("OUT", "bench_slides_ladder.json")


def compute():
    import aadc
    from correction_driver_vec2 import CorrectionDriverVec2
    from bench_slides_benchmarks import tape_b2
    t = tape_b2(); fn, rP, za = t["fn"], t["rP"], t["za"]; aS1 = t["th"][0]; thv = dict(t["thv"])
    Z = np.random.RandomState(42).randn(N, len(za))
    zin = {a: np.ascontiguousarray(Z[:, j]) for j, a in enumerate(za)}

    def corrected_delta(s):
        v = dict(thv); v[aS1] = s
        d = CorrectionDriverVec2(fn, rP, za, [aS1], num_threads=THREADS); d.precompute_directions(v)
        with contextlib.redirect_stdout(io.StringIO()):
            return float(d.run(Z)["total"][0])

    def price(inputs, s):
        inp = dict(inputs); inp.update(thv); inp[aS1] = s
        return aadc.evaluate(funcs=fn, request={rP: []}, inputs=inp, workers=THREADS)[0][rP]

    rows = []
    for s in LADDER:
        dm, d0, dp = corrected_delta(s - H), corrected_delta(s), corrected_delta(s + H)
        pm, p0, pp = (price(zin, s + k * H).mean() for k in (-1, 0, 1))
        rng = np.random.default_rng(20261115); acc = np.zeros(4); n = 0        # sums: delta, delta^2, gamma, gamma^2
        while n < REF_M * 1_000_000:
            Zr = rng.standard_normal((500_000, len(za))); inp = {a: np.ascontiguousarray(Zr[:, j]) for j, a in enumerate(za)}
            qm, q0, qp = (price(inp, s + k * H_REF) for k in (-1, 0, 1))
            dl = (qp - qm) / (2 * H_REF); gm = (qp - 2 * q0 + qm) / H_REF ** 2
            acc += [dl.sum(), (dl * dl).sum(), gm.sum(), (gm * gm).sum()]; n += len(Zr)
        rd, rg = acc[0] / n, acc[2] / n
        row = dict(S1=float(s), corr_delta=d0, corr_gamma=(dp - dm) / (2 * H),
                   bump_delta=(pp - pm) / (2 * H), bump_gamma=(pp - 2 * p0 + pm) / H ** 2,
                   ref_delta=rd, ref_delta_se=math.sqrt((acc[1] / n - rd * rd) / n),
                   ref_gamma=rg, ref_gamma_se=math.sqrt((acc[3] / n - rg * rg) / n))
        rows.append(row)
        print(f"S1={s:6.1f}  delta: corr {d0:+.4f}  bump {row['bump_delta']:+.4f}  ref {rd:+.4f}±{row['ref_delta_se']:.4f}   "
              f"gamma: corr {row['corr_gamma']:+.5f}  bump {row['bump_gamma']:+.5f}  ref {rg:+.5f}±{row['ref_gamma_se']:.5f}", flush=True)
    return dict(N=N, h=H, h_ref=H_REF, ref_paths_millions=REF_M, rows=rows)


def plot(J, out="fig_ladder_b2"):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    r = J["rows"]; S = np.array([x["S1"] for x in r])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    for ax, key, title in zip(axes, ("delta", "gamma"), ("Delta  ∂V/∂S₁", "Gamma  ∂²V/∂S₁²")):
        ref = np.array([x[f"ref_{key}"] for x in r]); se = np.array([x[f"ref_{key}_se"] for x in r])
        ax.plot(S, [x[f"bump_{key}"] for x in r], color="#eb6834", lw=1.4, marker="o", ms=3.5, label=f"bump, {J['N']:,} paths")
        ax.plot(S, [x[f"corr_{key}"] for x in r], color="#2a78d6", lw=2.4, marker="o", ms=4, label=f"correction, {J['N']:,} paths")
        ax.plot(S, ref, color="#0b0b0b", lw=1.1, ls=(0, (4, 2)), zorder=5, label=f"reference: bump, {J['ref_paths_millions']}M paths")
        ax.axhline(0, color="#8a8984", lw=0.6); ax.axvline(100, color="#8a8984", lw=0.6, ls=":")
        ax.set_title(title, loc="left"); ax.set_xlabel("spot S₁ (fixing 100)"); ax.grid(True, color="#e4e3df", lw=0.6)
    lo, hi = np.percentile([x["ref_gamma"] for x in r], [0, 100]); pad = 0.6 * (hi - lo)
    axes[1].set_ylim(lo - pad, hi + pad)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=150)
    print(f"figures -> {out}.pdf, {out}.png")


if __name__ == "__main__":
    if os.environ.get("FROM_JSON"):
        J = json.load(open(os.environ["FROM_JSON"]))
    else:
        J = compute(); json.dump(J, open(OUT, "w"), indent=1); print(f"raw results -> {OUT}")
    if os.environ.get("PLOTS") == "1" or os.environ.get("FROM_JSON"):
        plot(J)
