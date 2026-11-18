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
        N=1024 SOBOL=1 ...  the same ladder on N scrambled Sobol points (one scrambling, CorrectionDriverRQMC's
                            automatic ordering computed once at S1 = 100), correction and bump on the same points.
                            ORDERING=bridge: Brownian bridge instead (option A, no pilot paths; the tape layout
                            of Benchmark 2 is known: tape_b2()["bridge_factors"]).
        REF_FROM=<json>     reuse the reference of an earlier run on the same ladder (it does not depend on N).
        PLOTS=1 also writes fig_ladder_b2.pdf / .png (needs matplotlib).  FROM_JSON=<file>: plots only.
Raw results -> bench_slides_ladder.json (OUT=...).
"""
import sys, os, io, json, math, contextlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

QUICK = os.environ.get("QUICK", "0") == "1"
THREADS = int(os.environ.get("THREADS", "10"))
N = int(os.environ.get("N", "4000"))
SOBOL = os.environ.get("SOBOL", "0") == "1"
ORDERING = os.environ.get("ORDERING", "active_subspace")      # Sobol only: "active_subspace" (option B) or "bridge" (A)
N_PILOT = int(os.environ.get("N_PILOT", "2000"))              # option B: pilot paths for Q
LADDER = np.arange(60.0, 130.01, 10.0 if QUICK else 2.5)
H, H_REF = 1.0, 2.0
REF_M = 1 if QUICK else 8
OUT = os.environ.get("OUT", "bench_slides_ladder.json")


def compute():
    import aadc, time
    from correction_driver_vec2 import CorrectionDriverVec2
    from bench_slides_benchmarks import tape_b2
    t = tape_b2(); fn, rP, za = t["fn"], t["rP"], t["za"]; aS1 = t["th"][0]; thv = dict(t["thv"])
    if SOBOL:
        from rqmc import CorrectionDriverRQMC
        kw = dict(ordering="bridge", bridge_factors=t["bridge_factors"]) if ORDERING == "bridge" else dict(pilot_seed=999, n_pilot=N_PILOT)
        q = CorrectionDriverRQMC(fn, rP, za, [aS1], num_threads=THREADS, **kw)
        with contextlib.redirect_stdout(io.StringIO()):
            q.precompute_directions(thv)                   # warm-up; the second call is timed
            q.driver.precompute_directions(thv); t0 = time.perf_counter(); q.precompute_directions(thv)
        t_q = time.perf_counter() - t0                     # driver directions (also in every corrected delta) + Q
        Z = q.sample(N, 0)
    else:
        Z = np.random.RandomState(42).randn(N, len(za)); t_q = 0.0
    REF = {r["S1"]: r for r in json.load(open(os.environ["REF_FROM"]))["rows"]} if os.environ.get("REF_FROM") else None
    zin = {a: np.ascontiguousarray(Z[:, j]) for j, a in enumerate(za)}

    def corrected_delta(s):
        v = dict(thv); v[aS1] = s
        d = CorrectionDriverVec2(fn, rP, za, [aS1], num_threads=THREADS); d.precompute_directions(v)
        with contextlib.redirect_stdout(io.StringIO()):
            return float(d.run(Z)["total"][0])

    def price(inputs, s):
        inp = dict(inputs); inp.update(thv); inp[aS1] = s
        return aadc.evaluate(funcs=fn, request={rP: []}, inputs=inp, workers=THREADS)[0][rP]

    corrected_delta(100.0); price(zin, 100.0)                              # warm-up
    t0 = time.perf_counter(); corrected_delta(100.0); t_delta = time.perf_counter() - t0
    Zt = np.random.default_rng(3).standard_normal((1 << 18, len(za)))
    zt = {a: np.ascontiguousarray(Zt[:, j]) for j, a in enumerate(za)}
    t0 = time.perf_counter(); price(zt, 100.0); t_price_path = (time.perf_counter() - t0) / len(Zt)
    rows = []
    for s in LADDER:
        dm, d0, dp = corrected_delta(s - H), corrected_delta(s), corrected_delta(s + H)
        pm, p0, pp = (price(zin, s + k * H).mean() for k in (-1, 0, 1))
        rng = np.random.default_rng(20261115); acc = np.zeros(4); n = 0        # sums: delta, delta^2, gamma, gamma^2
        while REF is None and n < REF_M * 1_000_000:
            Zr = rng.standard_normal((500_000, len(za))); inp = {a: np.ascontiguousarray(Zr[:, j]) for j, a in enumerate(za)}
            qm, q0, qp = (price(inp, s + k * H_REF) for k in (-1, 0, 1))
            dl = (qp - qm) / (2 * H_REF); gm = (qp - 2 * q0 + qm) / H_REF ** 2
            acc += [dl.sum(), (dl * dl).sum(), gm.sum(), (gm * gm).sum()]; n += len(Zr)
        row = dict(S1=float(s), corr_delta=d0, corr_gamma=(dp - dm) / (2 * H),
                   bump_delta=(pp - pm) / (2 * H), bump_gamma=(pp - 2 * p0 + pm) / H ** 2)
        if REF is None:
            rd, rg = acc[0] / n, acc[2] / n
            row.update(ref_delta=rd, ref_delta_se=math.sqrt((acc[1] / n - rd * rd) / n),
                       ref_gamma=rg, ref_gamma_se=math.sqrt((acc[3] / n - rg * rg) / n))
        else:
            row.update({k: REF[float(s)][k] for k in ("ref_delta", "ref_delta_se", "ref_gamma", "ref_gamma_se")})
            rd, rg = row["ref_delta"], row["ref_gamma"]
        rows.append(row)
        print(f"S1={s:6.1f}  delta: corr {d0:+.4f}  bump {row['bump_delta']:+.4f}  ref {rd:+.4f}±{row['ref_delta_se']:.4f}   "
              f"gamma: corr {row['corr_gamma']:+.5f}  bump {row['bump_gamma']:+.5f}  ref {rg:+.5f}±{row['ref_gamma_se']:.5f}", flush=True)
    refm = json.load(open(os.environ["REF_FROM"]))["ref_paths_millions"] if REF is not None else REF_M
    return dict(N=N, sobol=SOBOL, ordering=ORDERING if SOBOL else None, n_pilot=N_PILOT if SOBOL and ORDERING != "bridge" else 0,
                h=H, h_ref=H_REF, ref_paths_millions=refm,
                t_q=t_q, t_delta=t_delta, t_price_path=t_price_path, rows=rows)


def summary(J):
    """RMSE over the ladder; CRN-bump paths (step h_ref) for the same RMSE, from the per-path spread of the reference
    bump; costs (one corrected delta incl. its directions, Q once, a bump of that size)."""
    r = J["rows"]; n_ref = J["ref_paths_millions"] * 1e6
    rmse = lambda a, b: math.sqrt(sum((x[a] - x[b]) ** 2 for x in r) / len(r))
    sd = lambda k: math.sqrt(sum(x[f"ref_{k}_se"] ** 2 * n_ref for x in r) / len(r))
    cd, cg, bd, bg = rmse("corr_delta", "ref_delta"), rmse("corr_gamma", "ref_gamma"), rmse("bump_delta", "ref_delta"), rmse("bump_gamma", "ref_gamma")
    nd, ng = (sd("delta") / cd) ** 2, (sd("gamma") / cg) ** 2
    pts = f"{J['N']} {'Sobol (' + (J.get('ordering') or 'active_subspace') + ')' if J.get('sobol') else 'MC'} points"
    print(f"\nRMSE over {len(r)} ladder points, {pts}: correction delta {cd:.4f} gamma {cg:.4f} | bump on the same points "
          f"delta {bd:.4f} gamma {bg:.4f}")
    print(f"CRN bump (h={J['h_ref']}) with the same RMSE: ~{nd / 1e3:.0f}K paths for delta, ~{ng / 1e3:.0f}K for gamma")
    if "t_delta" in J:
        tp = J["t_price_path"]
        print(f"cost ({THREADS} threads): corrected delta {J['t_delta']:.2f}s, Q once {J['t_q']:.3f}s, bump on the same "
              f"points {2 * J['N'] * tp * 1e3:.0f} ms; same-RMSE bump: delta {2 * nd * tp:.2f}s, gamma {3 * ng * tp:.2f}s "
              f"vs correction gamma (2 deltas + Q) {2 * J['t_delta'] + J['t_q']:.2f}s")


def plot(J, out=None):
    out = out or ("fig_ladder_b2_sobol%d" % J["N"] + ("_bridge" if J.get("ordering") == "bridge" else "")
                  if J.get("sobol") else "fig_ladder_b2")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 15})
    r = J["rows"]; S = np.array([x["S1"] for x in r]); pts = "Sobol paths" if J.get("sobol") else "paths"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, key, title in zip(axes, ("delta", "gamma"), ("Delta  ∂V/∂S₁", "Gamma  ∂²V/∂S₁²")):
        ref = np.array([x[f"ref_{key}"] for x in r]); se = np.array([x[f"ref_{key}_se"] for x in r])
        ax.plot(S, [x[f"bump_{key}"] for x in r], color="#eb6834", lw=1.4, marker="o", ms=3.5, label=f"bump, {J['N']:,} {pts}")
        ax.plot(S, [x[f"corr_{key}"] for x in r], color="#2a78d6", lw=2.4, marker="o", ms=4, label=f"correction, {J['N']:,} {pts}")
        ax.plot(S, ref, color="#0b0b0b", lw=1.1, ls=(0, (4, 2)), zorder=5, label=f"reference: bump, {J['ref_paths_millions']}M paths")
        ax.axhline(0, color="#8a8984", lw=0.6); ax.axvline(100, color="#8a8984", lw=0.6, ls=":")
        ax.set_title(title, loc="left"); ax.set_xlabel("spot S₁ (fixing 100)"); ax.grid(True, color="#e4e3df", lw=0.6)
    lo, hi = np.percentile([x["ref_gamma"] for x in r], [0, 100]); pad = 0.6 * (hi - lo)
    axes[1].set_ylim(lo - pad, hi + pad)
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out + ".pdf"); fig.savefig(out + ".png", dpi=150)
    print(f"figures -> {out}.pdf, {out}.png")


if __name__ == "__main__":
    if os.environ.get("FROM_JSON"):
        J = json.load(open(os.environ["FROM_JSON"]))
    else:
        J = compute(); json.dump(J, open(OUT, "w"), indent=1); print(f"raw results -> {OUT}")
    summary(J)
    if os.environ.get("PLOTS") == "1" or os.environ.get("FROM_JSON"):
        plot(J)
