"""
bench_slides_qmc.py — reproduce the numbers of the Quasi-Monte Carlo slides.

Product: 1-asset Heston digital 1{S_T > 100}, S0 = 100, T = 2, 8 steps, v0 = theta = 0.04, kappa = 2, xi = 0.3,
rho = -0.7, r = 0.03; Greeks dV/dS0, dV/dxi, dV/dkappa. Three variance schemes:
  euler_floor   Euler, v+ = max(v, 0.1 theta)                       (kink)
  smooth_floor  Euler, v+ = smooth max(v, 0.1 theta), width 0.1 theta
  alfonsi       drift-implicit scheme for sqrt(v), no floor           (smooth; 4 kappa theta > xi^2)

Slide "QMC Result: the Scheme Decides"
    For each scheme: correction driver with MC and with RQMC (CorrectionDriverRQMC, automatic ordering),
    N = 2^10..2^16, R = 16 replications. Error reduction RQMC/MC at 2^16, convergence order alpha of RQMC,
    max |bias| vs the reference in units of the combined standard error.
Slides "QMC Result: Equal Time vs Smoothing" / "Numbers"  (alfonsi scheme)
    RMSE (incl. bias) at wall-time budgets 0.05 s and 0.5 s per estimate for: correction + Sobol,
    correction + MC, smoothing + Sobol (Brownian bridge), smoothing + MC; smoothing = logistic of width eps,
    pathwise AAD, the best eps per budget (eps in 0.125..2, N up to 2^22).
Slide "QMC: Paths for the Same Accuracy"  (alfonsi)
    Paths per estimate (and wall time) needed to reach RMSE <= 1% and <= 0.5% of |Greek| for smoothing + MC,
    smoothing + Sobol (best eps for each target), correction + MC, correction + Sobol.
Reference: central bump with common random numbers on AADC, h = (0.5, 0.01, 0.05), 1.28e8 paths
    (second run with 2h as a check).

Seeds follow the original runs, so the driver estimates match them; wall times depend on the machine
(original: 10 physical cores of one socket, OPENBLAS_NUM_THREADS=1), so the equal-time numbers will move a bit.

Raw results are written to bench_slides_qmc.json (OUT=...). FROM_JSON=<file> prints the tables (and, with
PLOTS=1, writes fig_gain_by_scheme.pdf / fig_rmse_vs_time.pdf; needs matplotlib) from saved results
without recomputing.

Usage:  OPENBLAS_NUM_THREADS=1 python benchmarks/bench_slides_qmc.py           (~1 h)
        QUICK=1 ...                                                           (~5 min: 8M reference, smaller N)
        THREADS=10 (default)
"""
import sys, os, io, json, math, time, contextlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
if not os.environ.get("FROM_JSON"):         # tables/plots from saved results need neither aadc nor the drivers
    import aadc
    from correction_driver_vec2 import CorrectionDriverVec2
    from rqmc import CorrectionDriverRQMC, bridge_rotation, sobol_normals

QUICK = os.environ.get("QUICK", "0") == "1"
THREADS = int(os.environ.get("THREADS", "10"))
REF_M = 8 if QUICK else 128                     # reference paths, millions
M_DRV = (8, 14) if QUICK else (8, 16)           # driver N = 2^M_DRV
M_SMO = (8, 18) if QUICK else (8, 22)           # smoothing N
R = 16
EPS = [0.125, 0.25, 0.5, 1.0, 2.0]
TARGETS = (0.01, 0.005)                         # relative RMSE targets for the "same accuracy" slide
OUT = os.environ.get("OUT", "bench_slides_qmc.json")
NAMES = ["dV/dS0", "dV/dxi", "dV/dkappa"]
N_STEPS, FLOOR = 8, 0.004


def tape(scheme, smooth=None, xi0=0.3, kappa0=2.0, v0=0.04, rho=-0.7):
    fn = aadc.Functions(); fn.start_recording(register_switches=aadc.SWITCH_JUMP)
    S = aadc.idouble(100.0); aS = S.mark_as_input()
    xi = aadc.idouble(xi0); axi = xi.mark_as_input()
    ka = aadc.idouble(kappa0); aka = ka.mark_as_input()
    Z = [aadc.idouble(0.0) for _ in range(2 * N_STEPS)]; za = [q.mark_as_input() for q in Z]
    logS, v, dt = aadc.math.log(S), aadc.idouble(v0), 2.0 / N_STEPS
    y = aadc.math.sqrt(aadc.idouble(v0))
    P = aadc.idouble(0.0)
    def pay(logS):
        if smooth is None:
            return aadc.iif(aadc.math.exp(logS) > aadc.idouble(100.0), aadc.idouble(1.0), aadc.idouble(0.0))
        return aadc.idouble(0.5) * (aadc.idouble(1.0) + aadc.math.tanh((aadc.math.exp(logS) - aadc.idouble(100.0)) / aadc.idouble(2.0 * smooth)))
    for j in range(N_STEPS):
        zs, zv = Z[2 * j], Z[2 * j + 1]
        w = rho * zs + math.sqrt(1 - rho * rho) * zv
        if scheme == "alfonsi":
            sv = y; vp = y * y
            a = y + 0.5 * xi * math.sqrt(dt) * w
            c = (aadc.idouble(1.0) + 0.5 * ka * dt)
            y = (a + aadc.math.sqrt(a * a + 4.0 * c * (0.5 * ka * aadc.idouble(v0) - 0.125 * xi * xi) * dt)) / (2.0 * c)
            logS = logS + (aadc.idouble(0.03) - 0.5 * vp) * dt + sv * math.sqrt(dt) * zs
            if j == N_STEPS - 1:
                P = P + pay(logS)
            continue
        if scheme == "smooth_floor":
            vp = 0.5 * (v + aadc.idouble(FLOOR) + aadc.math.sqrt((v - aadc.idouble(FLOOR)) * (v - aadc.idouble(FLOOR)) + aadc.idouble(FLOOR * FLOOR)))
            sv = aadc.math.sqrt(vp)
        else:
            vp = aadc.math.max(v, aadc.idouble(FLOOR)); sv = aadc.math.sqrt(vp)
        v = v + ka * (aadc.idouble(v0) - vp) * dt + xi * sv * math.sqrt(dt) * w
        logS = logS + (aadc.idouble(0.03) - 0.5 * vp) * dt + sv * math.sqrt(dt) * zs
        if j == N_STEPS - 1:
            P = P + pay(logS)
    rP = P.mark_as_output(); fn.stop_recording()
    return fn, rP, za, [aS, axi, aka], {aS: 100.0, axi: xi0, aka: kappa0}


def reference(scheme):
    """CRN central bump on AADC (forward passes only), h and 2h."""
    fn, rP, za, th, thv = tape(scheme)
    out = []
    for H in ((0.5, 0.01, 0.05), (1.0, 0.02, 0.1)):
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
    (g, se), (g2, se2) = out
    print(f"  reference {scheme:12s} " + "  ".join(f"{nm} {a:+.5f}±{b:.1e}" for nm, a, b in zip(NAMES, g, se))
          + "   (h vs 2h: " + " ".join(f"{(a - b) / math.hypot(c, d):+.1f}σ" for a, b, c, d in zip(g, g2, se, se2)) + ")", flush=True)
    return g, se


def run_driver(scheme):
    """MC and RQMC (automatic ordering) estimates of the correction driver: {setup: [(N, mean, std, time/est)]}."""
    fn, rP, za, th, thv = tape(scheme)
    mc = CorrectionDriverVec2(fn, rP, za, th, num_threads=THREADS); mc.precompute_directions(thv)
    q = CorrectionDriverRQMC(fn, rP, za, th, num_threads=THREADS, pilot_seed=999); q.precompute_directions(thv)
    res = {"MC": [], "RQMC": []}
    for m in range(M_DRV[0], M_DRV[1] + 1):
        N = 2 ** m
        for setup in ("MC", "RQMC"):
            est, t = [], 0.0
            for r in range(R):
                Z = np.random.default_rng(10_000 + r).standard_normal((N, len(za))) if setup == "MC" else q.sample(N, r)
                t0 = time.perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    est.append((mc if setup == "MC" else q.driver).run(Z)["total"])
                t += time.perf_counter() - t0
            est = np.array(est); res[setup].append((N, est.mean(0), est.std(0, ddof=1), t / R))
    return res


def run_smoothing(scheme):
    """{setup: [(eps, N, mean, std, time/est)]} for smoothing + MC and smoothing + Sobol (Brownian bridge)."""
    Q = bridge_rotation(2 * N_STEPS, [list(range(0, 2 * N_STEPS, 2)), list(range(1, 2 * N_STEPS, 2))])
    res = {"MC": [], "RQMC": []}
    for eps in EPS:
        fn, rP, za, th, thv = tape(scheme, smooth=eps)
        def estimate(Z):
            inp = {a: np.ascontiguousarray(Z[:, j]) for j, a in enumerate(za)}; inp.update(thv)
            _, d, _ = aadc.evaluate(funcs=fn, request={rP: th}, inputs=inp, workers=THREADS)
            return np.array([np.mean(d[rP][a]) for a in th])
        for m in range(M_SMO[0], M_SMO[1] + 1):
            N = 2 ** m
            for setup in ("MC", "RQMC"):
                est, t = [], 0.0
                for r in range(R):
                    Z = np.random.default_rng(10_000 + r).standard_normal((N, len(za))) if setup == "MC" else sobol_normals(N, len(za), r) @ Q.T
                    t0 = time.perf_counter(); est.append(estimate(Z)); t += time.perf_counter() - t0
                est = np.array(est); res[setup].append((eps, N, est.mean(0), est.std(0, ddof=1), t / R))
    return res


def rmse(mean, std, ref):
    return np.sqrt(std ** 2 * (R - 1) / R + (mean - ref) ** 2)


def best_within(points, budget, k, ref):
    """Lowest RMSE among (time <= budget) points; points: iterable of (time, mean, std)."""
    c = [rmse(mn[k], sd[k], ref[k]) for t, mn, sd in points if t <= budget]
    return min(c) if c else float("nan")


def frontier_points(rows, k, ref, kind):
    """[(time, rmse, N, eps)] for driver rows (N, mean, std, t) or smoothing rows (eps, N, mean, std, t)."""
    if kind == "drv":
        return [(t, rmse(mn[k], sd[k], ref[k]), N, None) for N, mn, sd, t in rows]
    return [(t, rmse(mn[k], sd[k], ref[k]), N, e) for e, N, mn, sd, t in rows]


def first_to_reach(points, target):
    """Cheapest point (by paths, then time) with rmse <= target; None if never reached."""
    ok = [p for p in points if p[1] <= target]
    return min(ok, key=lambda p: (p[2], p[0])) if ok else None


def make_plots(refs, drv, smo, outdir="."):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    C = {"correction + Sobol": "#2a78d6", "correction + MC": "#1baf7a", "smoothing + Sobol (best ε)": "#eb6834", "smoothing + MC (best ε)": "#eda100"}
    g, se = refs["alfonsi"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1))
    for k, ax in enumerate(axes):
        def env(points):
            pts, out, best = sorted((t, e) for t, e, _, _ in points), [], math.inf
            for t, e in pts:
                if e < best: best = e; out.append((t, e))
            return np.array(out).T
        curves = {"correction + Sobol": env(frontier_points(drv["alfonsi"]["RQMC"], k, g, "drv")),
                  "correction + MC": env(frontier_points(drv["alfonsi"]["MC"], k, g, "drv")),
                  "smoothing + Sobol (best ε)": env(frontier_points(smo["RQMC"], k, g, "smo")),
                  "smoothing + MC (best ε)": env(frontier_points(smo["MC"], k, g, "smo"))}
        for nm, (t, e) in curves.items():
            ax.plot(t, e, "-" if "Sobol" in nm else "--", color=C[nm], lw=2, marker="o", ms=4, label=nm)
        ax.axhline(se[k], color="#52514e", lw=1, ls=":"); ax.text(2e-3, se[k] * 1.15, "reference s.e.", fontsize=8, color="#52514e")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(2e-3, 3); ax.set_title(["Δ = ∂V/∂S₀", "∂V/∂ξ (vol of vol)", "∂V/∂κ (mean reversion)"][k], loc="left")
        ax.set_xlabel("wall time per estimate, s"); ax.grid(True, color="#e4e3df")
        if k == 0: ax.set_ylabel("RMSE vs reference (incl. bias)")
    h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(outdir, "fig_rmse_vs_time.pdf"))
    gains = np.array([[drv[s]["MC"][-1][2][k] / drv[s]["RQMC"][-1][2][k] for k in range(3)] for s in ("euler_floor", "smooth_floor", "alfonsi")])
    fig, ax = plt.subplots(figsize=(7.2, 3.6)); x = np.arange(3); w = 0.26
    for i, (lab, col) in enumerate(zip(["Euler, max(v, 0.1θ)", "Euler, smooth floor", "Alfonsi (no floor)"], ["#eda100", "#eb6834", "#2a78d6"])):
        ax.bar(x + (i - 1) * w, gains[i], w * 0.92, color=col, label=lab)
        for xi, gg in zip(x + (i - 1) * w, gains[i]): ax.text(xi, gg + 0.25, f"{gg:.1f}×", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(["Δ", "∂V/∂ξ", "∂V/∂κ"]); ax.axhline(1, color="#52514e", lw=0.8)
    ax.set_ylabel(f"error reduction, Sobol vs MC\n(same N = $2^{{{M_DRV[1]}}}$)"); ax.set_ylim(0, max(18.5, gains.max() * 1.15))
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3); fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_gain_by_scheme.pdf"))


if __name__ == "__main__":
    print(f"Quasi-Monte Carlo slides — Heston digital ({'QUICK' if QUICK else 'full'}), {THREADS} threads\n")
    if os.environ.get("FROM_JSON"):
        J = json.load(open(os.environ["FROM_JSON"]))
        R = J["R"]
        refs = {k: (np.array(a), np.array(b)) for k, (a, b) in J["refs"].items()}
        drv = {s: {st: [(N, np.array(m), np.array(sd), t) for N, m, sd, t in rows] for st, rows in d.items()} for s, d in J["driver"].items()}
        smo = {st: [(e, N, np.array(m), np.array(sd), t) for e, N, m, sd, t in rows] for st, rows in J["smoothing"].items()}
        M_DRV = (int(math.log2(drv["alfonsi"]["MC"][0][0])), int(math.log2(drv["alfonsi"]["MC"][-1][0])))
        print(f"results loaded from {os.environ['FROM_JSON']}")
    else:
        refs, drv = {}, {}
        for scheme in ("euler_floor", "smooth_floor", "alfonsi"):
            refs[scheme] = reference(scheme)
            drv[scheme] = run_driver(scheme)
        smo = run_smoothing("alfonsi")
        tolist = lambda x: x.tolist() if isinstance(x, np.ndarray) else x
        json.dump(dict(quick=QUICK, threads=THREADS, R=R, names=NAMES, ref_paths_millions=REF_M,
                       refs={k: [tolist(a), tolist(b)] for k, (a, b) in refs.items()},
                       driver={s: {st: [[N, tolist(m), tolist(sd), t] for N, m, sd, t in rows] for st, rows in d.items()} for s, d in drv.items()},
                       smoothing={st: [[e, N, tolist(m), tolist(sd), t] for e, N, m, sd, t in rows] for st, rows in smo.items()}),
              open(OUT, "w"), indent=1)
        print(f"raw results -> {OUT}")

    print(f"\n{'=' * 78}\nSLIDE: QMC Result — the Scheme Decides  (RQMC error reduction over MC at N=2^{M_DRV[1]}, (alpha))\n{'=' * 78}")
    print(f"{'scheme':14s}" + "".join(f"{nm:>18s}" for nm in NAMES) + f"{'max |bias|':>12s}")
    for scheme in drv:
        g, se = refs[scheme]
        mcN, mcm, mcs, _ = drv[scheme]["MC"][-1]; qN, qm, qs, _ = drv[scheme]["RQMC"][-1]
        Ns = np.array([x[0] for x in drv[scheme]["RQMC"]])
        cells = []
        for k in range(3):
            alpha = -np.polyfit(np.log(Ns), np.log([x[2][k] for x in drv[scheme]["RQMC"]]), 1)[0]
            cells.append(f"{mcs[k] / qs[k]:8.1f}x ({alpha:.2f})")
        bias = max(abs(qm[k] - g[k]) / math.hypot(qs[k] / math.sqrt(R), se[k]) for k in range(3))
        print(f"{scheme:14s}" + "".join(f"{c:>18s}" for c in cells) + f"{bias:10.1f}σ")

    print(f"\n{'=' * 78}\nSLIDE: QMC Result — Equal Time vs Smoothing / Numbers  (alfonsi; RMSE incl. bias)\n{'=' * 78}")
    g, se = refs["alfonsi"]
    print("reference: " + ", ".join(f"{nm} {a:+.4f}±{b:.1e}" for nm, a, b in zip(NAMES, g, se)))
    print(f"{'Greek':10s} {'time':>6s} {'corr+Sobol':>11s} {'corr+MC':>9s} {'smooth+Sobol':>13s} {'smooth+MC':>10s} {'gain':>6s}")
    for k, nm in enumerate(NAMES):
        for b in (0.05, 0.5):
            cs = best_within([(x[3], x[1], x[2]) for x in drv["alfonsi"]["RQMC"]], b, k, g)
            cm = best_within([(x[3], x[1], x[2]) for x in drv["alfonsi"]["MC"]], b, k, g)
            ss = best_within([(x[4], x[2], x[3]) for x in smo["RQMC"]], b, k, g)
            sm = best_within([(x[4], x[2], x[3]) for x in smo["MC"]], b, k, g)
            print(f"{nm:10s} {b:5.2f}s {cs:11.1e} {cm:9.1e} {ss:13.1e} {sm:10.1e} {ss / cs:5.0f}x")

    print(f"\n{'=' * 78}\nSLIDE: QMC — Paths for the Same Accuracy  (alfonsi; RMSE incl. bias <= target · |Greek|)\n{'=' * 78}")
    methods = [("smoothing + MC", smo["MC"], "smo"), ("smoothing + Sobol", smo["RQMC"], "smo"),
               ("correction + MC", drv["alfonsi"]["MC"], "drv"), ("correction + Sobol", drv["alfonsi"]["RQMC"], "drv")]
    for tgt in TARGETS:
        print(f"target {tgt:.1%}:")
        for nm, rows, kind in methods:
            cells = []
            for k in range(3):
                hit = first_to_reach(frontier_points(rows, k, g, kind), tgt * abs(g[k]))
                if hit is None:
                    nmax = max(p[2] for p in frontier_points(rows, k, g, kind))
                    cells.append(f"not reached (N<=2^{int(math.log2(nmax))})")
                else:
                    t, e, N, eps = hit
                    nmin = min(p[2] for p in frontier_points(rows, k, g, kind))
                    cells.append(f"N{'<=' if N == nmin else '='}2^{int(math.log2(N))} {t:.3f}s" + (f" ε={eps}" if eps is not None else ""))
            print(f"  {nm:20s} " + " | ".join(f"{NAMES[k]}: {c}" for k, c in enumerate(cells)))
    if os.environ.get("PLOTS") == "1":
        make_plots(refs, drv, smo); print("figures -> fig_rmse_vs_time.pdf, fig_gain_by_scheme.pdf")
    print("\nDone. Compare with the slides (driver estimates match the original seeds; times are machine-dependent).")
