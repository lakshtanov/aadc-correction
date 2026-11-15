"""
example_heston_rqmc.py — δ-correction + randomized QMC (Sobol) on a Heston digital; the variance scheme matters.

Heston: S0 = K = 100, T = 2, 8 steps, v0 = theta = 0.04, kappa = 2, xi = 0.3, rho = -0.7; payoff 1{S_T > K}.
Greeks: delta, d/d(xi), d/d(kappa).

Two discretizations of the variance:
  euler_floor  Euler with truncation v+ = max(v, 0.1 theta)  — a kink: dg/dtheta jumps on oblique surfaces
  alfonsi      drift-implicit scheme for y = sqrt(v) (Alfonsi 2005), no floor, smooth; needs 4 kappa theta > xi^2

For each: standard error of the Greeks with MC and with RQMC (CorrectionDriverRQMC, automatic ordering),
same number of paths. Expected: RQMC gain ~1-3x with the floor, ~10-15x with the smooth scheme.
"""
import sys, os, io, math, contextlib, time
import numpy as np
import aadc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from correction_driver_vec2 import CorrectionDriverVec2
from rqmc import CorrectionDriverRQMC

S0, K, T, N_STEPS = 100.0, 100.0, 2.0, 8
V0, THETA_V, KAPPA, XI, RHO, R = 0.04, 0.04, 2.0, 0.3, -0.7, 0.03
N_PATHS, N_REP, THREADS = 2 ** 13, 16, 8


def record(scheme):
    dt = T / N_STEPS
    fn = aadc.Functions()
    fn.start_recording(register_switches=aadc.SWITCH_JUMP)
    S = aadc.idouble(S0); aS = S.mark_as_input()
    xi = aadc.idouble(XI); axi = xi.mark_as_input()
    ka = aadc.idouble(KAPPA); aka = ka.mark_as_input()
    Z = [aadc.idouble(0.0) for _ in range(2 * N_STEPS)]; za = [z.mark_as_input() for z in Z]
    logS, v, y = aadc.math.log(S), aadc.idouble(V0), aadc.math.sqrt(aadc.idouble(V0))
    c = aadc.idouble(1.0) + 0.5 * ka * dt
    for j in range(N_STEPS):
        zs, zv = Z[2 * j], Z[2 * j + 1]
        w = RHO * zs + math.sqrt(1 - RHO * RHO) * zv
        if scheme == "alfonsi":
            sv, vp = y, y * y
            a = y + 0.5 * xi * math.sqrt(dt) * w
            y = (a + aadc.math.sqrt(a * a + 4.0 * c * (0.5 * ka * aadc.idouble(THETA_V) - 0.125 * xi * xi) * dt)) / (2.0 * c)
        else:
            vp = aadc.math.max(v, aadc.idouble(0.1 * THETA_V)); sv = aadc.math.sqrt(vp)
            v = v + ka * (aadc.idouble(THETA_V) - vp) * dt + xi * sv * math.sqrt(dt) * w
        logS = logS + (aadc.idouble(R) - 0.5 * vp) * dt + sv * math.sqrt(dt) * zs
    P = aadc.iif(aadc.math.exp(logS) > aadc.idouble(K), aadc.idouble(1.0), aadc.idouble(0.0))
    rP = P.mark_as_output()
    fn.stop_recording()
    return fn, rP, za, [aS, axi, aka], {aS: S0, axi: XI, aka: KAPPA}


print(f"Heston digital, {N_STEPS} steps; {N_REP} x {N_PATHS} paths per method\n")
print(f"{'scheme':12s} {'method':5s} {'delta':>22s} {'d/dxi':>22s} {'d/dkappa':>22s}")
for scheme in ("euler_floor", "alfonsi"):
    fn, rP, za, th, thv = record(scheme)
    mc = CorrectionDriverVec2(fn, rP, za, th, num_threads=THREADS); mc.precompute_directions(thv)
    est = []
    for r in range(N_REP):
        with contextlib.redirect_stdout(io.StringIO()):
            est.append(mc.run(np.random.default_rng(100 + r).standard_normal((N_PATHS, len(za))))["total"])
    est = np.array(est); mc_m, mc_se = est.mean(0), est.std(0, ddof=1) / math.sqrt(N_REP)
    q = CorrectionDriverRQMC(fn, rP, za, th, num_threads=THREADS); q.precompute_directions(thv)
    out = q.run(N_PATHS, n_scramble=N_REP)
    fmt = lambda m, s: "  ".join(f"{a:+.5f}±{b:.1e}" for a, b in zip(m, s))
    print(f"{scheme:12s} {'MC':5s} {fmt(mc_m, mc_se)}")
    print(f"{'':12s} {'RQMC':5s} {fmt(out['total'], out['stderr'])}")
    print(f"{'':12s} {'gain':5s} " + "  ".join(f"{g:21.1f}x" for g in mc_se / out["stderr"]) + "\n")
