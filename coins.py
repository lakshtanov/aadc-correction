"""
coins.py — exact integration over "coins" for the delta-correction driver (CorrectionDriverVec2(..., coins="auto")).

A coin is an input normal z_j that enters the payoff only through comparisons, each of them depending on z_j alone:
g = A(other inputs) - f(z_j), f monotone. The standard example is the default threshold of an intensity (Cox) model,
E = -log Phi(-z_E) ~ Exp(1): default in the first step k with A_k = sum lambda dt >= E. Along the direction of the
coin every comparison g_k = A_k - E(z_E) is linear in E, so the delta correction of g_k taken along e_{z_E}
(Theorem 1 with v = e_{z_E}) has
    root      z* = -Phi^{-1}(exp(-A_k))            (closed form, no Newton; it exists for every path and every k)
    weight    phi(z*) / |dg_k/dz_E| = exp(-A_k)     (the density of E at the boundary; no division by the rate)
    jump      F(E = A_k - h) - F(E = A_k + h)       (default in step k minus default later)
    dg/dtheta dA_k/dtheta                           (the threshold does not depend on theta)
and the boundary terms of all steps of every path are integrated over E exactly. This is the same derivative as
the delta correction with the default directions (grid model), without root-finding and without the heavy tails of
estimators that divide by the intensity.

Coins are found on the tape (coins="auto") on pilot paths:
  stage 1, all candidates at once: inputs with d payoff/dz = 0; their comparison derivatives must not change when all
           other inputs are redrawn and must have one sign;
  stage 2, each survivor: separable (only this input redrawn kept), monotone, d payoff/dz = 0; law f = -log Phi(-z)
           checked to 1e-8 (exact term); any other monotone law -> the delta driver along +-e_{z_j} (Newton).

Fast path (optional): the product marks the default with CoxDefault (one call per step). Then the jumps between steps
that do not end on a date come from the step outputs at E = +inf (alive_k (pay_k - pay_{k+1})), date steps take two
forward passes, and sum_k w_k dA_k/dtheta comes from ONE reverse pass of the recorded output sum_k w_k A_k (w_k: weight
inputs). It is checked on pilot paths against the two-pass jumps; otherwise two passes on every step are used.
"""
import math
import warnings
import numpy as np
import aadc

_c = aadc.idouble


def _as_active(x):
    y = _c(0.0) + x
    return y if hasattr(y, "mark_as_output") else _c(float(y))


class CoxDefault:
    """Optional markup of a default with an exponential threshold (fast path of the coin term).

        dflt = CoxDefault()
        for k in range(1, N + 1):
            hit = dflt.step(k, A_prev=Lam, rate=lam, dt=dt, threshold=E, alive=alive, pay=pay_k,
                            on_date=(k in date_steps))
            ...                                  # the product uses hit as usual
        dflt.finalize()                          # before stop_recording

    pay_k: value paid if the default is in step k; alive: the note is alive before the default check of step k;
    on_date: cash flows of the product happen at the end of step k (after the default check)."""

    def __init__(self):
        self.steps = []                      # (k, g Result, on_date)
        self.step_pay, self.step_alive, self.step_w = [], [], []
        self._wsum = None
        self.wsum = None

    def step(self, k, A_prev, rate, dt, threshold, alive, pay, on_date=False):
        A_new = A_prev + rate * _c(dt)
        g = (A_new - threshold).mark_as_output()
        w = _c(0.0); self.step_w.append(w.mark_as_input_no_diff())
        self._wsum = w * A_new if self._wsum is None else self._wsum + w * A_new
        self.step_pay.append(_as_active(pay)); self.step_alive.append(_as_active(alive))
        self.steps.append((k, g, bool(on_date)))
        return aadc.iif(A_new >= threshold, _c(1.0), _c(0.0))

    def finalize(self):
        self.wsum = self._wsum.mark_as_output()
        self.step_pay = [v.mark_as_output() for v in self.step_pay]
        self.step_alive = [v.mark_as_output() for v in self.step_alive]

    def zero_inputs(self):
        """{weight input: 0.0}: add to the values passed to precompute_directions (every replay must set them)."""
        return {w: 0.0 for w in self.step_w}


class Coins:
    """Coin detection and the exact coin term for a CorrectionDriverVec2 (used by the driver, not directly)."""

    def __init__(self, drv, defaults=(), n_pilot=256, seed=20261117):
        self.drv, self.defaults, self.n_pilot, self.seed = drv, list(defaults), n_pilot, seed
        self.exp, self.generic, self.fast, self.diagnostics = [], [], {}, {}
        self.zpos = {a: j for j, a in enumerate(drv.z_args)}

    # -- evaluation ------------------------------------------------------------------------------------------------
    def _eval(self, request, Z, extra=None):
        drv = self.drv
        inp = {a: np.ascontiguousarray(Z[:, j]) for j, a in enumerate(drv.z_args)}
        inp.update({a: float(v) for a, v in drv.theta_values.items()})
        if extra:
            inp.update(extra)
        return drv.vf._evaluate(drv.funcs, request, inp, drv.num_threads)[:2]

    # -- setup -----------------------------------------------------------------------------------------------------
    def setup(self):
        """Find the coins among the driver's indicators; returns the set of indicator positions they cover."""
        from scipy.special import ndtr
        drv = self.drv
        Zp = np.random.default_rng(self.seed).standard_normal((self.n_pilot, drv.d))
        Zq = np.random.default_rng(self.seed + 1).standard_normal(Zp.shape)
        allg = list(drv.indicator_res)
        _, dP = self._eval({drv.payoff_res: list(drv.z_args)}, Zp)
        cand = [j for j, a in enumerate(drv.z_args) if np.max(np.abs(np.asarray(dP[drv.payoff_res][a]))) == 0.0]
        if not cand:
            return set()
        # stage 1: all candidates at once
        n1 = min(64, len(Zp)); ca = [drv.z_args[j] for j in cand]
        Zs = Zq[:n1].copy(); Zs[:, cand] = Zp[:n1, cand]
        _, D1 = self._eval({g: ca for g in allg}, Zp[:n1])
        _, D2 = self._eval({g: ca for g in allg}, Zs)
        keep = []
        for j, a in zip(cand, ca):
            involved, ok = False, True
            for g in allg:
                x1 = np.broadcast_to(np.asarray(D1[g][a]), (n1,))
                if np.max(np.abs(x1)) == 0.0:
                    continue
                involved = True
                x2 = np.broadcast_to(np.asarray(D2[g][a]), (n1,))
                if np.max(np.abs(x1 - x2)) > 1e-10 * np.max(np.abs(x1)) or not (np.all(x1 > 0) or np.all(x1 < 0)):
                    ok = False; break
            if involved and ok:
                keep.append(j)
        self.diagnostics["candidates"] = dict(screened=len(cand), passed=len(keep))
        covered = set()
        for j in keep:                                            # stage 2: one by one
            a = drv.z_args[j]
            Zr = Zq.copy(); Zr[:, j] = Zp[:, j]
            _, d1 = self._eval({g: [a] for g in allg}, Zp)
            _, d2 = self._eval({g: [a] for g in allg}, Zr)
            pos, ok = [], True
            for i, g in enumerate(allg):
                x1 = np.broadcast_to(np.asarray(d1[g][a]), (len(Zp),)); x2 = np.broadcast_to(np.asarray(d2[g][a]), (len(Zp),))
                if np.max(np.abs(x1)) == 0.0:
                    continue
                if np.max(np.abs(x1 - x2)) > 1e-10 * np.max(np.abs(x1)) or not (np.all(x1 > 0) or np.all(x1 < 0)):
                    ok = False; break
                pos.append(i)
            if not ok or not pos:
                continue
            z = Zp[:, j]; x1 = np.broadcast_to(np.asarray(d1[allg[pos[0]]][a]), z.shape)
            want = -np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) / ndtr(-z)
            exp_law = float(np.max(np.abs(x1 - want) / np.abs(want))) < 1e-8
            if exp_law:
                self.exp.append((j, pos)); covered |= set(pos)
            else:
                self.generic.append((j, pos, float(np.sign(x1[0])), float(np.mean(np.abs(x1)))))
            self.diagnostics[j] = dict(comparisons=len(pos), law="exp" if exp_law else "generic monotone")
        self._setup_fast(Zp)
        return covered

    def force_generic_directions(self):
        """Generic coin law: the delta driver along +-e_{z_j} for its comparisons (Newton; correct, not faster)."""
        for (j, pos, sign, nrm) in self.generic:
            v = np.zeros(self.drv.d); v[j] = sign
            for i in pos:
                self.drv.directions[i] = (v, max(nrm, 1e-12))

    def _setup_fast(self, Zp):
        """Map an exponential coin to a CoxDefault markup whose steps are its comparisons (by values on pilots)."""
        drv = self.drv
        if not self.defaults:
            return
        allg = list(drv.indicator_res)
        vals, _ = self._eval({g: [] for g in allg}, Zp[:64])
        for (j, pos) in self.exp:
            for dflt in self.defaults:
                mv, _ = self._eval({g: [] for (_, g, _) in dflt.steps}, Zp[:64])
                if len(dflt.steps) != len(pos):
                    continue
                if all(np.array_equal(np.asarray(vals[allg[i]]), np.asarray(mv[g])) for i, (_, g, _) in zip(pos, dflt.steps)):
                    Zc = Zp[:128]
                    J_slow = self._jumps_slow(j, pos, Zc)
                    J_fast = self._jumps_fast(dflt, j, Zc)
                    err = float(np.max(np.abs(J_fast - J_slow))); scale = max(float(np.max(np.abs(J_slow))), 1e-300)
                    ok = err <= 1e-9 * scale
                    self.diagnostics[j]["fast_path"] = dict(used=ok, max_jump_diff=err)
                    if ok:
                        self.fast[j] = dflt
                    else:
                        warnings.warn(f"coin {j}: CoxDefault outputs do not reproduce the jumps (diff {err:.1e}); "
                                      "two forward passes per step are used")
                    break

    # -- the coin term ---------------------------------------------------------------------------------------------
    def _A(self, j, gs, Z, want_grad=False):
        from scipy.special import ndtr
        th = self.drv.theta_args; M = Z.shape[0]
        vals, d = self._eval({g: (th if want_grad else []) for g in gs}, Z)
        E = -np.log(ndtr(-Z[:, j]))
        A = np.stack([np.asarray(vals[g]) + E for g in gs], 1)
        dA = None
        if want_grad:
            dA = np.stack([np.stack([np.broadcast_to(np.asarray(d[g][a]), (M,)) for a in th], 1) for g in gs], 1)
        return A, dA

    def _two_pass(self, j, A, kk, inc, Z):
        """F(E = A_k - h) - F(E = A_k + h) for step kk (0 where the step has no length)."""
        from scipy.special import ndtri
        M = Z.shape[0]; K = A.shape[1]
        nxt = inc[:, kk + 1] if kk + 1 < K else np.full(M, np.inf)
        h = np.minimum(1e-7 * np.maximum(A[:, kk], 1e-3), 0.25 * np.minimum(inc[:, kk], nxt))
        F = []
        for Ev in (A[:, kk] - h, A[:, kk] + h):
            Zs = Z.copy(); Zs[:, j] = -ndtri(np.exp(-np.maximum(Ev, 1e-300)))
            v, _ = self._eval({self.drv.payoff_res: []}, Zs); F.append(np.asarray(v[self.drv.payoff_res]))
        return np.where((inc[:, kk] > 0) & (h > 0), F[0] - F[1], 0.0)

    def _jumps_slow(self, j, pos, Z):
        gs = [self.drv.indicator_res[i] for i in pos]
        A, _ = self._A(j, gs, Z)
        inc = np.diff(np.concatenate([np.zeros((Z.shape[0], 1)), A], 1), axis=1)
        return np.stack([self._two_pass(j, A, kk, inc, Z) for kk in range(A.shape[1])], 1)

    def _jumps_fast(self, dflt, j, Z, A=None):
        M = Z.shape[0]; K = len(dflt.steps)
        if A is None:
            A, _ = self._A(j, [g for (_, g, _) in dflt.steps], Z)
        Zi = Z.copy(); Zi[:, j] = 8.5                                   # E ~ 39: no default
        v, _ = self._eval({**{r: [] for r in dflt.step_pay}, **{r: [] for r in dflt.step_alive}}, Zi)
        pay = np.stack([np.broadcast_to(np.asarray(v[r]), (M,)) for r in dflt.step_pay], 1)
        alive = np.stack([np.broadcast_to(np.asarray(v[r]), (M,)) for r in dflt.step_alive], 1)
        inc = np.diff(np.concatenate([np.zeros((M, 1)), A], 1), axis=1)
        J = np.zeros((M, K))
        for kk, (k, g, on_date) in enumerate(dflt.steps):
            if not on_date and kk + 1 < K:
                J[:, kk] = alive[:, kk] * (pay[:, kk] - pay[:, kk + 1])     # between dates only the recovery changes
            else:
                J[:, kk] = self._two_pass(j, A, kk, inc, Z)
        J[inc <= 0] = 0.0
        return J

    def term(self, Z):
        """Per-path sum of the coin boundary terms (M, n_theta) over all exponential coins."""
        th = self.drv.theta_args; M = Z.shape[0]
        out = np.zeros((M, len(th)))
        for (j, pos) in self.exp:
            dflt = self.fast.get(j)
            if dflt is not None:
                gs = [g for (_, g, _) in dflt.steps]
                A, _ = self._A(j, gs, Z)
                W = self._jumps_fast(dflt, j, Z, A) * np.exp(-A)
                extra = {w: np.ascontiguousarray(W[:, kk]) for kk, w in enumerate(dflt.step_w)}
                _, d = self._eval({dflt.wsum: th}, Z, extra)
                out += np.stack([np.broadcast_to(np.asarray(d[dflt.wsum][a]), (M,)) for a in th], 1)
            else:
                gs = [self.drv.indicator_res[i] for i in pos]
                A, dA = self._A(j, gs, Z, want_grad=True)
                inc = np.diff(np.concatenate([np.zeros((M, 1)), A], 1), axis=1)
                for kk in range(A.shape[1]):
                    jump = self._two_pass(j, A, kk, inc, Z)
                    out += (jump * np.exp(-A[:, kk]))[:, None] * dA[:, kk, :]
        return out
