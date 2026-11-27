"""
event_time.py — the event-time correction of Kovalov & Naumann for events of monotone accumulators, combined with the
delta correction of this repository for every other discontinuity.

Reference: A. Kovalov, U. Naumann, "Surrogate Greeks for Regime-Switching Processes via Corrected AD", RWTH Aachen,
STCE (poster and code: https://gitlab.stce.rwth-aachen.de/a.kovalov/hsde-sensitivity). There the coefficients of an SDE
switch when a barrier is hit, and pathwise AD misses the motion of the switching time; it is restored at the event,
d_p S_tau += d_p tau [F_1 - F_2]_tau with d_p tau = -d_p S_tau / d_t S_tau ~ -d_p S_tau / f_1(tau) (the drift as a
surrogate of d_t S_tau, "first order"); their next step is the Ito (Feynman-Kac) term. This module applies the same idea
to a payment triggered by an accumulator crossing a threshold (e.g. a default when int lambda dt exceeds an exponential
threshold, a variance budget) and offers the first-order surrogate and two second-order generators. The mathematics,
the implementation choices and the measurements: docs/comparison_kovalov_naumann.md.

Recording (inside the pricing code, between start_recording / stop_recording):

    ev = AccumulatorEvent("default")
    for k in range(1, N + 1):
        ...
        hit, first = ev.step(k, A_prev=Lam, rate=lam, dt=dt, threshold=E, alive=alive,
                             pay=pay_k, normals=[za[...] for this step], on_date=(k in date_steps))
        P = P + first * pay_k            # the product uses `first` as usual
        alive = alive * (1 - hit)
        Lam = Lam + lam * dt
    ev.finalize()                         # marks the aggregated outputs (before stop_recording)

Driver:

    drv = EventTimeDriver(fn, payoff, z_args, theta_args, events=[ev], monitoring="continuous", generator="drift")
    drv.precompute_directions(theta_values)
    res = drv.run(Z)    # total, pathwise, delta, event, counters, diagnostics

monitoring = the TARGET derivative:
  "continuous": continuous event time -> the event-time term for every path with the event, the delta correction on
      the indicators of date steps and on everything else. generator (d/dtau of the value paid at the event):
        "drift"        first order: the Kovalov-Naumann surrogate (drift only)
        "stein"        second order: + the Ito term by Stein's identity
        "quadrature"   second order, exact: Gauss-Hermite over the step normals of the payment
        "quadrature1d" second order, exact for a payment that depends on the step normals through one direction
  "discrete" (alias "cox"): the derivative of the grid model; a Cox threshold (a "coin") is integrated exactly, the
      rest by the delta correction -- the same as CorrectionDriverVec2(coins="auto") of this repository.
After each run: tail index of 1/rate of the event-time term (warning below 2.3: infinite variance when
2 kappa theta / xi^2 < 1 for a CIR rate). All tape replays: compiled (JIT), vectorised, multithreaded.
"""
import io
import math
import contextlib
import warnings
import numpy as np
import aadc
from correction_driver_vec2 import CorrectionDriverVec2

_c = aadc.idouble
try:                                       # the same vectorised, multithreaded replay as the delta driver (AVX-512 if any)
    from aadc_extensions import _evaluate_core, _best_vector_kind, compile_tape
    _VK = _best_vector_kind() if _evaluate_core is not None else None
except ImportError:                        # pragma: no cover
    _evaluate_core, _VK = None, None


def _replay(funcs, request, inputs, workers):
    """(values, derivatives) of aadc.evaluate on the widest vector workspace, all workers; falls back to aadc.evaluate."""
    global _VK
    if _VK is not None:
        try:
            return _evaluate_core(funcs, request, inputs, int(workers), compile_tape(funcs) or _VK)   # JIT kernels
        except Exception:                  # vector kind not supported by this engine build: use the default from now on
            _VK = None
    v, d, _ = aadc.evaluate(funcs=funcs, request=request, inputs=inputs, workers=workers)
    return v, d


def _as_active(x):
    """An idouble copy of x that can be marked as an output (a constant folds to a float otherwise)."""
    y = _c(0.0) + x
    return y if hasattr(y, "mark_as_output") else _c(float(y))


class AccumulatorEvent:
    """Event 'first k with A_k >= threshold' of a monotone accumulator A_k = A_{k-1} + rate_{k-1} dt (left point)."""

    def __init__(self, name="event", threshold_normal=None):
        """threshold_normal: the input Argument z_E of a Cox event, threshold = -log Phi(-z_E) ~ Exp(1) (for monitoring="cox")."""
        self.name = name
        self.threshold_normal = threshold_normal
        self.step_pay, self.step_alive, self.step_w = [], [], []   # per step: pay output, alive output, weight input
        self._wsum = None
        self.steps = []                   # (k, g Result, normal Arguments, on_date, dt)
        self._agg = None
        self._pay_prev = None
        self.res = None

    def step(self, k, A_prev, rate, dt, threshold, alive, pay, normals, on_date=False, pay_prev=None):
        """Record step k. Returns (hit, first): hit = 1{A_prev + rate dt >= threshold}, first = alive * hit.
        pay_prev: value paid had the event happened at the start of the first step (first call only; default: pay)."""
        one, zero = _c(1.0), _c(0.0)
        A_new = A_prev + rate * _c(dt)
        g = (A_new - threshold).mark_as_output()
        # for monitoring="cox": the payment and the alive flag of every step as outputs, and a weight input w_k with
        # the output sum_k w_k A_k (one reverse pass gives sum_k w_k dA_k/dtheta for any per-path weights)
        w = _c(0.0); self.step_w.append(w.mark_as_input_no_diff())
        self._wsum = w * A_new if self._wsum is None else self._wsum + w * A_new
        self.step_pay.append(_as_active(pay)); self.step_alive.append(_as_active(alive))
        hit = aadc.iif(A_new >= threshold, one, zero)
        first = alive * hit
        if self._agg is None:
            self._agg = dict(Aprev=zero, rate=zero, E=zero, pay=zero, pay_prev=zero, k=zero, fired=zero, on_date=zero,
                             dt=zero, rate_min=rate)
            self._pay_prev = pay if pay_prev is None else pay_prev     # 'event at t_0'
        a = self._agg
        a["Aprev"] = a["Aprev"] + first * A_prev
        a["rate"] = a["rate"] + first * rate
        a["E"] = a["E"] + first * threshold
        a["pay"] = a["pay"] + first * pay
        a["pay_prev"] = a["pay_prev"] + first * self._pay_prev
        a["k"] = a["k"] + first * _c(float(k))
        a["fired"] = a["fired"] + first
        a["dt"] = a["dt"] + first * _c(float(dt))
        if on_date:
            a["on_date"] = a["on_date"] + first
        a["rate_min"] = aadc.math.min(a["rate_min"], rate)
        self._pay_prev = pay
        self.steps.append((k, g, list(normals), bool(on_date), float(dt)))
        return hit, first

    def finalize(self):
        """Mark the aggregated outputs (call before stop_recording)."""
        self.res = {name: v.mark_as_output() for name, v in self._agg.items()}
        self.res["wsum"] = self._wsum.mark_as_output()
        self.step_pay = [v.mark_as_output() for v in self.step_pay]
        self.step_alive = [v.mark_as_output() for v in self.step_alive]


class EventTimeDriver:
    def __init__(self, funcs, payoff_res, z_args, theta_args, events, monitoring="continuous",
                 other_indicators=None, other_types=None, num_threads=10, n_pilot=256,
                 generator="stein", n_quad=9, coins="auto", **delta_kwargs):
        """generator: how d/dtau of the value paid at the event is estimated (default; can be overridden per run):
          "drift"      — Kovalov-Naumann surrogate, 1st order: pay_k(z_k = 0) - pay_{k-1} only (drift; one extra forward pass);
          "stein"      — 2nd order: pay_k(z_k = 0) - pay_{k-1} + 1/2 sum_j d pay_k / d z_kj z_kj (Ito term by
                         Stein's identity; one extra forward pass);
          "quadrature" — Gauss-Hermite over the step normals pay_k depends on: sum_q w_q pay_k(z_q) - pay_{k-1}
                         (n_quad^m forward passes, m = number of such normals; no z weight: less noise, QMC-friendly);
          "quadrature1d" — Gauss-Hermite along one fixed unit direction c of the active step normals (from pilot
                         gradients), the orthogonal part of the path's own normals kept: z_k -> x_q c + (I - c c^T) z_k.
                         Unbiased for any pay_k; exact when pay_k depends on the step normals only through c.z_k
                         (diagnostics['quadrature_1d'] reports how close the pilot gradients are to c); n_quad passes."""
        """other_indicators: delta indicators besides the events; None = the switch registry minus the switches of
        the events (identified on pilot paths by identical values)."""
        if monitoring not in ("continuous", "discrete", "cox"):
            raise ValueError("monitoring: 'continuous', 'discrete' or 'cox'")
        if coins not in ("auto", "declared", "off"):
            raise ValueError("coins: 'auto', 'declared' or 'off'")
        # monitoring picks the TARGET: "discrete" = exact derivative of the grid model, "continuous" = continuous event
        # time (event-time term). The algorithm for "discrete" is chosen by the driver: comparisons driven by a coin
        # (an input normal that only decides which side of its comparisons the path is on: Cox default threshold)
        # are integrated over the coin exactly; everything else by delta. "cox" is an alias of "discrete".
        # coins: "auto" (declared + found on the tape), "declared" (AccumulatorEvent(threshold_normal=...) only),
        # "off" (delta on every comparison, for comparison). With "continuous" coins are only reported.
        if monitoring == "cox":
            monitoring = "discrete"
        self.funcs, self.payoff_res, self.z_args, self.theta_args = funcs, payoff_res, list(z_args), list(theta_args)
        self.events, self.monitoring, self.num_threads, self.n_pilot = list(events), monitoring, num_threads, n_pilot
        self.other_indicators, self.other_types, self.delta_kwargs = other_indicators, other_types, delta_kwargs
        if generator not in ("stein", "drift", "quadrature", "quadrature1d"):
            raise ValueError("generator: 'stein', 'quadrature' or 'quadrature1d'")
        self.generator, self.n_quad = generator, int(n_quad)
        self.coins_mode = coins
        self.zpos = {a: j for j, a in enumerate(self.z_args)}
        for ev in self.events:
            if ev.res is None:
                raise ValueError(f"event '{ev.name}': call finalize() before stop_recording")

    # ---------------------------------------------------------------------------------------------------------
    def _eval(self, request, Z):
        inp = {a: np.ascontiguousarray(Z[:, j]) for j, a in enumerate(self.z_args)}
        inp.update({a: float(v) for a, v in self.theta_values.items()})
        return _replay(self.funcs, request, inp, self.num_threads)

    def precompute_directions(self, theta_values):
        theta_values = dict(theta_values)
        for ev in self.events:                     # weight inputs of the events: 0 unless set by the cox term
            for w in getattr(ev, "step_w", []):
                theta_values[w] = 0.0
        self.theta_values = dict(theta_values)
        Zp = np.random.default_rng(20261117).standard_normal((self.n_pilot, len(self.z_args)))
        self.diagnostics = {}
        # 1. checks of each event declaration on pilot paths -> events eligible for the event-time term
        self.event_time_events = []
        for ev in self.events:
            ok, why = True, []
            vals, _ = self._eval({ev.res["rate_min"]: []}, Zp)
            if np.min(vals[ev.res["rate_min"]]) < 0:
                ok = False; why.append("negative accumulator rate")
            sample = ev.steps[:: max(1, len(ev.steps) // 8)]
            vals, d = self._eval({g: [a for a in nrm] for (_, g, nrm, _, _) in sample}, Zp)
            for (k, g, nrm, _, _) in sample:
                if any(np.max(np.abs(np.asarray(d[g][a]))) > 1e-12 for a in nrm):
                    ok = False; why.append(f"step {k}: accumulator depends on its own step normals (not left point)")
                    break
            if self.monitoring == "continuous" and ok:
                self.event_time_events.append(ev)
            elif not ok:
                warnings.warn(f"event '{ev.name}' -> delta correction: " + "; ".join(why))
            self.diagnostics[ev.name] = dict(event_time=ok and self.monitoring == "continuous", checks=why or ["ok"])
        # normals of each step that the event payment depends on (quadrature dimensions), from pilot paths
        self.active_pos = {}
        for ev in self.events:
            pos = []
            sample = ev.steps[:: max(1, len(ev.steps) // 8)]
            normal_args = sorted({a for (_, _, nrm, _, _) in sample for a in nrm}, key=lambda a: self.zpos[a])
            # derivative of the aggregated payment with respect to each slot j of the step normals
            _, d = self._eval({ev.res["pay"]: normal_args}, Zp)
            m = max(len(nrm) for (_, _, nrm, _, _) in ev.steps)
            used = np.zeros(m, dtype=bool)
            for (_, _, nrm, _, _) in sample:
                for j, a in enumerate(nrm):
                    used[j] |= np.any(np.abs(np.asarray(d[ev.res["pay"]][a])) > 0)
            self.active_pos[ev.name] = [j for j in range(m) if used[j]]
        self.diagnostics["quadrature_dims"] = {k: len(v) for k, v in self.active_pos.items()}
        # direction of the 1D quadrature: mean of the sign-aligned unit gradients of the event payment with respect to
        # the active step normals, over pilot paths with the event (more pilot paths are drawn until >= 32 have it)
        self.quad_dir, self.diagnostics["quadrature_1d"] = {}, {}
        for ev in self.events:
            act = self.active_pos[ev.name]
            if not act:
                self.quad_dir[ev.name] = np.zeros(0); continue
            normal_args = sorted({a for (_, _, nrm, _, _) in ev.steps for a in nrm}, key=lambda a: self.zpos[a])
            step_norm = {k: [self.zpos[a] for a in nrm] for (k, _, nrm, _, _) in ev.steps}
            grads, rng = [], np.random.default_rng(20261121)
            for _ in range(16):
                Zq = rng.standard_normal((max(self.n_pilot, 1024), len(self.z_args)))
                vals, d = self._eval({ev.res["pay"]: normal_args, ev.res["fired"]: [], ev.res["k"]: []}, Zq)
                fired = np.asarray(vals[ev.res["fired"]]) > 0.5
                kk = np.rint(np.asarray(vals[ev.res["k"]])).astype(int)
                col_of = {self.zpos[a]: i for i, a in enumerate(normal_args)}
                D = np.stack([np.broadcast_to(np.asarray(d[ev.res["pay"]][a]), (len(Zq),)) for a in normal_args], 1)
                rows = np.where(fired)[0]
                if len(rows):
                    cols = np.array([[col_of[step_norm[kk[r]][j]] for j in act] for r in rows])   # index table only
                    Gr = D[rows[:, None], cols]
                    nr = np.linalg.norm(Gr, axis=1); keep = nr > 1e-300
                    grads.extend(list(Gr[keep] / nr[keep, None]))
                if len(grads) >= 32:
                    break
            if not grads:
                c = np.zeros(len(act)); c[0] = 1.0; cosmin = float("nan")
            else:
                G = np.array(grads); G *= np.sign(G @ G[0])[:, None]
                c = G.mean(0); c /= np.linalg.norm(c); cosmin = float(np.min(np.abs(G @ c)))
            self.quad_dir[ev.name] = c
            self.diagnostics["quadrature_1d"][ev.name] = dict(direction=np.round(c, 6).tolist(), pilot_paths=len(grads),
                                                              min_cos=cosmin, exact=bool(cosmin > 1 - 1e-10))
        # 2. delta indicators
        ev_g = [(ev, k, g, on_date) for ev in self.events for (k, g, _, on_date, _) in ev.steps]
        if self.other_indicators is None:
            sws = self.funcs.cmp_switches()
            cand = [(sw.g, sw.origin) for sw in sws]
            vals, _ = self._eval({**{g: [] for g, _ in cand}, **{g: [] for (_, _, g, _) in ev_g}}, Zp)
            ev_vals = [np.asarray(vals[g]) for (_, _, g, _) in ev_g]
            others, types = [], []
            for g, origin in cand:
                v = np.asarray(vals[g])
                if any(np.array_equal(v, w) for w in ev_vals):
                    continue                                   # the switch of an event step: not a separate indicator
                others.append(g); types.append(origin)
        else:
            others = list(self.other_indicators); types = list(self.other_types or ["jump"] * len(others))
        delta_ev = [(g, "jump") for (ev, k, g, on_date) in ev_g if ev not in self.event_time_events or on_date]
        self.coin_exp, self.coin_generic, self.diagnostics["coins"] = [], [], {}
        if self.coins_mode != "off":
            cand = others + [g for g, _ in delta_ev]
            used = self._setup_coins(Zp, cand, ev_g)
            if self.monitoring == "discrete":
                # exponential coins: integrated exactly (removed from delta); generic coins: delta along the coin
                drop = {id(g) for (_, gs) in self.coin_exp for g in gs}
                keep = [i for i, g in enumerate(others) if id(g) not in drop]
                types = [types[i] for i in keep]; others = [others[i] for i in keep]
                delta_ev = [(g, t) for (g, t) in delta_ev if id(g) not in drop]
            else:                                          # continuous target: report only
                for j in self.diagnostics["coins"]:
                    self.diagnostics["coins"][j]["used"] = "no (continuous target; monitoring='discrete' integrates it)"
                self.coin_exp, self.coin_generic = [], []
        self.delta_res = others + [g for g, _ in delta_ev]
        self.delta_types = types + [t for _, t in delta_ev]
        self.diagnostics["indicators"] = dict(delta_other=len(others), delta_event_steps=len(delta_ev),
                                              event_time_steps=sum(1 for (ev, k, g, od) in ev_g if ev in self.event_time_events and not od))
        import inspect
        kw = dict(self.delta_kwargs)
        if "coins" in inspect.signature(CorrectionDriverVec2.__init__).parameters:
            kw.setdefault("coins", "off")     # this module integrates coins itself; the inner driver keeps the default directions
        self.delta = CorrectionDriverVec2(self.funcs, self.payoff_res, self.z_args, self.theta_args,
                                          indicator_res=self.delta_res, indicator_types=self.delta_types,
                                          num_threads=self.num_threads, **kw)
        with contextlib.redirect_stdout(io.StringIO()):
            self.delta.precompute_directions(theta_values)
        if self.coin_generic:
            pos = {id(g): i for i, g in enumerate(self.delta_res)}
            for (j, gs, sign, nrm0) in self.coin_generic:
                v = np.zeros(len(self.z_args)); v[j] = sign
                for g in gs:
                    self.delta.directions[pos[id(g)]] = (v, max(nrm0, 1e-12))
        # fast path of an exponential coin whose comparisons are the steps of a marked event: jumps between
        # non-date steps from the step outputs, one weighted reverse pass; checked against the passes on pilots
        self.coin_fast = {}
        if True:
            for (j, gs) in self.coin_exp:
                for ev in self.events:
                    if [id(g) for (_, g, _, _, _) in ev.steps] == [id(g) for g in gs] and getattr(ev, "step_pay", None):
                        Zc = Zp[: min(len(Zp), 256)]
                        _, _, J_slow = self._cox_term_slow(j, gs, Zc, return_jumps=True)
                        J_fast = self._cox_jumps_fast(ev, j, Zc)
                        err = float(np.max(np.abs(J_fast - J_slow)))
                        scale = max(float(np.max(np.abs(J_slow))), 1e-300)
                        ok = err <= 1e-9 * scale
                        self.diagnostics["coins"][j]["fast_path"] = dict(used=ok, max_jump_diff=err, jump_scale=scale)
                        if ok:
                            self.coin_fast[j] = ev
                        else:
                            warnings.warn(f"coin {j}: step outputs do not reproduce the jumps (diff {err:.1e}); "
                                          "using forward passes on every step")

    # ---------------------------------------------------------------------------------------------------------
    def _setup_coins(self, Zp, cand_g, ev_g):
        """Coins: input normals z_j that enter the payoff only through comparisons, each comparison depending on z_j alone
        (g = A(other inputs) - f(z_j)). Declared ones (AccumulatorEvent threshold_normal) are checked, others are found
        on the tape (coins="auto"). Exponential threshold f = -log Phi(-z) -> exact term (self.coin_exp: root and weight
        in closed form); any other monotone f -> delta along the coin direction (self.coin_generic)."""
        from scipy.special import ndtr
        allg = list({id(g): g for g in [g for (_, _, g, _) in ev_g] + list(cand_g)}.values())
        declared = {}
        for ev in self.events:
            if ev.threshold_normal is not None:
                declared.setdefault(self.zpos[ev.threshold_normal], []).extend(g for (k, g, _, _, _) in ev.steps)
        cand = set(declared)
        if self.coins_mode == "auto":
            _, dP = self._eval({self.payoff_res: list(self.z_args)}, Zp)
            cand |= {j for j, a in enumerate(self.z_args) if np.max(np.abs(np.asarray(dP[self.payoff_res][a]))) == 0.0}
        Zq = np.random.default_rng(20261122).standard_normal(Zp.shape)
        # stage 1, all candidates at once (two Jacobian evaluations on 64 pilot paths): a coin's comparison derivatives
        # must not change when every non-candidate input is redrawn, and must have one sign. Removes e.g. the normals
        # of an intensity (they enter many comparisons through the path of other inputs). Declared coins always go on.
        if len(cand) > 1:
            cj = sorted(cand); ca = [self.z_args[j] for j in cj]; n1 = min(64, len(Zp))
            Zs = Zq[:n1].copy(); Zs[:, cj] = Zp[:n1, cj]
            _, D1 = self._eval({g: ca for g in allg}, Zp[:n1])
            _, D2 = self._eval({g: ca for g in allg}, Zs)
            keep = set(declared)
            for j, a in zip(cj, ca):
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
                    keep.add(j)
            self.diagnostics["coin_candidates"] = dict(screened=len(cj), passed=len(keep))
            cand = keep
        used = set()
        for j in sorted(cand):
            a = self.z_args[j]
            Zr = Zq.copy(); Zr[:, j] = Zp[:, j]                       # same coin, other inputs redrawn
            v1, d1 = self._eval({**{g: [a] for g in allg}, self.payoff_res: [a]}, Zp)
            _, d2 = self._eval({g: [a] for g in allg}, Zr)
            dpay = float(np.max(np.abs(np.asarray(d1[self.payoff_res][a]))))
            gs, ok = [], True
            for g in allg:
                x1 = np.broadcast_to(np.asarray(d1[g][a]), (len(Zp),)); x2 = np.broadcast_to(np.asarray(d2[g][a]), (len(Zp),))
                if np.max(np.abs(x1)) == 0.0:
                    continue                                          # comparison does not involve this input
                if np.max(np.abs(x1 - x2)) > 1e-10 * np.max(np.abs(x1)) or not (np.all(x1 > 0) or np.all(x1 < 0)):
                    ok = False; break                                 # not separable / not monotone in the coin
                gs.append(g)
            if j in declared and (not ok or dpay > 1e-10):
                raise ValueError(f"input {j} declared as a Cox threshold normal but d payoff/dz = {dpay:.1e}, separable "
                                 f"and monotone = {ok}; use monitoring='discrete'")
            if not ok or dpay > 1e-10 or not gs:
                continue
            z = Zp[:, j]
            x1 = np.broadcast_to(np.asarray(d1[gs[0]][a]), z.shape)
            want = -np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi) / ndtr(-z)
            exp_law = float(np.max(np.abs(x1 - want) / np.abs(want))) < 1e-8
            if exp_law:
                self.coin_exp.append((j, gs))
            else:
                self.coin_generic.append((j, gs, float(np.sign(x1[0])), float(np.mean(np.abs(x1)))))
            used |= {id(g) for g in gs}
            self.diagnostics["coins"][j] = dict(declared=j in declared, comparisons=len(gs),
                                                law="exp" if exp_law else "generic monotone")
        return used

    def _cox_term(self, j, gs, Z):
        ev = getattr(self, "coin_fast", {}).get(j)
        return self._cox_term_fast(ev, j, Z) if ev is not None else self._cox_term_slow(j, gs, Z)

    def _cox_jumps_fast(self, ev, j, Z, A=None):
        """(M, K) jumps D_k - D_{k+1}: steps not ending on a date from the step outputs at E = +inf
        (alive_k (pay_k - pay_{k+1})); date steps (and the last step) by two forward passes, as in the slow path."""
        from scipy.special import ndtr, ndtri
        M = Z.shape[0]; K = len(ev.steps)
        if A is None:
            vals, _ = self._eval({g: [] for (_, g, _, _, _) in ev.steps}, Z)
            E = -np.log(ndtr(-Z[:, j])); A = np.stack([np.asarray(vals[g]) + E for (_, g, _, _, _) in ev.steps], 1)
        Zi = Z.copy(); Zi[:, j] = 8.5                                             # E = -log Phi(-8.5) ~ 39: no event
        v, _ = self._eval({**{r: [] for r in ev.step_pay}, **{r: [] for r in ev.step_alive}}, Zi)
        pay = np.stack([np.broadcast_to(np.asarray(v[r]), (M,)) for r in ev.step_pay], 1)
        alive = np.stack([np.broadcast_to(np.asarray(v[r]), (M,)) for r in ev.step_alive], 1)
        inc = np.diff(np.concatenate([np.zeros((M, 1)), A], 1), axis=1)
        J = np.zeros((M, K))
        for kk, (k, g, _, on_date, _) in enumerate(ev.steps):
            if not on_date and kk + 1 < K:
                J[:, kk] = alive[:, kk] * (pay[:, kk] - pay[:, kk + 1])
                continue
            Ak = A[:, kk]; nxt = inc[:, kk + 1] if kk + 1 < K else np.full(M, np.inf)
            h = np.minimum(1e-7 * np.maximum(Ak, 1e-3), 0.25 * np.minimum(inc[:, kk], nxt))
            F = []
            for Ev in (Ak - h, Ak + h):
                Zs = Z.copy(); Zs[:, j] = -ndtri(np.exp(-np.maximum(Ev, 1e-300)))
                vv, _ = self._eval({self.payoff_res: []}, Zs); F.append(np.asarray(vv[self.payoff_res]))
            J[:, kk] = F[0] - F[1]
        J[inc <= 0] = 0.0
        return J

    def _cox_term_fast(self, ev, j, Z):
        """sum_k (D_k - D_{k+1}) exp(-A_k) dA_k/dtheta per path with one weighted reverse pass."""
        from scipy.special import ndtr
        M, th = Z.shape[0], self.theta_args
        vals, _ = self._eval({g: [] for (_, g, _, _, _) in ev.steps}, Z)
        E = -np.log(ndtr(-Z[:, j])); A = np.stack([np.asarray(vals[g]) + E for (_, g, _, _, _) in ev.steps], 1)
        W = self._cox_jumps_fast(ev, j, Z, A) * np.exp(-A)
        inp = {a: np.ascontiguousarray(Z[:, q]) for q, a in enumerate(self.z_args)}
        inp.update({a: float(x) for a, x in self.theta_values.items()})
        inp.update({w: np.ascontiguousarray(W[:, kk]) for kk, w in enumerate(ev.step_w)})
        _, d = _replay(self.funcs, {ev.res["wsum"]: th}, inp, self.num_threads)
        out = np.stack([np.broadcast_to(np.asarray(d[ev.res["wsum"]][a]), (M,)) for a in th], 1)
        return out, int(np.count_nonzero(W.any(axis=1)))

    def _cox_term_slow(self, j, gs, Z, return_jumps=False):
        """Per-path sum over the event steps k of (F(E = A_k - h) - F(E = A_k + h)) exp(-A_k) dA_k/dtheta: the delta
        term of g_k = A_k - E along z_E, i.e. the boundaries of the event integrated over E exactly."""
        from scipy.special import ndtr, ndtri
        M, th = Z.shape[0], self.theta_args
        vals, d = self._eval({g: th for g in gs}, Z)
        E = -np.log(ndtr(-Z[:, j]))
        A = np.stack([np.asarray(vals[g]) + E for g in gs], 1)                   # A_k per path (independent of z_E)
        dA = np.stack([np.stack([np.broadcast_to(np.asarray(d[g][a]), (M,)) for a in th], 1) for g in gs], 1)  # (M,K,n_th)
        inc = np.diff(np.concatenate([np.zeros((M, 1)), A], 1), axis=1)          # A_k - A_{k-1}
        out = np.zeros((M, len(th))); n_used = 0; Jall = np.zeros((M, len(gs)))
        for kk in range(len(gs)):
            Ak = A[:, kk]
            nxt = inc[:, kk + 1] if kk + 1 < len(gs) else np.full(M, np.inf)
            h = np.minimum(1e-7 * np.maximum(Ak, 1e-3), 0.25 * np.minimum(inc[:, kk], nxt))
            ok = (inc[:, kk] > 0) & (h > 0)
            if not ok.any():
                continue
            F = []
            for Ev in (Ak - h, Ak + h):
                Zs = Z.copy(); Zs[:, j] = -ndtri(np.exp(-np.maximum(Ev, 1e-300)))
                v, _ = self._eval({self.payoff_res: []}, Zs); F.append(np.asarray(v[self.payoff_res]))
            jump = np.where(ok, F[0] - F[1], 0.0)                                # default in step k minus later
            if return_jumps:
                Jall[:, kk] = jump
            out += (jump * np.exp(-Ak))[:, None] * dA[:, kk, :]
            n_used += int(ok.sum())
        return (out, n_used, Jall) if return_jumps else (out, n_used)

    def _event_term(self, ev, Z, generator=None, n_quad=None):
        """Per-path event-time term (M, n_theta): motion of the event time inside the event step, every path with the event."""
        R, th = ev.res, self.theta_args; M = Z.shape[0]
        normal_args = sorted({a for (_, _, nrm, _, _) in ev.steps for a in nrm}, key=lambda a: self.zpos[a])
        req = {R["pay"]: normal_args, R["pay_prev"]: [], R["Aprev"]: th, R["rate"]: th, R["E"]: [], R["k"]: [],
               R["fired"]: [], R["on_date"]: [], R["dt"]: []}
        vals, d = self._eval(req, Z)
        G_ = lambda r: np.stack([np.broadcast_to(np.asarray(d[r][a]), (M,)) for a in th], axis=1)
        # every path with the event: the term is the motion of tau inside its step; a date at the END of the step
        # is a separate boundary, handled by the delta indicator of that step
        use = np.asarray(vals[R["fired"]]) > 0.5
        rate = np.where(use, vals[R["rate"]], 1.0)
        frac_dt = np.where(use, (np.asarray(vals[R["E"]]) - vals[R["Aprev"]]) / rate, 0.0)
        dtau = -(G_(R["Aprev"]) + frac_dt[:, None] * G_(R["rate"])) / rate[:, None]
        kpath = np.rint(np.asarray(vals[R["k"]])).astype(int)
        step_norm = {k: [self.zpos[a] for a in nrm] for (k, _, nrm, _, _) in ev.steps}
        dts = np.where(use, vals[R["dt"]], 1.0)
        gen = generator or self.generator
        if gen in ("stein", "drift"):
            # "drift" (Kovalov-Naumann surrogate, 1st order): G dt = pay_k(z_k = 0) - pay_{k-1} (drift only);
            # "stein" (2nd order): + 1/2 sum_j d pay_k / d z_kj z_kj (Ito term by Stein's identity)
            stein = np.zeros(M); Z0 = Z.copy()
            for k, cols in step_norm.items():
                rows = np.where(use & (kpath == k))[0]
                if len(rows) == 0:
                    continue
                for j, a_idx in enumerate(cols):
                    dp = np.asarray(d[R["pay"]][self.z_args[a_idx]])
                    stein[rows] += 0.5 * np.broadcast_to(dp, (M,))[rows] * Z[rows, a_idx]
                    Z0[rows, a_idx] = 0.0
            v0, _ = self._eval({R["pay"]: []}, Z0)
            Epay = np.asarray(v0[R["pay"]]) + (stein if gen == "stein" else 0.0)
        elif gen == "quadrature1d":
            # Gauss-Hermite along the fixed direction c, the orthogonal part of the path's own normals kept
            nq = n_quad or self.n_quad
            x, w = np.polynomial.hermite_e.hermegauss(nq); w = w / np.sqrt(2.0 * np.pi)
            act, c = self.active_pos[ev.name], self.quad_dir[ev.name]
            rows = np.where(use)[0]; Epay = np.zeros(M)
            if len(rows) and act:
                cols = np.array([[step_norm[k][j] for j in act] for k in kpath[rows]])
                ridx = np.arange(len(rows))[:, None]
                zact = Z[rows][ridx, cols]                                  # (n, m) active normals of each path
                zperp = zact - np.outer(zact @ c, c)
                Zr = Z[rows].copy(); Epay_r = np.zeros(len(rows))
                for xq, wq in zip(x, w):
                    Zr[ridx, cols] = zperp + xq * c
                    vq, _ = self._eval({R["pay"]: []}, Zr)
                    Epay_r += wq * np.asarray(vq[R["pay"]])
                Epay[rows] = Epay_r
            elif len(rows):
                v0, _ = self._eval({R["pay"]: []}, Z[rows]); Epay[rows] = np.asarray(v0[R["pay"]])
        else:
            # Gauss-Hermite over the active normals of the event step: E[pay_k | F_{k-1}] by quadrature
            nq = n_quad or self.n_quad
            x, w = np.polynomial.hermite_e.hermegauss(nq); w = w / np.sqrt(2.0 * np.pi)
            act = self.active_pos[ev.name]
            rows = np.where(use)[0]
            Zr = Z[rows].copy(); Epay_r = np.zeros(len(rows))
            cols_of_row = np.array([[step_norm[k][j] for j in act] for k in kpath[rows]]) if len(rows) else np.zeros((0, len(act)), int)
            grid = np.array(np.meshgrid(*[np.arange(nq)] * len(act), indexing="ij")).reshape(len(act), -1).T if act else np.zeros((1, 0), int)
            for node in grid:
                Zq = Zr.copy()
                for jj, qi in enumerate(node):
                    Zq[np.arange(len(rows)), cols_of_row[:, jj]] = x[qi]
                vq, _ = self._eval({R["pay"]: []}, Zq)
                Epay_r += np.prod(w[node]) * np.asarray(vq[R["pay"]])
            Epay = np.zeros(M); Epay[rows] = Epay_r
        G = np.where(use, (Epay - vals[R["pay_prev"]]) / dts, 0.0)
        self._last_event_rates = np.asarray(vals[R["rate"]])[use]
        return G[:, None] * dtau, int(use.sum())

    def run(self, Z, generator=None, n_quad=None):
        """generator / n_quad: override the constructor's choice for this run (same delta part)."""
        with contextlib.redirect_stdout(io.StringIO()):
            o = self.delta.run(Z)
        out = dict(pathwise=np.asarray(o["pathwise"]), delta=np.asarray(o["correction"]),
                   newton_unconverged=o.get("newton_unconverged", 0), newton_far=o.get("newton_far", 0),
                   jump_unresolved=o.get("jump_unresolved", 0), event=np.zeros(len(self.theta_args)), event_paths=0)
        out["event_tail_index"] = {}
        if self.coin_exp:
            for (j, gs) in self.coin_exp:
                t, n = self._cox_term(j, gs, Z)
                out["event"] = out["event"] + t.mean(0); out["event_paths"] += n
        for ev in self.event_time_events:
            t, n = self._event_term(ev, Z, generator, n_quad)
            out["event"] = out["event"] + t.mean(0); out["event_paths"] += n
            # tail check: the term carries 1/a (rate of the accumulator at the event step), and an event in a step
            # with a small rate has probability ~ a. Hill index of 1/a over the top 1% of the paths with the event
            # (>= 20): for a CIR rate it is 1 + 2 kappa theta / xi^2 (measured: matches from 5K paths on). Below 2 the
            # variance of the event-time term is infinite: error bars and Sobol gains are unreliable -> use the delta
            # correction for this event (monitoring="discrete").
            idx = []
            ra = self._last_event_rates; ra = ra[ra > 0]
            if len(ra) >= 200:
                y = np.sort(1.0 / ra)[::-1]; kk = max(20, len(y) // 100)
                idx.append(1.0 / np.mean(np.log(y[:kk] / y[kk])))
            ti = float(min(idx)) if idx else float("nan")
            out["event_tail_index"][ev.name] = ti
            if ti < 2.3:
                warnings.warn(f"event '{ev.name}': heavy tails in the event-time term (Hill index of 1/rate {ti:.2f}; variance "
                              f"infinite below 2) -> error bars / Sobol gain unreliable; consider monitoring='discrete'")
        out["total"] = out["pathwise"] + out["delta"] + out["event"]
        for k in ("newton_unconverged", "jump_unresolved"):
            if out[k]:
                warnings.warn(f"{k} = {out[k]}")
        return out


HybridDriver = EventTimeDriver          # earlier name
