"""
correction_driver_vec2.py — Fully vectorized correction using selective evaluate.

Single VectorFunctionSelectiveAD replaces all per-indicator VFs.
evaluate() returns jac_z AND jac_theta — no scalar loops needed (step 1 only).
evaluate_selective_theta() for Newton (1 reverse pass per indicator, also
gives dg/dθ at the root — no full reverse after Newton).
evaluate_forward() for jump (0 reverse passes).
"""
import math
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import aadc
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aadc_extensions import VectorFunctionSelectiveAD


STEP_CLIP = 8.0       # max Newton step along v (rtsafe): stops a near-zero slope throwing the point far; 3 cost +3% work
U_FAR = 40.0          # |u| beyond which phi(u) == 0 in double precision (phi(38.6) ~ 1e-324)


def phi_normal_vec(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


class CorrectionDriverVec2:

    def __init__(self, funcs, payoff_res, z_args, theta_args,
                 indicator_res=None, indicator_types=None,
                 skip_sigma=None, jump_eps=1e-7, max_newton=50, num_threads=4, indicator_threads=3,
                 vector_kind="best", root_tol_u=1e-10, jump_tries=6, root_solver="rtsafe", linear_roots=True,
                 coins="auto", defaults=None):
        """
        Args:
            funcs: aadc.Functions with recorded tape.
            payoff_res: Result handle for the payoff output.
            z_args: list of Argument handles for stochastic inputs (normals).
            theta_args: list of Argument handles for model parameters.
            indicator_res: list of Result handles for g. If None, auto-discovered
                from funcs.cmp_switches() (requires register_switches at recording).
            indicator_types: list of 'jump' or 'kink'. If None, auto-discovered.
            indicator_threads: indicators processed concurrently (Python threads; AADC releases the GIL).
                Each aadc.evaluate call with num_threads > 1 costs ~2 ms of thread start-up, and the
                driver makes several calls per indicator; overlapping the calls of different indicators
                hides that cost. Results do not depend on it (contributions are summed in indicator order).
            vector_kind: AADC vector workspace: "best" (the tape is compiled -- aadc_extensions.compile_tape --
                and replayed on the workspace of its kernels; 5x faster than the interpreter), None (engine default),
                or "scalar" / "avx2" / "avx512". Results are bitwise identical across kinds.
            skip_sigma: optional screening — skip a (path, indicator) pair when the LINEARISED distance |g| / |dg/du|
                exceeds skip_sigma. Off by default (None): where g is flat at the path point the linearised
                distance is far too large and pairs with a nearby root and a non-negligible weight phi(u*) were
                dropped (timer option, xi = 0.5: -5% of dV/dv0).
            max_newton: Newton iterations per pair (only pairs not yet converged iterate, so a high cap is cheap).
                Every step is clipped to +-STEP_CLIP; a pair whose iterate leaves |u| <= 40 is dropped (phi = 0 there exactly),
                counted in result['newton_far']; result['newton_unconverged'] counts the others that did not converge.
                8 was too few where g is convex and flattens towards the root (default threshold E = -log Phi(-z) of
                a Cox model: 11% of the pairs did not converge, dV/dlambda0 lost 6%).
            root_solver: "rtsafe" (default): Newton while it converges, the Illinois method on the bracket once a
                Newton step leaves it or makes no progress twice in a row (|g| not below 0.9 x its best while the
                step reverses or |g| grows) (plain Newton cycles on piecewise-smooth g such
                as min(S_i) / S0 - B of a worst-of, and was dropping pairs whose root exists); "newton": the
                previous plain Newton (results as before).
            root_tol_u: Newton stops when the root is located to |g / dg_du| < root_tol_u along v_i
                (|g| < 1e-8 alone is not enough where g is flat along v_i); on the last allowed step |g| < 1e-8
                is accepted.
            linear_roots: (rtsafe only) an indicator whose gradient is parallel to its direction v_i on all pilot
                paths is a function of u = v_i . z alone (g = h(v_i . z): a barrier on a GBM, a corridor on an
                LGM / Hull-White rate); its boundary is u = U*_i for every path. U*_i is found once (bisection on
                the line u v_i) and used as the starting point of the solver, which then converges in one step.
                The solver still checks the root, so a wrong detection only costs iterations, never accuracy.
            coins: "auto" (default): comparisons driven by a coin -- an input normal that only decides which side of
                its comparisons the path is on, such as the threshold E = -log Phi(-z_E) of an intensity default --
                are integrated over the coin exactly (root and weight in closed form, no Newton; see coins.py);
                "off": the delta correction with the default directions on every comparison.
            defaults: optional CoxDefault markups (coins.py) of the product: fast path of the coin term.
            jump_eps, jump_tries: the jump is P(z* + eps v) - P(z* - eps v). It is accepted only if, between the two
                points, indicator i changes sign and no other indicator does; otherwise eps is reduced (another
                indicator crossed: two boundaries closer than eps) or increased (indicator i not bracketed), at
                most jump_tries times; eps never goes below 4x the root error. Pairs left unresolved are
                counted in result['jump_unresolved'] and contribute nothing.
        """
        self.funcs = funcs
        self.payoff_res = payoff_res
        self.z_args = z_args
        self.theta_args = theta_args
        self.skip_sigma = skip_sigma
        self.jump_eps = jump_eps
        self.max_newton = max_newton
        self.root_tol_u = root_tol_u
        if root_solver not in ("rtsafe", "newton"):
            raise ValueError("root_solver: 'rtsafe' or 'newton'")
        self.root_solver = root_solver
        self.linear_roots = bool(linear_roots) and root_solver == "rtsafe"
        self.u_star = None
        self.jump_tries = jump_tries
        if coins not in ("auto", "off"):
            raise ValueError("coins: 'auto' or 'off'")
        self.coins_mode, self.defaults = coins, list(defaults or [])
        self.coins, self.coin_skip = None, set()
        self.num_threads = num_threads
        self.indicator_threads = max(1, int(indicator_threads))

        # Auto-discover indicators from switch registry
        if indicator_res is None:
            switches = funcs.cmp_switches()
            indicator_res = [sw.g for sw in switches]
            indicator_types = [sw.origin for sw in switches]

        self.indicator_res = indicator_res
        self.indicator_types = indicator_types or ['jump'] * len(indicator_res)

        self.d = len(z_args)
        self.n_ind = len(indicator_res)
        self.n_theta = len(theta_args)

        # ONE VF for everything: all outputs, selective reverse
        all_res = [payoff_res] + list(indicator_res)
        self.vf = VectorFunctionSelectiveAD(
            funcs, z_args, all_res,
            param_args=theta_args, num_threads=num_threads, vector_kind=vector_kind
        )

        self.directions = None

    def precompute_directions(self, theta_values):
        self.theta_values = theta_values
        self.theta_list = [theta_values[a] for a in self.theta_args]
        self.vf.set_params(self.theta_list)
        # Other inputs the caller gave values for (e.g. a spot that is not differentiated):
        # they must reach evaluate too, aadc.evaluate reads garbage for a missing input.
        theta_set = set(self.theta_args)
        self.vf.set_fixed_inputs({a: v for a, v in theta_values.items() if a not in theta_set})

        ws = self.funcs.create_workspace("scalar")        # 1 lane: after compile_tape() the default is vector
        for j in range(self.d):
            ws.set_val(self.z_args[j], 0.0)
        for arg, val in theta_values.items():
            ws.set_val(arg, val)
        ws.forward()

        self.directions = []
        for i in range(self.n_ind):
            ws.reset_diff()
            ws.set_diff(self.indicator_res[i], 1.0)
            ws.reverse()
            v = np.zeros(self.d)
            for j in range(self.d):
                v[j] = ws.diff(self.z_args[j])
            norm = np.linalg.norm(v)
            if norm > 1e-15:
                v /= norm
            self.directions.append((v, norm))
        self.coins, self.coin_skip = None, set()
        if self.coins_mode == "auto":
            from coins import Coins
            self.coins = Coins(self, self.defaults)
            self.coin_skip = self.coins.setup()
            self.coins.force_generic_directions()
        if self.linear_roots:
            self._find_linear_roots()

    def _find_linear_roots(self, n_pilot=64, seed=20261124):
        """U*_i for the indicators that are functions of v_i . z alone (see linear_roots); nan for the others."""
        self.u_star = np.full(self.n_ind, np.nan)
        cand = [i for i, (v, nrm) in enumerate(self.directions) if nrm > 1e-15]
        if not cand:
            return
        Zp = np.random.default_rng(seed).standard_normal((n_pilot, self.d))
        _, J, _, _ = self.vf.evaluate(Zp)                         # (n_pilot, 1 + n_ind, d)
        lin = []
        for i in cand:
            v = self.directions[i][0]
            G = J[:, 1 + i, :]; gn = np.linalg.norm(G, axis=1)
            if np.all(gn > 1e-300) and np.all(G @ v >= gn * (1.0 - 1e-12)):   # parallel to v, same sign: monotone in u
                lin.append(i)
        if not lin:
            return
        # bisection on the lines u v_i, all linear indicators in one forward pass per iteration
        V = np.stack([self.directions[i][0] for i in lin])        # (n_lin, d)
        grid = np.linspace(-40.0, 40.0, 161)
        gv = np.stack([self.vf.evaluate_forward(u * V)[np.arange(len(lin)), [1 + i for i in lin]] for u in grid], 1)
        lo = np.full(len(lin), np.nan); hi = np.full(len(lin), np.nan)
        for r in range(len(lin)):
            ch = np.where(np.sign(gv[r, :-1]) != np.sign(gv[r, 1:]))[0]
            if len(ch) == 1:                                       # monotone along v: one sign change at most
                lo[r], hi[r] = grid[ch[0]], grid[ch[0] + 1]
        ok = np.isfinite(lo)
        if not ok.any():
            return
        idx = np.where(ok)[0]; rows = np.array([1 + lin[r] for r in idx])
        a, b = lo[idx].copy(), hi[idx].copy()
        ga = self.vf.evaluate_forward(a[:, None] * V[idx])[np.arange(len(idx)), rows]
        for _ in range(60):
            m = 0.5 * (a + b)
            gm = self.vf.evaluate_forward(m[:, None] * V[idx])[np.arange(len(idx)), rows]
            left = np.sign(gm) == np.sign(ga)
            a = np.where(left, m, a); ga = np.where(left, gm, ga); b = np.where(left, b, m)
        for r, u in zip(idx, 0.5 * (a + b)):
            self.u_star[lin[r]] = u

    def run(self, z_all):
        M = z_all.shape[0]
        t_start = time.time()

        # ── Step 1: one forward + reverse over the paths, in batches. Per indicator only its directional
        # derivative v_i . grad_z g_i is kept (that is all the solver needs at the path point); the payoff gets
        # its theta-derivatives (pathwise Greeks). Memory O(paths x indicators), not O(paths x indicators x d).
        n_out = 1 + self.n_ind
        vals_all = np.zeros((M, n_out))
        vdg_all = np.zeros((M, self.n_ind))
        pw_sum = np.zeros(self.n_theta)
        main = [i for i in range(self.n_ind) if i not in self.coin_skip and self.directions[i][1] > 1e-15]
        nz = {i: np.nonzero(self.directions[i][0])[0] for i in main}
        need = sorted({j for i in main for j in nz[i]})
        zargs_needed = [self.z_args[j] for j in need]
        BATCH = int(max(256, min(5000, 2.0e8 / max(1, (1 + len(main)) * max(1, len(need))))))
        req = {self.payoff_res: list(self.theta_args)}
        req.update({self.indicator_res[i]: [] for i in range(self.n_ind) if i not in self.coin_skip})   # values
        req.update({self.indicator_res[i]: zargs_needed for i in main})                                  # + reverse
        fixed = {a: float(v) for a, v in self.theta_values.items()}
        for start in range(0, M, BATCH):
            end = min(start + BATCH, M)
            inp = {a: np.ascontiguousarray(z_all[start:end, j]) for j, a in enumerate(self.z_args)}
            inp.update(fixed)
            vals, ders, _ = self.vf._evaluate(self.funcs, req, inp, self.num_threads)
            vals_all[start:end, 0] = vals[self.payoff_res]
            for i in range(self.n_ind):
                if i not in self.coin_skip:
                    vals_all[start:end, 1 + i] = vals[self.indicator_res[i]]
            for i in main:
                v_i = self.directions[i][0]; d_i = ders[self.indicator_res[i]]
                acc = np.zeros(end - start)
                for j in nz[i]:
                    acc += np.asarray(d_i[self.z_args[j]]) * v_i[j]
                vdg_all[start:end, i] = acc
            for q, a in enumerate(self.theta_args):
                pw_sum[q] += float(np.sum(np.asarray(ders[self.payoff_res][a])))
        t1 = time.time()
        print(f"  Step 1 (batch fwd + directional derivatives): {t1-t_start:.1f}s")

        prices = vals_all[:, 0]
        g_all = vals_all[:, 1:]
        price = prices.mean()
        pathwise = pw_sum / M
        t2 = time.time()
        print(f"  Step 1b (pathwise): {t2-t1:.3f}s")

        # ── Step 2: Per-indicator correction ──────────────────────────
        sum_correction = np.zeros(self.n_theta)
        total_active = 0
        jump_unresolved = 0
        newton_unconverged = 0
        newton_far = 0

        z_cols = np.asfortranarray(z_all)             # once: columns of the paths for the Newton/jump points
        jobs = lambda i: self._indicator_correction(i, z_all, z_cols, g_all, vdg_all)
        if self.indicator_threads > 1 and self.n_ind > 1:
            with ThreadPoolExecutor(self.indicator_threads) as ex:
                parts = list(ex.map(jobs, range(self.n_ind)))
        else:
            parts = [jobs(i) for i in range(self.n_ind)]
        for i, part in enumerate(parts):              # indicator order: same sums as the sequential loop
            if part is None:
                continue
            contrib_i, n_keep, n_unres, n_unconv, n_far = part
            sum_correction += contrib_i
            total_active += n_keep
            jump_unresolved += n_unres
            newton_unconverged += n_unconv
            newton_far += n_far
            if (i + 1) % 10 == 0:
                print(f"  Step 2: indicator {i+1}/{self.n_ind}, "
                      f"active={total_active}")

        coin_terms = 0
        if self.coins is not None and self.coins.exp:
            ct = self.coins.term(z_all)
            sum_correction += ct.sum(axis=0)
            coin_terms = len(self.coin_skip)
        t3 = time.time()
        print(f"  Step 2 (correction): {t3-t2:.1f}s")

        correction = sum_correction / M
        total = pathwise + correction

        print(f"  Total: {t3-t_start:.1f}s")

        return {
            'price': price,
            'pathwise': pathwise,
            'correction': correction,
            'total': total,
            'total_active': total_active,
            'jump_unresolved': jump_unresolved,
            'newton_unconverged': newton_unconverged,
            'newton_far': newton_far,
            'coin_comparisons': coin_terms,
        }

    def _indicator_correction(self, i, z_all, z_cols, g_all, vdg_all):
        """Correction sums (n_theta,) and the number of active paths for indicator i; None if inactive."""
        sum_correction = np.zeros(self.n_theta)
        v_i, norm_i = self.directions[i]
        if norm_i < 1e-15 or i in self.coin_skip:          # coin comparisons: integrated exactly in run()
            return None

        g_i = g_all[:, i]
        if self.skip_sigma is None:                 # default: no screening (see skip_sigma in __init__)
            near_idx = np.arange(len(g_i))
        else:
            near_idx = np.where(np.abs(g_i) / norm_i < self.skip_sigma)[0]
        n_near = len(near_idx)
        if n_near == 0:
            return None

        M = z_all.shape[0]
        z_near = z_all if n_near == M else z_all[near_idx]          # all paths near: no copy
        u1_orig = z_near @ v_i
        vdg = vdg_all[near_idx, i]                     # v_i . grad_z g_i at the path points

        ok = np.abs(vdg) > 1e-15
        if self.skip_sigma is None:
            keep = np.ones(n_near, dtype=bool)
        else:
            ratio = np.full(n_near, np.inf)
            ratio[ok] = np.abs(g_i[near_idx][ok]) / np.abs(vdg[ok])
            keep = ratio < self.skip_sigma
        if not np.any(keep):
            return None

        ki = np.where(keep)[0]
        n_keep = len(ki)

        kept = near_idx[ki]                                          # rows of z_all
        all_kept = n_keep == M
        u1 = u1_orig[ki].copy()
        g_vals = g_i[near_idx[ki]]
        vdg_k = vdg[ki].copy()
        u1_orig_k = u1_orig[ki]

        if self.root_solver == "newton":               # previous solver, unchanged
            step = np.clip(g_vals / vdg_k, -3.0, 3.0)
        elif self.u_star is not None and np.isfinite(self.u_star[i]):
            step = u1 - self.u_star[i]                  # g = h(v_i . z): the boundary is u = U*_i on every path
        else:
            flat = np.abs(vdg_k) <= 1e-15             # no slope at the path point: step towards the boundary
            step = np.where(flat, np.sign(g_vals) * 3.0, g_vals / np.where(flat, 1.0, vdg_k))
            step = np.clip(step, -3.0, 3.0)
        u1 -= step
        converged = np.zeros(n_keep, dtype=bool)

        # ── Newton: selective evaluate (1 indicator only) ──────
        # One reverse pass (indicator i) per unconverged path per iteration.
        # The same pass gives dg_i/dz and dg_i/dθ, stored at convergence,
        # so no extra passes are needed at the root.
        ind_res = self.indicator_res[i]
        vdg_star = np.zeros(n_keep)
        root_err_u = np.zeros(n_keep)
        n_far = 0
        dg_dt_star = np.zeros((n_keep, self.n_theta))
        active = np.arange(n_keep)
        # The Newton points z + du*v_i are built and passed to AADC column by column from the column-major
        # copy of the paths (no (n, d) temporaries). rows: sorted subset of 0..n_keep-1; when it is all of them
        # and every path is kept, the unshifted columns are passed as views.
        nz_v = v_i != 0.0

        def columns(rows, shift):
            if all_kept and len(rows) == n_keep and rows[0] == 0 and rows[-1] == n_keep - 1 and np.all(np.diff(rows) == 1):
                return [z_cols[:, j] + shift * v_i[j] if nz_v[j] else z_cols[:, j] for j in range(self.d)]
            r = kept[rows]
            return [z_cols[r, j] + shift * v_i[j] if nz_v[j] else z_cols[r, j] for j in range(self.d)]

        if self.root_solver == "newton":
            # previous solver (plain Newton, unclipped after the first step), kept for comparison
            for newton_iter in range(self.max_newton):
                du = u1[active] - u1_orig_k[active]
                g_at, vdg_new, jt_n = self.vf.newton_step_columns(columns(active, du), ind_res, v_i)
                g_abs = np.abs(g_at)
                last_step = newton_iter == self.max_newton - 1
                newly_converged = (g_abs < 1e-8) & ((g_abs <= self.root_tol_u * np.abs(vdg_new)) | last_step)
                done = active[newly_converged]
                converged[done] = True
                vdg_star[done] = vdg_new[newly_converged]
                root_err_u[done] = g_abs[newly_converged] / np.maximum(np.abs(vdg_new[newly_converged]), 1e-300)
                if jt_n is not None:
                    dg_dt_star[done] = jt_n[newly_converged]
                active = active[~newly_converged]
                if len(active) == 0:
                    break
                g_at = g_at[~newly_converged]
                vdg_new = vdg_new[~newly_converged]
                good = np.abs(vdg_new) > 1e-15
                u1[active[good]] -= g_at[good] / vdg_new[good]
        else:
            n_far = self._solve_rtsafe(active, u1, u1_orig_k, columns, ind_res, v_i,
                                       converged, vdg_star, dg_dt_star, root_err_u)

        # ── Jump for converged ───────────────────────────────
        conv = np.where(converged)[0]
        n_conv = len(conv)
        n_unconv = n_keep - n_conv                     # Newton did not converge: no contribution (counted)
        if n_conv == 0:
            return sum_correction, n_keep, 0, n_unconv - n_far, n_far

        du_conv = u1[conv] - u1_orig_k[conv]

        vdg_final = vdg_star[conv]
        dg_dt_final = dg_dt_star[conv]

        good_vdg = np.abs(vdg_final) > 1e-15
        if not np.any(good_vdg):
            return sum_correction, n_keep, 0, n_unconv - n_far, n_far

        phi_vals = phi_normal_vec(u1[conv]) / np.abs(vdg_final)

        # Bracket the boundary: z* +- eps v must differ in the sign of indicator i only.
        err = root_err_u[conv]
        eps = np.maximum(self.jump_eps, 4.0 * err)
        ok = np.zeros(n_conv, dtype=bool)
        p_up = np.zeros(n_conv); p_dn = np.zeros(n_conv)
        todo = np.arange(n_conv)
        for _ in range(self.jump_tries):
            if len(todo) == 0:
                break
            up = self.vf.forward_columns_all(columns(conv[todo], du_conv[todo] + eps[todo]))
            dn = self.vf.forward_columns_all(columns(conv[todo], du_conv[todo] - eps[todo]))
            flips = (up[:, 1:] >= 0.0) != (dn[:, 1:] >= 0.0)
            own = flips[:, i]
            others = flips.sum(axis=1) - own
            good = own & (others == 0)
            ok[todo[good]] = True
            p_up[todo[good]] = up[good, 0]; p_dn[todo[good]] = dn[good, 0]
            shrink = (others > 0) & (eps[todo] / 10.0 > 2.0 * err[todo])
            grow = ~own & (others == 0)
            eps[todo[shrink]] /= 10.0
            eps[todo[grow]] *= 10.0
            todo = todo[shrink | grow]
        n_unres = int(n_conv - ok.sum())
        use = good_vdg & ok

        if self.indicator_types[i] == 'jump':
            # JUMP: discontinuity in payoff VALUE, jump = V(z*+eps v) - V(z*-eps v)
            jump = p_up - p_dn
            for k in range(self.n_theta):
                contrib = jump * phi_vals * dg_dt_final[:, k]
                contrib[~use] = 0.0
                sum_correction[k] += contrib.sum()

        else:  # kink
            # KINK: discontinuity in payoff DERIVATIVE (slope)
            # slope_jump_k = dV/dθ_k(z*+εv) - dV/dθ_k(z*-εv), reverse for the payoff only
            z_star_conv = z_all[kept[conv]] + du_conv[:, None] * v_i[None, :]
            z_above = z_star_conv + eps[:, None] * v_i[None, :]
            z_below = z_star_conv - eps[:, None] * v_i[None, :]
            _, _, jt_above, _ = self.vf.evaluate_selective_theta(z_above, [self.payoff_res])
            _, _, jt_below, _ = self.vf.evaluate_selective_theta(z_below, [self.payoff_res])
            if jt_above is not None and jt_below is not None:
                slope_jump = jt_above[:, 0, :] - jt_below[:, 0, :]  # (n_conv, n_theta)
                for k in range(self.n_theta):
                    # For kink: no dg/dθ factor — the slope jump IS the correction integrand
                    contrib = slope_jump[:, k] * phi_vals
                    contrib[~use] = 0.0
                    sum_correction[k] += contrib.sum()

        return sum_correction, n_keep, n_unres, n_unconv - n_far, n_far

    def _solve_rtsafe(self, active, u1, u1_orig_k, columns, ind_res, v_i, converged, vdg_star, dg_dt_star, root_err_u):
        """Root of g_i along v_i for every active pair (in place). Newton (one forward + reverse per step, steps
        clipped to +-STEP_CLIP) while it converges; once the root is bracketed (g < 0 and g > 0 seen) a Newton step that
        leaves the bracket or shrinks by less than half switches the pair to the Illinois method on the bracket
        (forward passes only; one forward + reverse at the root). Exits: root located to root_tol_u along v
        (converged), |u| > U_FAR (phi(u) = 0 exactly: dropped, returned count), max_newton (left unconverged).
        If Newton stalls without a bracket (it can cycle in a valley of g that does not reach 0), the pair searches
        for a sign change at centre -+ 0.25, 0.5, 1, ... (forward passes); none within 80: no root with phi > 0.
        Returns the number of pairs dropped (|u| > U_FAR, or no sign change found)."""
        n = len(u1)
        mode = np.zeros(n, dtype=np.int8)                # 0 Newton, 1 Illinois on the bracket
        u_neg = np.full(n, np.nan); g_neg = np.full(n, np.nan)
        u_pos = np.full(n, np.nan); g_pos = np.full(n, np.nan)
        prev_step = np.full(n, np.inf); side_last = np.zeros(n, dtype=np.int8)
        w_hist = np.full((n, 2), np.inf)                 # bracket widths after the last two bracket steps
        stall = np.zeros(n, dtype=np.int8)               # Newton steps in a row without progress
        g_best = np.full(n, np.inf)                      # smallest |g| seen in Newton mode
        g_last = np.full(n, np.inf); step_sign = np.zeros(n, dtype=np.int8)   # previous |g| and Newton step direction
        s_c = np.zeros(n); s_g = np.zeros(n); s_r = np.zeros(n)   # search mode: centre, g(centre), radius
        n_noroot = 0
        last_move = np.full(n, np.inf)                   # size of the last bracket-mode move
        need_final = np.zeros(n, dtype=bool)
        n_far = 0
        tol = self.root_tol_u
        for it in range(self.max_newton):
            if len(active) == 0:
                break
            a_n = active[mode[active] == 0]; a_b = active[mode[active] == 1]; a_s = active[mode[active] == 2]
            # --- all forward-only points of this iteration in ONE call: bracket points, search centre -+ radius
            if len(a_b) or len(a_s):
                rows_f = np.concatenate([a_b, a_s, a_s])
                shift_f = np.concatenate([u1[a_b], s_c[a_s] - s_r[a_s], s_c[a_s] + s_r[a_s]]) - u1_orig_k[rows_f]
                g_f = self.vf.forward_columns(columns(rows_f, shift_f), ind_res)
                nb, ns_ = len(a_b), len(a_s)
                g_b_all, g_lo, g_hi = g_f[:nb], g_f[nb:nb + ns_], g_f[nb + ns_:]
            # --- search mode: a sign change against g(centre) gives the bracket
            if len(a_s):
                sg = np.sign(s_g[a_s])
                left = np.sign(g_lo) != sg; right = (np.sign(g_hi) != sg) & ~left
                for side_mask, x_other, g_other in ((left, s_c[a_s] - s_r[a_s], g_lo), (right, s_c[a_s] + s_r[a_s], g_hi)):
                    r_ = a_s[side_mask]
                    if len(r_) == 0:
                        continue
                    xo, go, gc, xc = x_other[side_mask], g_other[side_mask], s_g[r_], s_c[r_]
                    negc = gc < 0
                    u_neg[r_] = np.where(negc, xc, xo); g_neg[r_] = np.where(negc, gc, go)
                    u_pos[r_] = np.where(negc, xo, xc); g_pos[r_] = np.where(negc, go, gc)
                    mode[r_] = 1; side_last[r_] = 0; u1[r_] = 0.5 * (xc + xo)
                none = ~(left | right)
                r_ = a_s[none]; s_r[r_] *= 2.0
                gone = r_[s_r[r_] > 2 * U_FAR]               # no sign change within |u - centre| <= 80: no root with phi > 0
                n_noroot += len(gone)
                active = np.setdiff1d(active, gone, assume_unique=True)
            # --- evaluate
            if len(a_n):
                g_n, vdg_n, jt_n = self.vf.newton_step_columns(columns(a_n, u1[a_n] - u1_orig_k[a_n]), ind_res, v_i)
            if len(a_b):
                g_b = g_b_all                    # a_b: bracket pairs at the start of this iteration, evaluated above
            # --- bracket update (Illinois: the side kept twice in a row has its g halved)
            for rows, g in ((a_n, g_n if len(a_n) else None), (a_b, g_b if len(a_b) else None)):
                if g is None or len(rows) == 0:
                    continue
                neg, pos = g < 0, g > 0
                r = rows[neg]; again = side_last[r] == -1; g_pos[r[again]] *= 0.5
                u_neg[r] = u1[r]; g_neg[r] = g[neg]; side_last[r] = -1
                r = rows[pos]; again = side_last[r] == 1; g_neg[r[again]] *= 0.5
                u_pos[r] = u1[r]; g_pos[r] = g[pos]; side_last[r] = 1
            width = np.abs(u_pos - u_neg)                 # nan where not bracketed
            done_rows = []
            # --- Newton pairs
            if len(a_n):
                g_abs = np.abs(g_n)
                conv = ((g_abs < 1e-8) & (g_abs <= tol * np.abs(vdg_n))) | (width[a_n] < tol) | (g_n == 0.0)
                d = a_n[conv]
                converged[d] = True; vdg_star[d] = vdg_n[conv]
                if jt_n is not None:
                    dg_dt_star[d] = jt_n[conv]
                root_err_u[d] = np.fmin(g_abs[conv] / np.maximum(np.abs(vdg_n[conv]), 1e-300), np.nan_to_num(width[d], nan=np.inf))
                done_rows.append(d)
                rest = ~conv; r = a_n[rest]; g = g_n[rest]; dv = vdg_n[rest]
                flat = np.abs(dv) <= 1e-15
                step = np.clip(np.where(flat, np.sign(g) * STEP_CLIP, g / np.where(flat, 1.0, dv)), -STEP_CLIP, STEP_CLIP)
                cand = u1[r] - step
                br = np.isfinite(width[r])
                lo = np.fmin(u_neg[r], u_pos[r]); hi = np.fmax(u_neg[r], u_pos[r])
                # progress = |g| falls below 0.9 x the best |g| so far (monotone convergence always does;
                # cycling, e.g. in a valley of g that does not reach 0, does not)
                # no progress = |g| not below 0.9 x its best AND (the step reversed direction or |g| grew):
                # cycling does that; a monotone march towards a far root with |g| falling slowly does not
                sgn = np.sign(step).astype(np.int8)
                reversed_ = (step_sign[r] != 0) & (sgn != step_sign[r])
                no_progress = (np.abs(g) > 0.9 * g_best[r]) & (reversed_ | (np.abs(g) >= g_last[r]))
                g_best[r] = np.minimum(g_best[r], np.abs(g)); g_last[r] = np.abs(g); step_sign[r] = sgn
                stall[r] = np.where(no_progress, stall[r] + 1, 0)
                # bracketed: leave Newton if its step leaves the bracket or it stalled twice in a row;
                # not bracketed and stalled twice (e.g. cycling in a valley of g that does not reach 0): search
                switch = br & ((cand <= lo) | (cand >= hi) | (stall[r] >= 2))
                mode[r[switch]] = 1
                to_search = ~br & (stall[r] >= 2)
                rs = r[to_search]
                mode[rs] = 2; s_c[rs] = u1[rs]; s_g[rs] = g[to_search]; s_r[rs] = 0.25
                switch = switch | to_search
                keep_n = ~switch
                u1[r[keep_n]] = cand[keep_n]; prev_step[r[keep_n]] = np.abs(step[keep_n])
            # --- bracket pairs: converged when the bracket is narrow; the root needs one Newton evaluation
            if len(a_b):
                fin = (width[a_b] < tol) | (g_b == 0.0) | (last_move[a_b] < tol)    # narrow bracket or the point stopped
                need_final[a_b[fin]] = True
                done_rows.append(a_b[fin])
            done = np.concatenate(done_rows) if done_rows else np.zeros(0, dtype=int)
            active = np.setdiff1d(active, done, assume_unique=True)
            # --- next point of the pairs in bracket mode: Illinois (regula falsi with halving), midpoint if unsafe
            r = active[mode[active] == 1]
            if len(r):
                lo = np.fmin(u_neg[r], u_pos[r]); hi = np.fmax(u_neg[r], u_pos[r])
                den = g_pos[r] - g_neg[r]
                x = np.where(den != 0.0, (u_neg[r] * g_pos[r] - u_pos[r] * g_neg[r]) / np.where(den != 0.0, den, 1.0), 0.5 * (lo + hi))
                bad = ~((x > lo) & (x < hi))
                # bisect when the bracket did not halve over the last two bracket steps (one-sided false position)
                w_now = hi - lo
                slow = w_now > 0.5 * w_hist[r, 0]
                w_hist[r, 0] = w_hist[r, 1]; w_hist[r, 1] = w_now
                x = np.where(bad | slow, 0.5 * (lo + hi), x)
                last_move[r] = np.abs(x - u1[r])
                u1[r] = x
            # --- exit: |u| > U_FAR (Newton pairs without a bracket; bracket entirely beyond U_FAR)
            far = (np.abs(u1[active]) > U_FAR) & (mode[active] != 2)
            if np.any(far):
                n_far += int(far.sum()); active = active[~far]
        # pairs located by the bracket: one forward + reverse at the root for dg/du and dg/dtheta
        fin = np.where(need_final)[0]
        if len(fin):
            g_f, vdg_f, jt_f = self.vf.newton_step_columns(columns(fin, u1[fin] - u1_orig_k[fin]), ind_res, v_i)
            converged[fin] = True; vdg_star[fin] = vdg_f
            if jt_f is not None:
                dg_dt_star[fin] = jt_f
            root_err_u[fin] = np.minimum(np.nan_to_num(np.abs(u_pos[fin] - u_neg[fin]), nan=np.inf),
                                         np.maximum(np.abs(g_f) / np.maximum(np.abs(vdg_f), 1e-300), np.where(np.isfinite(last_move[fin]), last_move[fin], 0.0)))
        return n_far + n_noroot
