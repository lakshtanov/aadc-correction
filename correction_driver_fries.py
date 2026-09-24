"""
correction_driver_fries.py — Fries (2018) stochastic AAD correction.

Fully vectorized: batch evaluate returns jac_z AND jac_theta.
No scalar workspace loops.

Discretized delta: for each indicator g_i, collect near-boundary paths
(|g_i| < w), estimate density as n_near/(2w·M), compute jump via tape
replay (one Newton step + central difference). Universal for any payoff.

Biased (O(w)), but fast — no iterative Newton convergence needed.

Reference: Fries, C.P. (2018), arXiv:1811.05741, §3.3.
"""
import math
import time
import numpy as np
import aadc
import sys
sys.path.insert(0, "/tmp/aadc-correction-check")
from aadc_extensions import VectorFunctionSelectiveAD


class CorrectionDriverFries:

    def __init__(self, funcs, payoff_res, indicator_res, z_args, theta_args,
                 window_fraction=0.05, jump_eps=1e-4, num_threads=4,
                 adaptive=False, adaptive_windows=(0.02, 0.05, 0.10),
                 adaptive_tol=0.05):
        self.funcs = funcs
        self.payoff_res = payoff_res
        self.indicator_res = indicator_res
        self.z_args = z_args
        self.theta_args = theta_args
        self.window_fraction = window_fraction
        self.jump_eps = jump_eps
        self.num_threads = num_threads
        self.adaptive = adaptive
        self.adaptive_windows = adaptive_windows
        self.adaptive_tol = adaptive_tol

        self.d = len(z_args)
        self.n_ind = len(indicator_res)
        self.n_theta = len(theta_args)

        all_res = [payoff_res] + list(indicator_res)
        self.vf = VectorFunctionSelectiveAD(
            funcs, z_args, all_res,
            param_args=theta_args, num_threads=num_threads
        )

        self.directions = None
        self.theta_values = None

    def precompute_directions(self, theta_values):
        self.theta_values = theta_values
        theta_list = [theta_values[a] for a in self.theta_args]
        self.vf.set_params(theta_list)

        ws = self.funcs.create_workspace()
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

    def run(self, z_all):
        if self.adaptive:
            return self._run_adaptive(z_all)
        return self._run_fixed(z_all, self.window_fraction)

    def _run_adaptive(self, z_all):
        """Split-half adaptive window selection."""
        M = z_all.shape[0]
        half = M // 2
        z_half1 = z_all[:half]
        z_half2 = z_all[half:2*half]

        windows = sorted(self.adaptive_windows)
        stable_results = {}
        chosen = None

        for wf in windows:
            r_full = self._run_fixed(z_all, wf, quiet=True)
            r_h1 = self._run_fixed(z_half1, wf, quiet=True)
            r_h2 = self._run_fixed(z_half2, wf, quiet=True)

            is_stable = True
            for k in range(self.n_theta):
                ref = max(abs(r_h1['correction'][k]), abs(r_h2['correction'][k]), 1e-15)
                if abs(r_h1['correction'][k] - r_h2['correction'][k]) / ref > self.adaptive_tol:
                    is_stable = False
                    break

            stable_results[wf] = (r_full, is_stable)
            marker = "stable" if is_stable else "UNSTABLE"
            c = r_full['correction']
            c1 = r_h1['correction']
            c2 = r_h2['correction']
            print(f"    w={wf*100:5.1f}%: corr={[f'{x:+.4f}' for x in c]}  "
                  f"half1={[f'{x:+.4f}' for x in c1]}  "
                  f"half2={[f'{x:+.4f}' for x in c2]}  {marker}")

            if is_stable and chosen is None:
                chosen = wf

        if chosen is None:
            chosen = windows[-1]

        print(f"  Adaptive: chose w={chosen*100:.1f}%")
        return stable_results[chosen][0]

    def _run_fixed(self, z_all, window_fraction, quiet=False):
        M = z_all.shape[0]
        t_start = time.time()

        # ── Step 1: Batch forward + jac_theta ─────────────────────────
        BATCH = 5000
        n_out = 1 + self.n_ind
        vals_all = np.zeros((M, n_out))
        jac_theta_all = np.zeros((M, n_out, self.n_theta)) if self.n_theta > 0 else None
        for start in range(0, M, BATCH):
            end = min(start + BATCH, M)
            v_b, _, jt_b, _ = self.vf.evaluate(z_all[start:end])
            vals_all[start:end] = v_b
            if jac_theta_all is not None and jt_b is not None:
                jac_theta_all[start:end] = jt_b

        g_all = vals_all[:, 1:]
        price = vals_all[:, 0].mean()
        t1 = time.time()
        if not quiet: print(f"  Step 1 (batch fwd): {t1-t_start:.1f}s")

        # ── Step 2: Pathwise from jac_theta ───────────────────────────
        if jac_theta_all is not None:
            pathwise = jac_theta_all[:, 0, :].mean(axis=0)
        else:
            pathwise = np.zeros(self.n_theta)
        t2 = time.time()
        if not quiet: print(f"  Step 2 (pathwise): {t2-t1:.1f}s")

        # ── Step 3: Per-indicator Fries correction ────────────────────
        sum_correction = np.zeros(self.n_theta)
        total_near = 0
        eps = self.jump_eps

        for i in range(self.n_ind):
            v_i, norm_i = self.directions[i]
            if norm_i < 1e-15:
                continue

            g_i = g_all[:, i]
            abs_g = np.abs(g_i)
            n_window = max(20, int(M * window_fraction))
            partitioned = np.argpartition(abs_g, min(n_window, M-1))
            near_idx = partitioned[:n_window]
            w = abs_g[near_idx].max()
            if w < 1e-15:
                continue

            z_near = z_all[near_idx]
            u1_orig = z_near @ v_i
            total_near += n_window

            # Batch: get vdg for near paths via selective evaluate
            ind_res = self.indicator_res[i]
            v_n, j_n, _ = self.vf.evaluate_selective(z_near, [ind_res])
            g_near = v_n[:, 1 + i]
            dg_dz = j_n[:, 0, :]
            vdg = dg_dz @ v_i

            # Newton steps to boundary (2-3 iterations, batch)
            good = np.abs(vdg) > 1e-15
            u1 = u1_orig.copy()
            u1[good] -= np.clip(g_near[good] / vdg[good], -3.0, 3.0)

            for newton_it in range(3):
                du = u1 - u1_orig
                z_star_it = z_near + du[:, None] * v_i[None, :]
                v_it, j_it, _ = self.vf.evaluate_selective(z_star_it, [ind_res])
                g_it = v_it[:, 1 + i]
                converged = np.abs(g_it) < 1e-8
                if np.all(converged):
                    break
                vdg_it = j_it[:, 0, :] @ v_i
                update = (~converged) & (np.abs(vdg_it) > 1e-15)
                u1[update] -= g_it[update] / vdg_it[update]

            du = u1 - u1_orig
            z_star = z_near + du[:, None] * v_i[None, :]

            # Jump via batch forward (payoff-only)
            z_above = z_star + eps * v_i[None, :]
            z_below = z_star - eps * v_i[None, :]
            p_above = self.vf.evaluate_forward(z_above)
            p_below = self.vf.evaluate_forward(z_below)
            jump = p_above[:, 0] - p_below[:, 0]

            # dg/dθ from jac_theta (already computed in Step 1)
            dg_dt = jac_theta_all[near_idx, 1 + i, :] if jac_theta_all is not None \
                else np.zeros((n_window, self.n_theta))

            # Accumulate: Σ jump · dg/dθ / (2w)
            for k in range(self.n_theta):
                sum_correction[k] += np.sum(jump * dg_dt[:, k]) / (2.0 * w)

            if (i + 1) % max(1, self.n_ind // 5) == 0 or i == self.n_ind - 1:
                if not quiet: print(f"  Step 3: indicator {i+1}/{self.n_ind}")

        correction = sum_correction / M
        t3 = time.time()
        if not quiet:
            print(f"  Step 3 (Fries correction): {t3-t2:.1f}s")
            print(f"  Total: {t3-t_start:.1f}s")

        return {
            'price': price,
            'pathwise': pathwise,
            'correction': correction,
            'total': pathwise + correction,
            'total_near': total_near,
        }
