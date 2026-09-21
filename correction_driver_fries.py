"""
correction_driver_fries.py — Fries (2018) stochastic AAD correction.

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


class CorrectionDriverFries:

    def __init__(self, funcs, payoff_res, indicator_res, z_args, theta_args,
                 window_fraction=0.05, jump_eps=1e-4, num_threads=4):
        self.funcs = funcs
        self.payoff_res = payoff_res
        self.indicator_res = indicator_res
        self.z_args = z_args
        self.theta_args = theta_args
        self.window_fraction = window_fraction
        self.jump_eps = jump_eps
        self.num_threads = num_threads

        self.d = len(z_args)
        self.n_ind = len(indicator_res)
        self.n_theta = len(theta_args)

        all_res = [payoff_res] + list(indicator_res)
        self.vf = aadc.VectorFunctionWithAD(
            funcs, z_args, all_res,
            param_args=theta_args, num_threads=num_threads
        )

        # Per-indicator single-output VF for gradient direction
        self.vf_ind = []
        for i in range(self.n_ind):
            vfi = aadc.VectorFunctionWithAD(
                funcs, z_args, [indicator_res[i]],
                param_args=theta_args, num_threads=num_threads
            )
            self.vf_ind.append(vfi)

        # Payoff-only VF for jump evaluation
        self.vf_payoff = aadc.VectorFunctionWithAD(
            funcs, z_args, [payoff_res],
            param_args=theta_args, num_threads=num_threads
        )

        self.directions = None
        self.theta_values = None

    def precompute_directions(self, theta_values):
        """Compute gradient directions at z=0."""
        self.theta_values = theta_values
        theta_list = [theta_values[a] for a in self.theta_args]
        self.vf.set_params(theta_list)
        self.vf_payoff.set_params(theta_list)
        for vfi in self.vf_ind:
            vfi.set_params(theta_list)

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
        M = z_all.shape[0]
        t_start = time.time()

        # ── Step 1: Batch forward (values only) ──────────────────────
        BATCH = 5000
        n_out = 1 + self.n_ind
        vals_all = np.zeros((M, n_out))
        for start in range(0, M, BATCH):
            end = min(start + BATCH, M)
            v_b, _, _ = self.vf.evaluate(z_all[start:end])
            vals_all[start:end] = v_b

        g_all = vals_all[:, 1:]
        price = vals_all[:, 0].mean()
        t1 = time.time()
        print(f"  Step 1 (batch fwd): {t1-t_start:.1f}s")

        # ── Step 2: Pathwise via scalar ws ───────────────────────────
        ws = self.funcs.create_workspace()
        pathwise = np.zeros(self.n_theta)
        for m in range(M):
            for j in range(self.d):
                ws.set_val(self.z_args[j], float(z_all[m, j]))
            for arg, val in self.theta_values.items():
                ws.set_val(arg, val)
            ws.forward()
            ws.reset_diff()
            ws.set_diff(self.payoff_res, 1.0)
            ws.reverse()
            for k in range(self.n_theta):
                pathwise[k] += ws.diff(self.theta_args[k])
        pathwise /= M
        t2 = time.time()
        print(f"  Step 2 (pathwise): {t2-t1:.1f}s")

        # ── Step 3: Per-indicator Fries correction ───────────────────
        #
        # For each near-boundary path (|g_i| < w):
        #   1. One Newton step to boundary: u1 = u1_orig - g_i/vdg
        #   2. Jump via tape replay: P(z*+εv) - P(z*-εv)
        #   3. dg/dθ via scalar reverse
        #   4. Accumulate: jump · dg/dθ / (2w)
        #
        # This is Fries discretized delta but with tape-based jump
        # estimation — works for any payoff (barriers, autocallables).

        sum_correction = np.zeros(self.n_theta)
        total_near = 0
        eps = self.jump_eps

        for i in range(self.n_ind):
            v_i, norm_i = self.directions[i]
            if norm_i < 1e-15:
                continue

            g_i = g_all[:, i]
            abs_g = np.abs(g_i)
            n_window = max(20, int(M * self.window_fraction))
            partitioned = np.argpartition(abs_g, min(n_window, M-1))
            near_idx = partitioned[:n_window]
            w = abs_g[near_idx].max()
            if w < 1e-15:
                continue

            z_near = z_all[near_idx]  # (n_window, d)
            u1_orig = z_near @ v_i    # (n_window,)
            total_near += n_window

            # Batch: get vdg for near paths via single-output VF
            v_n, j_n, _ = self.vf_ind[i].evaluate(z_near)
            g_near = v_n[:, 0]
            dg_dz = j_n[:, 0, :]  # (n_window, d)
            vdg = dg_dz @ v_i     # (n_window,)

            # Newton steps to boundary (2-3 iterations, batch)
            good = np.abs(vdg) > 1e-15
            u1 = u1_orig.copy()
            u1[good] -= np.clip(g_near[good] / vdg[good], -3.0, 3.0)

            for newton_it in range(3):
                du = u1 - u1_orig
                z_star_it = z_near + du[:, None] * v_i[None, :]
                v_it, j_it, _ = self.vf_ind[i].evaluate(z_star_it)
                g_it = v_it[:, 0]
                converged = np.abs(g_it) < 1e-8
                if np.all(converged):
                    break
                vdg_it = j_it[:, 0, :] @ v_i
                update = (~converged) & (np.abs(vdg_it) > 1e-15)
                u1[update] -= g_it[update] / vdg_it[update]

            du = u1 - u1_orig
            z_star = z_near + du[:, None] * v_i[None, :]

            # Jump via batch tape replay (payoff-only VF)
            z_above = z_star + eps * v_i[None, :]
            z_below = z_star - eps * v_i[None, :]
            p_above, _, _ = self.vf_payoff.evaluate(z_above)
            p_below, _, _ = self.vf_payoff.evaluate(z_below)
            jump = p_above[:, 0] - p_below[:, 0]  # (n_window,)

            # dg/dθ at near-boundary paths via scalar ws
            dg_dt = np.zeros((n_window, self.n_theta))
            for pos in range(n_window):
                m = int(near_idx[pos])
                for j in range(self.d):
                    ws.set_val(self.z_args[j], float(z_all[m, j]))
                for arg, val in self.theta_values.items():
                    ws.set_val(arg, val)
                ws.forward()
                ws.reset_diff()
                ws.set_diff(self.indicator_res[i], 1.0)
                ws.reverse()
                for k in range(self.n_theta):
                    dg_dt[pos, k] = ws.diff(self.theta_args[k])

            # Accumulate: Σ jump · dg/dθ / (2w)
            for k in range(self.n_theta):
                sum_correction[k] += np.sum(jump * dg_dt[:, k]) / (2.0 * w)

            if (i + 1) % max(1, self.n_ind // 5) == 0 or i == self.n_ind - 1:
                print(f"  Step 3: indicator {i+1}/{self.n_ind}")

        correction = sum_correction / M
        t3 = time.time()
        print(f"  Step 3 (Fries correction): {t3-t2:.1f}s")
        print(f"  Total: {t3-t_start:.1f}s")

        return {
            'price': price,
            'pathwise': pathwise,
            'correction': correction,
            'total': pathwise + correction,
            'total_near': total_near,
        }
