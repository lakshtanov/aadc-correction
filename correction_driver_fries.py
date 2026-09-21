"""
correction_driver_fries.py — Fries (2018) stochastic AAD correction.

For each indicator g_i, the δ-function correction is:

    E[A · δ(g_i) · dg_i/dθ] ≈ (1/2w) · Σ_{|g_i| < w} [ΔP_i · dg_i/dθ] / M

where:
    w = window half-width (paths near boundary)
    ΔP_i = payoff jump when indicator i flips (estimated from near-boundary paths)
    dg_i/dθ = sensitivity of boundary to parameter (from adjoint)
    1/(2w) approximates the density φ(0)

No Newton, no root-finding. Just regression/counting near boundary.
Biased (O(w)), but fast and simple.

Reference: Fries, C.P. (2018), arXiv:1811.05741, §3.3.
"""
import math
import time
import numpy as np
import aadc


class CorrectionDriverFries:

    def __init__(self, funcs, payoff_res, indicator_res, z_args, theta_args,
                 window_fraction=0.1, num_threads=4):
        self.funcs = funcs
        self.payoff_res = payoff_res
        self.indicator_res = indicator_res
        self.z_args = z_args
        self.theta_args = theta_args
        self.window_fraction = window_fraction
        self.num_threads = num_threads

        self.d = len(z_args)
        self.n_ind = len(indicator_res)
        self.n_theta = len(theta_args)

        all_res = [payoff_res] + list(indicator_res)
        self.vf = aadc.VectorFunctionWithAD(
            funcs, z_args, all_res,
            param_args=theta_args, num_threads=num_threads
        )
        self.theta_values = None

    def precompute_directions(self, theta_values):
        """Store theta values. No directions needed for Fries."""
        self.theta_values = theta_values

    def run(self, z_all):
        M = z_all.shape[0]
        t_start = time.time()

        # ── Step 1: Batch forward (values only) ──────────────────────
        BATCH = 5000
        self.vf.set_params([self.theta_values[a] for a in self.theta_args])
        n_out = 1 + self.n_ind
        vals_all = np.zeros((M, n_out))
        for start in range(0, M, BATCH):
            end = min(start + BATCH, M)
            v_b, _, _ = self.vf.evaluate(z_all[start:end])
            vals_all[start:end] = v_b

        prices = vals_all[:, 0]
        g_all = vals_all[:, 1:]
        price = prices.mean()
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
        # For each indicator i with near-boundary paths:
        #
        # 1. Window w chosen so that ~window_fraction of paths are inside
        # 2. Density: φ(0) ≈ n_near / (2w · M)
        # 3. Jump ΔP: average payoff for g>0 minus average payoff for g<0
        #    among near-boundary paths
        # 4. dg/dθ: average of d(g_i)/d(θ) over near-boundary paths
        # 5. correction_i = ΔP · φ(0) · dg/dθ · M
        #                  = ΔP · (n_near / 2w) · dg/dθ

        sum_correction = np.zeros(self.n_theta)

        for i in range(self.n_ind):
            g_i = g_all[:, i]

            # Choose window: take closest window_fraction of paths to boundary
            abs_g = np.abs(g_i)
            n_window = max(20, int(M * self.window_fraction))
            # w = value of |g| at the n_window-th closest path
            partitioned = np.argpartition(abs_g, n_window)
            near_idx = partitioned[:n_window]
            w = abs_g[near_idx].max()

            if w < 1e-15:
                continue

            # For multi-indicator payoffs (barriers): only paths alive at
            # indicator i contribute to the jump. Alive = all earlier g_j > 0.
            alive_at_i = np.ones(M, dtype=bool)
            for j in range(i):
                alive_at_i &= (g_all[:, j] > 0)

            # Filter near paths to alive only
            alive_near = near_idx[alive_at_i[near_idx]]
            n_alive = len(alive_near)
            if n_alive < 5:
                continue

            # Split alive near paths
            near_pos = alive_near[g_i[alive_near] > 0]
            near_neg = alive_near[g_i[alive_near] <= 0]
            if len(near_pos) < 2 or len(near_neg) < 2:
                continue

            # Discretized delta (Fries §3.3):
            # correction = (1/M) · Σ_{near, alive} payoff_m · (dg_i/dθ)_m / (2w)
            #
            # payoff_m already includes all indicators.
            # Only alive_near paths with g_i > 0 have nonzero payoff.
            contrib = np.zeros(self.n_theta)
            for m_idx in alive_near:
                m = int(m_idx)
                P_m = prices[m]  # payoff (0 if knocked out at any step)
                if abs(P_m) < 1e-15:
                    continue
                for j in range(self.d):
                    ws.set_val(self.z_args[j], float(z_all[m, j]))
                for arg, val in self.theta_values.items():
                    ws.set_val(arg, val)
                ws.forward()
                ws.reset_diff()
                ws.set_diff(self.indicator_res[i], 1.0)
                ws.reverse()
                for k in range(self.n_theta):
                    contrib[k] += P_m * ws.diff(self.theta_args[k]) / (2.0 * w)
            for k in range(self.n_theta):
                sum_correction[k] += contrib[k]

        correction = sum_correction / M
        t3 = time.time()
        print(f"  Step 3 (Fries correction): {t3-t2:.1f}s")
        print(f"  Total: {t3-t_start:.1f}s")

        return {
            'price': price,
            'pathwise': pathwise,
            'correction': correction,
            'total': pathwise + correction,
        }
