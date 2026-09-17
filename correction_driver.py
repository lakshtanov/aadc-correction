"""
correction_driver.py — Delta-function correction for discontinuous MC payoffs.

Pure Python implementation using aadc (pip install aadc).

Usage:
    import aadc
    from correction_driver import CorrectionDriver

    # Record kernel B with z as active inputs
    fn = aadc.Functions()
    fn.start_recording()
    z = [aadc.idouble(0.0) for _ in range(d)]
    z_args = [zi.mark_as_input() for zi in z]         # z = Diff (for Newton)
    S0 = aadc.idouble(100.0)
    theta_args = [S0.mark_as_input()]                  # parameters for Greeks
    # ... build payoff with aadc.iIf ...
    payoff_res = payoff.mark_as_output()
    g_res = [gi.mark_as_output() for gi in indicators]
    fn.stop_recording()

    # Setup
    driver = CorrectionDriver(fn, payoff_res, g_res, z_args, theta_args)
    driver.precompute_directions()

    # MC loop
    for path in range(M):
        z_vals = np.random.randn(d)
        # Forward kernel B at this path's z
        for j, za in enumerate(z_args):
            ws.set_val(za, z_vals[j])
        ws.forward()

        # Compute correction for this path
        result = driver.compute_correction(z_vals)
        for k in range(n_theta):
            sum_correction[k] += result.correction[k]
"""

import math
import numpy as np


def phi_normal(x):
    """Standard normal density."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


class CorrectionResult:
    __slots__ = ('correction', 'num_active', 'num_newton_iters',
                 'num_forwards', 'num_reverses', 'num_newton_fail',
                 'u1_stars')

    def __init__(self, n_theta, n_ind):
        self.correction = [0.0] * n_theta
        self.num_active = 0
        self.num_newton_iters = 0
        self.num_forwards = 0
        self.num_reverses = 0
        self.num_newton_fail = 0
        self.u1_stars = [float('nan')] * n_ind


class CorrectionDriver:
    """Delta-function correction driver for aadc-ng Python API."""

    def __init__(self, funcs, payoff_res, indicator_res, z_args, theta_args,
                 skip_sigma=20.0, jump_eps=1e-4, max_newton=4):
        """
        Args:
            funcs:         aadc.Functions object (kernel B, z=Diff)
            payoff_res:    aadc.Result for the payoff output
            indicator_res: list of aadc.Result for each indicator g_i
            z_args:        list of aadc.Argument for z inputs (Diff)
            theta_args:    list of aadc.Argument for parameter inputs
            skip_sigma:    screening threshold (default 20)
            jump_eps:      central difference epsilon (default 1e-4)
            max_newton:    max Newton iterations (default 4)
        """
        self.funcs = funcs
        self.ws = funcs.create_workspace()
        self.payoff_res = payoff_res
        self.indicator_res = indicator_res
        self.z_args = z_args
        self.theta_args = theta_args
        self.skip_sigma = skip_sigma
        self.jump_eps = jump_eps
        self.max_newton = max_newton

        self.d = len(z_args)
        self.n_ind = len(indicator_res)
        self.n_theta = len(theta_args)

        # Precomputed directions: list of (sparse_entries, norm)
        # sparse_entries = list of (z_index, weight)
        self.directions = None

    def precompute_directions(self, theta_values=None):
        """
        Compute gradient directions at z=0.
        Call once before MC loop.

        Args:
            theta_values: dict {arg: value} for theta inputs (optional)
        """
        ws = self.ws

        # Set z = 0
        for j in range(self.d):
            ws.set_val(self.z_args[j], 0.0)

        # Set theta if provided
        if theta_values:
            for arg, val in theta_values.items():
                ws.set_val(arg, val)

        ws.forward()

        self.directions = []
        for i in range(self.n_ind):
            ws.reset_diff()
            ws.set_diff(self.indicator_res[i], 1.0)
            ws.reverse()

            entries = []
            sum_sq = 0.0
            for j in range(self.d):
                grad_j = ws.diff(self.z_args[j])
                if abs(grad_j) > 1e-15:
                    entries.append((j, grad_j))
                    sum_sq += grad_j * grad_j

            norm = math.sqrt(sum_sq)
            if norm > 1e-15:
                entries = [(idx, w / norm) for idx, w in entries]

            self.directions.append((entries, norm))

    def _set_zstar(self, z_values, ind_i, u1, u1_orig):
        """Set z* = z + (u1 - u1_orig) * v_i on workspace."""
        ws = self.ws
        du = u1 - u1_orig
        for j in range(self.d):
            ws.set_val(self.z_args[j], z_values[j])
        for idx, w in self.directions[ind_i][0]:
            ws.set_val(self.z_args[idx], z_values[idx] + du * w)

    def compute_correction(self, z_values):
        """
        Compute correction for one MC path.

        Args:
            z_values: list/array of d normal random values for this path.
                      Workspace must already have forward() called at these z.

        Returns:
            CorrectionResult
        """
        if self.directions is None:
            raise RuntimeError("Call precompute_directions() first")

        ws = self.ws
        result = CorrectionResult(self.n_theta, self.n_ind)

        # ── SCREENING ──
        screened = []
        for i in range(self.n_ind):
            g_val = ws.val(self.indicator_res[i])

            entries, grad_norm = self.directions[i]
            if grad_norm < 1e-15:
                continue
            if abs(g_val) / grad_norm > self.skip_sigma:
                continue

            # Project path onto direction
            u1_orig = sum(w * z_values[idx] for idx, w in entries)

            # Reverse for vdg and dg/dtheta
            ws.reset_diff()
            ws.set_diff(self.indicator_res[i], 1.0)
            ws.reverse()
            result.num_reverses += 1

            vdg = sum(w * ws.diff(self.z_args[idx]) for idx, w in entries)
            if abs(vdg) < 1e-15:
                continue
            if abs(g_val) / abs(vdg) > self.skip_sigma:
                continue

            dg_dt = [ws.diff(self.theta_args[k]) for k in range(self.n_theta)]
            screened.append((i, g_val, vdg, u1_orig, dg_dt))

        # ── NEWTON + JUMP ──
        for ind_i, g_val, vdg, u1_orig, dg_dtheta in screened:
            result.num_active += 1

            # Newton initial guess
            u1 = u1_orig
            step = g_val / vdg
            if abs(step) > 3.0:
                step = 3.0 if step > 0 else -3.0
            u1 -= step

            converged = False
            for it in range(self.max_newton):
                result.num_newton_iters += 1

                self._set_zstar(z_values, ind_i, u1, u1_orig)
                ws.forward()
                result.num_forwards += 1
                g_at = ws.val(self.indicator_res[ind_i])

                if abs(g_at) < 1e-8:
                    converged = True
                    break

                ws.reset_diff()
                ws.set_diff(self.indicator_res[ind_i], 1.0)
                ws.reverse()
                result.num_reverses += 1

                vdg = sum(w * ws.diff(self.z_args[idx])
                          for idx, w in self.directions[ind_i][0])
                if abs(vdg) < 1e-15:
                    break
                u1 -= g_at / vdg

            if not converged:
                result.num_newton_fail += 1
                continue

            result.u1_stars[ind_i] = u1

            # Final gradient at z*
            self._set_zstar(z_values, ind_i, u1, u1_orig)
            ws.forward()
            result.num_forwards += 1

            ws.reset_diff()
            ws.set_diff(self.indicator_res[ind_i], 1.0)
            ws.reverse()
            result.num_reverses += 1

            vdg = sum(w * ws.diff(self.z_args[idx])
                      for idx, w in self.directions[ind_i][0])
            dg_dtheta = [ws.diff(self.theta_args[k])
                         for k in range(self.n_theta)]

            if abs(vdg) < 1e-15:
                continue

            phi_oj = phi_normal(u1) / abs(vdg)

            # Jump: P(z*+eps*v) - P(z*-eps*v)
            eps = self.jump_eps
            self._set_zstar(z_values, ind_i, u1 + eps, u1_orig)
            ws.forward()
            result.num_forwards += 1
            pv_above = ws.val(self.payoff_res)

            self._set_zstar(z_values, ind_i, u1 - eps, u1_orig)
            ws.forward()
            result.num_forwards += 1
            pv_below = ws.val(self.payoff_res)

            payoff_jump = pv_above - pv_below

            # Accumulate
            for k in range(self.n_theta):
                result.correction[k] += payoff_jump * phi_oj * dg_dtheta[k]

        return result
