"""
correction_driver_vec.py — Vectorized correction using aadc VectorFunctionWithAD.

Key optimization: per-indicator VFs (1 output each) for Newton —
avoids computing full 51-output jacobian when only 1 indicator is needed.
Batch forward via AVX + multithreading.
"""
import math
import time
import numpy as np
import aadc


def phi_normal_vec(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


class CorrectionDriverVec:

    def __init__(self, funcs, payoff_res, indicator_res, z_args, theta_args,
                 skip_sigma=20.0, jump_eps=1e-4, max_newton=8, num_threads=4):
        self.funcs = funcs
        self.payoff_res = payoff_res
        self.indicator_res = indicator_res
        self.z_args = z_args
        self.theta_args = theta_args
        self.skip_sigma = skip_sigma
        self.jump_eps = jump_eps
        self.max_newton = max_newton
        self.num_threads = num_threads

        self.d = len(z_args)
        self.n_ind = len(indicator_res)
        self.n_theta = len(theta_args)

        # VF for initial screening: all outputs, jac w.r.t. z
        all_res = [payoff_res] + list(indicator_res)
        self.vf_all = aadc.VectorFunctionWithAD(
            funcs, z_args, all_res,
            param_args=theta_args, num_threads=num_threads
        )

        # VF for payoff-only (jump evaluation, 1 output → 1 reverse pass)
        self.vf_payoff = aadc.VectorFunctionWithAD(
            funcs, z_args, [payoff_res],
            param_args=theta_args, num_threads=num_threads
        )

        # Per-indicator VFs (1 output each → 1 reverse pass for Newton)
        self.vf_ind = []
        for i in range(self.n_ind):
            vfi = aadc.VectorFunctionWithAD(
                funcs, z_args, [indicator_res[i]],
                param_args=theta_args, num_threads=num_threads
            )
            self.vf_ind.append(vfi)

        # Per-indicator per-theta VFs for dg_i/d(theta_k)
        self.vf_ind_theta = []
        for i in range(self.n_ind):
            vf_row = []
            for k in range(self.n_theta):
                other_thetas = [a for j, a in enumerate(theta_args) if j != k]
                vf_dt = aadc.VectorFunctionWithAD(
                    funcs, z_args + [theta_args[k]], [indicator_res[i]],
                    param_args=other_thetas, num_threads=num_threads
                )
                vf_row.append(vf_dt)
            self.vf_ind_theta.append(vf_row)

        # Payoff per-theta VFs for pathwise
        self.vf_payoff_theta = []
        for k in range(self.n_theta):
            other_thetas = [a for j, a in enumerate(theta_args) if j != k]
            vf_pw = aadc.VectorFunctionWithAD(
                funcs, z_args + [theta_args[k]], [payoff_res],
                param_args=other_thetas, num_threads=num_threads
            )
            self.vf_payoff_theta.append(vf_pw)

        self.directions = None

    def precompute_directions(self, theta_values):
        self.theta_values = theta_values
        self.theta_list = [theta_values[a] for a in self.theta_args]
        self.vf_all.set_params(self.theta_list)
        self.vf_payoff.set_params(self.theta_list)
        for vfi in self.vf_ind:
            vfi.set_params(self.theta_list)
        for k in range(self.n_theta):
            other_vals = [self.theta_list[j] for j in range(self.n_theta) if j != k]
            if other_vals:
                self.vf_payoff_theta[k].set_params(other_vals)
            for i in range(self.n_ind):
                if other_vals:
                    self.vf_ind_theta[i][k].set_params(other_vals)

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

        # ── Step 1: Batch forward + jac (all outputs, chunked) ───────
        BATCH = 5000
        n_out = 1 + self.n_ind
        vals_all = np.zeros((M, n_out))
        jac_all = np.zeros((M, n_out, self.d))
        for start in range(0, M, BATCH):
            end = min(start + BATCH, M)
            v_b, j_b, _ = self.vf_all.evaluate(z_all[start:end])
            vals_all[start:end] = v_b
            jac_all[start:end] = j_b
        t1 = time.time()
        print(f"  Step 1 (batch fwd+jac): {t1-t_start:.1f}s")

        prices = vals_all[:, 0]
        g_all = vals_all[:, 1:]
        price = prices.mean()

        # ── Step 1b: Pathwise delta via per-theta VF ────────────────
        # For each theta_k, create VF with theta_k in args (not params)
        # so jacobian includes d(payoff)/d(theta_k)
        pathwise = np.zeros(self.n_theta)
        for k in range(self.n_theta):
            x_pw = np.zeros((M, self.d + 1))
            x_pw[:, :self.d] = z_all
            x_pw[:, self.d] = self.theta_list[k]
            # Batch in chunks
            pw_sum = 0.0
            for start in range(0, M, BATCH):
                end = min(start + BATCH, M)
                _, j_pw, _ = self.vf_payoff_theta[k].evaluate(x_pw[start:end])
                pw_sum += j_pw[:, 0, self.d].sum()  # d(payoff)/d(theta_k)
            pathwise[k] = pw_sum / M
        t2 = time.time()
        print(f"  Step 1b (pathwise batch): {t2-t1:.1f}s")
        ws = self.funcs.create_workspace()

        # ── Step 2: Per-indicator correction ──────────────────────────
        sum_correction = np.zeros(self.n_theta)
        total_active = 0

        for i in range(self.n_ind):
            v_i, norm_i = self.directions[i]
            if norm_i < 1e-15:
                continue

            g_i = g_all[:, i]
            near_mask = np.abs(g_i) / norm_i < self.skip_sigma
            near_idx = np.where(near_mask)[0]
            n_near = len(near_idx)
            if n_near == 0:
                continue

            z_near = z_all[near_idx]
            u1_orig = z_near @ v_i

            # vdg from precomputed jac (already have it from Step 1)
            dg_dz = jac_all[near_idx, 1 + i, :]
            vdg = dg_dz @ v_i

            ok = np.abs(vdg) > 1e-15
            ratio = np.full(n_near, np.inf)
            ratio[ok] = np.abs(g_i[near_idx][ok]) / np.abs(vdg[ok])
            keep = ratio < self.skip_sigma
            if not np.any(keep):
                continue

            ki = np.where(keep)[0]
            n_keep = len(ki)
            total_active += n_keep

            z_keep = z_near[ki]
            u1 = u1_orig[ki].copy()
            g_vals = g_i[near_idx[ki]]
            vdg_k = vdg[ki].copy()
            u1_orig_k = u1_orig[ki]

            step = g_vals / vdg_k
            step = np.clip(step, -3.0, 3.0)
            u1 -= step

            converged = np.zeros(n_keep, dtype=bool)

            # ── Newton: batch forward via 1-output VF ────────────
            for newton_iter in range(self.max_newton):
                du = u1 - u1_orig_k
                z_star = z_keep + du[:, None] * v_i[None, :]

                # Batch: 1 output VF → 1 reverse pass per batch
                v_n, j_n, _ = self.vf_ind[i].evaluate(z_star)
                g_at = v_n[:, 0]
                dg_dz_n = j_n[:, 0, :]  # (n_keep, d)

                newly_converged = np.abs(g_at) < 1e-8
                converged |= newly_converged
                if np.all(converged):
                    break

                vdg_new = dg_dz_n @ v_i
                good = (~converged) & (np.abs(vdg_new) > 1e-15)
                u1[good] -= g_at[good] / vdg_new[good]

            # ── Final gradient + jump for converged ──────────────
            conv = np.where(converged)[0]
            n_conv = len(conv)
            if n_conv == 0:
                continue

            du_conv = u1[conv] - u1_orig_k[conv]
            z_star_conv = z_keep[conv] + du_conv[:, None] * v_i[None, :]

            # Final vdg from batch VF (1-output, already have jac w.r.t. z)
            v_f, j_f, _ = self.vf_ind[i].evaluate(z_star_conv)
            vdg_final = j_f[:, 0, :] @ v_i  # (n_conv,)

            # dg/dtheta at z* via cached per-theta VF batch
            dg_dt_final = np.zeros((n_conv, self.n_theta))
            for k in range(self.n_theta):
                x_dt = np.zeros((n_conv, self.d + 1))
                x_dt[:, :self.d] = z_star_conv
                x_dt[:, self.d] = self.theta_list[k]
                _, j_dt, _ = self.vf_ind_theta[i][k].evaluate(x_dt)
                dg_dt_final[:, k] = j_dt[:, 0, self.d]  # d(g_i)/d(theta_k)

            good_vdg = np.abs(vdg_final) > 1e-15
            if not np.any(good_vdg):
                continue

            phi_vals = phi_normal_vec(u1[conv]) / np.abs(vdg_final)

            # Jump: batch payoff-only VF (1 output → fast)
            eps = self.jump_eps
            z_above = z_star_conv + eps * v_i[None, :]
            z_below = z_star_conv - eps * v_i[None, :]

            v_above, _, _ = self.vf_payoff.evaluate(z_above)
            v_below, _, _ = self.vf_payoff.evaluate(z_below)
            jump = v_above[:, 0] - v_below[:, 0]

            for k in range(self.n_theta):
                contrib = jump * phi_vals * dg_dt_final[:, k]
                contrib[~good_vdg] = 0.0
                sum_correction[k] += contrib.sum()

            if (i + 1) % 10 == 0:
                print(f"  Step 2: indicator {i+1}/{self.n_ind}, "
                      f"active={total_active}")

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
        }
