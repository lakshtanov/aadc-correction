"""
correction_driver_vec3.py — Path-batched correction driver.

Memory-friendly: processes paths in batches, all indicators per batch.
z_all loaded once per batch, stays in cache for all indicators.

vs vec2: vec2 loops indicators outer, paths inner (cold cache per indicator).
vec3 loops paths outer, indicators inner (hot cache per batch).

Newton: max iterations across indicators in batch (worst-case convergence).
"""
import math
import time
import numpy as np
import aadc
import sys
sys.path.insert(0, "/tmp/aadc-correction-check")
from aadc_extensions import VectorFunctionSelectiveAD


def phi_normal_vec(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


class CorrectionDriverVec3:

    def __init__(self, funcs, payoff_res, z_args, theta_args,
                 indicator_res=None, indicator_types=None,
                 skip_sigma=20.0, jump_eps=1e-4, max_newton=8, num_threads=4,
                 batch_size=5000):
        self.funcs = funcs
        self.payoff_res = payoff_res
        self.z_args = z_args
        self.theta_args = theta_args
        self.skip_sigma = skip_sigma
        self.jump_eps = jump_eps
        self.max_newton = max_newton
        self.num_threads = num_threads
        self.batch_size = batch_size

        if indicator_res is None:
            switches = funcs.cmp_switches()
            indicator_res = [sw.g for sw in switches]
            indicator_types = [sw.origin for sw in switches]

        self.indicator_res = indicator_res
        self.indicator_types = indicator_types or ['jump'] * len(indicator_res)

        self.d = len(z_args)
        self.n_ind = len(indicator_res)
        self.n_theta = len(theta_args)

        all_res = [payoff_res] + list(indicator_res)
        self.vf = VectorFunctionSelectiveAD(
            funcs, z_args, all_res,
            param_args=theta_args, num_threads=num_threads
        )

        self.directions = None

    def precompute_directions(self, theta_values):
        self.theta_values = theta_values
        self.theta_list = [theta_values[a] for a in self.theta_args]
        self.vf.set_params(self.theta_list)

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

        # ── Step 1: Batch forward + jac (same as vec2) ──
        BATCH = self.batch_size
        n_out = 1 + self.n_ind
        vals_all = np.zeros((M, n_out))
        jac_all = np.zeros((M, n_out, self.d))
        jac_theta_all = np.zeros((M, n_out, self.n_theta)) if self.n_theta > 0 else None
        for start in range(0, M, BATCH):
            end = min(start + BATCH, M)
            v_b, j_b, jt_b, _ = self.vf.evaluate(z_all[start:end])
            vals_all[start:end] = v_b
            jac_all[start:end] = j_b
            if jac_theta_all is not None and jt_b is not None:
                jac_theta_all[start:end] = jt_b
        t1 = time.time()
        print(f"  Step 1 (batch fwd+jac): {t1-t_start:.1f}s")

        prices = vals_all[:, 0]
        g_all = vals_all[:, 1:]
        price = prices.mean()

        if jac_theta_all is not None:
            pathwise = jac_theta_all[:, 0, :].mean(axis=0)
        else:
            pathwise = np.zeros(self.n_theta)
        t2 = time.time()

        # ── Step 2: Path-batched correction ──
        # For each batch of paths, process ALL indicators
        sum_correction = np.zeros(self.n_theta)
        total_active = 0
        eps = self.jump_eps

        for start in range(0, M, BATCH):
            end = min(start + BATCH, M)
            z_batch = z_all[start:end]
            n_batch = end - start

            for i in range(self.n_ind):
                v_i, norm_i = self.directions[i]
                if norm_i < 1e-15:
                    continue

                g_i = g_all[start:end, i]
                dg_dz = jac_all[start:end, 1 + i, :]
                vdg = dg_dz @ v_i

                # Screening
                ok = np.abs(vdg) > 1e-15
                ratio = np.full(n_batch, np.inf)
                ratio[ok] = np.abs(g_i[ok]) / np.abs(vdg[ok])
                keep = ratio < self.skip_sigma
                if not np.any(keep):
                    continue

                ki = np.where(keep)[0]
                n_keep = len(ki)
                total_active += n_keep

                z_keep = z_batch[ki]
                u1_orig = z_keep @ v_i
                u1 = u1_orig.copy()
                g_vals = g_i[ki]
                vdg_k = vdg[ki].copy()

                step = np.clip(g_vals / vdg_k, -3.0, 3.0)
                u1 -= step
                converged = np.zeros(n_keep, dtype=bool)

                # Newton: worst-case iterations across batch
                ind_res = self.indicator_res[i]
                for newton_iter in range(self.max_newton):
                    du = u1 - u1_orig
                    z_star = z_keep + du[:, None] * v_i[None, :]

                    v_n, j_n, _ = self.vf.evaluate_selective(z_star, [ind_res])
                    g_at = v_n[:, 1 + i]
                    dg_dz_n = j_n[:, 0, :]

                    newly_converged = np.abs(g_at) < 1e-8
                    converged |= newly_converged
                    if np.all(converged):
                        break

                    vdg_new = dg_dz_n @ v_i
                    good = (~converged) & (np.abs(vdg_new) > 1e-15)
                    u1[good] -= g_at[good] / vdg_new[good]

                conv = np.where(converged)[0]
                n_conv = len(conv)
                if n_conv == 0:
                    continue

                du_conv = u1[conv] - u1_orig[conv]
                z_star_conv = z_keep[conv] + du_conv[:, None] * v_i[None, :]

                _, j_f, _ = self.vf.evaluate_selective(z_star_conv, [ind_res])
                vdg_final = j_f[:, 0, :] @ v_i

                v_final, _, jt_final, _ = self.vf.evaluate(z_star_conv)
                if jt_final is not None:
                    dg_dt_final = jt_final[:, 1 + i, :]
                else:
                    dg_dt_final = np.zeros((n_conv, self.n_theta))

                good_vdg = np.abs(vdg_final) > 1e-15
                if not np.any(good_vdg):
                    continue

                phi_vals = phi_normal_vec(u1[conv]) / np.abs(vdg_final)

                z_above = z_star_conv + eps * v_i[None, :]
                z_below = z_star_conv - eps * v_i[None, :]

                if self.indicator_types[i] == 'jump':
                    v_above = self.vf.evaluate_forward(z_above)
                    v_below = self.vf.evaluate_forward(z_below)
                    jump = v_above[:, 0] - v_below[:, 0]
                    for k in range(self.n_theta):
                        contrib = jump * phi_vals * dg_dt_final[:, k]
                        contrib[~good_vdg] = 0.0
                        sum_correction[k] += contrib.sum()
                else:
                    _, _, jt_above, _ = self.vf.evaluate(z_above)
                    _, _, jt_below, _ = self.vf.evaluate(z_below)
                    if jt_above is not None and jt_below is not None:
                        slope_jump = jt_above[:, 0, :] - jt_below[:, 0, :]
                        for k in range(self.n_theta):
                            contrib = slope_jump[:, k] * phi_vals
                            contrib[~good_vdg] = 0.0
                            sum_correction[k] += contrib.sum()

            if (end) % (BATCH * 10) == 0 or end == M:
                print(f"  Step 2: batch {end}/{M}, active={total_active}")

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
