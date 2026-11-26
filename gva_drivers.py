"""gva_drivers.py -- the two GVA variants of U. Naumann ("Greeks Value Adjustment (GVA)", talk, QuantMinds International
2026, London) implemented on AADC tapes, for comparison with the correction of this repository
(docs/comparison_kovalov_naumann.md).

GVA-local (their tangent rule): at every branch c >= 0 the tangent gets  jump * phi_s(c) * dc/dtheta,  phi_s a Gaussian
  density of width s -- a mollified delta, biased O(s), the primal unchanged. On a tape whose payoff is multilinear in
  the indicators (all our products) this is, path by path,
      dV/dtheta ~ E[ dF/dtheta ] + sum_i E[ jump_i(Z) * phi_s(g_i(Z)) * dg_i/dtheta(Z) ],
  jump_i(Z) = payoff above minus below the boundary of g_i. As in the Fries driver the jump is taken at the boundary point
  reached from the path along the indicator's direction (one-to-three Newton steps), dg_i/dtheta at the path point.
  s = S_FRAC x (standard deviation of g_i over the paths); the best S_FRAC of a grid is chosen per target (oracle).
GVA-integral (their co-area rule, slides 15-17): jump * dc/dtheta / |dc/dx| at the root of c along ONE input variable.
  In a Gaussian Monte Carlo this is the boundary term of a one-dimensional slice -- our Theorem 1 -- with the slice taken
  along a coordinate axis (the input with the largest weight in the indicator's gradient) instead of the normal to the
  boundary; no coins (the exact integration of the default threshold is ours).
"""
import math, time
import numpy as np
from correction_driver_fries import CorrectionDriverFries
from correction_driver_vec2 import CorrectionDriverVec2


class GVALocal(CorrectionDriverFries):
    """Fries' machinery (near-boundary paths, Newton to the boundary, jump by two replays) with a Gaussian kernel."""

    def __init__(self, funcs, payoff_res, indicator_res, z_args, theta_args, s_frac=0.05, cut=6.0, **kw):
        super().__init__(funcs, payoff_res, indicator_res, z_args, theta_args, **kw)
        self.s_frac, self.cut = s_frac, cut

    def precompute_directions(self, theta_values):
        """As the parent, but on an explicit scalar workspace (after the JIT compile the default kind is vector)."""
        self.theta_values = theta_values
        self.vf.set_params([theta_values[a] for a in self.theta_args])
        ths = set(self.theta_args)
        self.vf.set_fixed_inputs({a: v for a, v in theta_values.items() if a not in ths})
        ws = self.funcs.create_workspace("scalar")
        for j in range(self.d):
            ws.set_val(self.z_args[j], 0.0)
        for arg, val in theta_values.items():
            ws.set_val(arg, val)
        ws.forward()
        self.directions = []
        for i in range(self.n_ind):
            ws.reset_diff(); ws.set_diff(self.indicator_res[i], 1.0); ws.reverse()
            v = np.array([ws.diff(self.z_args[j]) for j in range(self.d)], dtype=float)
            n = np.linalg.norm(v)
            self.directions.append((v / n if n > 1e-15 else v, n))

    def run(self, z_all):
        M = z_all.shape[0]
        BATCH = 5000
        n_out = 1 + self.n_ind
        vals = np.zeros((M, n_out)); jt = np.zeros((M, n_out, self.n_theta))
        outs = [self.payoff_res] + list(self.indicator_res)
        req = {r: list(self.theta_args) for r in outs}            # values + theta-derivatives only (no z-Jacobian)
        for a in range(0, M, BATCH):
            b = min(a + BATCH, M)
            inp = self.vf._inputs_from_columns([np.ascontiguousarray(z_all[a:b, j]) for j in range(self.d)])
            vv, dd, _ = self.vf._evaluate(self.funcs, req, inp, self.num_threads)
            for k, r in enumerate(outs):
                vals[a:b, k] = vv[r]
                for q, th in enumerate(self.theta_args):
                    jt[a:b, k, q] = dd[r][th]
        pathwise = jt[:, 0, :].mean(0)
        g_all = vals[:, 1:]
        corr = np.zeros(self.n_theta); eps = self.jump_eps
        for i in range(self.n_ind):
            v_i, norm_i = self.directions[i]
            if norm_i < 1e-15:
                continue
            g = g_all[:, i]; sd = g.std()
            if sd < 1e-15:
                continue
            s = self.s_frac * sd
            near = np.where(np.abs(g) < self.cut * s)[0]
            if len(near) == 0:
                continue
            ind_res = self.indicator_res[i]
            z_near = z_all[near]; u0 = z_near @ v_i
            v_n, j_n, _ = self.vf.evaluate_selective(z_near, [ind_res])
            gn = v_n[:, 1 + i]; vdg = j_n[:, 0, :] @ v_i
            u1 = u0.copy(); ok = np.abs(vdg) > 1e-15
            u1[ok] -= np.clip(gn[ok] / vdg[ok], -3.0, 3.0)
            for _ in range(3):
                z_it = z_near + (u1 - u0)[:, None] * v_i[None, :]
                v_it, j_it, _ = self.vf.evaluate_selective(z_it, [ind_res])
                g_it = v_it[:, 1 + i]
                conv = np.abs(g_it) < 1e-8
                if np.all(conv):
                    break
                vd = j_it[:, 0, :] @ v_i; up = (~conv) & (np.abs(vd) > 1e-15)
                u1[up] -= g_it[up] / vd[up]
            z_star = z_near + (u1 - u0)[:, None] * v_i[None, :]
            jump = self.vf.evaluate_forward(z_star + eps * v_i[None, :])[:, 0] - self.vf.evaluate_forward(z_star - eps * v_i[None, :])[:, 0]
            # orientation: jump is "above minus below" along v_i; the delta of 1{g > 0} needs the jump across g
            # increasing, i.e. sign(v_i . grad g) -- v_i is the normalised gradient at Z = 0, so the sign is +1 where
            # the direction is consistent; use the local sign to be safe
            sgn = np.sign(vdg); sgn[sgn == 0] = 1.0
            ker = np.exp(-0.5 * (gn / s) ** 2) / (s * math.sqrt(2 * math.pi))
            corr += ((sgn * jump * ker)[:, None] * jt[near, 1 + i, :]).sum(0)
        total = pathwise + corr / M
        return {"total": total, "pathwise": pathwise, "correction": corr / M}


def gva_integral(funcs, payoff_res, z_args, theta_args, theta_values, **kw):
    """CorrectionDriverVec2 with every slice along a coordinate axis (the input of largest |weight| in the indicator's
    gradient at Z = 0) instead of the normal; coins off; no linear-root shortcut (it assumes the normal direction)."""
    d = CorrectionDriverVec2(funcs, payoff_res, z_args, theta_args, coins="off", linear_roots=False, **kw)
    d.precompute_directions(theta_values)
    newdir = []
    for v, norm in d.directions:
        if norm < 1e-15:
            newdir.append((v, norm)); continue
        j = int(np.argmax(np.abs(v))); e = np.zeros_like(v); e[j] = 1.0 if v[j] > 0 else -1.0
        newdir.append((e, norm * abs(v[j])))          # norm: |dg/dz_j| at Z = 0 (only its size > 0 matters)
    d.directions = newdir
    d.u_star = None
    return d
