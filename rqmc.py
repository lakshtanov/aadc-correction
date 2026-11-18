"""
rqmc.py — randomized quasi-Monte Carlo (scrambled Sobol) inputs for the correction driver.

Why it works: the δ-function correction integrates the payoff jump analytically along the Newton line
(density φ(u*) at the root), so the per-path estimator no longer contains the indicator. What is left is as
smooth as the switch functions g_i and their θ-derivatives — QMC applies, provided the model discretization is
smooth too. A kink in the scheme (e.g. a variance floor max(v, v_min) in Heston Euler) makes ∂g/∂θ jump on
oblique surfaces and removes most of the gain; use a kink-free scheme (for Heston: the drift-implicit scheme
for sqrt(v), see examples/example_heston_rqmc.py).

Points:   scrambled Sobol (scipy.stats.qmc, LMS + digital shift), N = 2^m, R independent scramblings
          -> unbiased estimate and its standard error.
Ordering: an orthogonal map Z = Q x from Sobol normals x to tape inputs Z (orthogonal => still N(0, I),
          unbiased for any Q). It decides which directions get the best Sobol coordinates:
            "active_subspace"  automatic, from tape gradients on pilot paths (no knowledge of the z layout)
            "bridge"           Brownian bridge per Brownian factor; model-specific: the caller gives the tape indices
                               per factor (bridge_factors), only their range/uniqueness is checked — a wrong layout
                               keeps the estimator unbiased but silently loses the Sobol gain
            "none"             Sobol coordinates straight into the tape inputs

Usage:
    drv = CorrectionDriverRQMC(funcs, payoff_res, z_args, theta_args, num_threads=8)
    drv.precompute_directions(theta_values)
    res = drv.run(n_paths=2**14, n_scramble=16)
    res['total'], res['stderr']
"""
import io
import math
import time
import contextlib
from typing import Dict, List, Optional, Sequence

import numpy as np

from correction_driver_vec2 import CorrectionDriverVec2
from aadc_extensions import VectorFunctionSelectiveAD


def sobol_normals(n_paths: int, dim: int, seed: int) -> np.ndarray:
    """(n_paths, dim) standard normals from one scrambled Sobol sequence; n_paths must be a power of two."""
    from scipy.stats import qmc, norm
    m = int(round(math.log2(n_paths)))
    if 2 ** m != n_paths:
        raise ValueError("n_paths must be a power of two for Sobol (balance property)")
    u = qmc.Sobol(d=dim, scramble=True, seed=seed).random_base2(m)
    return norm.ppf(np.clip(u, 1e-15, 1.0 - 1e-15))


def _bridge_increments(n_steps: int) -> np.ndarray:
    """B (n_steps, n_steps): column p = unit-step Brownian increments produced by a unit normal at bridge position p
    (terminal value first, then midpoints level by level). B is orthogonal."""
    order = [(n_steps, 0, None)]
    queue = [(0, n_steps)]
    while queue:
        lo, hi = queue.pop(0)
        if hi - lo > 1:
            mid = (lo + hi) // 2
            order.append((mid, lo, hi))
            queue += [(lo, mid), (mid, hi)]
    B = np.zeros((n_steps, n_steps))
    for p in range(n_steps):
        W = np.zeros(n_steps + 1)
        gp = np.zeros(n_steps); gp[p] = 1.0
        for q, (k, lo, hi) in enumerate(order):
            if hi is None:
                W[k] = math.sqrt(k) * gp[q]
            else:
                W[k] = ((hi - k) * W[lo] + (k - lo) * W[hi]) / (hi - lo) + math.sqrt((k - lo) * (hi - k) / (hi - lo)) * gp[q]
        B[:, p] = np.diff(W)
    return B


def bridge_rotation(dim: int, factors: Sequence[Sequence[int]]) -> np.ndarray:
    """Orthogonal Q (dim, dim) for a Brownian-bridge construction.

    factors: for each Brownian factor, the tape indices of its per-step normals in time order
             (e.g. [[0, 2, 4, ...], [1, 3, 5, ...]] for interleaved spot / variance normals).
    Tape inputs not listed in any factor take the first Sobol coordinates (scalar normals, e.g. a default time).
    Sobol coordinate order after them: bridge level 0 (terminal values) of every factor, then level 1, ...
    """
    used = [i for f in factors for i in f]
    if len(set(used)) != len(used) or any(i < 0 or i >= dim for i in used):
        raise ValueError("bridge factors must be distinct tape indices in [0, dim)")
    extra = [i for i in range(dim) if i not in set(used)]
    Q = np.zeros((dim, dim))
    col = 0
    for i in extra:
        Q[i, col] = 1.0; col += 1
    Bs = [_bridge_increments(len(f)) for f in factors]
    for level in range(max(len(f) for f in factors)):
        for f, B in zip(factors, Bs):
            if level < len(f):
                Q[list(f), col] = B[:, level]; col += 1
    return Q


def active_subspace_rotation(funcs, payoff_res, z_args, theta_args, theta_values: Dict, indicator_res=None,
                             n_pilot: int = 2000, seed: int = 12345, num_threads: int = 4):
    """Orthogonal Q from tape gradients only (no knowledge of what each tape input means).

    C = mean over pilot paths of  (1/K) sum_i n_i n_i^T  +  grad P grad P^T / E|grad P|^2,
    n_i = grad_z g_i / |grad_z g_i| (normals to the switch surfaces), P = payoff.
    Columns of Q = eigenvectors of C, largest eigenvalue first. Returns (Q, eigenvalue shares).
    """
    gs = list(indicator_res) if indicator_res is not None else [sw.g for sw in funcs.cmp_switches()]
    vf = VectorFunctionSelectiveAD(funcs, z_args, [payoff_res] + gs, param_args=theta_args, num_threads=num_threads)
    vf.set_params([theta_values[a] for a in theta_args])
    theta_set = set(theta_args)
    vf.set_fixed_inputs({a: v for a, v in theta_values.items() if a not in theta_set})
    Zp = np.random.default_rng(seed).standard_normal((n_pilot, len(z_args)))
    _, J, _, _ = vf.evaluate(Zp)
    J = np.nan_to_num(J)
    Gi = J[:, 1:, :]
    nrm = np.linalg.norm(Gi, axis=2, keepdims=True)
    Ni = np.where(nrm > 1e-14, Gi / np.maximum(nrm, 1e-300), 0.0)
    C = np.einsum("pki,pkj->ij", Ni, Ni) / (n_pilot * max(len(gs), 1))
    gp = J[:, 0, :]
    sp = (gp * gp).sum(1).mean()
    if sp > 0:
        C += gp.T @ gp / n_pilot / sp
    lam, V = np.linalg.eigh(C)
    order = np.argsort(lam)[::-1]
    return V[:, order], lam[order] / max(lam.sum(), 1e-300)


class CorrectionDriverRQMC:
    """CorrectionDriverVec2 fed with randomized QMC points; returns Greeks and their standard error.

    The correction itself is CorrectionDriverVec2 (same Newton, same jumps, same Greeks); this class only
    generates the inputs and averages over independent Sobol scramblings.
    """

    def __init__(self, funcs, payoff_res, z_args, theta_args, indicator_res=None, indicator_types=None,
                 ordering: str = "active_subspace", bridge_factors: Optional[List[List[int]]] = None,
                 n_pilot: int = 2000, pilot_seed: int = 12345, **driver_kwargs):
        if ordering not in ("active_subspace", "bridge", "none"):
            raise ValueError("ordering: 'active_subspace', 'bridge' or 'none'")
        if ordering == "bridge" and bridge_factors is None:
            raise ValueError("ordering='bridge' needs bridge_factors (tape indices per Brownian factor)")
        self.funcs, self.payoff_res, self.z_args, self.theta_args = funcs, payoff_res, list(z_args), list(theta_args)
        self.indicator_res = indicator_res
        self.ordering, self.bridge_factors, self.n_pilot, self.pilot_seed = ordering, bridge_factors, n_pilot, pilot_seed
        self.num_threads = driver_kwargs.get("num_threads", 4)
        self.driver = CorrectionDriverVec2(funcs, payoff_res, z_args, theta_args, indicator_res=indicator_res,
                                           indicator_types=indicator_types, **driver_kwargs)
        self.Q = None
        self.eigen_share = None
        self.rotation_time = 0.0               # seconds spent on Q in the last precompute_directions (pilot paths)

    def precompute_directions(self, theta_values: Dict):
        self.driver.precompute_directions(theta_values)
        d = len(self.z_args)
        t0 = time.perf_counter()
        if self.ordering == "active_subspace":
            self.Q, self.eigen_share = active_subspace_rotation(
                self.funcs, self.payoff_res, self.z_args, self.theta_args, theta_values,
                indicator_res=self.indicator_res, n_pilot=self.n_pilot, seed=self.pilot_seed, num_threads=self.num_threads)
        elif self.ordering == "bridge":
            self.Q = bridge_rotation(d, self.bridge_factors)
        else:
            self.Q = np.eye(d)
        self.rotation_time = time.perf_counter() - t0

    def sample(self, n_paths: int, seed: int) -> np.ndarray:
        """(n_paths, d) tape inputs from scrambling `seed`."""
        if self.Q is None:
            raise RuntimeError("call precompute_directions() first")
        return sobol_normals(n_paths, len(self.z_args), seed) @ self.Q.T

    def run(self, n_paths: int, n_scramble: int = 16, seed: int = 0, verbose: bool = False) -> Dict:
        """Greeks averaged over n_scramble independent scramblings of n_paths Sobol points each.

        Returns dict with price, pathwise, correction, total (means), stderr (of total), totals (per scrambling).
        """
        if n_scramble < 2:
            raise ValueError("n_scramble >= 2 is needed for an error estimate")
        rows = {k: [] for k in ("price", "pathwise", "correction", "total")}
        for r in range(n_scramble):
            Z = self.sample(n_paths, seed + r)
            if verbose:
                res = self.driver.run(Z)
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    res = self.driver.run(Z)
            for k in rows:
                rows[k].append(np.atleast_1d(np.asarray(res[k], dtype=float)))
        out = {k: np.mean(v, axis=0) for k, v in rows.items()}
        totals = np.array(rows["total"])
        out["stderr"] = totals.std(axis=0, ddof=1) / math.sqrt(n_scramble)
        out["totals"] = totals
        out["n_paths"], out["n_scramble"] = n_paths, n_scramble
        return out
