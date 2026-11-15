"""
test_rqmc.py — CorrectionDriverRQMC (randomized QMC inputs for the correction driver).

1. Bridge and active-subspace rotations are orthogonal (=> inputs stay N(0, I), estimator stays unbiased).
2. Brownian-bridge increments are i.i.d. N(0, 1) (empirical check).
3. GBM down-and-out call, 10 monitoring dates (10 switches): RQMC (both orderings) agrees with a large MC run
   of the same driver within 4 combined s.e., and its s.e. is clearly below MC with the same number of paths.
   (A digital paying at T alone is not a test: on GBM its correction is exact on every path — zero variance.)
"""
import sys, os, math
import numpy as np
import aadc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from correction_driver_vec2 import CorrectionDriverVec2
from rqmc import CorrectionDriverRQMC, bridge_rotation, active_subspace_rotation

S0, K, B, SIGMA, R, T, N_STEPS = 100.0, 90.0, 80.0, 0.25, 0.05, 1.0, 10


def barrier_tape():
    """Down-and-out call, discretely monitored on 10 dates (10 switches S_j > B)."""
    dt = T / N_STEPS
    fn = aadc.Functions()
    fn.start_recording(register_switches=aadc.SWITCH_JUMP)
    S = aadc.idouble(S0); aS = S.mark_as_input()
    zs, za = [], []
    for _ in range(N_STEPS):
        z = aadc.idouble(0.0); za.append(z.mark_as_input()); zs.append(z)
    logS = aadc.math.log(S)
    alive = aadc.idouble(1.0)
    for j in range(N_STEPS):
        logS = logS + aadc.idouble((R - 0.5 * SIGMA ** 2) * dt) + aadc.idouble(SIGMA * math.sqrt(dt)) * zs[j]
        alive = alive * aadc.iif(aadc.math.exp(logS) > aadc.idouble(B), aadc.idouble(1.0), aadc.idouble(0.0))
    P = alive * aadc.math.max(aadc.math.exp(logS) - aadc.idouble(K), aadc.idouble(0.0)) * aadc.idouble(math.exp(-R * T))
    rP = P.mark_as_output()
    fn.stop_recording()
    return fn, rP, za, aS


def mc_driver(fn, rP, za, aS, n, reps, seed):
    import io, contextlib
    drv = CorrectionDriverVec2(fn, rP, za, [aS], num_threads=4)
    drv.precompute_directions({aS: S0})
    est = []
    for r in range(reps):
        with contextlib.redirect_stdout(io.StringIO()):
            est.append(drv.run(np.random.default_rng(seed + r).standard_normal((n, N_STEPS)))["total"][0])
    return np.mean(est), np.std(est, ddof=1) / math.sqrt(reps)


def test_rotations_orthogonal():
    print("Test 1: rotations are orthogonal")
    Q = bridge_rotation(7, [[0, 2, 4, 6], [1, 3, 5]])
    assert np.allclose(Q.T @ Q, np.eye(7), atol=1e-12), "bridge rotation not orthogonal"
    Q = bridge_rotation(5, [[1, 2, 3, 4]])          # input 0 is a scalar normal -> first coordinate
    assert np.allclose(Q.T @ Q, np.eye(5), atol=1e-12) and Q[0, 0] == 1.0
    fn, rP, za, aS = barrier_tape()
    Q, share = active_subspace_rotation(fn, rP, za, [aS], {aS: S0}, n_pilot=500, num_threads=4)
    assert np.allclose(Q.T @ Q, np.eye(len(za)), atol=1e-10), "active-subspace rotation not orthogonal"
    print(f"  OK (active subspace: first eigenvalue share {share[0]:.2f})")


def test_bridge_increments_iid():
    print("Test 2: bridge increments are i.i.d. N(0,1)")
    Q = bridge_rotation(10, [list(range(10))])
    D = np.random.default_rng(0).standard_normal((200_000, 10)) @ Q.T
    C = np.cov(D.T)
    assert abs(D.mean(0)).max() < 0.01 and np.allclose(C, np.eye(10), atol=0.015)
    print("  OK")


def test_barrier_unbiased_and_better_than_mc():
    print("Test 3: GBM down-and-out call — RQMC unbiased, smaller error than MC")
    fn, rP, za, aS = barrier_tape()
    ref, ref_se = mc_driver(fn, rP, za, aS, 2 ** 16, 8, seed=5000)       # same driver, many MC paths
    n, reps = 2 ** 12, 16
    mc_m, mc_se = mc_driver(fn, rP, za, aS, n, reps, seed=1000)
    print(f"  reference (MC, 8 x 2^16) {ref:.5f} ± {ref_se:.1e};  MC 2^12 x 16: {mc_m:.5f} ± {mc_se:.1e}")
    for ordering in ("active_subspace", "bridge"):
        drv = CorrectionDriverRQMC(fn, rP, za, [aS], ordering=ordering, bridge_factors=[list(range(N_STEPS))], num_threads=4)
        drv.precompute_directions({aS: S0})
        out = drv.run(n, n_scramble=reps)
        m, se = out["total"][0], out["stderr"][0]
        print(f"  RQMC {ordering:16s} {m:.5f} ± {se:.1e}   (s.e. {mc_se / se:.1f}x smaller than MC)")
        assert abs(m - ref) < 4 * math.hypot(se, ref_se), f"{ordering}: disagrees with the MC reference"
        assert se < mc_se / 2, f"{ordering}: no RQMC gain"
    print("  OK")


if __name__ == "__main__":
    test_rotations_orthogonal()
    test_bridge_increments_iid()
    test_barrier_unbiased_and_better_than_mc()
    print("\nAll RQMC tests passed.")
