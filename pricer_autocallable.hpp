// pricer_autocallable.hpp — MC autocallable with BranchManager (AADCNG_IF).
//
// A Phoenix autocallable note with N observation dates:
//   - At each date: if S(t_i) > autocall_barrier → early redemption
//   - At maturity:  if S(T)   > put_barrier      → notional
//                   else                          → S(T) / S(0)
//
// Uses AADCNG_IF for all stochastic branches: both sides recorded on tape,
// replay selects per-lane via mask. This is how real pricing libraries
// write structured products — imperative if/else, not expression-level iIf.
//
// The switch registry (register_switches=SWITCH_JUMP) discovers all
// comparisons automatically. The correction driver finds every discontinuity
// without any manual indicator code.
//
// Monte Carlo: caller provides z (normals) as a flat vector.
// One call = one path. Batch externally for AVX + multithreading.

#pragma once

#include <aadcNG/aadcNG.h>
#include <aadcNG/aadc_ifelseNG.h>
#include <vector>
#include <cmath>

namespace aadcex {

using Real = idoubleNG;

/// Single-path MC autocallable pricing.
///
/// @param spot         current spot S(0)
/// @param strike       put barrier (e.g. 70)
/// @param autocall     autocall barrier (e.g. 100)
/// @param coupon       annual coupon rate (e.g. 0.05)
/// @param vol          Black-Scholes volatility
/// @param rate         risk-free rate (continuous)
/// @param ttm          total time to maturity (years)
/// @param n_obs        number of observation dates (equally spaced)
/// @param normals      n_obs standard normals for this path
/// @return             discounted payoff
Real autocallableNpv(Real spot, Real strike, Real autocall,
                     double coupon, Real vol, Real rate, Real ttm,
                     int n_obs, const std::vector<Real>& normals)
{
    AADCNG_BRANCHLESS_FUNCTION

    const double dt = static_cast<double>(ttm) / n_obs;
    const Real sqrtDt = std::sqrt(Real(dt));
    const Real drift  = (rate - Real(0.5) * vol * vol) * Real(dt);

    Real logS = std::log(spot);
    Real disc(0.0);
    Real payoff(0.0);
    Real alive(1.0);   // 1 = not yet autocalled

    for (int i = 0; i < n_obs; ++i) {
        logS = logS + drift + vol * sqrtDt * normals[i];
        disc = disc + rate * Real(dt);
        Real S_i = std::exp(logS);

        // Autocall check — BranchManager records both sides
        AADCNG_IF(S_i > autocall)
            Real redemption = alive
                * (Real(1.0) + Real(coupon * (i + 1) * dt))
                * std::exp(-disc);
            payoff = payoff + redemption;
            alive  = Real(0.0);
        AADCNG_ELSE
            // Not autocalled — continue (no-op, but both arms recorded)
        AADCNG_IF_END
    }

    // Final payoff at maturity
    Real S_T = std::exp(logS);
    Real df  = std::exp(-disc);

    AADCNG_IF(S_T > strike)
        payoff = payoff + alive * Real(1.0) * df;       // above put barrier
    AADCNG_ELSE
        payoff = payoff + alive * (S_T / spot) * df;     // below: loss
    AADCNG_IF_END

    return payoff;
}

}  // namespace aadcex
