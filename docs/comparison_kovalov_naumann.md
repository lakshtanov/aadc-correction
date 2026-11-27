# The Kovalov–Naumann corrections and the correction of this repository

Two recent pieces of work from the STCE group at RWTH Aachen address the same problem as this repository — the part of
a pathwise (AD) Greek that is lost at a discontinuity — and both cite our preprint
(E. Lakshtanov, *Unbiased Monte Carlo Greeks for Discontinuous Payoffs*, arXiv:2609.06137):

1. A. Kovalov, U. Naumann, *Surrogate Greeks for Regime-Switching Processes via Corrected AD*, poster, RWTH Aachen,
   STCE; poster and code: <https://gitlab.stce.rwth-aachen.de/a.kovalov/hsde-sensitivity>.
2. U. Naumann, *Greeks Value Adjustment (GVA) — or: Algorithmic Differentiation Software beyond Algorithmic
   Differentiability*, talk, [QuantMinds International 2026](https://informaconnect.com/quantminds-international/),
   London, 16–19 November 2026.

This note gives the mathematics of both methods, how they are implemented here (`event_time.py`, `gva_drivers.py`),
where an implementation choice is ours, and what we measured on the benchmarks of this repository. Every number is
produced by the scripts listed in Section 6.

---

## 1. The event-time correction (Kovalov–Naumann)

### 1.1 Their setting

An SDE switches its coefficients when a level is hit:
$$
dS_t = F(S_t,p)\cdot dX_t,\qquad F=\begin{cases}F_1=(f_1,g_1) & t\le\tau\\ F_2=(f_2,g_2) & t>\tau\end{cases},\qquad
\tau=\inf\{t:\ c(S_t,p)=B\},\quad dX_t=(dt,dW_t).
$$
Writing $S_T$ in integral form and differentiating with the Leibniz rule,
$$
d_pS_T = d_pS_0+\int_0^\tau d_pF_1(S_t,p)\,dX_t+\int_\tau^T d_pF_2(S_t,p)\,dX_t+\underbrace{d_p\tau\,\big[F_1-F_2\big]_\tau}_{\text{missed by AD}} .
$$
The motion of the switching time follows from the implicit function theorem applied to the barrier condition,
$d_p\tau=-\partial_pS_\tau/\partial_tS_\tau$. On an Itô path $\partial_tS_\tau$ does not exist; the poster replaces it
by the drift before the event, $d_p\tau\approx-\partial_pS_\tau/f_1(\tau)$ — the **surrogate** (first order). The event is
located inside the time step by interpolation, and the correction is added to the tangent at the event. The poster names
the next step: account for the missing (Itô) term in expectation via the Feynman–Kac formula.

### 1.2 Applied to an event of a monotone accumulator

The products of this repository do not switch coefficients; they **pay** when an accumulator crosses a threshold:
a default when $\Lambda_t=\int_0^t\lambda_s\,ds$ exceeds an exponential threshold $E$, a variance budget
$\int_0^t v_s\,ds\ge V^*$. On the simulation grid ($\Delta t=t_k-t_{k-1}$, left point)
$$
A_k=A_{k-1}+a_{k-1}\,\Delta t,\qquad a\ge0,\qquad \text{event in step }k\iff A_{k-1}<E\le A_k ,
$$
and $\mathrm{pay}_k$ is what is paid if the event is in step $k$ (it may depend on the normals $z_k$ of that step).

**Event time inside the step.** With $A$ linear inside the step, $\tau=t_{k-1}+(E-A_{k-1})/a_{k-1}$ and, by the
implicit function theorem on $A(\tau)=E$,
$$
\partial_\theta\tau=-\frac{\partial_\theta A_{k-1}+(\tau-t_{k-1})\,\partial_\theta a_{k-1}}{a_{k-1}} .
$$
This is the poster's $-\partial_pS_\tau/f_1(\tau)$ with $S\to A$ and the drift $f_1\to a_{k-1}$; for an accumulator of
bounded variation it is exact (no surrogate is needed here). Both derivatives come from one reverse pass over two
outputs recorded on the tape, $\sum_k\mathrm{first}_k A_{k-1}$ and $\sum_k\mathrm{first}_k a_{k-1}$.

**What the moving event time changes.** If the event moves from $\tau$ to $\tau+d\tau$, the value paid changes by
$\frac{d}{d\tau}\mathbb{E}\big[e^{-r\tau}u(X_\tau)\big]\,d\tau=e^{-r\tau}(\mathcal L-r)u(X_\tau)\,d\tau$, with $u$ the payment as
a function of the state and $\mathcal L$ the generator of the state process. The correction term is
$$
\mathcal A=\mathbb{E}\Big[\mathbf 1\{\text{event}\}\;\partial_\theta\tau\cdot G\Big],\qquad G\approx(\mathcal L-r)u\ \text{at the event}.
$$
On the grid, $G\,\Delta t=\mathbb{E}[\mathrm{pay}_k\,|\,\mathcal F_{k-1}]-\mathrm{pay}_{k-1}$.

### 1.3 Orders of the generator

Let $p(z_k)=\mathrm{pay}_k$. The step normals enter through $\sigma\sqrt{\Delta t}\,z_k$, so $\partial^2_{jj}p=O(\Delta t)$ and
$$
\mathbb{E}[p(z_k)]=p(0)+\tfrac12\sum_j\mathbb{E}\big[\partial^2_{jj}p\big]+O(\Delta t^2).
$$

| generator (`event_time.py`) | $G\,\Delta t$ | property |
|---|---|---|
| `"drift"` — **1st order, the surrogate** | $p(0)-\mathrm{pay}_{k-1}$ | drift and discount only. The omitted $\tfrac12\sum_j\mathbb{E}[\partial^2_{jj}p]$ is $O(\Delta t)$, the same order as $G\,\Delta t$: a bias that does **not** vanish as $\Delta t\to0$; its limit is $-\mathbb{E}\big[\partial_\theta\tau\,\tfrac12\mathrm{tr}(\sigma\sigma^\top\nabla^2u)\big]$, zero only if the payment is affine in the state that moves within the step. |
| `"stein"` — 2nd order | $p(0)-\mathrm{pay}_{k-1}+\tfrac12\sum_j\partial_jp\,z_{k,j}$ | Stein's identity $\mathbb{E}[\partial_jp\,z_j]=\mathbb{E}[\partial^2_{jj}p]$: unbiased to $O(\Delta t^2)$ per step. The added term is $O(\sqrt{\Delta t})$ per path against a mean $O(\Delta t)$: relative noise $\sim\Delta t^{-1/2}$; the factor $z_k$ removes the gain from quasi-Monte Carlo. |
| `"quadrature"` — 2nd order, exact | $\sum_q w_q\,p(z_k=x_q)-\mathrm{pay}_{k-1}$ | Gauss–Hermite over the step normals the payment depends on ($n^m$ passes). |
| `"quadrature1d"` — 2nd order, exact in one direction | $\sum_q w_q\,p\big(x_qc+(I-cc^\top)z_k\big)-\mathrm{pay}_{k-1}$ | $c$: mean of the unit gradients of the payment on pilot paths with the event. Since $c$ does not depend on the path, $c\cdot z_k$ and $(I-cc^\top)z_k$ are independent: unbiased for any payment, exact when it depends on $z_k$ through $c\cdot z_k$; $n$ passes. |

The selection of the event step and $\partial_\theta\tau$ do not depend on $z_k$ (left point), so the estimators are fair.
Second order by quadrature is our implementation of the Itô term the poster points to.

### 1.4 Dates, targets, tails

* **Date steps.** If the event falls in a step that ends on a payment date, the event-time term still applies; the date
  only adds the delta-correction term of that step's indicator (the jump of the date payment).
* **Target.** The event-time term is the derivative with a *continuous* event time. The correction of this repository
  (and its exact integration of the exponential threshold, the "coin") is the derivative of the *grid* model, in which
  the event is checked at the grid points. For an increasing accumulator both select the same step; they differ in where
  inside the step the payment is valued. The difference is reported as measured.
* **Tails.** $\partial_\theta\tau$ carries $1/a_{k-1}$. An event falls in a step with rate $a$ with probability
  $\approx f_A(E)\,a\,\Delta t$; if the density of the rate near zero behaves as $a^{\nu-1}$ — for a CIR intensity or
  variance $\nu=2\kappa\theta/\xi^2$ — then $P(|\text{term}|>y)\sim y^{-(1+\nu)}$: tail index $1+\nu$, **infinite
  variance for $\nu<1$** (Feller condition violated). The order of the generator does not change this. `event_time.py`
  reports the Hill index of $1/a$ at the event after every run and warns below 2.3.

### 1.5 Where it applies

| benchmark | event inside a step? | event-time term |
|---|---|---|
| 1, 3: down-and-out call (GBM; GBM + Hull–White), 50 dates | no — a barrier of a diffusion observed on dates | not applicable |
| 2: Phoenix autocallable (Heston), 8 dates | no — coupon / autocall on dates | not applicable |
| 4: credit-linked autocallable | default: $\int\lambda$ crosses $E$ | applies (+ delta on dates) |
| 5: range accrual on a 3M LGM rate, daily | default, as above | applies (+ delta on the corridor) |
| timer option (Heston) | realised variance reaches $V^*$ (fixed threshold) | applies |

---

## 2. GVA (Naumann, QuantMinds 2026)

For a branch `if (c(x,p) >= 0) x = g(x,p); else x = h(x,p);` AD returns
$\frac{dg}{dp}\mathbf 1(c)+\frac{dh}{dp}(1-\mathbf 1(c))$ and misses $(g-h)\,\delta(c)\,\partial_pc$.

* **GVA-local** (tangent rule). Add `jump * exp(-c^2/(2 s^2)) / (s sqrt(2 pi)) * dc/dp` to the tangent: a Gaussian
  mollification of $\delta(c)$ of width $s$. The primal is unchanged, the Greek is regularised — biased, $O(s)$.
* **GVA-integral** (co-area formula). For $\frac{d}{dp}\int_\Omega F\,dx$,
  $$
  \frac{d}{dp}\int_\Omega F\,dx=\int_{\Omega_1}\frac{dg}{dp}\,dx+\int_{\Omega_0}\frac{dh}{dp}\,dx
  +\int_\Gamma (g-h)\,\frac{\partial_pc}{\lVert\nabla_xc\rVert}\,d\Gamma ,
  $$
  $\Gamma=\{c=0\}$; in code a root of $c$ along one variable and `jump * dc/dp / |dc/dx|` (fixed bounds, one simple
  interior root).

## 3. Relation to the correction of this repository

The GVA-integral boundary term is the classical one (Leibniz rule, Reynolds transport theorem, co-area formula) and is
the same quantity as the correction term of Theorem 1 of our preprint. Parametrise $\Gamma$ by the projection along a
fixed unit direction $v$ with $v\cdot\nabla g>0$, $Z=u\,v+w$: the surface element is
$d\Gamma=\frac{\lVert\nabla g\rVert}{\partial_ug}\,dw$, so with the Gaussian density $\varphi_d(Z)=\varphi(u)\varphi_{d-1}(w)$
$$
\int_\Gamma \Delta F\,\frac{\partial_\theta g}{\lVert\nabla g\rVert}\,\varphi_d\,d\Gamma
=\mathbb{E}_w\Big[\Delta F\,\frac{\varphi(u^*)}{\partial_ug}\,\partial_\theta g\Big]_{Z^*=u^*v+w} .
$$
What the repository adds is the Monte Carlo estimator of this term — a one-dimensional slice along a fixed direction,
every path contributing the root and the jump at its own $w$ (no surface sampling) — its automatic application to
every comparison found on the AADC tape for arbitrary compositions of indicators (Theorem 2), the exact integration of
independent thresholds ("coins"), and the compatibility with randomized quasi-Monte Carlo.

## 4. Implementation here, and what is ours

| method | implementation | faithful to the source? |
|---|---|---|
| event-time, 1st order | `EventTimeDriver(monitoring="continuous", generator="drift")`; $\partial_\theta\tau$ above; the delta correction on all other discontinuities | the poster's surrogate, transferred from a coefficient switch to a payment at the event of an accumulator |
| event-time, 2nd order | `generator="stein"` / `"quadrature1d"` | the poster's "future direction 1" (Itô / Feynman–Kac term), our implementation |
| GVA-local | `gva_drivers.GVALocal`: per indicator, paths with $\lvert g_i\rvert<6s$; jump by two replays at the boundary point reached along the indicator's direction; weight $\varphi_s(g_i)\,\partial_\theta g_i$; $s$ = fraction × sd of $g_i$, the best fraction of a grid per Greek and target (oracle) | the tangent rule of the talk; the jump taken at the boundary instead of the path point (the same to $O(s)$) |
| GVA-integral | `gva_drivers.gva_integral`: the correction driver with every slice along a **coordinate axis** (the input of largest weight in the indicator's gradient) instead of the normal; no coins | the co-area rule with "a root along one variable"; for many variables the choice of the variable is ours |

## 5. Results

All runs: 10 threads, compiled tapes, generation of the random inputs excluded; MC and scrambled Sobol with the
rotation of option B (512 pilot paths); the same paths for all methods of a product.

### 5.1 GVA on benchmarks 1–5

$N$ = 8192 paths (B5: 1024), 16 repetitions (B5: 8). Bias and RMSE (including the bias) in % of the Greek, against the
corrected estimate of this repository at the largest $N$ of `bench_paths_for_accuracy.py`; for each GVA variant the
better of MC and Sobol, for GVA-local also the best width $s$ of {0.01, 0.03, 0.1, 0.3} × sd$(g_i)$ (an oracle in its
favour). Last column: this repository (correction + Sobol) at the same $N$ — unbiased, so its RMSE is its standard
deviation.

| benchmark, Greek | GVA-local: bias / RMSE | GVA-integral (axis): bias / RMSE | correction + Sobol: RMSE |
|---|---|---|---|
| B1 $\partial V/\partial S_0$ | +0.07% / 0.57% | +0.04% / 0.29% | 0.16% |
| B1 $\partial V/\partial\sigma$ | +0.72% / 2.24% | -0.13% / 1.58% | 0.66% |
| B2 $\partial V/\partial S_1$ | -1.87% / 3.94% | **+100%** (no root along the axis) | 1.44% |
| B2 $\partial V/\partial S_3$ | +0.68% / 4.78% | +0.40% / 2.04% | 1.57% |
| B2 $\partial V/\partial\xi_1$ | +2.07% / 14.49% | **+100%** (no root along the axis) | 5.79% |
| B2 $\partial V/\partial\kappa_1$ | -0.28% / 6.52% | **-100%** (no root along the axis) | 2.44% |
| B3 $\partial V/\partial S_0$ | +0.12% / 0.56% | -0.15% / 0.68% | 0.20% |
| B3 $\partial V/\partial\sigma$ | +1.88% / 2.12% | +0.72% / 2.84% | 0.70% |
| B3 $\partial V/\partial r_0$ | -0.01% / 0.28% | -0.07% / 0.44% | 0.22% |
| B3 $\partial V/\partial\sigma_r$ | -2.26% / 4.42% | -1.26% / 4.59% | 4.36% |
| B4 $\partial V/\partial\lambda_0$ | +9.78% / 11.15% | +0.02% / 0.28% | 0.42% |
| B4 $\partial V/\partial\sigma_\lambda$ | +10.33% / 11.65% | -0.15% / 2.15% | 2.40% |
| B4 $\partial V/\partial S_1$ | +2.40% / 5.73% | **+107%** (no root along the axis) | 2.90% |
| B5 $\partial V/\partial R_0$ | -1.07% / 1.24% | -1.67% / 2.35% | 0.15% |
| B5 $\partial V/\partial\sigma_r$ | +2.22% / 2.43% | +0.47% / 2.58% | 0.13% |
| B5 $\partial V/\partial\lambda_0$ | +9.34% / 9.76% | +0.00% / 0.03% | 0.01% |

* **GVA-local** has the bias of a mollification: a few % on the credit Greeks at the best width (B4 $\lambda_0$, $\sigma_\lambda$:
  +10%; B5 $\lambda_0$: +9%), and its noise is not reduced much by Sobol.
* **GVA-integral** with an axis slice agrees with the correction wherever the axis crosses the boundary (B1, B3, the
  default Greeks of B4/B5). On worst-of products the boundary is often not crossed along the chosen axis and the Greek is
  lost entirely (B2 $S_1$, $\xi_1$, $\kappa_1$; B4 $S_1$): **the direction of the slice matters**; the normal of the
  boundary (as in the correction) does not have this failure.

### 5.2 The event-time correction

**Benchmark 4, credit-linked autocallable** (16 × 8192 paths, Sobol option B; mean ± s.e.; difference to the correction on the same paths; time per estimate)

| Greek | correction | event-time 1st (surrogate) | event-time 2nd (Stein) | event-time 2nd (quadrature) |
|---|---|---|---|---|
| $\partial V/\partial\lambda_0$ | -39.1 ± 0.029; 2.5 s | -39.64 ± 0.031; diff -0.545 ± 0.01; 3.4 s | -38.92 ± 0.28; diff +0.183 ± 0.28; 3.4 s | -39.09 ± 0.058; diff +0.00753 ± 0.049; 3.4 s |
| $\partial V/\partial\sigma_\lambda$ | -0.7697 ± 0.0043; 2.5 s | -0.7645 ± 0.0046; diff +0.00515 ± 0.0018; 3.4 s | -0.7674 ± 0.021; diff +0.00229 ± 0.02; 3.4 s | -0.7701 ± 0.0053; diff -0.000371 ± 0.0039; 3.4 s |

**Benchmark 5, LGM range accrual** (8 × 1024 paths, Sobol option B; mean ± s.e.; difference to the correction on the same paths; time per estimate)

| Greek | correction | event-time 1st (surrogate) | event-time 2nd (Stein) | event-time 2nd (quadrature) |
|---|---|---|---|---|
| $\partial V/\partial\lambda_0$ | -37.45 ± 0.0041; 34 s | -37.48 ± 0.025; diff -0.0291 ± 0.024; 31 s | -37.48 ± 0.025; diff -0.0291 ± 0.024; 31 s | -37.48 ± 0.025; diff -0.0291 ± 0.024; 31 s |
| $\partial V/\partial\sigma_\lambda$ | +0.03043 ± 0.011; 34 s | +0.02828 ± 0.0093; diff -0.00215 ± 0.002; 31 s | +0.02828 ± 0.0093; diff -0.00215 ± 0.002; 31 s | +0.02828 ± 0.0093; diff -0.00215 ± 0.002; 31 s |

**Timer option, $\xi=0.3$, 52 steps (correction = delta on every step = the grid model)** (16 × 16384 paths, Sobol option B; mean ± s.e.; difference to the correction on the same paths; time per estimate)

| Greek | correction | event-time 1st (surrogate) | event-time 2nd (Stein) | event-time 2nd (quadrature) |
|---|---|---|---|---|
| $\partial V/\partial v_0$ | -16.1 ± 0.059; 3.9 s | +33.99 ± 0.061; diff +50.1 ± 0.049; 0.11 s | -16.05 ± 0.83; diff +0.0453 ± 0.84; 0.11 s | -16.51 ± 0.18; diff -0.411 ± 0.18; 0.25 s |
| $\partial V/\partial\theta$ | -28.3 ± 0.14; 3.9 s | +46.74 ± 0.092; diff +75 ± 0.098; 0.11 s | -28.9 ± 1.9; diff -0.602 ± 1.9; 0.11 s | -29.08 ± 0.27; diff -0.777 ± 0.27; 0.25 s |

**Timer option, $\xi=0.3$, 208 steps (correction = delta on every step = the grid model)** (16 × 16384 paths, Sobol option B; mean ± s.e.; difference to the correction on the same paths; time per estimate)

| Greek | correction | event-time 1st (surrogate) | event-time 2nd (Stein) | event-time 2nd (quadrature) |
|---|---|---|---|---|
| $\partial V/\partial v_0$ | -15.33 ± 0.052; 53 s | +32.71 ± 0.057; diff +48 ± 0.049; 0.56 s | -16.9 ± 1.6; diff -1.57 ± 1.6; 0.52 s | -15.52 ± 0.22; diff -0.194 ± 0.2; 1.2 s |
| $\partial V/\partial\theta$ | -30.68 ± 0.13; 53 s | +47.26 ± 0.12; diff +77.9 ± 0.082; 0.56 s | -32.04 ± 2.3; diff -1.37 ± 2.3; 0.52 s | -30.98 ± 0.35; diff -0.299 ± 0.3; 1.2 s |

**Timer option, $\xi=0.5$, 52 steps (correction = delta on every step = the grid model)** (16 × 16384 paths, Sobol option B; mean ± s.e.; difference to the correction on the same paths; time per estimate)

| Greek | correction | event-time 1st (surrogate) | event-time 2nd (Stein) | event-time 2nd (quadrature) |
|---|---|---|---|---|
| $\partial V/\partial v_0$ | -5.726 ± 0.068; 4.1 s | +33.32 ± 0.28; diff +39 ± 0.3; 0.12 s | -6.297 ± 0.72; diff -0.571 ± 0.71; 0.11 s | -6.411 ± 0.39; diff -0.686 ± 0.42; 0.2 s |
| $\partial V/\partial\theta$ | -3.989 ± 0.24; 4.1 s | +64.35 ± 0.43; diff +68.3 ± 0.48; 0.12 s | -4.413 ± 1.5; diff -0.424 ± 1.5; 0.11 s | -5.468 ± 0.62; diff -1.48 ± 0.69; 0.2 s |

**Timer option, $\xi=0.5$, 208 steps (correction = delta on every step = the grid model)** (16 × 16384 paths, Sobol option B; mean ± s.e.; difference to the correction on the same paths; time per estimate)

| Greek | correction | event-time 1st (surrogate) | event-time 2nd (Stein) | event-time 2nd (quadrature) |
|---|---|---|---|---|
| $\partial V/\partial v_0$ | -7.065 ± 0.063; 55 s | +30.85 ± 0.31; diff +37.9 ± 0.32; 0.49 s | -6.349 ± 1.7; diff +0.716 ± 1.7; 0.45 s | -7.475 ± 0.39; diff -0.41 ± 0.39; 1 s |
| $\partial V/\partial\theta$ | -9.951 ± 0.19; 55 s | +60.93 ± 0.59; diff +70.9 ± 0.6; 0.49 s | -10.72 ± 3.4; diff -0.768 ± 3.4; 0.45 s | -10.74 ± 0.61; diff -0.791 ± 0.64; 1 s |

**Reading.**
* *First order (the surrogate).* Biased whenever the payment at the event has curvature in the step's noise:
  benchmark 4, $\partial V/\partial\lambda_0$: $-0.545\pm0.010$ ($1.4\%$, $>50$ s.e.) against the correction; the timer
  (a convex payment $(S_\tau-K)^+$): **the wrong sign** at every grid. Benchmark 5: the payment at default is known at the
  start of the step, the Itô term vanishes and all orders coincide.
* *Second order.* Stein's identity removes the bias but is $4$--$10\times$ noisier; the 1-D quadrature is unbiased with
  a noise close to the correction: benchmark 4, $\partial V/\partial\lambda_0$: $+0.008\pm0.049$ against the correction.
* *Cost.* With an exponential (Cox) threshold the correction integrates the threshold exactly (the coin): benchmark 4
  2.5 s against 3.4 s per estimate, benchmark 5 with Sobol $\approx 6\times$ less noise on $\partial V/\partial\lambda_0$ at the
  same time. With a **fixed** threshold (the timer) there is no coin: the correction solves a root on every step of the
  accumulator (52 / 208 steps: 4 s / 52–55 s per estimate) while the event-time term is computed once at the event
  (0.2 s / 1.1–1.3 s) — on this product the second-order event-time correction is $16$–$20\times$ (52 steps) and
  $44$–$54\times$ (208 steps) cheaper per estimate; its noise is $2$–$6\times$ larger, so in work (time × variance) it
  needs $1.4$–$5.5\times$ less on the timer Greeks except one case ($\xi=0.5$, 52 steps, $\partial V/\partial v_0$:
  $0.6\times$, i.e. more), and it approaches the grid model as the grid is refined.
* *Targets.* The event-time correction targets the continuous event time, the correction the grid model; on the timer
  the gap of the quadrature variant to the grid shrinks from 52 to 208 steps (e.g. $\xi=0.3$, $\partial V/\partial v_0$:
  $-0.41\pm0.29$ → $+0.10\pm0.39$ on MC).



## 6. Reproduce

```
OPENBLAS_NUM_THREADS=1 PRODUCT=credit R=16 NPATH=8192 python benchmarks/bench_event_time.py
OPENBLAS_NUM_THREADS=1 PRODUCT=lgm    R=8  NPATH=1024 python benchmarks/bench_event_time.py
OPENBLAS_NUM_THREADS=1 PRODUCT=timer XI=0.3 NSTEP=52  R=16 NPATH=16384 python benchmarks/bench_event_time.py   (also XI=0.5, NSTEP=208)
OPENBLAS_NUM_THREADS=1 REF_DIR=<folder of bench_paths_for_accuracy.py OUT files> python benchmarks/bench_gva.py
```
