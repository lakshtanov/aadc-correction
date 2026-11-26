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

RESULTS_PLACEHOLDER

## 6. Reproduce

```
OPENBLAS_NUM_THREADS=1 PRODUCT=credit R=16 NPATH=8192 python benchmarks/bench_event_time.py
OPENBLAS_NUM_THREADS=1 PRODUCT=lgm    R=8  NPATH=1024 python benchmarks/bench_event_time.py
OPENBLAS_NUM_THREADS=1 PRODUCT=timer XI=0.3 NSTEP=52  R=16 NPATH=20000 python benchmarks/bench_event_time.py   (also XI=0.5, NSTEP=208)
OPENBLAS_NUM_THREADS=1 REF_DIR=<folder of bench_paths_for_accuracy.py OUT files> python benchmarks/bench_gva.py
```
