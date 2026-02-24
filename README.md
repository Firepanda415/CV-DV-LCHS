# CV-DV LCHS Heat-Equation (PDE-First)

This repository is configured to solve and validate the original heat equation target, not a modified operator.

## Scope

- PDE target: `u(t) = exp(-alpha * t * T) u(0)` with fixed `T` from the 2-qubit finite-difference mapping.
- Active objective priority:
1. minimize `pde_error`
2. maximize `post_prob` (tie-break)
- `fidelity` is kept only as a diagnostic.

## Active files

- `heat_eq_postselect.py`: main CV-DV postselected simulation and correctness report.
- `heat_eq_sensitivity_refine.py`: PDE-focused sweeps and optional optimizer.
- `heat_eq_systematic_optimize.py`: theory-guided, systematic global+local PDE-priority optimizer.
- `PARAMETER_FIDELITY_NOTES.tex`: compact PDE-oriented theory/metric notes.

## Environment

```bash
conda activate cvdv
```

## Run

1. Single PDE correctness run:

```bash
python heat_eq_postselect.py
```

2. PDE sensitivity sweeps:

```bash
python heat_eq_sensitivity_refine.py --profile quick --min-post-prob 1e-3
python heat_eq_sensitivity_refine.py --profile default --min-post-prob 1e-3
python heat_eq_sensitivity_refine.py --profile full --min-post-prob 1e-3
```

3. Optional 3-parameter optimization (`r_target`, `r_prime`, `kernel_beta`):

```bash
python heat_eq_sensitivity_refine.py --optimize --maxiter 200 --min-post-prob 1e-3 --low-post-penalty 100
```

4. Systematic theory-guided optimization (new):

```bash
python heat_eq_systematic_optimize.py --global-samples 42 --n-starts 8 --local-maxiter 80 --min-post-prob 1e-3
```

Outputs are written to `sensitivity_refine_results/`.
Useful PDE-priority outputs:
- `refine_top_candidates.csv` (ranked by feasibility, then `pde_error`, then `post_prob`)
- `refine_pareto_front.csv` (non-dominated `pde_error`/`post_prob` trade-off points)
- `refine_tradeoff_pde_vs_post.png` (global trade-off view)
Systematic optimizer outputs:
- `systematic_top.csv`, `systematic_pareto_front.csv`
- `systematic_tradeoff_pde_vs_post.png`, `systematic_landscape_rtarget_rprime.png`

## Metrics to report

1. `pde_error` (primary)
2. `post_prob` (secondary)
3. `purity`
4. `fidelity` (diagnostic only)

## Physical Meaning Of Search Region

For the systematic optimizer (`heat_eq_systematic_optimize.py`), the search region is theory-guided:

- `r_target` (`r`): post-selection squeezing strength.
- `r_prime` (`r'`): preparation-basis squeezing strength.
- Physical consistency requires:

\[
r' < r
\]

- The key shaping parameter is:

\[
\gamma(r,r') = e^{-2r'} - e^{-2r} > 0
\]

Small `\gamma` (i.e., `r' \approx r`) is numerically unstable for coefficient shaping.

Why the extra constraints are used:

- `gamma_min`: enforce

\[
\gamma(r,r') \ge \gamma_{\min}
\]

- Effective-mode constraint:

\[
n_{\mathrm{eff}} = \frac{1}{\sum_n |C_n|^4}, \qquad n_{\mathrm{eff}} \le n_{\mathrm{eff,max}}
\]

This keeps Fock support compatible with finite cutoff (`N=32`) and reduces truncation artifacts.

- Feasibility constraint:

\[
p_{\mathrm{post}} \ge p_{\min}
\]

so solutions are not only accurate but also practically post-selectable.

Objective interpretation:

- Primary metric:

\[
\varepsilon_{\mathrm{PDE}} = \frac{\|u_{\mathrm{PDE}}-\sqrt{p_{\mathrm{post}}}\,\psi_{\mathrm{principal}}\|}{\|u_{\mathrm{PDE}}\|}
\]

- Optimized objective:

\[
J(\theta)=\varepsilon_{\mathrm{PDE}}
 + \lambda_p \frac{[p_{\min}-p_{\mathrm{post}}]_+}{p_{\min}}
 + \lambda_\gamma \frac{[\gamma_{\min}-\gamma]_+}{\gamma_{\min}}
 + \lambda_n \frac{[n_{\mathrm{eff}}-n_{\mathrm{eff,max}}]_+}{n_{\mathrm{eff,max}}}
 - 10^{-3} p_{\mathrm{post}}
\]

where \([x]_+ = \max(0,x)\) and \(\theta=(r,r',\beta_k)\).

- Secondary preference: larger `post_prob` when `pde_error` is similar.
- Penalty weights are algorithmic tuning weights, not physical constants.

## Important implementation note

With `use_fock_expansion=True`, CV loading is evaluated via incoherent `|C_n|^2` averaging. This is the feasible backend path in this project and can limit absolute PDE-vector matching.
