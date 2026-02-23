# CV-DV LCHS Heat-Equation Experiments

This repo studies CV-DV postselected simulation for a 2-qubit heat-equation model using HybridLane.

## Reference files

`res_base` contains our latest working paper. `res_refs` contains papers that are very important to our research, should be refered in forming research framework and directions.

## Files

| File                              | Purpose                                                                      |
| --------------------------------- | ---------------------------------------------------------------------------- |
| `heat_eq_postselect.py`         | Main simulation: CV-DV circuit, Fock expansion, post-selection, metrics      |
| `heat_eq_sensitivity_refine.py` | Refinement sweeps, PDE-error/post-prob tracking, joint Nelder-Mead optimizer |
| `PARAMETER_FIDELITY_NOTES.tex`  | Theory notes on parameter effects and metric definitions                     |

## Current reference setup and result

Reference settings (in `heat_eq_postselect.py`):

- `total_time=1`, `n_steps=100`, `max_fock_level=32`
- `r_target=1.2`, `r_prime=0.3`, `beta=0.7`
- `kernel_beta=0.4`, `alpha_disp=1.4`, `energy_shift=-1.0`
- `use_gaussian_prep=False`, `use_fock_expansion=True`

Representative output:

- `post_prob = 0.004257`
- `purity = 0.94886` (mixed output)
- `F = <u_hat|rho_post|u_hat> = 0.95093`
- `relative PDE-vector error = 0.29034`

Interpretation:

- Directional agreement with target is high (`F`).
- Absolute vector matching is weaker (`~29%` relative error).
- Because state is mixed, fidelity and vector error must be read together.

## Bug fixes applied to sensitivity tooling

Three critical bugs were found and fixed in `heat_eq_sensitivity_refine.py`:

1. **Missing `kernel_beta`**: `lchs_coefficients()` was called without `kernel_beta`, defaulting to `0.0` instead of tuned `0.4`.
2. **Missing `alpha_disp`/`energy_shift` in circuit**: `cvdv_heat_postselect_fock_component()` was called without these, defaulting to `1.0`/`0.0` instead of tuned `1.4`/`-1.0`.
3. **Missing `alpha_disp`/`energy_shift` in theory**: `dv_generator_matrix()` was called without arguments, so the theory comparison used wrong Hamiltonian.

Additional fixes:

- `n_dim` corrected from 64 to 32 (matching `MAX_FOCK_LEVEL`).
- `_safe_float` replaced with `_safe_numeric` to avoid `TypeError` when sweeping integer parameters (`n_steps`, `n_quad_points`).
- Refinement evaluator now uses `sanitize_density_matrix()` so metrics match the main script pipeline.

## Additions

- **PDE vector error metric** added to refinement evaluator: `||u_theory - sqrt(p_post)*psi_principal|| / ||u_theory||`, tracked in CSV outputs and candidate ranking.
- **`alpha_disp`, `energy_shift`, `kernel_beta` sweeps** in refine profiles (quick/default/full).
- **Joint Nelder-Mead optimizer** (`--optimize` flag) for 5-parameter search over `(r_target, r_prime, alpha_disp, energy_shift, kernel_beta)`, with objective priority:
  1. minimize `pde_error`
  2. maximize `post_prob` (tie-break)
  3. fidelity kept only as diagnostic

## Parameter effects

Use `heat_eq_sensitivity_refine.py` outputs as the active source of trends:

- `refine_*_pde_error.png` for primary tuning signal.
- `refine_*_post_prob.png` for secondary practicality.
- `refine_top_candidates.csv` ranked by `(pde_error asc, post_prob desc)`.

## Reproduce (cvdv environment)

```bash
conda activate cvdv
```

1. Single-run evaluation:

```bash
python heat_eq_postselect.py
```

2. Refinement sweeps:

```bash
python heat_eq_sensitivity_refine.py --profile quick
python heat_eq_sensitivity_refine.py --profile default
python heat_eq_sensitivity_refine.py --profile full
```

Outputs: `sensitivity_refine_results/`.

3. Joint optimizer:

```bash
python heat_eq_sensitivity_refine.py --optimize --maxiter 200
```

Outputs: `sensitivity_refine_results/optimize_log.csv`, `optimize_top.csv`.

## What to report when repeating experiments

Report these together:

1. Settings (`r_target`, `r_prime`, `kernel_beta`, `alpha_disp`, `energy_shift`, `n_steps`, `cutoff`).
2. Relative PDE-vector error (primary).
3. `post_prob` (secondary).
4. `purity`.
5. Mixed-state fidelity `F = <u_hat|rho_post|u_hat>` (diagnostic).

## Important caveat

With `use_fock_expansion=True`, the CV state is evaluated via incoherent `|C_n|^2` averaging. This is currently the feasible backend path here, but it removes coherent cross terms and can limit PDE-vector accuracy even when fidelity is high.
