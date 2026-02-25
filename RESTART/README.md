# RESTART: 1D Heat Equation CV-DV LCHS

This folder is a clean reimplementation for the 1D heat-equation experiment.

## Files
- `HEAT1D_LCHS_EXPERIMENT.tex`
  - derivation + operator mapping + error budget + theorem-constrained search model.
- `heat1d_lchs_cv_dv.py`
  - core CV-DV LCHS circuit implementation and single-run PDE-priority report.
- `heat1d_lchs_sensitivity.py`
  - deterministic theorem-guided sensitivity and parameter optimization.

## Quick start (cvdv env)
```bash
python RESTART/heat1d_lchs_cv_dv.py --calibrate-phase
```

## Sensitivity / optimization
```bash
python RESTART/heat1d_lchs_sensitivity.py \
  --output-dir RESTART/results \
  --n-r-target 5 --n-r-prime 5 --n-beta 7
```

## Project policy used in this restart
- Primary metric: `pde_error`.
- Secondary metric: `post_prob`.
- No random search: deterministic grids + deterministic local refinement.
- Hard constraints from derivation:
  - `r_prime < r_target`,
  - `gamma = exp(-2r_prime)-exp(-2r_target) > 0`,
  - `beta in (0, 1)`,
  - truncation and `n_eff` constraints.
