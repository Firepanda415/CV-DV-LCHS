# 1D Heat Equation with CV-DV LCHS

This repository contains a PDE-priority implementation of the 1D heat equation
using a hybrid CV-DV LCHS circuit (`hybridlane` + `bosonicqiskit.hybrid`).

## Main files
- `heat1d_lchs_cv_dv.py`
  - core CV-DV LCHS circuit, classical target, phase calibration, and metrics.
- `heat1d_lchs_sensitivity.py`
  - deterministic theorem-constrained sensitivity/optimization.
- `HEAT1D_LCHS_EXPERIMENT.tex`
  - derivation, operator mapping, and error-budget model.
- `tests/test_gate_math.py`
  - fast unit tests that verify gate math directly (no full PDE run).

## Core run (cvdv env)
```bash
python heat1d_lchs_cv_dv.py --calibrate-phase
```

## Sensitivity run
```bash
python heat1d_lchs_sensitivity.py \
  --output-dir results/sensitivity \
  --n-r-target 5 --n-r-prime 5 --n-beta 7 \
  --calibrate-phase
```

## Gate-math tests (recommended before long runs)
```bash
pytest -q tests/test_gate_math.py
```

## Project policy
- Primary metric: `pde_error`.
- Secondary metric: `post_prob`.
- Search method: deterministic coarse grid + deterministic local refinement.
- Derivation constraints:
  - `r_prime < r_target`,
  - `gamma = exp(-2r_prime) - exp(-2r_target) > 0`,
  - `kernel_beta in (0, 1)`,
  - truncation controls via `tail_mass` and `n_eff`.

## Convention guardrails
- Displacement phase is explicit and auditable in both scripts.
- Verified mapping (HybridLane Heisenberg rep, `hbar=2`):
  - `Delta x = 2 a cos(phi)`,
  - `Delta p = 2 a sin(phi)`.
