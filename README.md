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
python heat_eq_sensitivity_refine.py --profile quick
python heat_eq_sensitivity_refine.py --profile default
python heat_eq_sensitivity_refine.py --profile full
```

3. Optional 3-parameter optimization (`r_target`, `r_prime`, `kernel_beta`):

```bash
python heat_eq_sensitivity_refine.py --optimize --maxiter 200
```

Outputs are written to `sensitivity_refine_results/`.

## Metrics to report

1. `pde_error` (primary)
2. `post_prob` (secondary)
3. `purity`
4. `fidelity` (diagnostic only)

## Important implementation note

With `use_fock_expansion=True`, CV loading is evaluated via incoherent `|C_n|^2` averaging. This is the feasible backend path in this project and can limit absolute PDE-vector matching.
