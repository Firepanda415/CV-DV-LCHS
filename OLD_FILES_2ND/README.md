# 1D Heat Equation with CV-DV LCHS

This repository contains a PDE-priority implementation of the 1D heat equation
using a hybrid CV-DV LCHS circuit (`hybridlane` + `bosonicqiskit.hybrid`).

## Runtime models (math)
Let \(C_n(r,r',\beta)\) be the prepared-state coefficients, and let
\(\lvert \psi_n\rangle\) be the unnormalized postselected DV branch from
input Fock level \(n\):
\[
\lvert \psi_n\rangle :=
(\langle 0|S(-r)\otimes I)\,U\,(S(r')|n\rangle\otimes |u_0\rangle).
\]

- Model A (`--model A`, legacy baseline):
\[
p_{\text{post}}^{(A)}=\sum_n |C_n|^2\,\langle\psi_n|\psi_n\rangle,\qquad
\rho_A \propto \sum_n |C_n|^2\,|\psi_n\rangle\langle\psi_n|.
\]
- Model B (`--model B`, default):
\[
|\psi_{\text{post}}^{(B)}\rangle=\sum_n C_n\,|\psi_n\rangle,\quad
p_{\text{post}}^{(B)}=\langle\psi_{\text{post}}^{(B)}|\psi_{\text{post}}^{(B)}\rangle,\quad
\rho_B=\frac{|\psi_{\text{post}}^{(B)}\rangle\langle\psi_{\text{post}}^{(B)}|}{p_{\text{post}}^{(B)}}.
\]

Primary metric remains:
\[
\epsilon_{\text{PDE}}=\frac{\|u_{\text{ref}}-\sqrt{p_{\text{post}}}\,\psi_{\max}(\rho_{\text{post}})\|_2}{\|u_{\text{ref}}\|_2}.
\]

## Main files
- `heat1d_lchs_cv_dv.py`
  - core CV-DV LCHS runner, model selector, classical target, phase calibration, and metrics.
- `heat1d_lchs_sensitivity.py`
  - deterministic theorem-constrained sensitivity/optimization.
- `heat1d_lchs_model_b.py`
  - coherent Model B aggregation module (default path).
- `deep-research-report.tex`
  - derivation, operator mapping, and error-budget model.
- `tests/test_gate_math.py`
  - fast unit tests that verify gate math directly (no full PDE run).

## Core run (cvdv env)
```bash
python heat1d_lchs_cv_dv.py --calibrate-phase
```
Default solver mode is `B` (coherent). To force legacy mode:
```bash
python heat1d_lchs_cv_dv.py --model A
```

## Sensitivity run
```bash
python heat1d_lchs_sensitivity.py \
  --output-dir results/sensitivity \
  --n-r-target 5 --n-r-prime 5 --n-beta 7 \
  --model B \
  --calibrate-phase
```

## Gate-math tests (recommended before long runs)
```bash
pytest -q tests/test_gate_math.py
```

## Recommended run order
1. Gate sanity (fast):
```bash
pytest -q tests/test_gate_math.py
```

2. Phase calibration + baseline JSON:
```bash
python heat1d_lchs_cv_dv.py --calibrate-phase --output-json results/baseline_phase.json
```
Use the selected `disp_phase` from this run for all later runs.

3. Sensitivity sweep with fixed phase:
```bash
python heat1d_lchs_sensitivity.py \
  --output-dir results/sensitivity \
  --n-r-target 7 --n-r-prime 7 --n-beta 9 \
  --disp-phase <PHASE_FROM_STEP2>
```

4. Trotter convergence at fixed parameters:
```bash
for N in 10 25 50 100 200 500; do
  echo "=== n_steps=$N ==="
  python heat1d_lchs_cv_dv.py \
    --n-steps $N \
    --disp-phase <PHASE_FROM_STEP2> \
    --r-target <R> --r-prime <RP> --kernel-beta <KB> \
    --output-json results/trotter_n${N}.json
done
```

## Project policy
- Primary metric: `pde_error`.
- Secondary metric: `post_prob`.
- Default runtime model: `B` (coherent aggregation).
- Legacy baseline model: `A` (incoherent aggregation).
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
