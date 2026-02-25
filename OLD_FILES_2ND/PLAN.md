# PDE-First Implementation Plan

1. Read and extract equations from `res_refs` and `res_base`.
2. Fix mismatches between draft assumptions and package/operator definitions.
3. Build a clean 1D-heat CV-DV LCHS circuit implementation in `heat1d_lchs_cv_dv.py`.
4. Add explicit math-to-gate comments and a phase-branch calibration guard.
5. Add direct gate-math unit tests (generator/decomposition/tape checks) before long PDE runs.
6. Define PDE-priority correctness metrics and error diagnostics.
7. Build deterministic theorem-constrained sensitivity in `heat1d_lchs_sensitivity.py`.
8. Document full derivation and experiment protocol in `deep-research-report.tex`.

## Key corrections from prior iterations
- Use improved kernel branch consistently: `exp(-(1+ik)^beta)/(C_beta*(1-ik))`.
- Enforce squeeze geometry `r_prime < r_target` (`gamma > 0`).
- Use current hybridlane APIs (`FockState`, `FockStateProjector([0])`).
- Skip identity factors in projector-tensor observables to avoid backend typing issues.
- Keep phase branch explicit and calibratable to avoid hidden quadrature-convention mistakes.

## Execution protocol
1. `pytest -q tests/test_gate_math.py`
2. `python heat1d_lchs_cv_dv.py --calibrate-phase --output-json results/baseline_phase.json`
3. `python heat1d_lchs_sensitivity.py --output-dir results/sensitivity --n-r-target 7 --n-r-prime 7 --n-beta 9 --disp-phase <PHASE_FROM_STEP2>`
4. Trotter convergence at fixed `(r, r_prime, beta, phase)` over `n_steps in {10,25,50,100,200,500}`.

Interpretation rule:
- If PDE error decreases strongly with `n_steps`, Trotter error is dominant.
- If PDE error plateaus, bottleneck is non-Trotter (mixture/truncation/model mismatch).
