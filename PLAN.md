# PDE-First Implementation Plan

1. Read and extract equations from `res_refs` and `res_base`.
2. Fix mismatches between draft assumptions and package/operator definitions.
3. Build a clean 1D-heat CV-DV LCHS circuit implementation in `heat1d_lchs_cv_dv.py`.
4. Add explicit math-to-gate comments and a phase-branch calibration guard.
5. Add direct gate-math unit tests (generator/decomposition/tape checks) before long PDE runs.
6. Define PDE-priority correctness metrics and error diagnostics.
7. Build deterministic theorem-constrained sensitivity in `heat1d_lchs_sensitivity.py`.
8. Document full derivation and experiment protocol in `HEAT1D_LCHS_EXPERIMENT.tex`.

## Key corrections from prior iterations
- Use improved kernel branch consistently: `exp(-(1+ik)^beta)/(C_beta*(1-ik))`.
- Enforce squeeze geometry `r_prime < r_target` (`gamma > 0`).
- Use current hybridlane APIs (`FockState`, `FockStateProjector([0])`).
- Skip identity factors in projector-tensor observables to avoid backend typing issues.
- Keep phase branch explicit and calibratable to avoid hidden quadrature-convention mistakes.
