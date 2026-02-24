# CV-DV LCHS Heat Equation: Build Notes

Records for rebuilding the work and avoiding repeated mistakes.

---

## 1. File Inventory and Dependency Graph

```
heat_eq_postselect.py           <- core simulation (no external deps besides pennylane/hybridlane/qutip)
  |
  +-- heat_eq_sensitivity_refine.py   <- Evaluator + Settings classes, 1D sweeps, Nelder-Mead optimizer
  |     |
  |     +-- heat_eq_theory_optimize.py      <- theory-guided 4-phase optimizer (imports Evaluator, Settings)
  |     +-- heat_eq_systematic_optimize.py  <- penalty-objective optimizer  (imports Evaluator, Settings)
  |
PARAMETER_FIDELITY_NOTES.tex    <- math reference (standalone)
```

All optimizer scripts import `Evaluator` and `Settings` from `heat_eq_sensitivity_refine.py`.
Any change to `Settings` fields or `Evaluator.evaluate()` return keys propagates downstream.

---

## 2. Critical Constraints (Do Not Violate)

| Constraint | Why |
|---|---|
| `r_prime < r_target` | Otherwise `gamma = e^{-2r'} - e^{-2r} <= 0`, and `lchs_coefficients()` raises `ValueError` |
| `n_dim == MAX_FOCK_LEVEL == 32` | The bosonic backend allocates a fixed Hilbert space; mismatch crashes |
| `MAX_FOCK_LEVEL` must be a power of 2 | Explicit check at module import time in `heat_eq_postselect.py` |
| `kernel_beta >= 0` | Negative values produce divergent kernel integrands |
| `init_qubits` is length-2 tuple of 0/1 | Hardcoded for 2-qubit (4-point) finite-difference grid |

---

## 3. PDE-First Design (Current State)

The codebase was refactored to a "PDE-first" design:

- `dv_generator_matrix()` takes **no arguments** — returns the pure Laplacian `T`.
- Old parameters `alpha_disp` and `energy_shift` were removed entirely.
- The operator is: `T = 2(I x I) - (I x X) - 0.5[(X x X) + (Y x Y)]`.
- The PDE solution is: `u(t) = exp(-alpha * t * T) * u(0)` with `alpha = 1.0`, `h = 1.0`.
- **Do not** re-introduce `alpha_disp` or `energy_shift` — they were intentionally removed.

---

## 4. Bug History (Mistakes to Avoid)

### Bug 1: Missing `kernel_beta` in function calls
**Symptom**: LCHS coefficients computed with `kernel_beta=0.0` (the default) regardless of settings.
**Root cause**: `kernel_beta` not passed to `hep.lchs_coefficients()`.
**Fix**: Always pass `kernel_beta=cfg.kernel_beta` (or equivalent) when calling `lchs_coefficients()` or `get_lchs_states()`.
**Rule**: Any new code calling `lchs_coefficients()` must pass `kernel_beta`.

### Bug 2: Missing parameters in `cvdv_heat_postselect_fock_component()`
**Symptom**: Simulations ran with default parameter values instead of swept values.
**Root cause**: Not forwarding `r_target`/`r_prime` from the Settings config to the circuit call.
**Fix**: Always pass `r_target=cfg.r_target, r_prime=cfg.r_prime` explicitly.

### Bug 3: Stale `dv_generator_matrix()` calls with old arguments
**Symptom**: Theory comparison used a different operator than the circuit.
**Root cause**: After PDE-first refactor, `dv_generator_matrix()` takes no args, but old code still passed `alpha_disp`/`energy_shift`.
**Fix**: Call `hep.dv_generator_matrix()` with no arguments.

### Bug 4: Sensitivity script not matching main script parameters
**Symptom**: Sweep results disagree with the main `heat_eq_postselect.py` output.
**Root cause**: `Settings` dataclass defaults were out of sync with the main script's hardcoded values.
**Fix**: Keep `Settings` defaults aligned with the values in `heat_eq_postselect.py` `__main__`.

### Bug 5: Using unsupported coherent `FockStateVector` path on the bosonic backend
**Symptom**: Decomposition/device error when running with `use_gaussian_prep=False` and `use_fock_expansion=False`.
**Root cause**: The `FockStateVector` op path attempts to load a full non-Gaussian state vector directly, which the bosonic backend cannot always decompose.
**Fix**: Use the Fock-component evaluation path (`cvdv_heat_postselect_fock_component`) with incoherent `|C_n|^2` aggregation. This is the only reliable run mode (`use_fock_expansion=True`).

---

## 5. Key Numerical Findings

### Spectral properties of T
- Eigenvalues: `lambda_k = 2 - 2*cos(k*pi/5)` for `k = 1..4`
  - `{0.382, 1.382, 2.618, 3.618}`
- Spectral norm `||T|| = 3.618`
- Condition number `kappa = 9.472`
- `||exp(-T) * u0|| = 0.422` for `u0 = [0,1,0,0]`

### Error budget at baseline (`r=1.2, r'=0.3, beta_k=0.4, n_steps=100`)
| Component | Magnitude | Note |
|---|---|---|
| Trotter bound | 0.005 (operator), 0.012 (PDE-scaled) | Negligible |
| Truncation error | ~1e-8 | Negligible at cutoff=1e-8 |
| LCHS + mixture | ~0.26 | **Dominant** |
| **Total observed** | **~0.27** | |

**Key insight**: Trotter error is negligible. The dominant error comes from the LCHS integral approximation combined with the incoherent Fock mixture decoherence (`rho ~ sum_n |C_n|^2 |n><n|`).

### Commutator structure
Only one non-zero commutator among Hamiltonian terms:
```
[H_IX, H_YY] = i(Y x Z),  norm = 1.0
```
All other pairs commute. So `C_comm = 1`.

### Empirically good parameter bounds (from optimization runs)
```
r_target:    [0.25, 1.2]   (shifted down from original [0.8, 2.0])
r_prime:     [0.05, 0.45]  (shifted down from original [0.05, 0.9])
kernel_beta: [0.0, 1.0]
min_gap:     r_target - r_prime >= 0.02
```

---

## 6. Settings Dataclass (Source of Truth)

From `heat_eq_sensitivity_refine.py`:

```python
@dataclass(frozen=True)
class Settings:
    total_time: float = 1.0
    n_steps: int = 100
    init_qubits: tuple[int, int] = (0, 1)
    n_dim: int = 32
    r_target: float = 1.2
    r_prime: float = 0.3
    kernel_beta: float = 0.4
    fock_expansion_cutoff: float = 1e-8
    n_quad_points: int = 220
```

---

## 7. Evaluator Pattern

All optimization scripts share the same evaluation pattern:

```python
ev = Evaluator()             # has coeff_cache
cfg = replace(base, r_target=..., r_prime=..., kernel_beta=...)
metrics = ev.evaluate(cfg)   # returns dict with: fidelity, post_prob, purity, pde_error, used_fock_terms
```

The `Evaluator.evaluate()` method:
1. Checks `r_prime < r_target` (returns NaN dict if violated)
2. Computes LCHS coefficients (cached by `get_coeffs()`)
3. Runs Fock-component circuit simulations for each `n` with `|C_n|^2 >= cutoff`
4. Rebuilds density matrix from Pauli tomography
5. Computes PDE vector error against `exp(-alpha*t*T) * u0`

---

## 8. Optimization Scripts Comparison

| Feature | `heat_eq_sensitivity_refine.py` | `heat_eq_systematic_optimize.py` | `heat_eq_theory_optimize.py` |
|---|---|---|---|
| Global search | Grid sweeps (1D + 2D coupled) | LHS sampling | LHS sampling |
| Local optimizer | Single Nelder-Mead | Multi-start Nelder-Mead | Multi-start Nelder-Mead |
| Theory diagnostics | None | gamma, n_eff | gamma, n_eff, coherence_frac, error budget |
| Convergence checks | None | None | Phase 1: n_steps, n_quad, cutoff sweeps |
| Resume support | No | Yes (CSV checkpoints) | Yes (CSV checkpoints) |
| Penalty terms | post_prob only | post, gamma, n_eff | post_prob only |
| Plots | 1D curves, 3D surface, Pareto | Landscape, Pareto | Convergence, error budget, landscape, correlations, Pareto |

---

## 9. Running the Scripts

### Environment
```bash
conda activate cvdv
```

### Main simulation
```bash
python heat_eq_postselect.py
```

### Sensitivity sweeps (quick test)
```bash
python heat_eq_sensitivity_refine.py --profile quick --output-dir sensitivity_refine_results
```

### Systematic optimizer
```bash
python heat_eq_systematic_optimize.py --output-dir systematic_opt_results --resume
```

### Theory-guided optimizer (skip Phase 1 for faster runs)
```bash
python heat_eq_theory_optimize.py --output-dir theory_opt_results --skip-convergence --resume
```

### Full theory run (all phases)
```bash
python heat_eq_theory_optimize.py --output-dir theory_opt_results --n-global 30 --n-starts 4 --local-maxiter 30
```

---

## 10. Metrics Glossary

| Metric | Formula | Primary/Secondary |
|---|---|---|
| `pde_error` | `||u_theory - sqrt(p_post) * psi_principal|| / ||u_theory||` | **Primary** |
| `post_prob` | `sum_n |C_n|^2 * <0|S(-r)^dag U_n^dag proj U_n S(-r)|0>` | Secondary (feasibility) |
| `fidelity` | `<u_hat|rho_post|u_hat>` | Diagnostic only |
| `purity` | `Tr(rho_post^2)` | Diagnostic only |
| `gamma` | `e^{-2r'} - e^{-2r}` | Theory diagnostic |
| `n_eff` | `1 / sum_n |C_n|^4` | Theory diagnostic |
| `coherence_frac` | `1 - 1/n_eff` | Fraction of coherence lost in mixture approx |

**Ranking rule**: Sort by (1) smallest `pde_error`, (2) largest `post_prob` as tiebreak.
Use fidelity only as a consistency check, never as the optimization target.

---

## 11. Quick Regression Checklist

- [ ] `epsilon_alg == 0` in main script output (theory target matches classical PDE target).
- [ ] `pde_error` is finite and reported.
- [ ] `post_prob` is finite and above threshold for the selected candidate.
- [ ] Candidate comes from PDE-priority ranking, not fidelity-only ranking.
- [ ] Script outputs are resumable and checkpointed for long runs.
- [ ] `dv_generator_matrix()` called with no arguments (PDE-first design).
- [ ] `kernel_beta` passed explicitly in all `lchs_coefficients()` calls.

---

## 12. Artifacts to Archive for Reproducibility

- Main run text output (settings + all metrics)
- Optimizer CSVs:
  - `optimize_log.csv`, `optimize_top.csv` (from `heat_eq_sensitivity_refine.py --optimize`)
  - `systematic_global.csv`, `systematic_local.csv`, `systematic_top.csv`, `systematic_pareto_front.csv`
  - `phase2_global.csv`, `phase3_local.csv`, `final_top.csv`, `final_all.csv` (from `heat_eq_theory_optimize.py`)
- Plots:
  - `refine_tradeoff_pde_vs_post.png`
  - `systematic_tradeoff_pde_vs_post.png`, `systematic_landscape_rtarget_rprime.png`
  - `landscape_pde.png`, `theory_correlations.png`, `error_budget.png`, `tradeoff_pde_vs_post.png`

---

## 13. Derivation Hygiene (for Math Writeups)

- Keep derivations in final form only; do not leave scratch corrections inside finished notes.
- For commutator calculations, verify tensor dimension consistency term-by-term before publishing.
- Mark mixture/coherence relations as:
  - **proxy/indicator** when coefficient-only (e.g., `1 - 1/n_eff`),
  - **bound** only when operator-norm assumptions are explicitly stated.
- If a quantitative claim is used in ranking or runtime decisions, ensure it is computed in code (not just argued in text).
