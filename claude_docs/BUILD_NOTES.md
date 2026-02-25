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
  +-- heat_eq_surrogate.py             <- surrogate model: precompute Fock components, sweep kernel_beta free
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

### Bug 6: Wrong sign in LCHS kernel function (CRITICAL — Feb 2026)
**Symptom**: pde_error stuck at ~0.069 floor; appeared to be a "fundamental" LCHS/mixture limit.
**Root cause**: `kernel_function()` in `heat_eq_postselect.py` had `exp(+(1+ik)^β)` instead of the correct `exp(-(1+ik)^β)`. The positive exponent produces an exponentially growing (wrong-direction) kernel, opposite to what the LCHS identity requires. The missing normalization constant `C_β = 2π exp(-2β)` was also absent.
**Old (wrong)**: `np.exp((1.0 + 1.0j * k_points) ** kernel_beta) / (1.0 - 1.0j * k_points)`
**New (correct)**: `np.exp(-((1.0 + 1.0j * k_points) ** kernel_beta)) / (c_beta * (1.0 - 1.0j * k_points))` with `c_beta = 2π exp(-2 kernel_beta)`
**Reference**: LCHS working paper (res_refs), Eq. for `g(k) = f(k)/(1-ik)` with `f(k) = exp(-(1+ik)^β) / C_β`.
**Impact**: All previous optimization results (Basin A ~0.069, Basin B ~0.070-0.073) were computed with the wrong kernel. The "0.069 error floor" was an artifact of the sign bug, not a fundamental limit.
**Propagation**: The fix in `heat_eq_postselect.py` propagates to all scripts (sensitivity_refine, systematic_optimize, theory_optimize) since they all import from it.

---

## 5. Key Numerical Findings

### Spectral properties of T
- Eigenvalues: `lambda_k = 2 - 2*cos(k*pi/5)` for `k = 1..4`
  - `{0.382, 1.382, 2.618, 3.618}`
- Spectral norm `||T|| = 3.618`
- Condition number `kappa = 9.472`
- `||exp(-T) * u0|| = 0.422` for `u0 = [0,1,0,0]`

### Error budget at baseline (PRE-FIX values, wrong kernel sign — superseded)

> **WARNING**: These numbers were computed with the wrong kernel sign (Bug 6).
> They are kept for reference only. See post-fix results below.

| Component | Magnitude | Note |
|---|---|---|
| Trotter bound | 0.005 (operator), 0.012 (PDE-scaled) | Negligible |
| Truncation error | ~1e-8 | Negligible at cutoff=1e-8 |
| LCHS + mixture | ~0.26 | **Dominant** (wrong kernel) |
| **Total observed** | **~0.27** | |

### Error budget post-fix (`r=1.2, r'=0.3, beta_k=0.8, n_steps=100`)
At the default parameters with the corrected kernel:

| Metric | Value |
|---|---|
| pde_error | 0.216 |
| fidelity | 0.948 |
| post_prob | 0.142 |
| purity | 0.972 |

Sensitivity sweeps (corrected kernel, single-parameter variations from default):

| Parameter | Best value tested | Best pde_error |
|---|---|---|
| kernel_beta | 0.8 (in [0.0, 0.8]) | 0.216 |
| r_target | 1.0 (in [1.0, 1.5]) | 0.189 |
| r_prime | 0.15 (in [0.15, 0.55]) | 0.190 |

**Key insight**: The corrected kernel produces higher pde_error at old default params, but the parameter landscape has shifted. Optimization over the full (r_target, r_prime, kernel_beta) space with the correct kernel is in progress.

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

### Best known optimum — PRE-FIX (wrong kernel sign, superseded)

> **WARNING**: These results used the wrong kernel sign (Bug 6). Kept for reference only.

The overnight run (`heat_eq_systematic_optimize.py`, 180 global LHS + 10 local starts,
broad bounds `r_target in [0.15, 1.4]`, `r_prime in [0.03, 0.9]`) found two basins:

| Basin | r_target | r_prime | kernel_beta | pde_error | fidelity |
|-------|----------|---------|-------------|-----------|----------|
| A (pre-fix) | ~0.467 | ~0.030 | ~0.494 | 0.0690 | ~0.926 |
| B (pre-fix) | ~0.20-0.25 | ~0.18-0.22 | ~0.18 | 0.070-0.073 | ~0.960 |

### Best known optimum — POST-FIX (correct kernel)

Systematic optimization with corrected kernel is in progress
(`systematic_opt_results_v3_corrected/`, bounds: `r_target in [0.6, 1.2]`,
`r_prime in [0.03, 0.20]`, `kernel_beta in [0.55, 0.90]`, 84 global + 6 local starts).

Results will be updated here when the run completes.

### Convergence verification at Basin A optimum — PRE-FIX (superseded)

> **WARNING**: These sweeps used the wrong kernel sign (Bug 6). Convergence behavior may differ with the corrected kernel.

**n_dim (Fock truncation dimension): fully converged, no benefit from increasing.**

| n_dim | pde_error | used_terms |
|-------|-----------|------------|
| 16 | 0.068959 | 16 |
| 24 | 0.068957 | 24 |
| 32 | 0.068957 | 32 |
| 64 | 0.068957 | 50 |

At the Basin A optimum (small r_prime ~ 0.03), the Fock distribution is extremely
concentrated. Even n_dim=16 is sufficient. Increasing n_dim is wasted computation.

**n_steps (Trotter steps): marginal improvement, ~0.5% going from 100 to 500.**

| n_steps | pde_error | delta from n=500 |
|---------|-----------|------------------|
| 50 | 0.069487 | +0.000891 (1.3%) |
| 100 | 0.068957 | +0.000361 (0.5%) |
| 200 | 0.068725 | +0.000129 (0.2%) |
| 500 | 0.068596 | baseline |

The Trotter error contribution at n_steps=100 is real but tiny (~0.0004 actual vs
~0.012 upper bound). The dominant error (~0.0686) is the **LCHS integral / mixture
decoherence error**, which is independent of n_steps and n_dim.

**Conclusion (PRE-FIX, now invalidated)**: The 0.069 error floor was believed to be a
fundamental limit of the LCHS/mixture path, but it was actually an artifact of the
wrong kernel sign (Bug 6). With the corrected kernel, the error landscape is different
and needs re-exploration.

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
    kernel_beta: float = 0.8
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

### Reproducibility manifest template

Save one manifest per major run:

```yaml
project: CV-DV-LCHS
goal: original PDE target (no modified operator)
timestamp_local: "YYYY-MM-DD HH:MM"
git_commit: "<commit-hash>"
conda_env: "cvdv"
python: "<python-version>"
script: "heat_eq_systematic_optimize.py"
command: "python ... (full command)"
output_dir: "systematic_opt_results_v2"
fixed_invariants:
  max_fock_level: 32
  n_dim: 32
  use_fock_expansion: true
optimization_settings:
  min_post_prob: 1e-3
  post_penalty: 100.0
  gamma_min: 0.03
  neff_max: 16.0
  bounds:
    r_target: [0.25, 1.2]
    r_prime: [0.05, 0.45]
    kernel_beta: [0.0, 1.0]
key_outputs:
  best_pde_error: "<value>"
  best_post_prob: "<value>"
  best_params: {r_target: "<v>", r_prime: "<v>", kernel_beta: "<v>"}
```

---

## 13. Rigor Labels

Use these labels consistently in notes, reports, and code comments:

| Label | Meaning | Example |
|---|---|---|
| **STRICT-BOUND** | Mathematically proved inequality under stated assumptions | Trotter bound with computed `C_comm` |
| **PROXY** | Diagnostic indicator, not a guaranteed bound on state error | `n_eff`, coefficient-only coherence fraction `1 - 1/n_eff` |
| **EMPIRICAL** | Observed numerically; must reference run settings or manifest | "pde_error = 0.069 at Basin A optimum" |

---

## 14. Robustness Rules for Long Runs

Backend evaluations can fail (Qiskit transpiler errors, NaN from extreme parameters).
Design rules to keep optimization runs alive:

1. **Catch backend exceptions inside objective evaluation** — return `objective = inf`
   and log the error string, do not let the exception propagate and kill the run.
2. **Record invalid points** with `eval_error` field in CSV for post-mortem analysis.
3. **Cap local evaluations** with `--local-maxfev` to prevent a single start from
   consuming the entire time budget.
4. **Diversify local starts** with `--start-min-dist` to avoid redundant convergence
   to the same basin.
5. **Resume safety**: only resume within the same output directory and search domain.
   If bounds or penalties changed materially, start a new output directory.

### Overnight run statistics (Feb 2026)

| Metric | Value |
|---|---|
| Global LHS samples | 180 |
| Local starts attempted | 10 (9 completed, 1 crashed) |
| Total local evaluations | 1044 |
| Finite evaluations | 918 (88%) |
| NaN/inf evaluations | 126 (12%) |
| Crash cause | `TranspilerError: cD synthesis` at small r_prime |

---

## 15. Theoretical Parameter Interaction Model

### How (r_target, r_prime, kernel_beta) couple

The three optimization parameters enter through the LCHS coefficient integral:

```
C_n ∝ ∫ dk  exp(-γ k²) · g_β(k) · H_n(k / e^{r'}) / √(2^n n!)
```

The integrand is a product of three factors with different parameter dependencies:

| Factor | Role | Parameters |
|---|---|---|
| `exp(-γ k²)` | Gaussian envelope in k-space | γ = e^{-2r'} - e^{-2r} — **couples r and r'** |
| `g_β(k) = exp(-(1+ik)^β) / [C_β(1-ik)]` | Spectral shaping | **kernel_beta only** |
| `H_n(k/e^{r'}) / √(2^n n!)` | Fock basis projection | **r_prime only** (basis scale) |

Key interactions:
- **r and r' couple through γ**: Only their difference matters for the Gaussian width.
  Large γ (big gap r−r') → narrow Gaussian → concentrated coefficients → low n_eff.
- **r' has dual role**: Appears in both γ (envelope) and Hermite argument (basis scale).
  Increasing r' simultaneously widens the envelope and stretches the Hermite basis.
- **kernel_beta is modulated by γ**: If γ is large (narrow Gaussian), kernel shape
  matters less — the Gaussian already kills large-|k| contributions.

### Downstream chain and surrogate factorization

```
(r, r', β_k) → C_n            [cheap, analytical]
                 ↓
(r, r')       → (p_n, ρ_n)    [expensive circuit, independent of β_k]
                 ↓
              p_post = Σ |C_n|² p_n
              ρ_post = (1/p_post) Σ |C_n|² p_n ρ_n
                 ↓
              pde_error, fidelity, purity
```

**Critical factorization**: kernel_beta only affects the weights C_n, not the per-Fock
circuit outputs (p_n, ρ_n). Those depend on (r_target, r_prime, n) through the squeezing
operations in the circuit. This enables the surrogate model in `heat_eq_surrogate.py`:

1. Precompute {p_n, paulis_n} on a (r_target, r_prime) grid — expensive, one-time.
2. For any kernel_beta, recompute C_n (milliseconds) and reconstruct all metrics.

### Surrogate script: `heat_eq_surrogate.py`

Usage:
```bash
# Phase 1: precompute Fock components on grid (slow, ~32 × grid_size circuit evals)
python heat_eq_surrogate.py --precompute --output-dir surrogate_data

# Phase 2: sweep kernel_beta at each grid point (fast, no circuits)
python heat_eq_surrogate.py --sweep --output-dir surrogate_data

# Phase 3: optimize kernel_beta per grid point (fast)
python heat_eq_surrogate.py --optimize --output-dir surrogate_data

# All phases:
python heat_eq_surrogate.py --all --output-dir surrogate_data
```

Custom grids:
```bash
python heat_eq_surrogate.py --all --output-dir surrogate_data \
  --r-target-grid "0.6,0.8,1.0,1.2" \
  --r-prime-grid "0.05,0.10,0.15,0.20" \
  --n-steps 100
```

---

## 16. Derivation Hygiene (for Math Writeups)

- Keep derivations in final form only; do not leave scratch corrections inside finished notes.
- For commutator calculations, verify tensor dimension consistency term-by-term before publishing.
- Mark mixture/coherence relations with the appropriate rigor label (Section 13):
  - **PROXY** when coefficient-only (e.g., `1 - 1/n_eff`),
  - **STRICT-BOUND** only when operator-norm assumptions are explicitly stated.
- If a quantitative claim is used in ranking or runtime decisions, ensure it is computed in code (not just argued in text).
