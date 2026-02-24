# CV-DV LCHS Heat-Equation Rebuild Log (PDE Priority)

## 1) Objective (frozen)

Solve the original mapped heat-equation target, not a modified operator:

- Target vector:
  - `u_PDE(t) = exp(-alpha * t * T) u(0)`
- Fixed generator:
  - `T = 2(I⊗I) - (I⊗X) - 1/2 (X⊗X + Y⊗Y)`
- Priority:
  1. minimize `pde_error`
  2. maximize `post_prob` (tie-break)
  3. `fidelity` is diagnostic only

## 2) Environment and invariants

- Conda env: `cvdv`
- Fixed truncation:
  - `MAX_FOCK_LEVEL = 32`
  - `n_dim = 32`
- Core run mode:
  - `use_fock_expansion = True`
  - mixture approximation via `|C_n|^2`

## 3) Canonical scripts (current)

- `/Users/zhen002/GitHub/CV-DV-LCHS/heat_eq_postselect.py`
  - single-run correctness report against original PDE target
- `/Users/zhen002/GitHub/CV-DV-LCHS/heat_eq_sensitivity_refine.py`
  - PDE-priority sweeps and 3-parameter optimization
- `/Users/zhen002/GitHub/CV-DV-LCHS/heat_eq_systematic_optimize.py`
  - theory-guided global+local optimization with resume/checkpoint

## 4) Rebuild protocol (exact order)

1. Activate environment

```bash
conda activate cvdv
```

2. Sanity run of the main pipeline

```bash
python /Users/zhen002/GitHub/CV-DV-LCHS/heat_eq_postselect.py
```

Expected key pattern:
- `ε_alg = 0.0` (theory target == classical PDE target)
- finite `post_prob`
- finite `pde_error` (non-zero due mixed-state approximation path)

3. PDE-priority refinement optimizer

```bash
python /Users/zhen002/GitHub/CV-DV-LCHS/heat_eq_sensitivity_refine.py \
  --optimize --maxiter 200 --min-post-prob 1e-3 --low-post-penalty 100
```

4. Systematic theory-guided optimizer

```bash
python /Users/zhen002/GitHub/CV-DV-LCHS/heat_eq_systematic_optimize.py \
  --output-dir /Users/zhen002/GitHub/CV-DV-LCHS/systematic_opt_results_v2 \
  --global-samples 30 --n-starts 4 --local-maxiter 30 \
  --min-post-prob 1e-3
```

5. Resume after interruption

```bash
python /Users/zhen002/GitHub/CV-DV-LCHS/heat_eq_systematic_optimize.py \
  --output-dir /Users/zhen002/GitHub/CV-DV-LCHS/systematic_opt_results_v2 \
  --resume
```

## 4.1) Run profiles (time-control)

- Quick sanity (small search, fast iteration):
```bash
python /Users/zhen002/GitHub/CV-DV-LCHS/heat_eq_systematic_optimize.py \
  --output-dir /Users/zhen002/GitHub/CV-DV-LCHS/systematic_opt_results_quick \
  --global-samples 12 --n-starts 2 --local-maxiter 10 --min-post-prob 1e-3
```

- Standard project run:
```bash
python /Users/zhen002/GitHub/CV-DV-LCHS/heat_eq_systematic_optimize.py \
  --output-dir /Users/zhen002/GitHub/CV-DV-LCHS/systematic_opt_results_v2 \
  --global-samples 30 --n-starts 4 --local-maxiter 30 --min-post-prob 1e-3
```

- Higher-coverage run (slower):
```bash
python /Users/zhen002/GitHub/CV-DV-LCHS/heat_eq_systematic_optimize.py \
  --output-dir /Users/zhen002/GitHub/CV-DV-LCHS/systematic_opt_results_full \
  --global-samples 60 --n-starts 8 --local-maxiter 50 --min-post-prob 1e-3
```

Resume safety:
- Resume only within the same output directory and same search domain/settings.
- If bounds or penalties changed materially, start a new output directory.

## 5) Critical mistakes to avoid (from this project)

- Mistake: optimizing against modified operator while claiming original PDE correctness.
  - Fix: keep PDE target fixed; in main script, generator is fixed to `T`.

- Mistake: using unsupported coherent state loading op path (`FockStateVector`) on backend run path.
  - Symptom: decomposition/device error.
  - Fix: use fock-component evaluation path (`cvdv_heat_postselect_fock_component`) with incoherent `|C_n|^2` aggregation.

- Mistake: mismatch between `n_dim` and backend cutoff.
  - Fix: enforce `n_dim == MAX_FOCK_LEVEL == 32`.

- Mistake: treating fidelity as primary success criterion.
  - Fix: use `pde_error` first, `post_prob` second; fidelity only diagnostic for mixed outputs.

- Mistake: allowing invalid parameter geometry (`r_prime >= r_target`).
  - Fix: enforce `r_prime < r_target` and practical `min_gap`.

- Mistake: systematic optimizer domain excluding empirically good region.
  - Observation: good region appeared around lower `r_target` than initial high-only bounds.
  - Fix: default bounds shifted and exposed in CLI.

- Mistake: long runs without progress/checkpoint.
  - Fix: systematic optimizer now has periodic progress prints, checkpoint writes, and resume.

## 6) Decision rules for parameter acceptance

- Accept a candidate only if:
  - finite `pde_error`
  - `post_prob >= min_post_prob`
  - stable diagnostics (`purity`, Pauli reconstruction sanity)
- Rank candidates by:
  1. feasible/infeasible (`post_prob` threshold)
  2. lowest `pde_error`
  3. highest `post_prob`

## 7) Artifacts to archive for reproducibility

- Main run text output (settings + metrics)
- Optimizer CSVs:
  - `optimize_log.csv`, `optimize_top.csv`
  - `systematic_global.csv`, `systematic_local.csv`, `systematic_top.csv`, `systematic_pareto_front.csv`
- Plots:
  - `refine_tradeoff_pde_vs_post.png`
  - `systematic_tradeoff_pde_vs_post.png`
  - `systematic_landscape_rtarget_rprime.png`

## 7.1) Reproducibility manifest template

Save one manifest with every major run:

```yaml
project: CV-DV-LCHS
goal: original PDE target (no modified operator)
timestamp_local: "YYYY-MM-DD HH:MM"
git_commit: "<commit-hash>"
conda_env: "cvdv"
python: "<python-version>"
script: "/Users/zhen002/GitHub/CV-DV-LCHS/heat_eq_systematic_optimize.py"
command: "python ... (full command)"
output_dir: "/Users/zhen002/GitHub/CV-DV-LCHS/systematic_opt_results_v2"
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

## 8) Quick regression checklist

- `ε_alg == 0` in main script output.
- `pde_error` is finite and reported.
- `post_prob` is finite and above threshold for selected candidate.
- Candidate comes from PDE-priority ranking, not fidelity-only ranking.
- Script outputs are resumable and checkpointed for long runs.

## 9) Derivation hygiene checks (math writeups)

- Keep derivations final-form only; do not leave scratch corrections inside final notes.
- For commutators, verify tensor dimension consistency term-by-term before publishing equations.
- Mark mixture/coherence relations as:
  - `proxy/indicator` when coefficient-only,
  - `bound` only when operator-norm assumptions are explicitly stated.
- If a quantitative claim is used in ranking or runtime decisions, ensure it is computed in code (not just argued in text).

## 10) Rigor labels (strict vs heuristic)

Use these labels in notes and reports:

- `STRICT-BOUND`:
  - mathematically proved inequality under stated assumptions.
  - example: first-order Trotter operator bound with computed commutator constant.

- `PROXY`:
  - diagnostic indicator, not a guaranteed bound.
  - examples: `n_eff`, coefficient-only coherence-loss indicators.

- `EMPIRICAL`:
  - observed numerically in current experiments/runs.
  - must include run settings or manifest reference.

## 11) Overnight run postmortem (2026-02-24)

Observed from `logs/overnight_opt.log` and `systematic_opt_results_overnight`:

- Run ended by backend exception, not convergence:
  - `TranspilerError: HighLevelSynthesis is unable to synthesize "cD"`
  - happened during local stage at `start 10/10`.
- Local stage was too expensive:
  - `local_rows = 1044`
  - finite rows: `918`
  - invalid rows (`nan/inf`): `126`
- Many local starts converged to the same basin:
  - approximately `r_target ~ 0.467`, `r_prime ~ 0.030`, `kernel_beta ~ 0.494`
  - best observed `pde_error ~ 0.06896`, `post_prob ~ 0.1777`

Actions applied in optimizer code (`heat_eq_systematic_optimize.py`):

- Crash resilience:
  - backend eval exceptions are now caught inside objective evaluation
  - failed points are recorded as invalid with `objective = inf` and `eval_error`
  - run continues instead of terminating the full job
- Runtime controls:
  - added `--local-maxfev` (hard cap on objective evaluations per local start)
  - added `--start-min-dist` (diverse local-start selection to avoid redundant basins)
  - local Nelder-Mead now receives explicit parameter bounds

Recommended next command (faster PDE-priority profile):

```bash
python /Users/zhen002/GitHub/CV-DV-LCHS/heat_eq_systematic_optimize.py \
  --output-dir /Users/zhen002/GitHub/CV-DV-LCHS/systematic_opt_results_v3 \
  --r-target-min 0.15 --r-target-max 1.4 \
  --r-prime-min 0.03 --r-prime-max 0.9 \
  --global-samples 96 --n-starts 6 \
  --local-maxiter 40 --local-maxfev 70 \
  --start-min-dist 0.12 \
  --min-post-prob 1e-3
```

## 12) Fock cutoff check at current optimum basin

Question: does Fock level matter for current optimum region?

- Using coefficient diagnostics at
  - `r_target = 0.4676405762`
  - `r_prime = 0.0302221766`
  - `kernel_beta = 0.4941015201`
- With `n_dim=96` reference coefficients:
  - `sum_{n=0}^{31} |C_n|^2 = 0.9999973462`
  - tail mass `sum_{n>=32} |C_n|^2 = 2.6538e-6`
  - max single tail weight at `n=32`: `5.6419e-7`

Conclusion:

- For this basin, `MAX_FOCK_LEVEL = 32` is already tight.
- Keep optimization at 32 for speed.
- Re-evaluate only top 1-3 candidates at higher cutoff as a final truncation sanity check.

## 13) Cross-review corrections (from claude_docs + heat_eq_theory_optimize.py)

These points are useful, but must be interpreted carefully:

- Basin-specific claims must stay basin-specific:
  - Statements like “`n_dim=16` is sufficient” are true for the low-`r_prime` best basin, not globally.
  - Keep the project invariant at `n_dim=MAX_FOCK_LEVEL=32` unless explicitly running a truncation study.

- Do not call the observed `~0.069` floor “fundamental” without qualification:
  - It is an empirical floor under current ansatz/backend/mixture path.
  - It is not a proof-level lower bound for all CV-DV LCHS formulations.

- Theory optimizer default bounds can miss the known best basin:
  - `heat_eq_theory_optimize.py` currently uses `r_prime_lo=0.05`.
  - Best observed basin is near `r_prime~0.03`.
  - If using theory optimizer, lower `r_prime_lo` accordingly.

- Backend robustness is mandatory for long runs:
  - Treat transpiler/backend failures as invalid points (`objective=inf`) and continue.
  - Do not allow one failing point to terminate the full run.

- Keep rigor labels strict:
  - `n_eff`, coherence-fraction, and related mixture diagnostics are `PROXY`.
  - Trotter commutator inequality is `STRICT-BOUND`.

Practical policy:

- Use `heat_eq_systematic_optimize.py` for production runs (now robust and budget-controlled).
- Use `heat_eq_theory_optimize.py` as an analysis script, and only with corrected bounds and runtime caps.
