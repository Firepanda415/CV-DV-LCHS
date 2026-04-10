# Paper-Facing Consistency Audit for `res_base`

Date: 2026-04-09

## Scope
- Manuscript sources:
  - `res_base/main.tex`
  - `res_base/experiments.tex`
- Paper-facing code paths:
  - `clean_core.py`
  - `clean_oracles.py`
  - `clean_hybrid.py`
  - `clean_sweep.py`
  - `clean_dv_lchs_nwq.py`
  - `clean_dv_lchs_dirichlet.py`
- Retained paper data:
  - folders listed in `WORKSPACE_GUIDE.md`

## Executive Summary
- Overall paper-facing status: mostly consistent.
- High-confidence matches:
  - benchmark definition
  - ideal-loading oracle baseline table
  - bound-related empirical checks
  - hybrid resource counts
  - DV quadrature table
  - DV circuit-count table
  - abstract headline numbers, numerically
- Paper-facing issues that need attention:
  - `inconsistent`: the SNAP+D table/text in `res_base/experiments.tex` describes the chosen depth as the "best depth identified" at each kernel point, but the retained `results_clean_snap_recreate_*` folders are ranked by `oracle_only`, not PDE fidelity, and Neumann/periodic have different PDE-optimal depths.
  - `qualified match`: the manuscript describes a broader initial injection sweep domain than the paper-facing repo currently retains. The retained sweep folders only cover the refined high-fidelity subdomain.

## Findings Requiring Action

### 1. SNAP+D depth-selection wording is inconsistent with retained paper data
Status: `inconsistent`

Evidence:
- `res_base/experiments.tex` states that the SNAP+D row "uses the best depth identified at that kernel point" and presents depth 30 for all three boundaries.
- The retained result folders `results_clean_snap_recreate_*` store `ranking_objective = "oracle_only"` in `sweep_summary.json`.
- In those same files:
  - Dirichlet: `best_rows_by_objective["pde"]` also has depth 30.
  - Neumann: `best_rows_by_objective["pde"]` has depth 18, while `best_row`/`oracle_only` has depth 30.
  - Periodic: `best_rows_by_objective["pde"]` has depth 14, while `best_row`/`oracle_only` has depth 30.

Implication:
- The table values in `res_base/experiments.tex` match the retained `best_row` numerically, but for Neumann and periodic they are not the PDE-optimal depths.
- If "best depth" is intended to mean best end-to-end PDE fidelity, the current manuscript wording is wrong.
- If the intended selection criterion is best oracle fidelity, the wording should say that explicitly.

Recommended fix:
- Update the text and caption around the SNAP+D table to say the retained rows are selected by oracle-fidelity ranking (`oracle_only`), or regenerate/select rows by PDE fidelity and update the table values accordingly.

### 2. The broad initial injection sweep described in the paper is not retained in the paper-facing repo
Status: `qualified match`

Evidence:
- `res_base/experiments.tex` describes a "full search domain"
  - `r_target in [1.92, 8.4]`
  - `r_prime in [0.975, 4.5]`
  - `beta in [0.3, 0.9]`
- The retained paper data folders listed in `WORKSPACE_GUIDE.md` for the ideal-loading sweep are `results_clean_refined_*`.
- The retained `results_clean_refined_*/sweep_all.csv` files cover only:
  - `r_target in [7.8, 8.4]`
  - `r_prime in [4.0, 4.5]`
  - `beta in {0.3, 0.5, 0.6, 0.675, 0.725, 0.75, 0.8, 0.9}`
  - `n_coeff in {24, 48}`
  - `n_trotter = 100`
- `clean_sweep.py` can run arbitrary grids, so the broader search is still representable in code, but it is not retained in the paper-facing result folders.

Implication:
- The refined high-fidelity subdomain and the figure/table values are supported by repo data.
- The claim that the broader initial sweep was performed is not fully supported by retained paper-facing artifacts in this repo.

Recommended fix:
- Either soften the paper wording to say the repo retains the refined subdomain used for the reported figure/table, or retain/add the broader initial sweep outputs if full repo reproducibility is required.

## Claim-by-Claim Audit

### Benchmark definition
Status: `match`

Claim:
- Two-qubit benchmark, `M=4`, `alpha=1`, `h=1`, `T=1`, `u(0)=|01>`, with the stated Dirichlet/Neumann/periodic matrices, `n_fock=64`, and `n_trotter=100`.

Validation:
- `res_base/experiments.tex` states these values directly.
- `clean_core.py` constructs the three heat-equation matrices exactly as written in the manuscript.
- `clean_core.py` uses `init_basis_index=1` by default for all three heat builders.
- `clean_core.py` stores basis states in physics ordering; index `1` corresponds to `|01>`.
- `clean_hybrid.py` permutes the physics-ordered state into Qiskit ordering before circuit initialization, so the paper notation and circuit initialization are aligned.

Conclusion:
- The benchmark implementation matches the manuscript.

### Fidelity definitions and exact/truncated references
Status: `match`

Claim:
- `F_exact`, `F_trunc`, `F_oracle`, `F_snap`, `F_givens`, `F_DV`, and `p_post` are used as defined in the experiments section.

Validation:
- `clean_hybrid.py` computes the exact DV reference from `exp(-(L+iH)T)`, the exact truncated CV reference induced by the prepared seed, and the reported fidelities/postselection probability.
- The data fields in the retained `sweep_summary.json` files match the manuscript’s naming and interpretation.

Conclusion:
- The reported metric definitions are implemented consistently.

### Ideal-loading sweep optima and Table `clean_oracle_baseline`
Status: `match`

Claim:
- Best injection points:
  - Dirichlet `(7.9, 4.1, 0.5, 48)` with `F_exact=0.998961`, `F_trunc=0.999992`, `p_post=0.06279`
  - Neumann `(7.9, 4.0, 0.3, 48)` with `F_exact=0.999716`, `F_trunc=0.999984`, `p_post=0.05616`
  - Periodic `(8.1, 4.1, 0.3, 48)` with `F_exact=0.999725`, `F_trunc=1.000000`, `p_post=0.04842`

Validation:
- These values match the retained `results_clean_refined_*/sweep_summary.json` files to rounding precision.
- The sensitivity figure range in the manuscript matches the retained refined-domain sweep folders and included PDF figure.

Conclusion:
- Table values and refined-domain figure claims match the retained paper data.

### Bound-related empirical checks
Status: `match`

Claim:
- Increasing `n_coeff` from 24 to 48 improves the best exact fidelity in all three cases by the quoted amounts.
- The Dirichlet Trotter check gives relative errors `3.0835e-3`, `1.5428e-3`, `7.7171e-4` for `n_trotter=8,16,32`.

Validation:
- These numbers match `results_clean_bounds/section6_stateprep_consistency.csv` and `results_clean_bounds/section6_trotter_tightness.csv`.

Conclusion:
- The bound-related discussion is supported by the retained data.

### Givens baseline table and hybrid resource counts
Status: `match`

Claim:
- Givens rows reproduce the ideal-loading fidelities.
- Givens uses 47 JC pulses and 47 qubit rotations in all three cases.
- Hybrid 100-step Trotter block counts are:
  - Dirichlet: `1Q=1400`, `CNOT=400`, `D=100`, `CD=300`
  - Neumann: `1Q=1400`, `CNOT=600`, `D=100`, `CD=400`
  - Periodic: `1Q=600`, `CNOT=200`, `D=100`, `CD=200`

Validation:
- The Givens fidelities and JC/qubit-rotation counts match `results_clean_givens_*/sweep_summary.json`.
- Rebuilding the hybrid circuits from `clean_hybrid.py` and inspecting `circuit_resource_report` reproduces the exact operation histograms:
  - Dirichlet: `h=1000`, `rz=400`, `cx=400`, `D=100`, `cD=300`
  - Neumann: `h=1000`, `rz=400`, `cx=600`, `D=100`, `cD=400`
  - Periodic: `h=600`, `cx=200`, `D=100`, `cD=200`
- Aggregating the one-qubit DV basis-change/rotation gates gives the table values above.

Conclusion:
- The hybrid resource table is consistent with the current implementation.

### SNAP+D numerical rows
Status: `qualified match`

Claim:
- The manuscript reports:
  - Dirichlet: `F_snap=0.996817`, `F_oracle=0.994380`, depth 30
  - Neumann: `F_snap=0.999678`, `F_oracle=0.995048`, depth 30
  - Periodic: `F_snap=0.999531`, `F_oracle=0.996856`, depth 30

Validation:
- These numbers match the retained `best_row` entries in `results_clean_snap_recreate_*/sweep_summary.json`.
- However, for Neumann and periodic, the same files record different PDE-optimal depths under `best_rows_by_objective["pde"]`.

Conclusion:
- The table is numerically consistent with the retained saved `best_row`, but the selection criterion is inconsistent with the manuscript wording.

### DV quadrature table
Status: `match`

Claim:
- Optimal DV classical beta scan gives:
  - Dirichlet: `beta=0.60`, `M_DV=320`, `m_c=9`, `||c||_1=0.9357`, `F_DV=0.997666`
  - Neumann: `beta=0.90`, `M_DV=192`, `m_c=8`, `||c||_1=1.2073`, `F_DV=0.997740`
  - Periodic: `beta=0.80`, `M_DV=248`, `m_c=8`, `||c||_1=1.0740`, `F_DV=0.999722`

Validation:
- These values match `results_clean_dv_lchs_nwq_beta_scan_fine_eta1/best_by_boundary_beta_scan_eps0p1_eta1.csv` to rounding precision.

Conclusion:
- The DV quadrature-comparison table is consistent with the retained paper data.

### DV circuit-count table and comparison text
Status: `match`

Claim:
- Representative Dirichlet DV circuit in the nine-control-qubit regime uses:
  - ancilla PREP: depth 71, `1Q=130`, `CNOT=48`
  - one controlled-evolution slice: depth 91, `1Q=53`, `CNOT=47`
  - full 100-step circuit: depth 9227, `1Q=5561`, `CNOT=4796`

Validation:
- `results_clean_dv_lchs_dirichlet/default_summary.json` and `trotter100_summary.json` support these numbers.
- The per-op counts aggregate exactly as described in the manuscript:
  - one slice `1Q = rz 36 + p 13 + h 4 = 53`
  - one slice `CNOT-type = cx 38 + crz 9 = 47`
  - full circuit `1Q = 260 + 3600 + 1300 + 400 + 1 = 5561`
  - full circuit `CNOT-type = 3896 + 900 = 4796`

Conclusion:
- The DV circuit-comparison text and table are consistent with the retained implementation outputs.

### Abstract numerical summary
Status: `qualified match`

Claim:
- Abstract numbers include the 400 vs nearly 4800 CNOT comparison, the three ideal-loading fidelities above `0.9989`, the SNAP+D fidelity triplet, and the `M_DV = 320, 192, 248` comparison against `n_coeff=48`.

Validation:
- All abstract numbers match the retained paper-facing data and the circuit reconstructions used above.
- The same SNAP-depth selection caveat applies to the SNAP+D triplet because those values come from the saved `best_row` selection.

Conclusion:
- The abstract is numerically consistent with the retained data, subject to the same SNAP-depth wording issue noted above.

## Additional Notes

### OAT SNAP-depth reruns are not exact duplicates of `sweep_all.csv`
Status: `qualified match`

Observation:
- `WORKSPACE_GUIDE.md` describes `oat_*.csv` files as one-at-a-time reruns around the selected best row.
- For the SNAP folders, `oat_snap_depth.csv` does not numerically match `sweep_all.csv` at the same depth.

Explanation:
- `clean_oracles.py` uses unseeded random restarts for SNAP+D optimization.
- `clean_sweep.py` reruns the depth-continuation sweep when producing `oat_snap_depth.csv`.
- Because there is no fixed seed, those reruns are not guaranteed to reproduce the original `sweep_all.csv` numerically.

Implication:
- This is not, by itself, evidence that the code is wrong.
- It does mean `oat_snap_depth.csv` should be treated as a fresh rerun artifact, not as an exact replay of `sweep_all.csv`.

### `clean_sweep.py` CLI defaults are generic, not paper defaults
Status: `qualified match`

Observation:
- The CLI defaults in `clean_sweep.py` are generic toy grids, not the paper grids.
- The actual paper runs are represented by the retained result folders and their saved `grid_config`.

Implication:
- This is consistent with the repo being a reusable clean stack plus retained paper outputs.
- It should not be interpreted as evidence that the paper used the CLI defaults.

## Bottom Line
- If the question is "do the main paper tables and headline numbers in `res_base` agree with the retained paper-facing code and data in this repo?", the answer is mostly yes.
- The two paper-facing issues worth fixing are:
  - the SNAP+D depth-selection wording in `res_base/experiments.tex`
  - the broad initial sweep-domain claim, if the repo is expected to retain all supporting data for that claim
