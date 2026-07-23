# results_revision_v2/ — Paper data provenance (revised submission)

This directory is the authoritative artifact tree for **every number that
enters the revised manuscript**. All experiments use one shared operating
point, (r, r', beta, N) = (1.6, 0.25, 0.5, 32), with N_Fock = 64, T = 1,
n_t = 100, identical for all three boundary conditions. The selection record
is `../results_revision/selected_params_v2.json` (selection rule, input data
hashes, supersession note).

## Environments

- **All CV-chain and classical DV-chain artifacts**: miniconda **base**
  (python 3.12.4, qiskit 2.0.0, numpy 2.3.4, scipy 1.16.3,
  bosonic_qiskit 15.1). Every CSV has a `.meta.json` sidecar recording the git
  HEAD, package versions, and input hashes.
- **Single exception**: the DV circuit inventory (Section 9 below) runs in the
  legacy **cvdv** conda environment because the upstream MPS state preparation
  requires `scikit_tt`, which is absent from base. Gate counts are
  environment-independent integers.
- Test baseline: `python3 -m pytest -q` at the repo root -> **40 passed**
  (35 at the v2 freeze; +3 from `tests/test_2d_circuit.py`, Section 12;
  +2 from `tests/test_m32.py`, Section 13).

## Reproduction recipe (working-directory pattern)

The pipeline uses relative paths throughout. Reproduce in a **fresh
directory** so no frozen artifact is ever touched:

```bash
mkdir -p rerun/results_revision && cd rerun
cp ../results_revision/selected_params_v2.json results_revision/selected_params.json

# 1. Benchmark suite (Table 2, error-chain/worst-input paragraph, resource ledger)
python3 ../revision_eval.py finite-map --suite m4 --boundaries dirichlet,neumann,periodic \
  --params-file results_revision/selected_params.json --output results_revision/finite_map_m4.csv

# 2. Prerequisite for breadth: write the gate stub read by _gate_b_anchor,
#    with per-boundary eps_t taken from the eps_t column of step 1:
#    results_revision/gate_summary.json = [{"gate":"B","gate_evaluation":{"per_boundary":
#      {"dirichlet":{"eps_t":...},"neumann":{...},"periodic":{...}}}}]

# 3. Breadth / scaling / SNAP / addendum cases / dense sweep (figure source)
python3 ../revision_eval.py breadth  --output results_revision/breadth.csv
python3 ../revision_eval.py scaling  --output results_revision/scaling.csv
python3 ../revision_eval.py snap --n-coeff 16 --seed 11 --depth 30 --restarts 9 --maxiter 1000000 \
  --maxfun 100000000 --wall-time-seconds 14400 \
  --output results_revision/snap_N16_seed11.json    # same with --n-coeff 32
python3 ../make_breadth_addendum.py
python3 ../make_scaling_dense.py
```

## Paper element by element

### 1. Table 2 (tab:clean_oracle_baseline, LE benchmark) and the error-chain / worst-input paragraph
- **Artifacts**: `finite_map_m4.csv` (+ per-boundary `finite_map_*_raw.npz`).
- **Generation**: recipe step 1. eps_F, benchmark-input 1-F (per_input
  index 1), p, 1/p, eps_model/eps_synth/eps_t, and the worst-input columns are
  read directly from the CSV.
- Paper-facing (hbar = 2) values go through `paper_hbar2/finite_map_table.csv`
  (Section 10).

### 2. delta_nG and stellar rank (Table 2 caption)
- **Artifact**: `delta_ng_shared.json` (delta_nG = 1.3319, stellar rank 31;
  one shared kernel state, boundary-independent).
- **Generation**: `python3 clean_delta_ng_values.py --coefficients
  results_revision_v2/coefficients_dirichlet_r1p6_rp0p25_b0p5_N32_nq240.json
  --format json --output <out>.json`

### 3. Breadth table (tab:clean_breadth, seven circuit-level rows)
- **Artifacts**: `breadth.csv` (rows d8_1d_heat, heat_2d_4x4) and
  `breadth_addendum.csv` (rows d16_1d_heat_circuit, advdiff_d8, advdiff_d16;
  the kappa(V) and Henrici columns live here), plus one `breadth_*_raw.npz`
  per row. The two M = 32 rows (heat_d32, advdiff_d32) live in
  `breadth_m32.csv` (Section 13).
- **Generation**: CLI breadth (needs the gate stub) and
  `make_breadth_addendum.py`. Advection-diffusion definition: nu = c = h = 1,
  Dirichlet, central differences, mesh Peclet 0.5. Every row carries a dense
  product-formula wiring crosscheck (<= 1e-8).
- The synthetic non-normal case (row nonnormal_d4 in breadth.csv) predates the advection-diffusion family and is superseded by it as the non-normality test. It is not used in the paper and stays in the tree only because the frozen breadth.csv and its recorded hash were generated with it.

### 4. SNAP convergence-statistics paragraph (Section 7.1) and the R1-B response numbers
- **Artifacts**: `snap_N16_seed11.json` (fidelity 0.9999997, 3022 iterations, 1.7e6 function evaluations, 4 h) and `snap_N32_seed11.json` (0.999697, 1590 iterations, 1.7e6 function evaluations, 4 h).
- **Generation**: the two snap commands in the recipe (depth 30, restarts 9, seed 11, 14400 s wall-time budget split evenly over the nine restarts, per-restart evaluation cap lifted to 1e8).
- Every restart in both runs used its full 26.7 min share and stopped on the wall-time budget rather than on the L-BFGS-B convergence tests. These runs supersede the earlier 20 min artifacts (0.999407 and 0.97987), whose restarts stopped on the scipy default cap of 15000 objective evaluations, and use the same seed and hence the same nine starting points.
- Threshold crossings for the N16 run come from deterministic replays of its two sub-1e-6 starts (`snap_trace_replay_N16_idx6.json`, `snap_trace_replay_N16_idx7.json`, generated by `python3 ../replay_snap_trace.py --n-coeff 16 --start-index 6` and `... 7` in the working directory), validated by the replayed final objective matching the recorded one bit for bit. Start 7 first drops below 1e-6 infidelity at evaluation 108742 (14.9 min at that start's measured evaluation rate), start 6 at evaluation 169875 (23.6 min), so the sequential nine-start run first holds a sub-1e-6 point after 183.6 min. No other N16 start and no N32 start reaches 1e-6 within its share (the recorded per-start objective is the minimum over all its evaluations).
- Runs produced after the trace instrumentation carry `improvement_trace` and `decade_crossings` in every start record, so these questions read directly off the artifact without a replay.
- The best N32 start (index 4, 3.03e-4 after its 26.7 min share) was continued as a single line with no random restarts, in two warm-started segments (`snap_N32_seed11_warm2h.json`, 7200 s, then `snap_N32_seed11_warm3h.json`, 3600 s, each generated with `--restarts 0 --warm-start-from <previous artifact> --wall-time-seconds <budget>`). On the cumulative clock of that line it first drops below 1e-4 at 45.1 min, below 1e-5 at 98.8 min, below 1e-6 at 175.5 min, and ends at 8.67e-7 (fidelity 0.9999991) after 206.7 min, still descending at the final budget stop. The crossings inside the segments are recorded natively in `decade_crossings`. Each warm start resets the L-BFGS-B curvature memory, so the cumulative time is that of a segmented continuation of one line, not of one uninterrupted run.

### 5. Resource table (tab:clean_resource_counts, LE-only)
- **Artifact**: the `resource_ledger` column of `finite_map_m4.csv`
  (four-segment ledger, reconciled against compiled circuits).
- JC/R = 31/31; the four Trotter columns are C_run's h+rz / cx / D / cD.

### 6. Sensitivity figure (figures/sensitivity_landscape_v2.pdf)
- **Data**: `../results_revision/viz_dense/landscape_dense.csv` and
  `landscape_dense_ext.csv` (the coarse grid search itself, >= 8 values per
  axis; cross-validated bitwise, 243/243, against the registered subgrid).
  Raw maps behind every valid row are frozen in the companion
  `landscape_dense{,_ext}_raw.npz` (per-boundary stacks aligned with
  `grid_index`; sha256 in the meta sidecars).
- **Generation**: `python3 make_landscape_dense.py [grid.json out.csv]`
  (per-point try/except for vanishing zeroth-moment overlap), then
  `python3 make_sensitivity_figure.py`. Figure meta sidecars are in
  `../results_revision/paper_hbar2/`.

### 7. Scaling figure (figures/scaling_study.pdf) and scaling text numbers
- **Data**: `viz_dense/scaling_dense.csv` (eight T values x three cutoff pairs
  at exact-map level; eight n_t values at T = 1 at circuit level) and
  `scaling.csv` (CLI version, 15 rows). Every row carries a `raw_artifact`
  npz: `scaling_dense_T*_N*_NF*_exact_raw.npz` for the exact-map rows,
  `scaling_dense_T1.0_N*_NF*_nt*_raw.npz` for the circuit rows.
- **Generation**: `make_scaling_dense.py` (working directory), then
  `python3 make_scaling_figure.py`.

### 8. New rows of the DV quadrature table (tab:dv_resource_compare)
- **Artifact**: `dv_extension.csv`.
- **Generation**: `python3 -W ignore make_dv_extension.py` (base environment).
  **Validation gate**: the M = 4 rows must reproduce the published table
  digit-for-digit (320/192/248 and 2.334e-3/2.260e-3/2.777e-4) or the script
  raises without producing new rows. beta scan [0.60, 0.95], epsilon = 0.1,
  eta = 1, u0 = basis index 1; for advection-diffusion, H != 0 enters
  exp(-i T (k L + H)).
- The two M = 32 rows of the same table live in `dv_extension_m32.csv`
  (Section 13), generated behind the same M = 4 gate.

### 9. New rows of the DV circuit table (tab:dv_circuit_compare, heat M = 8/16) and the cost-structure paragraph (CO-7b)
- **Artifacts**: `dv_circuit_extension.csv`,
  `dv_circuit_m{8,16}_summary.json`, and `dv_slice_anatomy.md` (gate-spectrum
  anatomy and the O(m_c log^2 M) argument).
- `dv_circuit_full_depths.json` (generated by `../make_dv_full_depths.py`, cvdv environment) holds directly measured full-circuit depths of the complex-kernel circuits with all 100 slices instantiated, in the {1Q, CX} basis with each CRZ decomposed. The measured depths are 12874, 36975, 77466, 132964 for M = 4, 8, 16, 32, equal at every size to the one- and two-slice extrapolation previously quoted in the table, and the full-circuit 1Q and CX counts reproduce the published inventory as a hard gate.
- `dv_circuit_opt2_transpile.json` (generated by `../make_dv_transpile_opt2.py`, cvdv environment) transpiles the same full circuits with Qiskit at optimization level 2 in the {U, CX} basis, fixed transpiler seed, no coupling map. The CX count is unchanged at every size (hard gate), so the published identity-decomposition CX counts already coincide with the optimized ones, while one-qubit counts shrink by 8 to 27 percent and depths by 5 to 22 percent through adjacent one-qubit merging.
- **Generation** (cvdv environment plus the two upstream repos):
  ```bash
  PY=/Users/zhen002/miniconda3/envs/cvdv/bin/python
  NWQ=/Users/zhen002/GitHub/nwtoolchain/prototypes_nwqlib/prototypes/lchs/src
  $PY clean_dv_lchs_dirichlet.py --nwq-src $NWQ --num-system-qubits 3 \
    --kernel parametrized --kernel-projection positive_real --beta 0.6 --num-layers 2 \
    --num-time-steps 100 --initial-basis-index 1 --no-measure --nwq-trunc-multiplier 1.0 \
    --output-json results_revision_v2/dv_circuit_m8_summary.json   # same with 4 -> m16
  $PY -W ignore make_dv_circuit_ext.py
  ```
  `make_dv_circuit_ext.py` builds the n_t = 1 and n_t = 2 circuits, decomposes
  multi-controlled gates to the {1Q, CX, CRZ} reporting basis, differences out
  the per-slice and PREP+unPREP costs, and extrapolates to 100 slices.
  **Validation gate**: M = 4 must reproduce 53/38/9 (slice) and 5561/3896/900
  (full circuit). The M = 4 depth 9227 is printed by this script but gated
  only in `make_m32_dv_circuit.py` (Section 13), which also produces the
  M = 32 column of the table.

### 10. Paper-convention (hbar = 2) exports — manuscript source for the CV finite-map / breadth / scaling tables
- **Artifacts**: `paper_hbar2/{finite_map,breadth,scaling,breadth_addendum}_table.csv`.
- **Generation** (inside the working directory):
  `python3 ../revision_eval.py export-paper --table <t> --input <src>
  --output results_revision/paper_hbar2/<t>_table.csv`
- Scope: these exports feed Table 2, the M <= 16 breadth rows, and the
  scaling table. The DV comparison tables (Sections 8, 9, 13), the 2D
  full-circuit row (Section 12), and the M = 32 breadth rows (Section 13)
  are read directly from their artifacts: those entries are gate counts,
  percentages, and fidelities, which the hbar convention does not touch.
- Do not confuse these with the same-named old-point exports under
  `../results_revision/paper_hbar2/`.

## Miscellaneous

- `logs/` — run logs plus `postfix_spotcheck_dirichlet.csv` (invariance spot
  check after the Y basis-change fix; differences are at the floating-point
  last-ulp level only). The fix is documented in the basis-change comment in
  `clean_hybrid.py` and by the unit test
  `test_dv_pauli_rotation_matches_dense_for_odd_and_even_y`.
- `coefficients_*.json` — frozen coefficient files per cutoff (write-once,
  hashed).
- The Gate A convention evidence (`convention.json`) and the coefficient
  backend justification (`coefficient_backend.csv`) live in
  `../results_revision/`; both are inherited unchanged by this tree.

### 11. Oscillator photon-loss analysis (noise subsection and fig:noise_loss)
- **Artifacts**: `noise_loss.csv` (+ `noise_loss.meta.json`); figure
  `figures/noise_loss.pdf`.
- **Generation**: `python3 -W ignore make_noise_loss.py` (base environment),
  then `python3 make_noise_figure.py`.
- **Model**: one lumped photon-loss channel on the prepared kernel state
  before the H = 0 evolution, evaluated by exact Kraus summation through the
  postselected-map machinery (no Lindblad simulation). For the heat
  benchmarks this endpoint model is exhaustive for oscillator loss up to a
  deterministic, compensable rescaling of the effective evolution time
  (displacement covariance of the loss channel; exact at H = 0).
- **Validation gates**: at gamma = 0 the branch-0 map reproduces
  `exact_truncated_cv_map_h0` to machine precision for every boundary, and
  the ideal fixed-scale errors match the selection-grid values
  (0.6694/0.4838/0.4691 percent).
- Key quantities: kernel-state photon moments <n> = 1.6068,
  sqrt(<n^2>) = 6.7170; first-order distortion bound
  (gamma/2) sqrt(<n^2>) verified with a 3-7x margin.

### 12. 2D full-circuit upgrade (breadth 2D row and the DV-comparison 2D gap)
- **Artifacts**: `breadth_2d_circuit.csv` (+ `breadth_2d_circuit.meta.json`);
  raw payload `breadth_2d_4x4_circuit_raw.npz` (target/prepared/circuit maps,
  per-input wiring errors, probabilities, per-column wall times).
- **Generation** (base environment, fresh working directory; total circuit
  wall time ~133 s):
  ```bash
  mkdir -p workdir_2d/results_revision
  cp results_revision_v2/coefficients_dirichlet_r1p6_rp0p25_b0p5_N32_nq240.json \
     workdir_2d/results_revision/
  cp results_revision/selected_params_v2.json workdir_2d/results_revision/
  cp results_revision/selected_params_v2.json \
     workdir_2d/results_revision/selected_params.json
  cp results_revision/gate_summary.json workdir_2d/results_revision/
  cd workdir_2d && python3 ../make_2d_circuit.py
  ```
- **What it is**: the frozen `breadth.csv` row `heat_2d_4x4` (exact finite map
  plus one circuit-level anchor at basis input 5) extended to a full
  16-column circuit-level map through the shared `reconstruct_circuit_map` /
  `run_clean_lchs` flow. The frozen `breadth.csv` row is unchanged and
  remains the exact-map record.
- **Validation gates** (enforced by `make_2d_circuit.py`): per-input dense
  product-formula wiring crosscheck max 1.41e-13 against the 1e-8 bound;
  basis-5 anchor continuity exact to all printed digits
  (eps_t_2d = 3.552982606303818e-4, relative 2.6103688480868586e-3). Two
  further checks were performed manually in the job-6 audit, not by the
  script: per-input postselection probabilities within 0.65 percent relative
  of the frozen exact-map values, and compiled per-input resources identical
  to the frozen row.
- **Paper mapping**: the breadth-table 2D row reads eps_F = 7.40 percent,
  worst 1-F = 6.20e-3, p range 0.48--1.85 percent, and measured
  eps_t = 8.07e-4 from this artifact (`eps_t_kind =
  full_circuit_map_measured`). The `tab:dv_resource_compare` 2D fidelity gap
  F_LE - F_DV = 3.58e-3 combines `one_minus_F_DV` = 9.0766e-3 from
  `dv_extension.csv` with the circuit-level basis-1 conditional infidelity
  5.4948e-3 from this artifact. It supersedes the exact-map-based 3.5545e-3,
  which combined the same `one_minus_F_DV` with the exact-map basis-1
  conditional infidelity 5.5222e-3 from the frozen `breadth.csv` per-input
  record (`dv_extension.csv` itself stores no gap column).
- Ledger: `../results_revision/gate_summary.json`, entry
  `breadth-2D-circuit-addendum`. 
  
### 13. M = 32 extension (breadth rows, DV quadrature rows, DV circuit column)
- **Artifacts**: `breadth_m32.csv` (+ `breadth_m32.meta.json`; rows heat_d32
  and advdiff_d32) with raw payloads `breadth_d32_raw.npz` and
  `breadth_advdiff_d32_raw.npz` (target/prepared/circuit maps, per-input
  wiring errors, probabilities, per-column wall times);
  `dv_extension_m32.csv` (+ `.meta.json`; classical DV quadrature rows
  heat_m32_dirichlet and advdiff_m32); `dv_circuit_extension_m32.csv`
  (+ `.meta.json`) and `dv_circuit_m32_summary.json` (M = 32 heat DV
  circuit inventory).
- **Generation** (fresh working directory; the breadth script hash-checks
  the seed copy-ins below against pinned sha256 values before running;
  circuit wall time ~23 min per breadth case):
  ```bash
  mkdir -p workdir_m32/results_revision
  cp results_revision_v2/coefficients_dirichlet_r1p6_rp0p25_b0p5_N32_nq240.json \
     workdir_m32/results_revision/
  cp results_revision/selected_params_v2.json workdir_m32/results_revision/
  cp results_revision/selected_params_v2.json \
     workdir_m32/results_revision/selected_params.json
  cp results_revision/gate_summary.json workdir_m32/results_revision/
  cd workdir_m32
  python3 ../make_m32_breadth.py                 # base environment
  python3 -W ignore ../make_m32_dv_extension.py  # base environment
  /Users/zhen002/miniconda3/envs/cvdv/bin/python -W ignore \
      ../make_m32_dv_circuit.py                  # cvdv environment (Section 9)
  ```
  All three scripts write into `results_revision/` under the working
  directory; the artifacts above were promoted from there byte-identical.
- **Validation gates** (all enforced in-script): breadth heat_d32 — eps_F
  1.4554 percent inside [0.8, 5.0] percent, p_min 5.27 percent >= 2 percent,
  per-input resource ledger exactly 386/196/1/31 (1q/cx/D/cD) per Trotter
  step and 38600/19600/100/3100 over 100 steps, eps_synth <= 1e-12, wiring
  crosscheck max 2.01e-12 against the 1e-8 bound; advdiff_d32 — advisory
  eps_F 1.4706 percent inside [0.5, 15] percent, eps_synth <= 1e-12, wiring
  max 4.85e-12; spectral pins ||T_32||_2 = 2 + 2 cos(pi/33),
  lambda_min = 2 - 2 cos(pi/33), and Pauli round trips <= 1e-10 (also unit
  tests in `tests/test_m32.py`); DV quadrature — the M = 4 rows must
  reproduce the published table digit-for-digit (320/192/248 and
  2.334e-3/2.260e-3/2.777e-4) before the M = 32 rows are written, then
  M_DV inside [350, 370] and m_c = 9; DV circuit — M = 4 must reproduce
  53/38/9 (slice), 5561/3896/900 (full circuit), and depth 9227, then
  M = 32 must give LCU width 9, slice CRZ = 9, and PREP+unPREP
  261 1Q + 96 CX + 0 CRZ. The M = 8/16 row continuity (heat eps_F
  1.0439/1.4577/1.4554 percent, advdiff 1.1519/1.4843/1.4706 percent at
  M = 8/16/32) was checked in the job-7 audit, not by the scripts.
- **Paper mapping**: the two breadth-table M = 32 rows — heat eps_F 1.46
  percent, worst 1-F 2.58e-4, p 5.27--12.47 percent, measured eps_t
  1.78e-3; advection-diffusion eps_F 1.47 percent, worst 1-F 3.51e-4,
  p 2.68--12.46 percent, eps_t 2.07e-3, kappa(V) 2.63e7, Henrici 0.3884
  (columns `kappa_eigenvector` / `henrici_departure`; these also feed the
  kappa(V)-growth sentence). `tab:dv_resource_compare` M = 32 rows read
  beta_opt / h1 / K / Q / M_DV / m_c / ||c||_1 / one_minus_F_DV from
  `dv_extension_m32.csv`; their fidelity-gap column combines one_minus_F_DV
  (5.00e-3 heat, 4.10e-3 advdiff) with the circuit-level basis-1
  conditional infidelities from the `breadth_m32.csv` per-input records
  (1.97e-4 and 1.93e-4), giving 4.81e-3 and 3.91e-3.
  `tab:dv_circuit_compare` M = 32 column and the cost-structure paragraph
  read slice 1127/758/9, PREP+unPREP 261/96/0, full-circuit
  112961/75896/900, and depth ~129317 from `dv_circuit_extension_m32.csv`
  (the measured slice CX 758 equals the prior m = 5 probe, delta 0). The
  hybrid-side ledger sentence (38600/19600/100/3100 at M = 32) reads the
  gated resource ledger of the heat_d32 row.
- The `.meta.json` sidecars record working-directory-relative output paths
  and generation-time script hashes. `make_m32_dv_circuit.py` was edited
  after generation to write into the working directory (its output path was
  previously anchored to a repo-local `workdir_m32/`), so its recorded
  source hash refers to the generation-time version in git history.
- Ledger: `../results_revision/gate_summary.json`, entry
  `breadth-dv-m32-extension`. 

### 14. Complex-kernel DV circuit inventory and dyadic-grid kernel fidelity (tab:dv_circuit_compare, removal of the projection caveat)
- **Artifacts**: `dv_circuit_extension_complex_kernel.csv` (+ `.meta.json`; heat M = 4/8/16/32 inventory with the complex kernel realized exactly, magnitude amplitudes plus the ancilla phase diagonal, three-type {1Q, CX, CRZ} basis), `dv_circuit_extension_complex_kernel_cx.csv` (+ `.meta.json`; the same circuits in the two-type {1Q, CX} basis, each CRZ decomposed via its standard identity into two CX and two rotations, depths measured in that basis; this is the paper-facing version), and `dv_kernel_fidelity.csv` (+ `.meta.json`; conditional fidelity of the dyadic-grid LCHS map for the complex kernel and for the positive-real projection, exact branch propagators).
- **Generation** (repo root):
  ```bash
  /Users/zhen002/miniconda3/envs/cvdv/bin/python -W ignore \
      make_dv_circuit_complex_kernel.py      # cvdv environment (Section 9)
  /Users/zhen002/miniconda3/envs/cvdv/bin/python -W ignore \
      make_dv_circuit_complex_kernel_cx.py   # cvdv environment, reads the CSV above
  python3 make_dv_kernel_fidelity.py         # base environment
  ```
- **Validation gates** (in-script): per-slice counts must equal the phase-free inventory at every size (53/38/9, 255/168/9, 619/390/9, 1127/758/9), LCU width 9; the cx-basis script requires every slice and PREP+unPREP count to equal the three-type value with CX + 2 crz and 1Q + 2 crz exactly; the fidelity script requires the k grids read from the four circuit summaries to be identical and equal to the closed form (512 nodes, lsb_pos = -3, k in [-32, 31.875]). Direct M = 4 builds at n_t = 100 reproduce the extrapolated counts and depth exactly in both bases (6072/4406/900 at depth 10196, and 7872/6206 at depth 12874). Unit tests: `tests/test_dv_complex_kernel.py` (slice parity, the constant 511/510 diagonal overhead, full-circuit arithmetic, the CRZ-conversion identities, and the fidelity separation).
- **Results**: the phase layer is one diagonal on the 9 ancilla qubits, applied once between the last slice and unPREP, so PREP+unPREP+phase = 772 1Q + 606 CX at every size (511 rotations and 510 CX over the phase-free 261/96/0) in both bases, since the diagonal contains no CRZ. Three-type basis: slice 53/38/9, 255/168/9, 619/390/9, 1127/758/9 and full circuit 6072/4406/900, 26272/17406/900, 62672/39606/900, 113472/76406/900 at depth ~10196/34297/74788/130286. Two-type basis (paper-facing, CRZ = 2 CX + 2 rotations): slice 71/56, 273/186, 637/408, 1145/776 and full circuit 7872/6206, 28072/19206, 64472/41406, 115272/78206 at depth ~12874/36975/77466/132964. Dyadic-grid kernel fidelity at beta = 0.6: complex kernel 1-F = 4.37e-8 / 4.56e-7 / 7.19e-7 / 6.92e-7 at M = 4/8/16/32 with ||c||_1 = 1.1664, positive-real projection 8.64e-3 / 1.61e-2 / 2.04e-2 / 2.03e-2 with ||c||_1 = 1.0657. Branch propagators are exact matrix exponentials, so these values isolate the kernel-quadrature error and bound the circuit-level fidelity from above (Trotter and PREP-compression error enter separately, as elsewhere in the study).
- **Paper mapping**: `tab:dv_circuit_compare` reads slice / PREP+unPREP / full-circuit / depth from `dv_circuit_extension_complex_kernel_cx.csv` (the caption states the CRZ = 2 CX + 2 rotations convention), the Trotter-block CX ratio sentence becomes 5600/400 = 14 at M = 4 narrowing to 77600/19600 = 4.0 at M = 32, and the DV comparison text states the matched complex-kernel baseline directly (dyadic-grid map reproduces exp(-AT)u0 to 1-F <= 7.2e-7 across all four sizes), which removes the sentence presenting the comparison as an architecture-dependent inventory with phases disabled.
