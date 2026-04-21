# Workspace Guide

The minimal code and retained outputs needed to support the present paper
draft.

## Data-Backed Manuscript Map

Paths below are repo-relative. This table covers the parts of the current
`res_base/` manuscript that depend on retained generated data.

| Manuscript item                                                         | `res_base` location        | Retained source file(s)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Notes                                                                                                                                                                                                                                           |
| ----------------------------------------------------------------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Abstract fidelity claims                                                | `res_base/main.tex`        | `results_clean_givens_dirichlet/sweep_summary.json<br>``results_clean_givens_neumann/sweep_summary.json<br>``results_clean_givens_periodic/sweep_summary.json<br>``results_clean_snap_recreate_dirichlet/sweep_summary.json<br>``results_clean_snap_recreate_neumann/sweep_summary.json<br>``results_clean_snap_recreate_periodic/sweep_summary.json`                                                                                                                                                                      | The abstract's end-to-end Givens and `SNAP+D` fidelity statements come from these retained summaries.                                                                                                                                         |
| `tab:lchs_comparison`                                                 | `res_base/main.tex`        | `results_clean_givens_dirichlet/sweep_summary.json<br>``results_clean_dv_lchs_nwq_beta_scan_fine_eta1/best_by_boundary_beta_scan_eps0p1_eta1.csv<br>``results_clean_dv_lchs_dirichlet/trotter100_summary.json`                                                                                                                                                                                                                                                                                                             | Benchmark-level hybrid-versus-DV comparison for the representative 100-step Dirichlet example.                                                                                                                                                  |
| `fig:clean_sensitivity_exact`                                         | `res_base/experiments.tex` | `results_clean_refined_dirichlet/sweep_all.csv<br>``results_clean_refined_neumann/sweep_all.csv<br>``results_clean_refined_periodic/sweep_all.csv<br>``res_base/Pics/section6_sensitivity_exact.pdf`                                                                                                                                                                                                                                                                                                                       | The checked-in manuscript still imports this PDF. The same refined sweep CSVs also feed `clean_sensitivity_analysis.py`, which writes the alternative `results_clean_sensitivity/section6_sensitivity_landscape.{pdf,png}` figure.              |
| Bound-related discussion on `n_coeff` and first-order Trotter scaling | `res_base/experiments.tex` | `results_clean_bounds/section6_stateprep_consistency.csv<br>``results_clean_bounds/section6_trotter_tightness.csv`                                                                                                                                                                                                                                                                                                                                                                                                         | These CSVs support the prose discussion. The current manuscript does not import a separate bound-analysis figure asset.                                                                                                                         |
| `tab:clean_oracle_baseline`                                           | `res_base/experiments.tex` | `results_clean_refined_dirichlet/sweep_summary.json<br>``results_clean_refined_neumann/sweep_summary.json<br>``results_clean_refined_periodic/sweep_summary.json<br>``results_clean_givens_dirichlet/sweep_summary.json<br>``results_clean_givens_neumann/sweep_summary.json<br>``results_clean_givens_periodic/sweep_summary.json<br>``results_clean_snap_recreate_dirichlet/sweep_summary.json<br>``results_clean_snap_recreate_neumann/sweep_summary.json<br>``results_clean_snap_recreate_periodic/sweep_summary.json` | Optimal kernel parameters come from the refined injection sweeps; the Givens and `SNAP+D` table entries come from the per-boundary retained summaries at those kernel points.                                                                 |
| `tab:clean_resource_counts`                                           | `res_base/experiments.tex` | `results_clean_snap_recreate_dirichlet/sweep_summary.json<br>``results_clean_snap_recreate_neumann/sweep_summary.json<br>``results_clean_snap_recreate_periodic/sweep_summary.json<br>``results_clean_givens_dirichlet/sweep_summary.json<br>``results_clean_givens_neumann/sweep_summary.json<br>``results_clean_givens_periodic/sweep_summary.json`                                                                                                                                                                      | Optimizer iterations,`SNAP+D` layer counts, and Givens pulse/rotation counts are retained in the summaries. The split hybrid Trotter-block operator counts are stated directly in `res_base/experiments.tex` for the same benchmark points. |
| `tab:dv_resource_compare`                                             | `res_base/experiments.tex` | `results_clean_dv_lchs_nwq_beta_scan_fine_eta1/best_by_boundary_beta_scan_eps0p1_eta1.csv<br>``results_clean_givens_dirichlet/sweep_summary.json<br>``results_clean_givens_neumann/sweep_summary.json<br>``results_clean_givens_periodic/sweep_summary.json`                                                                                                                                                                                                                                                               | The DV quadrature parameters come from the retained `eta=1` beta scan; the final column uses the retained CV-DV Givens ceilings as the comparison baseline.                                                                                   |
| `tab:dv_circuit_compare`                                              | `res_base/experiments.tex` | `results_clean_dv_lchs_dirichlet/default_summary.json<br>``results_clean_dv_lchs_dirichlet/trotter100_summary.json`                                                                                                                                                                                                                                                                                                                                                                                                        | `default_summary.json` retains the ancilla-preparation block counts; `trotter100_summary.json` retains the repeated-slice circuit counts for the 100-step Dirichlet comparison.                                                             |

## Inline-Only Manuscript Content

The current `res_base/` also contains paper elements that are authored directly
in LaTeX and are not backed by separate retained CSV or JSON artifacts in this
trimmed repository.

| Manuscript item                                                                           | `res_base` location     | Notes                                                                            |
| ----------------------------------------------------------------------------------------- | ------------------------- | -------------------------------------------------------------------------------- |
| `fig:main circuit`, `fig:heat_h2_block`, `tab:heat-gate-scaling`, `alg:givens-le` | `res_base/main.tex`     | Theory and compilation content written directly in LaTeX.                        |
| `tab:lower_bounds`, `tab:upper_bounds`                                                | `res_base/Appendix.tex` | Analytical-versus-numerical coefficient tables written directly in the appendix. |

## Generation Commands

### Refined ideal-loading sweeps

Produces:

- `results_clean_refined_dirichlet`
- `results_clean_refined_neumann`
- `results_clean_refined_periodic`

```bash
RT="7.8,7.9,8.0,8.1,8.16,8.2,8.3,8.4"
RP="4.0,4.1,4.19,4.25,4.35,4.5"
BETA="0.3,0.5,0.6,0.675,0.725,0.75,0.8,0.9"

for bc in dirichlet neumann periodic; do
  python clean_sweep.py \
    --output-dir "results_clean_refined_${bc}" \
    --boundary-condition "$bc" \
    --num-qubits 2 \
    --n-fock 64 \
    --n-quad 240 \
    --r-target-grid "$RT" \
    --r-prime-grid "$RP" \
    --beta-grid "$BETA" \
    --n-coeff-grid 24,48 \
    --n-trotter-grid 100 \
    --prep-method-grid injection \
    --ranking-objective pde \
    --top-k 15
done
```

### Alternative Section 6 sensitivity figure

Produces:

- `results_clean_sensitivity/section6_sensitivity_landscape.png`
- `results_clean_sensitivity/section6_sensitivity_landscape.pdf`
- `results_clean_sensitivity/section6_surface_rows.csv`
- `results_clean_sensitivity/section6_beta_profile_rows.csv`

This figure uses the refined `injection` sweeps above. It plots exact
infidelity `1-F`, where `F = F(u_{\mathrm{alg}}, u_{\mathrm{exact}}) =
|\langle \hat u_{\mathrm{alg}} | \hat u_{\mathrm{exact}} \rangle|^2` is the
normalized overlap with the exact discretized solution. In the retained sweep
tables, this is the `fidelity` column populated from `fidelity_vs_exact`.

The left column shows `1-F` over `(r_target, r_prime)` at `n_coeff = 48` and
the boundary-specific `beta` that minimizes exact infidelity. The right column
shows `1-F` versus `beta` at fixed `(r_target, r_prime)`, with one fixed-squeeze
curve for each `n_coeff in {24, 48}`.

```bash
python clean_sensitivity_analysis.py
```

### Givens baselines

Produces:

- `results_clean_givens_dirichlet`
- `results_clean_givens_neumann`
- `results_clean_givens_periodic`

```bash
for entry in \
  "dirichlet,7.9,4.1,0.5,48" \
  "neumann,7.9,4.0,0.3,48" \
  "periodic,8.1,4.1,0.3,48"
do
  IFS=, read -r bc rt rp beta ncoeff <<< "$entry"
  python clean_sweep.py \
    --output-dir "results_clean_givens_${bc}" \
    --boundary-condition "$bc" \
    --num-qubits 2 \
    --n-fock 64 \
    --n-quad 240 \
    --r-target-grid "$rt" \
    --r-prime-grid "$rp" \
    --beta-grid "$beta" \
    --n-coeff-grid "$ncoeff" \
    --n-trotter-grid 100 \
    --prep-method-grid givens \
    --ranking-objective pde \
    --top-k 1
done
```

### Replayable `SNAP+D` sweeps

Produces:

- `results_clean_snap_recreate_dirichlet`
- `results_clean_snap_recreate_neumann`
- `results_clean_snap_recreate_periodic`

`sweep_all.csv` in these folders retains `snap_parameter_payload_json`, which is
the replayable optimized `SNAP+D` payload used by the current manuscript.

```bash
for entry in \
  "periodic,8.1,4.1,0.3,48" \
  "dirichlet,7.9,4.1,0.5,48" \
  "neumann,7.9,4.0,0.3,48"
do
  IFS=, read -r bc rt rp beta ncoeff <<< "$entry"
  python clean_sweep.py \
    --output-dir "results_clean_snap_recreate_${bc}" \
    --boundary-condition "$bc" \
    --num-qubits 2 \
    --n-fock 64 \
    --n-quad 240 \
    --r-target-grid "$rt" \
    --r-prime-grid "$rp" \
    --beta-grid "$beta" \
    --n-coeff-grid "$ncoeff" \
    --n-trotter-grid 100 \
    --prep-method-grid snap_d \
    --snap-depth-grid 4,6,8,10,14,18,30 \
    --snap-restarts 9 \
    --snap-maxiter 3000 \
    --ranking-objective oracle_only \
    --snap-warm-start-depths \
    --top-k 8
done
```

### Bound-analysis CSVs

Produces:

- `results_clean_bounds/section6_stateprep_consistency.csv`
- `results_clean_bounds/section6_trotter_tightness.csv`

`clean_bound_analysis.py` also writes
`results_clean_bounds/section6_trotter_tightness.pdf` and `.png`, but the
current `res_base/` manuscript does not import those files.

Reproducing `section6_trotter_tightness.csv` requires the historical
`results_clean_refine/sweep_all.csv`, which is not part of the trimmed retained
dataset.

```bash
python clean_bound_analysis.py \
  --trotter-sweep results_clean_refine/sweep_all.csv \
  --refined-sweeps results_clean_refined_dirichlet/sweep_all.csv,results_clean_refined_neumann/sweep_all.csv,results_clean_refined_periodic/sweep_all.csv \
  --output-dir results_clean_bounds
```

### Representative Dirichlet DV circuit summaries

Produces:

- `results_clean_dv_lchs_dirichlet/default_summary.json`
- `results_clean_dv_lchs_dirichlet/trotter100_summary.json`

This block uses the separate DV environment because
`clean_dv_lchs_dirichlet.py` depends on `scikit_tt` and the
`lchs-quantum-pde` stack.

```bash
conda activate dvlchs
python clean_dv_lchs_dirichlet.py \
  --beta 0.8 \
  --no-measure \
  --output-json results_clean_dv_lchs_dirichlet/default_summary.json
python clean_dv_lchs_dirichlet.py \
  --beta 0.8 \
  --num-time-steps 100 \
  --no-measure \
  --output-json results_clean_dv_lchs_dirichlet/trotter100_summary.json
```

### Fine classical DV beta scan with `eta=1`

Produces:

- `results_clean_dv_lchs_nwq_beta_scan_fine_eta1/*.json`
- `results_clean_dv_lchs_nwq_beta_scan_fine_eta1/summary_beta_scan_eps0p1_eta1.csv`
- `results_clean_dv_lchs_nwq_beta_scan_fine_eta1/best_by_boundary_beta_scan_eps0p1_eta1.csv`

```bash
python - <<'PY'
import csv
import json
from pathlib import Path

from clean_dv_lchs_nwq import (
    DEFAULT_NWQ_LCHS_SRC,
    build_nwq_dv_lchs_dirichlet,
)

root = Path("results_clean_dv_lchs_nwq_beta_scan_fine_eta1")
root.mkdir(parents=True, exist_ok=True)

betas = [round(0.60 + 0.05 * i, 2) for i in range(8)]
boundaries = ["dirichlet", "neumann", "periodic"]
rows = []

for bc in boundaries:
    for beta in betas:
        _circ, summary = build_nwq_dv_lchs_dirichlet(
            nwq_src=DEFAULT_NWQ_LCHS_SRC,
            num_system_qubits=2,
            boundary_condition=bc,
            total_time=1.0,
            alpha=1.0,
            grid_spacing=1.0,
            beta=beta,
            epsilon=0.1,
            trunc_multiplier=1.0,
            qiskit_api=False,
            trotter_lh=False,
            build_circuit=False,
            init_basis_index=1,
        )
        data = summary.__dict__.copy()
        out = root / f"{bc}_beta{beta:.2f}_eps0p1_eta1.json"
        out.write_text(json.dumps(data, indent=2) + "\n")
        rows.append(
            {
                "boundary_condition": bc,
                "beta": beta,
                "epsilon": data["epsilon"],
                "trunc_multiplier": data["trunc_multiplier"],
                "h1": data["h1"],
                "K": data["K"],
                "Q": data["Q"],
                "M_DV": data["M"],
                "m_c": data["control_qubits"],
                "coeff_l1_norm": data["coeff_l1_norm"],
                "classical_exact_fidelity": data["classical_exact_fidelity"],
                "classical_relative_error": data["classical_relative_error"],
            }
        )

rows.sort(key=lambda row: (row["boundary_condition"], row["beta"]))

summary_csv = root / "summary_beta_scan_eps0p1_eta1.csv"
with summary_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

best_rows = []
for bc in boundaries:
    candidates = [row for row in rows if row["boundary_condition"] == bc]
    best_rows.append(max(candidates, key=lambda row: row["classical_exact_fidelity"]))

best_csv = root / "best_by_boundary_beta_scan_eps0p1_eta1.csv"
with best_csv.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(best_rows[0].keys()))
    writer.writeheader()
    writer.writerows(best_rows)

for row in best_rows:
    print(row)
PY
```

## Core Files

| Path                           | Purpose                                                                                                                             |
| ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `clean_core.py`              | Reference mathematics, shared dataclasses, exact and truncated models, and heat-equation system builders.                           |
| `clean_oracles.py`           | CV oracle preparation for `injection`, `SNAP+D`, and `givens`, including replayable `SNAP+D` payload support.               |
| `clean_hybrid.py`            | Hybrid CV-DV circuit construction, Trotterized evolution, postselection, and fidelity/resource extraction.                          |
| `clean_sweep.py`             | Sweep driver used to generate the retained `injection`, `givens`, and `SNAP+D` datasets.                                      |
| `clean_sensitivity_analysis.py` | Builds the alternative Section 6 sensitivity figure from the retained refined `injection` sweeps.                              |
| `clean_bound_analysis.py`    | Rebuilds the retained Section 6 bound-analysis CSVs and optional bound-analysis figure from existing sweep outputs.                 |
| `clean_dv_lchs_nwq.py`       | Computes classical DV LCHS quadrature quantities such as `h1`, `K`, `Q`, `M_DV`, `m_c`, and classical fidelity summaries. |
| `clean_dv_lchs_dirichlet.py` | Builds the retained Dirichlet DV circuit summaries used for the circuit-comparison discussion.                                      |
| `clean_demo.ipynb`           | Reader-facing notebook that walks through a small live example and then loads the retained paper summaries.                         |
| `requirements-clean.txt`     | Minimal package list for the retained clean code path.                                                                              |

## Retained Result Directories

These folders are the retained outputs that are currently cited by the active
`res_base/` manuscript:

| Path                                              | Purpose                                                                      |
| ------------------------------------------------- | ---------------------------------------------------------------------------- |
| `results_clean_refined_dirichlet`               | Dirichlet ideal-loading parameter sweep.                                     |
| `results_clean_refined_neumann`                 | Neumann ideal-loading parameter sweep.                                       |
| `results_clean_refined_periodic`                | Periodic ideal-loading parameter sweep.                                      |
| `results_clean_givens_dirichlet`                | Dirichlet Givens baseline at the selected kernel point.                      |
| `results_clean_givens_neumann`                  | Neumann Givens baseline at the selected kernel point.                        |
| `results_clean_givens_periodic`                 | Periodic Givens baseline at the selected kernel point.                       |
| `results_clean_snap_recreate_dirichlet`         | Replayable Dirichlet `SNAP+D` sweep used by the manuscript.                |
| `results_clean_snap_recreate_neumann`           | Replayable Neumann `SNAP+D` sweep used by the manuscript.                  |
| `results_clean_snap_recreate_periodic`          | Replayable periodic `SNAP+D` sweep used by the manuscript.                 |
| `results_clean_bounds`                          | Bound-analysis CSVs supporting the experiments discussion.                   |
| `results_clean_dv_lchs_dirichlet`               | JSON summaries for the representative Dirichlet DV circuit comparison.       |
| `results_clean_dv_lchs_nwq_beta_scan_fine_eta1` | Fine classical DV beta scan used for the per-boundary optimal DV comparison. |

Additional retained outputs currently present in the repo but not cited by the
active `res_base/` draft include:

- `results_clean_sensitivity`
- `results_clean_dv_lchs_nwq`
- `results_clean_dv_lchs_nwq_beta_scan`
- `results_clean_dv_lchs_nwq_beta_scan_fine`

## What Is Kept Inside The Result Folders

For the retained sweep folders, the authoritative files are:

| File                   | Meaning                                            |
| ---------------------- | -------------------------------------------------- |
| `sweep_all.csv`      | Full evaluated sweep table.                        |
| `sweep_summary.json` | Best row plus summary metadata.                    |
| `oat_*.csv`          | One-at-a-time reruns around the selected best row. |

For the replayable `SNAP+D` sweep folders, `sweep_all.csv` also contains:

| Field                           | Meaning                                                                                    |
| ------------------------------- | ------------------------------------------------------------------------------------------ |
| `snap_parameter_payload_json` | Serialized optimized `SNAP+D` layer parameters for exact replay without re-optimization. |

For the DV comparison folder:

| File                        | Meaning                                                                                           |
| --------------------------- | ------------------------------------------------------------------------------------------------- |
| `default_summary.json`    | One-step Dirichlet DV LCHS benchmark summary, including kernel-preparation counts.                |
| `trotter100_summary.json` | Repeated-100-slice Dirichlet DV summary, including the retained circuit-count comparison numbers. |

For the fine classical DV beta-scan folder:

| File                                           | Meaning                                                                               |
| ---------------------------------------------- | ------------------------------------------------------------------------------------- |
| `*_beta*_eps0p1_eta1.json`                   | Per-boundary classical DV summaries across the retained `beta` scan with `eta=1`. |
| `summary_beta_scan_eps0p1_eta1.csv`          | Full retained scan table for all three boundary conditions.                           |
| `best_by_boundary_beta_scan_eps0p1_eta1.csv` | Best retained `beta` choice per boundary condition.                                 |
