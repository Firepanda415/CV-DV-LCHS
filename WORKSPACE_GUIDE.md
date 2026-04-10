# Workspace Guide

This repository has been reduced to the files needed for the current paper and
for reproducing the retained paper data.

## Paper Data Source Map

Paths below are repo-relative. This section points readers to the retained
source file(s) for each paper-facing dataset used in `res_base`.

| Paper-facing item                                                               | Source file(s)                                                                                                                                                                                                                                                                                                                                                                                                                                                                | Notes                                                                                                                                                       |
| ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Abstract ideal-loading fidelity claims and Table `clean_oracle_baseline`      | `GitHub/CV-DV-LCHS/results_clean_refined_dirichlet/sweep_summary.json<br>``GitHub/CV-DV-LCHS/results_clean_refined_neumann/sweep_summary.json<br>``GitHub/CV-DV-LCHS/results_clean_refined_periodic/sweep_summary.json`                                                                                                                                                                                                                                                 | Best retained ideal-loading rows for the three boundary conditions.                                                                                         |
| Figure `section6_sensitivity_exact`                                           | `GitHub/CV-DV-LCHS/results_clean_refined_dirichlet/sweep_all.csv<br>``GitHub/CV-DV-LCHS/results_clean_refined_neumann/sweep_all.csv<br>``GitHub/CV-DV-LCHS/results_clean_refined_periodic/sweep_all.csv<br>``GitHub/CV-DV-LCHS/res_base/Pics/section6_sensitivity_exact.pdf`                                                                                                                                                                                          | The CSV files are the retained sweep data; the PDF is the figure included in the manuscript.                                                                |
| Bound-related discussion on `n_coeff` and Trotter scaling                     | `GitHub/CV-DV-LCHS/results_clean_bounds/section6_stateprep_consistency.csv<br>``GitHub/CV-DV-LCHS/results_clean_bounds/section6_trotter_tightness.csv`                                                                                                                                                                                                                                                                                                                    | Supports the empirical checks discussed in the experiments section.                                                                                         |
| Givens rows in Table `clean_snap_comparison` and Givens pulse/rotation counts | `GitHub/CV-DV-LCHS/results_clean_givens_dirichlet/sweep_summary.json<br>``GitHub/CV-DV-LCHS/results_clean_givens_neumann/sweep_summary.json<br>``GitHub/CV-DV-LCHS/results_clean_givens_periodic/sweep_summary.json`                                                                                                                                                                                                                                                    | Per-boundary retained Givens summaries at the selected kernel points.                                                                                       |
| SNAP+D rows in Table `clean_snap_comparison`                                  | `GitHub/CV-DV-LCHS/results_clean_snap_recreate_dirichlet/sweep_summary.json<br>``GitHub/CV-DV-LCHS/results_clean_snap_recreate_neumann/sweep_summary.json<br>``GitHub/CV-DV-LCHS/results_clean_snap_recreate_periodic/sweep_summary.json`                                                                                                                                                                                                                               | Per-boundary retained replayable `SNAP+D` summaries.                                                                                                      |
| Replayable SNAP+D parameter payloads                                            | `GitHub/CV-DV-LCHS/results_clean_snap_recreate_dirichlet/sweep_all.csv<br>``GitHub/CV-DV-LCHS/results_clean_snap_recreate_neumann/sweep_all.csv<br>``GitHub/CV-DV-LCHS/results_clean_snap_recreate_periodic/sweep_all.csv`                                                                                                                                                                                                                                              | The `snap_parameter_payload_json` field stores the replayable optimized parameters.                                                                       |
| Table `clean_resource_counts`                                                 | `GitHub/CV-DV-LCHS/results_clean_snap_recreate_dirichlet/sweep_summary.json<br>``GitHub/CV-DV-LCHS/results_clean_snap_recreate_neumann/sweep_summary.json<br>``GitHub/CV-DV-LCHS/results_clean_snap_recreate_periodic/sweep_summary.json<br>``GitHub/CV-DV-LCHS/results_clean_givens_dirichlet/sweep_summary.json<br>``GitHub/CV-DV-LCHS/results_clean_givens_neumann/sweep_summary.json<br>``GitHub/CV-DV-LCHS/results_clean_givens_periodic/sweep_summary.json` | Optimizer iterations and selected depths are in the summaries; the hybrid Trotter counts are reproduced from `clean_hybrid.py` at the same kernel points. |
| Table `dv_resource_compare` and abstract DV `M_{\mathrm{DV}}` values        | `GitHub/CV-DV-LCHS/results_clean_dv_lchs_nwq_beta_scan_fine_eta1/best_by_boundary_beta_scan_eps0p1_eta1.csv`                                                                                                                                                                                                                                                                                                                                                                | Best retained classical DV quadrature result per boundary condition.                                                                                        |
| Table `dv_circuit_compare` and representative Dirichlet DV circuit discussion | `GitHub/CV-DV-LCHS/results_clean_dv_lchs_dirichlet/default_summary.json<br>``GitHub/CV-DV-LCHS/results_clean_dv_lchs_dirichlet/trotter100_summary.json`                                                                                                                                                                                                                                                                                                                   | One-step and repeated-100-slice Dirichlet DV circuit summaries.                                                                                             |

## Generation Commands

Refined Ideal-Loading Sweeps

Produces:

- `GitHub/CV-DV-LCHS/results_clean_refined_dirichlet`
- `GitHub/CV-DV-LCHS/results_clean_refined_neumann`
- `GitHub/CV-DV-LCHS/results_clean_refined_periodic`

```bash
RT="7.8,7.9,8.0,8.1,8.16,8.2,8.3,8.4" && RP="4.0,4.1,4.19,4.25,4.35,4.5" && BETA="0.3,0.5,0.6,0.675,0.725,0.75,0.8,0.9" && for bc in dirichlet neumann periodic; do python clean_sweep.py --output-dir "results_clean_refined_${bc}" --boundary-condition "$bc" --num-qubits 2 --n-fock 64 --n-quad 240 --r-target-grid "$RT" --r-prime-grid "$RP" --beta-grid "$BETA" --n-coeff-grid 24,48 --n-trotter-grid 100 --prep-method-grid injection --ranking-objective pde --top-k 15; done
```

### Givens Baselines

Produces:

- `GitHub/CV-DV-LCHS/results_clean_givens_dirichlet`
- `GitHub/CV-DV-LCHS/results_clean_givens_neumann`
- `GitHub/CV-DV-LCHS/results_clean_givens_periodic`

```bash
for entry in "dirichlet,7.9,4.1,0.5,48" "neumann,7.9,4.0,0.3,48" "periodic,8.1,4.1,0.3,48"; do IFS=, read -r bc rt rp beta ncoeff <<< "$entry"; python clean_sweep.py --output-dir "results_clean_givens_${bc}" --boundary-condition "$bc" --num-qubits 2 --n-fock 64 --n-quad 240 --r-target-grid "$rt" --r-prime-grid "$rp" --beta-grid "$beta" --n-coeff-grid "$ncoeff" --n-trotter-grid 100 --prep-method-grid givens --ranking-objective pde --top-k 1; done
```

### Replayable SNAP+D Sweeps

Produces:

- `GitHub/CV-DV-LCHS/results_clean_snap_recreate_dirichlet`
- `GitHub/CV-DV-LCHS/results_clean_snap_recreate_neumann`
- `GitHub/CV-DV-LCHS/results_clean_snap_recreate_periodic`

`sweep_all.csv` in these folders retains `snap_parameter_payload_json`, which is the replayable optimized `SNAP+D` payload.

```bash
for entry in "periodic,8.1,4.1,0.3,48" "dirichlet,7.9,4.1,0.5,48" "neumann,7.9,4.0,0.3,48"; do IFS=, read -r bc rt rp beta ncoeff <<< "$entry"; python clean_sweep.py --output-dir "results_clean_snap_recreate_${bc}" --boundary-condition "$bc" --num-qubits 2 --n-fock 64 --n-quad 240 --r-target-grid "$rt" --r-prime-grid "$rp" --beta-grid "$beta" --n-coeff-grid "$ncoeff" --n-trotter-grid 100 --prep-method-grid snap_d --snap-depth-grid 4,6,8,10,14,18,30 --snap-restarts 9 --snap-maxiter 3000 --ranking-objective oracle_only --snap-warm-start-depths --top-k 8; done
```

### Bound-Analysis CSVs

Produces:

- `GitHub/CV-DV-LCHS/results_clean_bounds/section6_stateprep_consistency.csv`
- `GitHub/CV-DV-LCHS/results_clean_bounds/section6_trotter_tightness.csv`

The recovered `clean_bound_analysis.py` reads existing sweep outputs and writes fresh CSVs under `results_clean_bounds`. Reproducing `section6_trotter_tightness.csv` also requires the historical Trotter sweep file `results_clean_refine/sweep_all.csv`, which is not part of the trimmed retained dataset.

```bash
python clean_bound_analysis.py --trotter-sweep results_clean_refine/sweep_all.csv --refined-sweeps results_clean_refined_dirichlet/sweep_all.csv,results_clean_refined_neumann/sweep_all.csv,results_clean_refined_periodic/sweep_all.csv --output-dir results_clean_bounds
```

### Representative Dirichlet DV Circuit Summaries

Produces:

- `GitHub/CV-DV-LCHS/results_clean_dv_lchs_dirichlet/default_summary.json`
- `GitHub/CV-DV-LCHS/results_clean_dv_lchs_dirichlet/trotter100_summary.json`

This block uses the separate DV environment because `clean_dv_lchs_dirichlet.py` depends on `scikit_tt` and the `lchs-quantum-pde` stack.

```bash
conda activate dvlchs && python clean_dv_lchs_dirichlet.py --beta 0.8 --no-measure --output-json results_clean_dv_lchs_dirichlet/default_summary.json && python clean_dv_lchs_dirichlet.py --beta 0.8 --num-time-steps 100 --no-measure --output-json results_clean_dv_lchs_dirichlet/trotter100_summary.json
```

### Fine Classical DV Beta Scan With `eta=1`

Produces:

- `GitHub/CV-DV-LCHS/results_clean_dv_lchs_nwq_beta_scan_fine_eta1/*.json`
- `GitHub/CV-DV-LCHS/results_clean_dv_lchs_nwq_beta_scan_fine_eta1/summary_beta_scan_eps0p1_eta1.csv`
- `GitHub/CV-DV-LCHS/results_clean_dv_lchs_nwq_beta_scan_fine_eta1/best_by_boundary_beta_scan_eps0p1_eta1.csv`

```bash
python - <<'PY'
import csv
import json
from pathlib import Path
from clean_dv_lchs_nwq import build_nwq_dv_lchs_dirichlet, DEFAULT_NWQ_LCHS_SRC

root = Path("/GitHub/CV-DV-LCHS/results_clean_dv_lchs_nwq_beta_scan_fine_eta1")
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

## Core Scripts

| Path                                             | Purpose                                                                                                                      |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| `GitHub/CV-DV-LCHS/clean_core.py`              | Reference mathematics, shared dataclasses, exact/truncated models, and heat-equation system builders.                        |
| `GitHub/CV-DV-LCHS/clean_oracles.py`           | CV oracle preparation for `injection`, `SNAP+D`, and `givens`, including replayable `SNAP+D` payloads.               |
| `GitHub/CV-DV-LCHS/clean_hybrid.py`            | Hybrid CV-DV circuit construction, Trotterized evolution, postselection, and fidelity/resource extraction.                   |
| `GitHub/CV-DV-LCHS/clean_sweep.py`             | Sweep driver used to generate the retained `injection`, `givens`, and `SNAP+D` datasets.                               |
| `GitHub/CV-DV-LCHS/clean_bound_analysis.py`    | Rebuilds the retained Section 6 bound-analysis CSVs and figure from existing sweep outputs.                                  |
| `GitHub/CV-DV-LCHS/clean_dv_lchs_nwq.py`       | Computes DV LCHS quadrature quantities such as `h_1`, `K`, `Q`, `M_DV`, `m_c`, and `                               |
| `GitHub/CV-DV-LCHS/clean_dv_lchs_dirichlet.py` | Builds the practical Dirichlet DV LCHS circuit summaries used for the DV comparison.                                         |
| `GitHub/CV-DV-LCHS/clean_demo.ipynb`           | Reader-facing notebook that walks through the algorithm on a small live example and then loads the retained paper summaries. |
| `GitHub/CV-DV-LCHS/requirements-clean.txt`     | Minimal package list for the retained clean code path.                                                                       |

## Manuscript Files

| Path                                                               | Purpose                                                           |
| ------------------------------------------------------------------ | ----------------------------------------------------------------- |
| `GitHub/CV-DV-LCHS/res_base/main.tex`                            | Main manuscript entry point.                                      |
| `GitHub/CV-DV-LCHS/res_base/experiments.tex`                     | Current experiments section used by the manuscript.               |
| `GitHub/CV-DV-LCHS/res_base/Appendix.tex`                        | Appendix included by `main.tex`.                                |
| `GitHub/CV-DV-LCHS/res_base/refs.bib`                            | Bibliography file.                                                |
| `GitHub/CV-DV-LCHS/res_base/Pics/section6_sensitivity_exact.pdf` | Sensitivity figure currently included in the experiments section. |

## Retained Paper Data

| Path                                                                | Purpose                                                                                        |
| ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `GitHub/CV-DV-LCHS/results_clean_refined_dirichlet`               | Dirichlet ideal-loading parameter sweep.                                                       |
| `GitHub/CV-DV-LCHS/results_clean_refined_neumann`                 | Neumann ideal-loading parameter sweep.                                                         |
| `GitHub/CV-DV-LCHS/results_clean_refined_periodic`                | Periodic ideal-loading parameter sweep.                                                        |
| `GitHub/CV-DV-LCHS/results_clean_givens_dirichlet`                | Dirichlet Givens baseline at the selected kernel point.                                        |
| `GitHub/CV-DV-LCHS/results_clean_givens_neumann`                  | Neumann Givens baseline at the selected kernel point.                                          |
| `GitHub/CV-DV-LCHS/results_clean_givens_periodic`                 | Periodic Givens baseline at the selected kernel point.                                         |
| `GitHub/CV-DV-LCHS/results_clean_snap_recreate_dirichlet`         | Final replayable Dirichlet `SNAP+D` sweep used in the paper.                                 |
| `GitHub/CV-DV-LCHS/results_clean_snap_recreate_neumann`           | Final replayable Neumann `SNAP+D` sweep used in the paper.                                   |
| `GitHub/CV-DV-LCHS/results_clean_snap_recreate_periodic`          | Final replayable periodic `SNAP+D` sweep used in the paper.                                  |
| `GitHub/CV-DV-LCHS/results_clean_bounds`                          | CSV summaries supporting the bound-related discussion in the experiments section.              |
| `GitHub/CV-DV-LCHS/results_clean_dv_lchs_dirichlet`               | JSON summaries for the one-step and 100-slice DV Dirichlet comparison.                         |
| `GitHub/CV-DV-LCHS/results_clean_dv_lchs_nwq_beta_scan_fine_eta1` | Fine classical DV beta scan used for the per-boundary optimal DV comparison in the manuscript. |

## What Is Kept Inside The Result Folders

For the retained sweep folders, the authoritative files are:

| File                   | Meaning                                            |
| ---------------------- | -------------------------------------------------- |
| `sweep_all.csv`      | Full evaluated sweep table.                        |
| `sweep_summary.json` | Best row and summary metadata.                     |
| `oat_*.csv`          | One-at-a-time reruns around the selected best row. |

For the replayable `SNAP+D` sweep folders, `sweep_all.csv` also contains:

| Field                           | Meaning                                                                                    |
| ------------------------------- | ------------------------------------------------------------------------------------------ |
| `snap_parameter_payload_json` | Serialized optimized `SNAP+D` layer parameters for exact replay without re-optimization. |

For the DV comparison folder:

| File                        | Meaning                                                     |
| --------------------------- | ----------------------------------------------------------- |
| `default_summary.json`    | One-step DV LCHS summary for the Dirichlet benchmark.       |
| `trotter100_summary.json` | DV summary for the repeated 100-slice Dirichlet comparison. |

For the fine NWQlib beta scan folder:

| File                                           | Meaning                                                                               |
| ---------------------------------------------- | ------------------------------------------------------------------------------------- |
| `*_beta*_eps0p1_eta1.json`                   | Per-boundary classical DV summaries across the retained `beta` scan with `eta=1`. |
| `summary_beta_scan_eps0p1_eta1.csv`          | Full retained scan table for all three boundary conditions.                           |
| `best_by_boundary_beta_scan_eps0p1_eta1.csv` | Best retained `beta` choice per boundary condition.                                 |

## Other Retained Material

| Path                           | Purpose                                                             |
| ------------------------------ | ------------------------------------------------------------------- |
| `GitHub/CV-DV-LCHS/res_refs` | Local reference PDFs kept as background reading for the manuscript. |

## Removed

The repository no longer contains the old exploratory folders, archived scripts,
tests, superseded `SNAP+D` result folders, noise-scan artifacts, or redundant
exported circuit files.
