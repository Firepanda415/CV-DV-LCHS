# Workspace Guide

This repository has been reduced to the files needed for the current paper and
for reproducing the retained paper data.

## Paper Data Source Map

Paths below are repo-relative. This section points readers to the retained
source file(s) for each paper-facing dataset used in `res_base`.

| Paper-facing item | Source file(s) | Notes |
| --- | --- | --- |
| Abstract ideal-loading fidelity claims and Table `clean_oracle_baseline` | `GitHub/CV-DV-LCHS/results_clean_refined_dirichlet/sweep_summary.json`<br>`GitHub/CV-DV-LCHS/results_clean_refined_neumann/sweep_summary.json`<br>`GitHub/CV-DV-LCHS/results_clean_refined_periodic/sweep_summary.json` | Best retained ideal-loading rows for the three boundary conditions. |
| Figure `section6_sensitivity_exact` | `GitHub/CV-DV-LCHS/results_clean_refined_dirichlet/sweep_all.csv`<br>`GitHub/CV-DV-LCHS/results_clean_refined_neumann/sweep_all.csv`<br>`GitHub/CV-DV-LCHS/results_clean_refined_periodic/sweep_all.csv`<br>`GitHub/CV-DV-LCHS/res_base/Pics/section6_sensitivity_exact.pdf` | The CSV files are the retained sweep data; the PDF is the figure included in the manuscript. |
| Bound-related discussion on `n_coeff` and Trotter scaling | `GitHub/CV-DV-LCHS/results_clean_bounds/section6_stateprep_consistency.csv`<br>`GitHub/CV-DV-LCHS/results_clean_bounds/section6_trotter_tightness.csv` | Supports the empirical checks discussed in the experiments section. |
| Givens rows in Table `clean_snap_comparison` and Givens pulse/rotation counts | `GitHub/CV-DV-LCHS/results_clean_givens_dirichlet/sweep_summary.json`<br>`GitHub/CV-DV-LCHS/results_clean_givens_neumann/sweep_summary.json`<br>`GitHub/CV-DV-LCHS/results_clean_givens_periodic/sweep_summary.json` | Per-boundary retained Givens summaries at the selected kernel points. |
| SNAP+D rows in Table `clean_snap_comparison` | `GitHub/CV-DV-LCHS/results_clean_snap_recreate_dirichlet/sweep_summary.json`<br>`GitHub/CV-DV-LCHS/results_clean_snap_recreate_neumann/sweep_summary.json`<br>`GitHub/CV-DV-LCHS/results_clean_snap_recreate_periodic/sweep_summary.json` | Per-boundary retained replayable `SNAP+D` summaries. |
| Replayable SNAP+D parameter payloads | `GitHub/CV-DV-LCHS/results_clean_snap_recreate_dirichlet/sweep_all.csv`<br>`GitHub/CV-DV-LCHS/results_clean_snap_recreate_neumann/sweep_all.csv`<br>`GitHub/CV-DV-LCHS/results_clean_snap_recreate_periodic/sweep_all.csv` | The `snap_parameter_payload_json` field stores the replayable optimized parameters. |
| Table `clean_resource_counts` | `GitHub/CV-DV-LCHS/results_clean_snap_recreate_dirichlet/sweep_summary.json`<br>`GitHub/CV-DV-LCHS/results_clean_snap_recreate_neumann/sweep_summary.json`<br>`GitHub/CV-DV-LCHS/results_clean_snap_recreate_periodic/sweep_summary.json`<br>`GitHub/CV-DV-LCHS/results_clean_givens_dirichlet/sweep_summary.json`<br>`GitHub/CV-DV-LCHS/results_clean_givens_neumann/sweep_summary.json`<br>`GitHub/CV-DV-LCHS/results_clean_givens_periodic/sweep_summary.json` | Optimizer iterations and selected depths are in the summaries; the hybrid Trotter counts are reproduced from `clean_hybrid.py` at the same kernel points. |
| Table `dv_resource_compare` and abstract DV `M_{\mathrm{DV}}` values | `GitHub/CV-DV-LCHS/results_clean_dv_lchs_nwq_beta_scan_fine_eta1/best_by_boundary_beta_scan_eps0p1_eta1.csv` | Best retained classical DV quadrature result per boundary condition. |
| Table `dv_circuit_compare` and representative Dirichlet DV circuit discussion | `GitHub/CV-DV-LCHS/results_clean_dv_lchs_dirichlet/default_summary.json`<br>`GitHub/CV-DV-LCHS/results_clean_dv_lchs_dirichlet/trotter100_summary.json` | One-step and repeated-100-slice Dirichlet DV circuit summaries. |

## Core Scripts

| Path | Purpose |
| --- | --- |
| `GitHub/CV-DV-LCHS/clean_core.py` | Reference mathematics, shared dataclasses, exact/truncated models, and heat-equation system builders. |
| `GitHub/CV-DV-LCHS/clean_oracles.py` | CV oracle preparation for `injection`, `SNAP+D`, and `givens`, including replayable `SNAP+D` payloads. |
| `GitHub/CV-DV-LCHS/clean_hybrid.py` | Hybrid CV-DV circuit construction, Trotterized evolution, postselection, and fidelity/resource extraction. |
| `GitHub/CV-DV-LCHS/clean_sweep.py` | Sweep driver used to generate the retained `injection`, `givens`, and `SNAP+D` datasets. |
| `GitHub/CV-DV-LCHS/clean_dv_lchs_nwq.py` | Computes DV LCHS quadrature quantities such as `h_1`, `K`, `Q`, `M_DV`, `m_c`, and `||c||_1`. |
| `GitHub/CV-DV-LCHS/clean_dv_lchs_dirichlet.py` | Builds the practical Dirichlet DV LCHS circuit summaries used for the DV comparison. |
| `GitHub/CV-DV-LCHS/clean_demo.ipynb` | Reader-facing notebook that walks through the algorithm on a small live example and then loads the retained paper summaries. |
| `GitHub/CV-DV-LCHS/requirements-clean.txt` | Minimal package list for the retained clean code path. |

## Manuscript Files

| Path | Purpose |
| --- | --- |
| `GitHub/CV-DV-LCHS/res_base/main.tex` | Main manuscript entry point. |
| `GitHub/CV-DV-LCHS/res_base/experiments.tex` | Current experiments section used by the manuscript. |
| `GitHub/CV-DV-LCHS/res_base/Appendix.tex` | Appendix included by `main.tex`. |
| `GitHub/CV-DV-LCHS/res_base/refs.bib` | Bibliography file. |
| `GitHub/CV-DV-LCHS/res_base/Pics/section6_sensitivity_exact.pdf` | Sensitivity figure currently included in the experiments section. |

## Retained Paper Data

| Path | Purpose |
| --- | --- |
| `GitHub/CV-DV-LCHS/results_clean_refined_dirichlet` | Dirichlet ideal-loading parameter sweep. |
| `GitHub/CV-DV-LCHS/results_clean_refined_neumann` | Neumann ideal-loading parameter sweep. |
| `GitHub/CV-DV-LCHS/results_clean_refined_periodic` | Periodic ideal-loading parameter sweep. |
| `GitHub/CV-DV-LCHS/results_clean_givens_dirichlet` | Dirichlet Givens baseline at the selected kernel point. |
| `GitHub/CV-DV-LCHS/results_clean_givens_neumann` | Neumann Givens baseline at the selected kernel point. |
| `GitHub/CV-DV-LCHS/results_clean_givens_periodic` | Periodic Givens baseline at the selected kernel point. |
| `GitHub/CV-DV-LCHS/results_clean_snap_recreate_dirichlet` | Final replayable Dirichlet `SNAP+D` sweep used in the paper. |
| `GitHub/CV-DV-LCHS/results_clean_snap_recreate_neumann` | Final replayable Neumann `SNAP+D` sweep used in the paper. |
| `GitHub/CV-DV-LCHS/results_clean_snap_recreate_periodic` | Final replayable periodic `SNAP+D` sweep used in the paper. |
| `GitHub/CV-DV-LCHS/results_clean_bounds` | CSV summaries supporting the bound-related discussion in the experiments section. |
| `GitHub/CV-DV-LCHS/results_clean_dv_lchs_dirichlet` | JSON summaries for the one-step and 100-slice DV Dirichlet comparison. |
| `GitHub/CV-DV-LCHS/results_clean_dv_lchs_nwq_beta_scan_fine_eta1` | Fine classical DV beta scan used for the per-boundary optimal DV comparison in the manuscript. |

## What Is Kept Inside The Result Folders

For the retained sweep folders, the authoritative files are:

| File | Meaning |
| --- | --- |
| `sweep_all.csv` | Full evaluated sweep table. |
| `sweep_summary.json` | Best row and summary metadata. |
| `oat_*.csv` | One-at-a-time reruns around the selected best row. |

For the replayable `SNAP+D` sweep folders, `sweep_all.csv` also contains:

| Field | Meaning |
| --- | --- |
| `snap_parameter_payload_json` | Serialized optimized `SNAP+D` layer parameters for exact replay without re-optimization. |

For the DV comparison folder:

| File | Meaning |
| --- | --- |
| `default_summary.json` | One-step DV LCHS summary for the Dirichlet benchmark. |
| `trotter100_summary.json` | DV summary for the repeated 100-slice Dirichlet comparison. |

For the fine NWQlib beta scan folder:

| File | Meaning |
| --- | --- |
| `*_beta*_eps0p1_eta1.json` | Per-boundary classical DV summaries across the retained `beta` scan with `eta=1`. |
| `summary_beta_scan_eps0p1_eta1.csv` | Full retained scan table for all three boundary conditions. |
| `best_by_boundary_beta_scan_eps0p1_eta1.csv` | Best retained `beta` choice per boundary condition. |

## Other Retained Material

| Path | Purpose |
| --- | --- |
| `GitHub/CV-DV-LCHS/res_refs` | Local reference PDFs kept as background reading for the manuscript. |

## Removed

The repository no longer contains the old exploratory folders, archived scripts,
tests, superseded `SNAP+D` result folders, noise-scan artifacts, or redundant
exported circuit files.
