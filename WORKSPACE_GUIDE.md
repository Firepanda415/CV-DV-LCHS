# Workspace Guide

This file maps the retained files and folders in this repository to their role
in the current clean CV-DV LCHS workflow and manuscript.

## Core Runtime

| Path | Purpose |
| --- | --- |
| `GitHub/CV-DV-LCHS/clean_core.py` | Reference-side mathematics, shared dataclasses, heat-equation builders, coefficient generation, and exact/truncated model definitions. |
| `GitHub/CV-DV-LCHS/clean_oracles.py` | CV state-preparation logic for `injection`, `SNAP+D`, and `givens`, including warm-start continuation and replayable `SNAP+D` payloads. |
| `GitHub/CV-DV-LCHS/clean_hybrid.py` | Hybrid CV-DV circuit construction, Trotterized evolution, postselection, and end-to-end fidelity/resource extraction. |
| `GitHub/CV-DV-LCHS/clean_sweep.py` | Hyperparameter sweep driver for the clean stack. Produces `sweep_all.csv`, `sweep_summary.json`, OAT sensitivity files, and plots. |
| `GitHub/CV-DV-LCHS/clean_demo.ipynb` | Notebook demonstration of the clean stack and the Section 6 style examples. |
| `GitHub/CV-DV-LCHS/requirements-clean.txt` | Minimal Python package list for the clean code path. |

## Analysis And Figure Scripts

| Path | Purpose |
| --- | --- |
| `GitHub/CV-DV-LCHS/clean_paper_plots.py` | Builds the parameter-sensitivity figures used by the paper assets. |
| `GitHub/CV-DV-LCHS/clean_bound_analysis.py` | Generates the empirical bound-check figure and CSV summaries used in Section 6. |
| `GitHub/CV-DV-LCHS/clean_dv_lchs_nwq.py` | Computes DV LCHS quadrature-side quantities such as `h_1`, `K`, `Q`, `M_DV`, `m_c`, and `||c||_1`. |
| `GitHub/CV-DV-LCHS/clean_dv_lchs_dirichlet.py` | Builds the practical Dirichlet DV LCHS circuit baseline using the `lchs-quantum-pde` implementation, including the repeated middle-block construction used for the DV comparison in Section 6. |

## Paper Sources

| Path | Purpose |
| --- | --- |
| `GitHub/CV-DV-LCHS/res_base/main.tex` | Main paper entry point. |
| `GitHub/CV-DV-LCHS/res_base/experiments.tex` | Current Section 6 experiments section used by the manuscript. This is the file you have been editing manually. |
| `GitHub/CV-DV-LCHS/res_base/refs.bib` | Bibliography for the manuscript. |
| `GitHub/CV-DV-LCHS/res_base/Pics/section6` | Figures copied or generated for Section 6. |
| `GitHub/CV-DV-LCHS/res_refs` | Reference materials associated with the manuscript. |

## Current Paper Data Folders

These are the result folders that support the present Section 6 text.

| Path | Role In Paper |
| --- | --- |
| `GitHub/CV-DV-LCHS/results_clean_refined_dirichlet` | Injection-based parameter study for Dirichlet. Supports the sensitivity plot and Table 1. |
| `GitHub/CV-DV-LCHS/results_clean_refined_neumann` | Injection-based parameter study for Neumann. Supports the sensitivity plot and Table 1. |
| `GitHub/CV-DV-LCHS/results_clean_refined_periodic` | Injection-based parameter study for periodic boundary conditions. Supports the sensitivity plot and Table 1. |
| `GitHub/CV-DV-LCHS/results_clean_givens_dirichlet` | Givens baseline at the selected Dirichlet kernel point. |
| `GitHub/CV-DV-LCHS/results_clean_givens_neumann` | Givens baseline at the selected Neumann kernel point. |
| `GitHub/CV-DV-LCHS/results_clean_givens_periodic` | Givens baseline at the selected periodic kernel point. |
| `GitHub/CV-DV-LCHS/results_clean_snap_recreate_dirichlet` | Final replayable Dirichlet `SNAP+D` dataset used in the paper, including saved parameter payloads. |
| `GitHub/CV-DV-LCHS/results_clean_snap_recreate_neumann` | Final replayable Neumann `SNAP+D` dataset used in the paper, including saved parameter payloads. |
| `GitHub/CV-DV-LCHS/results_clean_snap_recreate_periodic` | Final replayable periodic `SNAP+D` dataset used in the paper, including saved parameter payloads. |
| `GitHub/CV-DV-LCHS/results_clean_bounds` | Bound-check plots and CSV summaries for the empirical discussion in Section 6. |
| `GitHub/CV-DV-LCHS/results_clean_dv_lchs_dirichlet` | Practical DV Dirichlet circuit artifacts and summaries, including the 100-slice repeated-evolution comparison. |

## What Each Result Folder Typically Contains

For the sweep-style folders above, the standard files are:

| File | Meaning |
| --- | --- |
| `sweep_all.csv` | Full table of every evaluated row and all recorded metrics. |
| `sweep_top.csv` | Highest-ranked rows under the chosen sweep objective. |
| `sweep_summary.json` | Summary of the run, including the best row and OAT outputs. |
| `oat_*.csv` | One-at-a-time sensitivity reruns around the selected best row. |
| `*.png` | Sweep and OAT plots generated directly from the run. |
| `snap_parameter_payload_json` | Serialized `SNAP+D` layer parameters stored inside the recreate sweep tables for exact replay without re-optimization. |

For the DV circuit folder:

| File | Meaning |
| --- | --- |
| `default_summary.json` | One-step DV LCHS summary for the Dirichlet benchmark. |
| `trotter100_summary.json` | Dirichlet DV summary for the repeated 100-slice middle-block construction. |
| `*.qpy` | Serialized Qiskit circuits. |
| `*.txt` | ASCII circuit drawings. |

## Tests

| Path | Purpose |
| --- | --- |
| `GitHub/CV-DV-LCHS/tests/test_clean_core.py` | Regression tests for the reference math and boundary-condition builders. |
| `GitHub/CV-DV-LCHS/tests/test_clean_oracles.py` | Regression tests for `injection`, `SNAP+D`, Givens, warm starts, and replay payloads. |
| `GitHub/CV-DV-LCHS/tests/test_clean_sweep.py` | Regression tests for sweep scoring, boundary-condition selection, and `SNAP+D` payload persistence. |

## Archive

| Path | Purpose |
| --- | --- |
| `GitHub/CV-DV-LCHS/OLD_FILES` | Legacy scripts, old experiments, and archival material kept for historical reference. This folder is not part of the clean paper pipeline. |

## Disposable Artifacts

These are not part of the scientific record and may be deleted when they
reappear:

- `__pycache__` folders
- `.DS_Store`
- `.pytest_cache`
- empty `*.log` files from quiet sweep runs

The authoritative experiment outputs are the data folders listed above, not
the old log files.
