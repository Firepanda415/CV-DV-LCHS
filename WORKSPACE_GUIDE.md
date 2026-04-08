# Workspace Guide

This repository has been reduced to the files needed for the current paper and
for reproducing the retained paper data.

## Core Scripts

| Path | Purpose |
| --- | --- |
| `GitHub/CV-DV-LCHS/clean_core.py` | Reference mathematics, shared dataclasses, exact/truncated models, and heat-equation system builders. |
| `GitHub/CV-DV-LCHS/clean_oracles.py` | CV oracle preparation for `injection`, `SNAP+D`, and `givens`, including replayable `SNAP+D` payloads. |
| `GitHub/CV-DV-LCHS/clean_hybrid.py` | Hybrid CV-DV circuit construction, Trotterized evolution, postselection, and fidelity/resource extraction. |
| `GitHub/CV-DV-LCHS/clean_sweep.py` | Sweep driver used to generate the retained `injection`, `givens`, and `SNAP+D` datasets. |
| `GitHub/CV-DV-LCHS/clean_dv_lchs_nwq.py` | Computes DV LCHS quadrature quantities such as `h_1`, `K`, `Q`, `M_DV`, `m_c`, and `||c||_1`. |
| `GitHub/CV-DV-LCHS/clean_dv_lchs_dirichlet.py` | Builds the practical Dirichlet DV LCHS circuit summaries used for the DV comparison. |
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

## Other Retained Material

| Path | Purpose |
| --- | --- |
| `GitHub/CV-DV-LCHS/res_refs` | Local reference PDFs kept as background reading for the manuscript. |

## Removed

The repository no longer contains the old exploratory folders, archived scripts,
tests, demo notebooks, superseded `SNAP+D` result folders, noise-scan artifacts,
or redundant exported circuit files.
