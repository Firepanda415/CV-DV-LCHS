# Workspace Guide

The minimal clean code path and retained outputs for the CV-DV LCHS
benchmarks.

## Paper Data Correspondence

Paths below are repo-relative. This list maps working-paper tables and figures
to the retained data folders that support them, without listing manuscript
source-file locations.

- `fig:clean_sensitivity_exact`: `results_clean_refined_dirichlet`,
  `results_clean_refined_neumann`, `results_clean_refined_periodic`, and
  `results_clean_sensitivity`. The refined `injection` sweep CSVs provide the
  exact-infidelity data; `results_clean_sensitivity` contains regenerated
  sensitivity plot outputs and extracted plotting rows.

- `tab:lchs_comparison`: `results_clean_law_eberly_dirichlet`,
  `results_clean_dv_lchs_nwq_beta_scan_fine_eta1`, and
  `results_clean_dv_lchs_dirichlet`. These support the hybrid-versus-DV
  comparison for the scan-optimal 100-step Dirichlet example.

- Bound-related discussion on `n_coeff` and first-order Trotter scaling:
  `results_clean_bounds`. The key files are
  `section6_stateprep_consistency.csv` and `section6_trotter_tightness.csv`.

- `tab:clean_oracle_baseline`: `results_clean_refined_dirichlet`,
  `results_clean_refined_neumann`, `results_clean_refined_periodic`,
  `results_clean_law_eberly_dirichlet`,
  `results_clean_law_eberly_neumann`,
  `results_clean_law_eberly_periodic`,
  `results_clean_snap_recreate_dirichlet`,
  `results_clean_snap_recreate_neumann`, and
  `results_clean_snap_recreate_periodic`. Refined injection sweeps provide
  selected kernel parameters; Law-Eberly and `SNAP+D` summaries provide the
  baseline entries at those points. The six relative-entropy non-Gaussianity
  values are recomputed by `clean_delta_ng_values.py` from the retained
  Law-Eberly kernel parameters and replayable `SNAP+D` payloads.

- `tab:clean_resource_counts`: `results_clean_snap_recreate_dirichlet`,
  `results_clean_snap_recreate_neumann`,
  `results_clean_snap_recreate_periodic`,
  `results_clean_law_eberly_dirichlet`,
  `results_clean_law_eberly_neumann`, and
  `results_clean_law_eberly_periodic`. Summaries retain optimizer iterations,
  `SNAP+D` layer counts, and Law-Eberly JC/qubit-rotation pulse counts.

- `tab:dv_resource_compare`: `results_clean_dv_lchs_nwq_beta_scan_fine_eta1`,
  `results_clean_law_eberly_dirichlet`,
  `results_clean_law_eberly_neumann`, and
  `results_clean_law_eberly_periodic`. The DV quadrature parameters come from
  the retained `eta=1` beta scan; retained Law-Eberly summaries provide the
  comparison baseline.

- `tab:dv_circuit_compare`: `results_clean_dv_lchs_dirichlet`.
  `default_summary.json` retains one-step ancilla-preparation counts;
  `trotter100_summary.json` retains the repeated-slice circuit counts for the
  scan-optimal 100-step Dirichlet comparison.

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

### Sensitivity figure

Produces:

- `results_clean_sensitivity/section6_sensitivity_landscape.png`
- `results_clean_sensitivity/section6_sensitivity_landscape.pdf`
- `results_clean_sensitivity/section6_surface_rows.csv`
- `results_clean_sensitivity/section6_beta_profile_rows.csv`

This figure uses the refined `injection` sweeps above. It plots exact
infidelity `1-F`, where `F = F(u_{\mathrm{alg}}, u_{\mathrm{exact}}) = |\langle \hat u_{\mathrm{alg}} | \hat u_{\mathrm{exact}} \rangle|^2` is the
normalized overlap with the exact discretized solution. In the retained sweep
tables, this is the `fidelity` column populated from `fidelity_vs_exact`.

The left column shows `1-F` over `(r_target, r_prime)` at `n_coeff = 48` and
the boundary-specific `beta` that minimizes exact infidelity. The right column
shows `1-F` versus `beta` at fixed `(r_target, r_prime)`, with one fixed-squeeze
curve for each `n_coeff in {24, 48}`.

```bash
python clean_sensitivity_analysis.py
```

### Law-Eberly circuit baselines

Produces:

- `results_clean_law_eberly_dirichlet`
- `results_clean_law_eberly_neumann`
- `results_clean_law_eberly_periodic`
- `results_clean_law_eberly_selective_dirichlet`
- `results_clean_law_eberly_selective_neumann`
- `results_clean_law_eberly_selective_periodic`

The `law_eberly` state-preparation method implements the recursive
Law-Eberly protocol for arbitrary finite oscillator states
([Law and Eberly, PRL 76, 1055](https://doi.org/10.1103/PhysRevLett.76.1055);
[Hofheinz et al., Nature 459, 546](https://doi.org/10.1038/nature08005)).
For a target

```math
|g\rangle \otimes |\psi\rangle
  = |g\rangle \otimes \sum_{n=0}^{N} c_n |n\rangle ,
```

the compiler solves the time-reversed problem by repeatedly removing the
highest occupied Fock amplitude. Each inverse step first applies a
Jaynes-Cummings exchange that couples

```math
|e,n-1\rangle \leftrightarrow |g,n\rangle ,
```

then applies a global auxiliary-qubit rotation that removes the remaining
qubit excitation. The preparation circuit is the adjoint of that
unpreparation sequence. This is the original Law-Eberly pulse set at the ideal
circuit level.

Law and Eberly denote the JC channel by `Q_j` and the classical atomic drive by
`C_j`. In this repository, `S_n` denotes the JC exchange, while `R_n` denotes
the corresponding auxiliary-qubit rotation emitted as a standard Qiskit qubit
`r` gate. The separate `law_eberly_selective` method keeps the previous
Bosonic-Qiskit selective variant, where that second step is replaced by
`cv_sqr`; this is a selective-rotation alternative, not the original
Law-Eberly `C_j` pulse.

At the Bosonic Qiskit level, the exchange pulse is

```math
S_n(\alpha,\phi)=
\exp\!\left[-i\frac{\alpha}{\sqrt n}
\left(e^{i\phi}\sigma_-a^\dagger+e^{-i\phi}\sigma_+a\right)\right],
```

implemented by `cv_jc(alpha / sqrt(n), phi, qumode, qubit)`. The original
Law-Eberly qubit rotation is

```math
R_n(\theta,\varphi)=
I_{\rm osc}\otimes
\exp\!\left[-\frac{i\theta}{2}
\left(\cos\varphi\,X+\sin\varphi\,Y\right)\right]
```

implemented by `r(theta, phi, qubit)` on the auxiliary qubit. This is an ideal
circuit-level Law-Eberly model; it does not include the calibrated flux
pulses, detuning trajectories, leakage, or decoherence model used in the
Hofheinz hardware experiment. The retained `results_clean_law_eberly_*`
folders use this original `cv_jc` plus auxiliary-qubit-rotation route. The
retained `results_clean_law_eberly_selective_*` folders preserve the previous
`cv_jc` plus `cv_sqr` selective variant for provenance checks.

```bash
for entry in \
  "dirichlet,7.9,4.1,0.5,48" \
  "neumann,7.9,4.0,0.3,48" \
  "periodic,8.1,4.1,0.3,48"
do
  IFS=, read -r bc rt rp beta ncoeff <<< "$entry"
  python clean_sweep.py \
    --output-dir "results_clean_law_eberly_${bc}" \
    --boundary-condition "$bc" \
    --num-qubits 2 \
    --n-fock 64 \
    --n-quad 240 \
    --r-target-grid "$rt" \
    --r-prime-grid "$rp" \
    --beta-grid "$beta" \
    --n-coeff-grid "$ncoeff" \
    --n-trotter-grid 100 \
    --prep-method-grid law_eberly \
    --ranking-objective pde \
    --top-k 1
done
```

To regenerate the selective-rotation provenance folders, use the same loop with
`--output-dir "results_clean_law_eberly_selective_${bc}"` and
`--prep-method-grid law_eberly_selective`.

### Replayable `SNAP+D` sweeps

Produces:

- `results_clean_snap_recreate_dirichlet`
- `results_clean_snap_recreate_neumann`
- `results_clean_snap_recreate_periodic`

`sweep_all.csv` in these folders retains `snap_parameter_payload_json`, which is
the replayable optimized `SNAP+D` payload.

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

### Non-Gaussianity diagnostics

Recomputes the six relative-entropy non-Gaussianity values reported with the
Law-Eberly target and replayed `SNAP+D` rows. The target/LE value is computed
from the normalized finite oscillator seed state; the `SNAP+D` value is
computed after replaying the saved 30-layer payload, without rerunning the
optimizer.

```bash
python clean_delta_ng_values.py --format markdown
```

Use `--format csv --output results_clean_delta_ng_values.csv` to retain a CSV
copy of the recomputed diagnostics.

### Bound-analysis CSVs

Produces:

- `results_clean_bounds/section6_stateprep_consistency.csv`
- `results_clean_bounds/section6_trotter_tightness.csv`

`clean_bound_analysis.py` also writes
`results_clean_bounds/section6_trotter_tightness.pdf` and `.png` as optional
diagnostic plots.

Reproducing `section6_trotter_tightness.csv` requires the historical
`results_clean_refine/sweep_all.csv`, which is not part of the trimmed retained
dataset.

```bash
python clean_bound_analysis.py \
  --trotter-sweep results_clean_refine/sweep_all.csv \
  --refined-sweeps results_clean_refined_dirichlet/sweep_all.csv,results_clean_refined_neumann/sweep_all.csv,results_clean_refined_periodic/sweep_all.csv \
  --output-dir results_clean_bounds
```

### Scan-optimal Dirichlet DV circuit summaries

Produces:

- `results_clean_dv_lchs_dirichlet/default_summary.json`
- `results_clean_dv_lchs_dirichlet/trotter100_summary.json`

This block uses the separate DV environment because
`clean_dv_lchs_dirichlet.py` depends on `scikit_tt` and the
`lchs-quantum-pde` stack. The commands below use the Dirichlet optimum from
the retained `eta=1` classical beta scan: `beta=0.6`, `M_DV=320`, and
`m_c=9`.

```bash
conda activate dvlchs
python clean_dv_lchs_dirichlet.py \
  --beta 0.6 \
  --nwq-trunc-multiplier 1.0 \
  --no-measure \
  --output-json results_clean_dv_lchs_dirichlet/default_summary.json
python clean_dv_lchs_dirichlet.py \
  --beta 0.6 \
  --nwq-trunc-multiplier 1.0 \
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

| Path                              | Purpose                                                                                                                             |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `clean_core.py`                 | Reference mathematics, shared dataclasses, exact and truncated models, and heat-equation system builders.                           |
| `clean_oracles.py`              | CV oracle preparation for `injection`, `SNAP+D`, `law_eberly`, and `law_eberly_selective`, including replayable `SNAP+D` payload support.           |
| `clean_hybrid.py`               | Hybrid CV-DV circuit construction, Trotterized evolution, postselection, and fidelity/resource extraction.                          |
| `clean_sweep.py`                | Sweep driver used to generate the retained `injection`, `law_eberly`, and `SNAP+D` datasets.                                  |
| `clean_sensitivity_analysis.py` | Builds the sensitivity figure from the retained refined `injection` sweeps.                                                        |
| `clean_bound_analysis.py`       | Rebuilds the retained bound-analysis CSVs and optional bound-analysis figure from existing sweep outputs.                           |
| `clean_delta_ng_values.py`      | Recomputes target/LE and replayed `SNAP+D` relative-entropy non-Gaussianity diagnostics from retained results.                    |
| `clean_dv_lchs_nwq.py`          | Computes classical DV LCHS quadrature quantities such as `h1`, `K`, `Q`, `M_DV`, `m_c`, and classical fidelity summaries. |
| `clean_dv_lchs_dirichlet.py`    | Builds the retained Dirichlet DV circuit summaries used for the circuit-comparison discussion.                                      |
| `clean_demo.ipynb`              | Reader-facing notebook that walks through a small live example and then loads the retained benchmark summaries.                     |
| `requirements-clean.txt`        | Minimal package list for the retained clean code path.                                                                              |

## Retained Result Directories

These folders are the retained outputs for the clean benchmark workflow:

| Path                                              | Purpose                                                                      |
| ------------------------------------------------- | ---------------------------------------------------------------------------- |
| `results_clean_refined_dirichlet`               | Dirichlet ideal-loading parameter sweep.                                     |
| `results_clean_refined_neumann`                 | Neumann ideal-loading parameter sweep.                                       |
| `results_clean_refined_periodic`                | Periodic ideal-loading parameter sweep.                                      |
| `results_clean_law_eberly_dirichlet`            | Dirichlet Law-Eberly circuit baseline at the selected kernel point.          |
| `results_clean_law_eberly_neumann`              | Neumann Law-Eberly circuit baseline at the selected kernel point.            |
| `results_clean_law_eberly_periodic`             | Periodic Law-Eberly circuit baseline at the selected kernel point.           |
| `results_clean_law_eberly_selective_dirichlet`  | Dirichlet selective-rotation Law-Eberly variant at the selected kernel point.|
| `results_clean_law_eberly_selective_neumann`    | Neumann selective-rotation Law-Eberly variant at the selected kernel point.  |
| `results_clean_law_eberly_selective_periodic`   | Periodic selective-rotation Law-Eberly variant at the selected kernel point. |
| `results_clean_snap_recreate_dirichlet`         | Replayable Dirichlet `SNAP+D` sweep.                                       |
| `results_clean_snap_recreate_neumann`           | Replayable Neumann `SNAP+D` sweep.                                         |
| `results_clean_snap_recreate_periodic`          | Replayable periodic `SNAP+D` sweep.                                        |
| `results_clean_bounds`                          | Bound-analysis CSVs.                                                        |
| `results_clean_dv_lchs_dirichlet`               | JSON summaries for the representative Dirichlet DV circuit comparison.       |
| `results_clean_dv_lchs_nwq_beta_scan_fine_eta1` | Fine classical DV beta scan used for the per-boundary optimal DV comparison. |

Additional retained outputs currently present in the repo include:

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
