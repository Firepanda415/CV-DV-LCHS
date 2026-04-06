#!/usr/bin/env python3
"""
Hyperparameter sweep and sensitivity analysis for the clean CV-DV LCHS stack.

This script imports only the independent clean modules. It supports:
  - compact Cartesian grid search,
  - optional local refinement around the top seeds,
  - one-at-a-time sensitivity scans around the best point.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

from clean_core import (
    EvolutionSpec,
    KernelSpec,
    StatePrepSpec,
    build_dirichlet_heat_system,
)
from clean_hybrid import run_clean_lchs


RANKING_OBJECTIVES = ("balanced", "pde", "prep_pde", "oracle", "truncated")


@dataclass(frozen=True)
class SweepCandidate:
    r_target: float
    r_prime: float
    beta: float
    n_coeff: int
    n_trotter_steps: int
    state_prep_method: str
    snap_depth: int


def _parse_float_list(text: str) -> List[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _parse_int_list(text: str) -> List[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_str_list(text: str) -> List[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _score(row: Dict[str, Any], objective: str) -> float:
    fid = max(0.0, float(row["fidelity"]))
    post = max(0.0, float(row["postselection_probability"]))
    oracle = max(0.0, float(row.get("oracle_fidelity", 0.0)))
    truncated = max(0.0, float(row.get("fidelity_vs_truncated", 0.0)))

    if objective == "balanced":
        return float(np.sqrt(fid * post))
    if objective == "pde":
        return fid
    if objective == "prep_pde":
        return float(np.sqrt(fid * oracle))
    if objective == "oracle":
        return oracle
    if objective == "truncated":
        return truncated
    raise ValueError(
        f"Unknown ranking objective '{objective}'. Expected one of {RANKING_OBJECTIVES}."
    )


def _score_columns(row: Dict[str, Any]) -> Dict[str, float]:
    return {f"score_{objective}": _score(row, objective) for objective in RANKING_OBJECTIVES}


def _best_row_for_objective(
    rows: Sequence[Dict[str, Any]], objective: str
) -> Dict[str, Any] | None:
    valid_rows = [row for row in rows if row.get("valid")]
    if not valid_rows:
        return None
    return max(valid_rows, key=lambda row: (_score(row, objective), -float(row["rel_error_vs_exact"])))


def _candidate_to_specs(
    candidate: SweepCandidate,
    *,
    n_fock: int,
    n_quad: int,
    coeff_backend: str,
    snap_restarts: int,
    snap_maxiter: int,
) -> Tuple[KernelSpec, StatePrepSpec, EvolutionSpec]:
    kernel = KernelSpec(
        r_target=candidate.r_target,
        r_prime=candidate.r_prime,
        beta=candidate.beta,
        n_coeff=candidate.n_coeff,
        n_fock=n_fock,
        n_quad=n_quad,
        coeff_backend=coeff_backend,
    )
    prep = StatePrepSpec(
        method=candidate.state_prep_method,
        snap_depth=candidate.snap_depth,
        snap_restarts=snap_restarts,
        snap_maxiter=snap_maxiter,
    )
    evolution = EvolutionSpec(
        n_trotter_steps=candidate.n_trotter_steps,
        readout_mode="postselect_statevector",
    )
    return kernel, prep, evolution


def evaluate_candidate(
    candidate: SweepCandidate,
    *,
    num_qubits: int,
    alpha: float,
    grid_spacing: float,
    total_time: float,
    init_basis_index: int,
    n_fock: int,
    n_quad: int,
    coeff_backend: str,
    snap_restarts: int,
    snap_maxiter: int,
    ranking_objective: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "r_target": candidate.r_target,
        "r_prime": candidate.r_prime,
        "beta": candidate.beta,
        "n_coeff": candidate.n_coeff,
        "n_trotter_steps": candidate.n_trotter_steps,
        "state_prep_method": candidate.state_prep_method,
        "snap_depth": candidate.snap_depth,
    }

    if candidate.r_target <= candidate.r_prime:
        row.update({"valid": False, "error": "r_target_must_exceed_r_prime"})
        return row

    try:
        system = build_dirichlet_heat_system(
            num_qubits=num_qubits,
            alpha=alpha,
            grid_spacing=grid_spacing,
            total_time=total_time,
            init_basis_index=init_basis_index,
        )
        kernel, prep, evolution = _candidate_to_specs(
            candidate,
            n_fock=n_fock,
            n_quad=n_quad,
            coeff_backend=coeff_backend,
            snap_restarts=snap_restarts,
            snap_maxiter=snap_maxiter,
        )
        result = run_clean_lchs(system, kernel, prep, evolution)
        oracle_metadata = dict(result.metadata.get("oracle_metadata", {}))
        row.update(
            {
                "valid": True,
                "error": "",
                "fidelity": float(result.fidelity_vs_exact),
                "fidelity_vs_truncated": float(result.fidelity_vs_truncated),
                "postselection_probability": float(result.postselection_probability),
                "rel_error_vs_exact": float(result.rel_error_vs_exact),
                "rel_error_vs_truncated": float(result.rel_error_vs_truncated),
                "oracle_fidelity": float(result.oracle_fidelity),
                "coeff_backend_gap": float(result.coeff_backend_gap),
                "circuit_depth": int(result.circuit_depth),
                "circuit_size": int(result.circuit_size),
                "oracle_apply_mode": str(result.metadata.get("oracle_apply_mode", "")),
                "oracle_n_active_fock_levels": oracle_metadata.get("n_active_fock_levels", ""),
                "oracle_n_jc_pulses": oracle_metadata.get("n_jc_pulses", ""),
                "oracle_n_qubit_rotations": oracle_metadata.get("n_qubit_rotations", ""),
                "oracle_vs_ideal_fidelity": float(
                    result.metadata.get("oracle_vs_ideal_fidelity", np.nan)
                ),
                "ideal_vs_reference_fidelity": float(
                    result.metadata.get("ideal_vs_reference_fidelity", np.nan)
                ),
                "snap_n_snap": oracle_metadata.get("snap_n_snap", ""),
                "snap_total_iterations": oracle_metadata.get("snap_total_iterations", ""),
                "snap_restarts": oracle_metadata.get("snap_restarts", prep.snap_restarts),
                "snap_maxiter": oracle_metadata.get("snap_maxiter", prep.snap_maxiter),
                "score_objective": ranking_objective,
            }
        )
        row.update(_score_columns(row))
        row["score"] = _score(row, ranking_objective)
    except Exception as exc:
        row.update({"valid": False, "error": f"{type(exc).__name__}: {exc}"})
    return row


def _rows_to_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_parameter_curve(
    rows: Sequence[Dict[str, Any]],
    parameter: str,
    out_path: Path,
    *,
    metric: str,
    ylabel: str | None = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to generate sweep plots.") from exc

    valid_rows = [row for row in rows if row.get("valid") and metric in row]
    if not valid_rows:
        return

    valid_rows = sorted(valid_rows, key=lambda row: row[parameter])
    x = [row[parameter] for row in valid_rows]
    y = [row[metric] for row in valid_rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.0, 4.0))
    plt.plot(x, y, marker="o")
    plt.xlabel(parameter)
    plt.ylabel(ylabel or metric)
    plt.title(f"{ylabel or metric} vs {parameter}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def _local_refine(
    seeds: Sequence[Dict[str, Any]],
    *,
    num_qubits: int,
    alpha: float,
    grid_spacing: float,
    total_time: float,
    init_basis_index: int,
    n_fock: int,
    n_quad: int,
    coeff_backend: str,
    snap_restarts: int,
    snap_maxiter: int,
    ranking_objective: str,
    local_refine_maxiter: int,
) -> List[Dict[str, Any]]:
    refined_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        if not seed.get("valid"):
            continue

        method = str(seed["state_prep_method"])
        n_coeff = int(seed["n_coeff"])
        n_trotter = int(seed["n_trotter_steps"])
        snap_depth = int(seed["snap_depth"])

        def objective(x: np.ndarray) -> float:
            r_target = float(x[0])
            r_prime = float(x[1])
            beta = float(x[2])
            if r_target <= r_prime or r_prime < 0.0 or not (0.0 < beta < 1.0):
                return 1e6

            candidate = SweepCandidate(
                r_target=r_target,
                r_prime=r_prime,
                beta=beta,
                n_coeff=n_coeff,
                n_trotter_steps=n_trotter,
                state_prep_method=method,
                snap_depth=snap_depth,
            )
            row = evaluate_candidate(
                candidate,
                num_qubits=num_qubits,
                alpha=alpha,
                grid_spacing=grid_spacing,
                total_time=total_time,
                init_basis_index=init_basis_index,
                n_fock=n_fock,
                n_quad=n_quad,
                coeff_backend=coeff_backend,
                snap_restarts=snap_restarts,
                snap_maxiter=snap_maxiter,
                ranking_objective=ranking_objective,
            )
            if not row.get("valid"):
                return 1e6
            return -_score(row, ranking_objective)

        x0 = np.array([seed["r_target"], seed["r_prime"], seed["beta"]], dtype=float)
        bounds = [(0.05, 4.0), (0.0, 3.5), (0.05, 0.99)]
        result = minimize(
            objective,
            x0,
            method="Powell",
            bounds=bounds,
            options={"maxiter": local_refine_maxiter, "xtol": 1e-3, "ftol": 1e-3},
        )

        candidate = SweepCandidate(
            r_target=float(result.x[0]),
            r_prime=float(result.x[1]),
            beta=float(result.x[2]),
            n_coeff=n_coeff,
            n_trotter_steps=n_trotter,
            state_prep_method=method,
            snap_depth=snap_depth,
        )
        row = evaluate_candidate(
            candidate,
            num_qubits=num_qubits,
            alpha=alpha,
            grid_spacing=grid_spacing,
            total_time=total_time,
            init_basis_index=init_basis_index,
            n_fock=n_fock,
            n_quad=n_quad,
            coeff_backend=coeff_backend,
            snap_restarts=snap_restarts,
            snap_maxiter=snap_maxiter,
            ranking_objective=ranking_objective,
        )
        row["source"] = "local_refine"
        refined_rows.append(row)

    return refined_rows


def _oat_sensitivity_rows(
    best_row: Dict[str, Any],
    *,
    parameter: str,
    values: Sequence[Any],
    num_qubits: int,
    alpha: float,
    grid_spacing: float,
    total_time: float,
    init_basis_index: int,
    n_fock: int,
    n_quad: int,
    coeff_backend: str,
    snap_restarts: int,
    snap_maxiter: int,
    ranking_objective: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for value in values:
        candidate = SweepCandidate(
            r_target=float(best_row["r_target"]) if parameter != "r_target" else float(value),
            r_prime=float(best_row["r_prime"]) if parameter != "r_prime" else float(value),
            beta=float(best_row["beta"]) if parameter != "beta" else float(value),
            n_coeff=int(best_row["n_coeff"]) if parameter != "n_coeff" else int(value),
            n_trotter_steps=int(best_row["n_trotter_steps"]) if parameter != "n_trotter_steps" else int(value),
            state_prep_method=str(best_row["state_prep_method"])
            if parameter != "state_prep_method"
            else str(value),
            snap_depth=int(best_row["snap_depth"]) if parameter != "snap_depth" else int(value),
        )
        row = evaluate_candidate(
            candidate,
            num_qubits=num_qubits,
            alpha=alpha,
            grid_spacing=grid_spacing,
            total_time=total_time,
            init_basis_index=init_basis_index,
            n_fock=n_fock,
            n_quad=n_quad,
            coeff_backend=coeff_backend,
            snap_restarts=snap_restarts,
            snap_maxiter=snap_maxiter,
            ranking_objective=ranking_objective,
        )
        row["source"] = f"oat_{parameter}"
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent clean CV-DV LCHS hyperparameter sweep")
    parser.add_argument("--output-dir", default="results_clean", help="Output directory")
    parser.add_argument("--num-qubits", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--grid-spacing", type=float, default=1.0)
    parser.add_argument("--total-time", type=float, default=1.0)
    parser.add_argument("--init-basis-index", type=int, default=1)
    parser.add_argument("--n-fock", type=int, default=32)
    parser.add_argument("--n-quad", type=int, default=180)
    parser.add_argument("--coeff-backend", default="explicit_overlap")
    parser.add_argument("--r-target-grid", default="0.8,1.0,1.2")
    parser.add_argument("--r-prime-grid", default="0.05,0.1,0.2")
    parser.add_argument("--beta-grid", default="0.5,0.7,0.9")
    parser.add_argument("--n-coeff-grid", default="8,12,16")
    parser.add_argument("--n-trotter-grid", default="10,20,40")
    parser.add_argument("--prep-method-grid", default="injection,snap_d,givens")
    parser.add_argument("--snap-depth-grid", default="2,4")
    parser.add_argument(
        "--snap-restarts",
        type=int,
        default=3,
        help="Number of random restarts for SNAP+D optimization.",
    )
    parser.add_argument(
        "--snap-maxiter",
        type=int,
        default=1000,
        help="Maximum iterations per SNAP+D optimizer run.",
    )
    parser.add_argument(
        "--ranking-objective",
        choices=RANKING_OBJECTIVES,
        default="prep_pde",
        help="Objective used to rank sweep rows.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--local-refine", action="store_true")
    parser.add_argument(
        "--local-refine-maxiter",
        type=int,
        default=100,
        help="Maximum Powell iterations for continuous local refinement.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    r_target_grid = _parse_float_list(args.r_target_grid)
    r_prime_grid = _parse_float_list(args.r_prime_grid)
    beta_grid = _parse_float_list(args.beta_grid)
    n_coeff_grid = _parse_int_list(args.n_coeff_grid)
    n_trotter_grid = _parse_int_list(args.n_trotter_grid)
    prep_method_grid = _parse_str_list(args.prep_method_grid)
    snap_depth_grid = _parse_int_list(args.snap_depth_grid)

    sweep_rows: List[Dict[str, Any]] = []
    for r_target, r_prime, beta, n_coeff, n_trotter, method in product(
        r_target_grid,
        r_prime_grid,
        beta_grid,
        n_coeff_grid,
        n_trotter_grid,
        prep_method_grid,
    ):
        relevant_depths = snap_depth_grid if method == "snap_d" else [0]
        for snap_depth in relevant_depths:
            candidate = SweepCandidate(
                r_target=r_target,
                r_prime=r_prime,
                beta=beta,
                n_coeff=n_coeff,
                n_trotter_steps=n_trotter,
                state_prep_method=method,
                snap_depth=snap_depth,
            )
            row = evaluate_candidate(
                candidate,
                num_qubits=args.num_qubits,
                alpha=args.alpha,
                grid_spacing=args.grid_spacing,
                total_time=args.total_time,
                init_basis_index=args.init_basis_index,
                n_fock=args.n_fock,
                n_quad=args.n_quad,
                coeff_backend=args.coeff_backend,
                snap_restarts=args.snap_restarts,
                snap_maxiter=args.snap_maxiter,
                ranking_objective=args.ranking_objective,
            )
            row["source"] = "grid"
            sweep_rows.append(row)

    valid_rows = [row for row in sweep_rows if row.get("valid")]
    valid_rows.sort(
        key=lambda row: (-_score(row, args.ranking_objective), row["rel_error_vs_exact"])
    )
    top_rows = valid_rows[: args.top_k]

    refined_rows: List[Dict[str, Any]] = []
    if args.local_refine and top_rows:
        refined_rows = _local_refine(
            top_rows,
            num_qubits=args.num_qubits,
            alpha=args.alpha,
            grid_spacing=args.grid_spacing,
            total_time=args.total_time,
            init_basis_index=args.init_basis_index,
            n_fock=args.n_fock,
            n_quad=args.n_quad,
            coeff_backend=args.coeff_backend,
            snap_restarts=args.snap_restarts,
            snap_maxiter=args.snap_maxiter,
            ranking_objective=args.ranking_objective,
            local_refine_maxiter=args.local_refine_maxiter,
        )

    all_rows = sweep_rows + refined_rows
    _rows_to_csv(all_rows, out_dir / "sweep_all.csv")
    _rows_to_csv(top_rows, out_dir / "sweep_top.csv")

    summary = {
        "num_total_rows": len(all_rows),
        "num_valid_rows": len([row for row in all_rows if row.get("valid")]),
        "top_k": args.top_k,
        "ranking_objective": args.ranking_objective,
        "best_row": top_rows[0] if top_rows else None,
        "best_rows_by_objective": {
            objective: _best_row_for_objective(valid_rows, objective)
            for objective in RANKING_OBJECTIVES
        },
        "grid_config": {
            "r_target_grid": r_target_grid,
            "r_prime_grid": r_prime_grid,
            "beta_grid": beta_grid,
            "n_coeff_grid": n_coeff_grid,
            "n_trotter_grid": n_trotter_grid,
            "prep_method_grid": prep_method_grid,
            "snap_depth_grid": snap_depth_grid,
            "snap_restarts": args.snap_restarts,
            "snap_maxiter": args.snap_maxiter,
        },
    }

    oat_outputs: Dict[str, Dict[str, Any]] = {}
    if top_rows:
        best = top_rows[0]
        oat_specs = {
            "r_target": r_target_grid,
            "r_prime": r_prime_grid,
            "beta": beta_grid,
            "n_coeff": n_coeff_grid,
            "n_trotter_steps": n_trotter_grid,
            "state_prep_method": prep_method_grid,
            "snap_depth": snap_depth_grid,
        }
        for parameter, values in oat_specs.items():
            if parameter == "snap_depth" and best["state_prep_method"] != "snap_d":
                continue
            rows = _oat_sensitivity_rows(
                best,
                parameter=parameter,
                values=values,
                num_qubits=args.num_qubits,
                alpha=args.alpha,
                grid_spacing=args.grid_spacing,
                total_time=args.total_time,
                init_basis_index=args.init_basis_index,
                n_fock=args.n_fock,
                n_quad=args.n_quad,
                coeff_backend=args.coeff_backend,
                snap_restarts=args.snap_restarts,
                snap_maxiter=args.snap_maxiter,
                ranking_objective=args.ranking_objective,
            )
            csv_path = out_dir / f"oat_{parameter}.csv"
            _rows_to_csv(rows, csv_path)
            plot_paths: Dict[str, str] = {}
            for metric, ylabel in (
                ("fidelity", "fidelity"),
                ("oracle_fidelity", "oracle_fidelity"),
                ("score", f"score ({args.ranking_objective})"),
            ):
                png_path = out_dir / f"{metric}_vs_{parameter}.png"
                _plot_parameter_curve(rows, parameter, png_path, metric=metric, ylabel=ylabel)
                plot_paths[metric] = str(png_path)
            oat_outputs[parameter] = {"csv": str(csv_path), "plots": plot_paths}

    summary["oat_outputs"] = oat_outputs
    with (out_dir / "sweep_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
