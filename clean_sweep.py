#!/usr/bin/env python3
"""Hyperparameter sweep and sensitivity analysis for the clean stack.

This client script evaluates the independent clean runtime on a fixed benchmark
while varying kernel, Trotter, and state-preparation hyperparameters.

The ranking objectives are intentionally separated because the notion of "best"
depends on the task:

- ``pde``: maximize fidelity against the exact DV reference ``exp(-A T)``,
- ``oracle``: maximize CV state-preparation fidelity only,
- ``oracle_only``: explicit alias of ``oracle`` for state-preparation-only
  scans,
- ``prep_pde``: balance state-preparation fidelity and PDE fidelity,
- ``balanced``: balance PDE fidelity and postselection probability,
- ``truncated``: maximize fidelity to the truncated CV reference.
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
    build_neumann_heat_system,
    build_periodic_heat_system,
)
from clean_oracles import (
    OraclePreparation,
    compute_lchs_coefficients,
    prepare_cv_oracle,
    snap_parameter_payload,
)
from clean_hybrid import run_clean_lchs


RANKING_OBJECTIVES = (
    "balanced",
    "pde",
    "prep_pde",
    "oracle",
    "oracle_only",
    "truncated",
)
BOUNDARY_CONDITIONS = ("dirichlet", "neumann", "periodic")


@dataclass(frozen=True)
class SweepCandidate:
    """One point in the sweep hyperparameter grid."""

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


def _build_heat_system(
    boundary_condition: str,
    *,
    num_qubits: int,
    alpha: float,
    grid_spacing: float,
    total_time: float,
    init_basis_index: int,
):
    """Build the requested heat-equation benchmark system.

    Args:
        boundary_condition: One of ``dirichlet``, ``neumann``, or ``periodic``.
        num_qubits: Number of DV qubits in the benchmark.
        alpha: Diffusion coefficient.
        grid_spacing: Spatial lattice spacing.
        total_time: Evolution time.
        init_basis_index: Initial DV basis state when no explicit initial state
            is supplied.

    Returns:
        Pauli-system specification for the chosen heat boundary condition.

    Raises:
        ValueError: If ``boundary_condition`` is unsupported.
    """

    if boundary_condition == "dirichlet":
        return build_dirichlet_heat_system(
            num_qubits=num_qubits,
            alpha=alpha,
            grid_spacing=grid_spacing,
            total_time=total_time,
            init_basis_index=init_basis_index,
        )
    if boundary_condition == "neumann":
        return build_neumann_heat_system(
            num_qubits=num_qubits,
            alpha=alpha,
            grid_spacing=grid_spacing,
            total_time=total_time,
            init_basis_index=init_basis_index,
        )
    if boundary_condition == "periodic":
        return build_periodic_heat_system(
            num_qubits=num_qubits,
            alpha=alpha,
            grid_spacing=grid_spacing,
            total_time=total_time,
            init_basis_index=init_basis_index,
        )
    raise ValueError(
        f"Unknown boundary_condition '{boundary_condition}'. Expected one of {BOUNDARY_CONDITIONS}."
    )


def _score(row: Dict[str, Any], objective: str) -> float:
    """Return the scalar score for one sweep row under a chosen objective."""

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
    if objective in {"oracle", "oracle_only"}:
        return oracle
    if objective == "truncated":
        return truncated
    raise ValueError(
        f"Unknown ranking objective '{objective}'. Expected one of {RANKING_OBJECTIVES}."
    )


def _score_columns(row: Dict[str, Any]) -> Dict[str, float]:
    """Compute all supported ranking scores for one row."""

    return {f"score_{objective}": _score(row, objective) for objective in RANKING_OBJECTIVES}


def _best_row_for_objective(
    rows: Sequence[Dict[str, Any]], objective: str
) -> Dict[str, Any] | None:
    """Return the best valid row under one ranking objective."""

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
    """Translate a sweep candidate into clean-stack specs."""

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


def _evaluate_candidate_internal(
    candidate: SweepCandidate,
    *,
    boundary_condition: str,
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
    warm_start_oracle: OraclePreparation | None = None,
) -> Tuple[Dict[str, Any], OraclePreparation | None]:
    """Evaluate one sweep candidate on the selected heat benchmark.

    Args:
        candidate: Hyperparameter point to evaluate.
        boundary_condition: Heat boundary condition for the benchmark.
        num_qubits: Number of DV qubits in the benchmark.
        alpha: Diffusion coefficient of the heat equation.
        grid_spacing: Spatial lattice spacing.
        total_time: Evolution time.
        init_basis_index: Initial DV basis state when no explicit initial state
            is supplied.
        n_fock: Oscillator truncation dimension.
        n_quad: Numerical quadrature budget for coefficient generation.
        coeff_backend: Coefficient backend name.
        snap_restarts: Number of random restarts for SNAP+D optimization.
        snap_maxiter: Maximum iterations per SNAP+D restart.
        ranking_objective: Objective used to fill the ``score`` column.

    Returns:
        Tuple ``(row, oracle)`` containing the flat summary row and the
        resolved oracle preparation used for this evaluation. The returned
        oracle lets a sweep reuse a shallower SNAP+D solution as a warm start
        for a deeper depth at the same kernel point.
    """

    row: Dict[str, Any] = {
        "boundary_condition": boundary_condition,
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
        return row, None

    try:
        system = _build_heat_system(
            boundary_condition,
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
        coeffs = compute_lchs_coefficients(kernel)
        oracle = prepare_cv_oracle(kernel, prep, coeffs=coeffs, warm_start=warm_start_oracle)
        result = run_clean_lchs(
            system,
            kernel,
            prep,
            evolution,
            coeffs=coeffs,
            oracle=oracle,
        )
        oracle_metadata = dict(result.metadata.get("oracle_metadata", {}))
        snap_payload = snap_parameter_payload(oracle)
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
                "oracle_n_sqr_pulses": oracle_metadata.get("n_sqr_pulses", ""),
                "oracle_n_qubit_rotations": oracle_metadata.get("n_qubit_rotations", ""),
                "oracle_le_aux_ground_probability": oracle_metadata.get(
                    "le_aux_ground_probability", ""
                ),
                "oracle_vs_ideal_fidelity": float(
                    result.metadata.get("oracle_vs_ideal_fidelity", np.nan)
                ),
                "ideal_vs_reference_fidelity": float(
                    result.metadata.get("ideal_vs_reference_fidelity", np.nan)
                ),
                "snap_n_snap": oracle_metadata.get("snap_n_snap", ""),
                "snap_total_iterations": oracle_metadata.get("snap_total_iterations", ""),
                "snap_total_starts": oracle_metadata.get("snap_total_starts", ""),
                "snap_used_warm_start": oracle_metadata.get("snap_used_warm_start", ""),
                "snap_restarts": oracle_metadata.get("snap_restarts", prep.snap_restarts),
                "snap_maxiter": oracle_metadata.get("snap_maxiter", prep.snap_maxiter),
                "snap_parameter_payload_json": (
                    json.dumps(snap_payload, separators=(",", ":"))
                    if snap_payload is not None
                    else ""
                ),
                "score_objective": ranking_objective,
            }
        )
        row.update(_score_columns(row))
        row["score"] = _score(row, ranking_objective)
    except Exception as exc:
        row.update({"valid": False, "error": f"{type(exc).__name__}: {exc}"})
        oracle = None
    return row, oracle


def evaluate_candidate(
    candidate: SweepCandidate,
    *,
    boundary_condition: str,
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
    warm_start_oracle: OraclePreparation | None = None,
) -> Dict[str, Any]:
    """Evaluate one sweep candidate and return only the flat output row."""

    row, _ = _evaluate_candidate_internal(
        candidate,
        boundary_condition=boundary_condition,
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
        warm_start_oracle=warm_start_oracle,
    )
    return row


def _evaluate_snap_depth_continuation(
    *,
    r_target: float,
    r_prime: float,
    beta: float,
    n_coeff: int,
    n_trotter_steps: int,
    boundary_condition: str,
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
    snap_depths: Sequence[int],
    warm_start_depths: bool,
) -> List[Dict[str, Any]]:
    """Evaluate one fixed kernel point across multiple SNAP+D depths.

    When ``warm_start_depths`` is enabled, each deeper ansatz is initialized
    from the best shallower oracle by copying existing layers and padding the
    extra layers with identity actions.
    """

    rows: List[Dict[str, Any]] = []
    warm_start_oracle: OraclePreparation | None = None

    for snap_depth in sorted(set(int(depth) for depth in snap_depths)):
        candidate = SweepCandidate(
            r_target=r_target,
            r_prime=r_prime,
            beta=beta,
            n_coeff=n_coeff,
            n_trotter_steps=n_trotter_steps,
            state_prep_method="snap_d",
            snap_depth=snap_depth,
        )
        row, oracle = _evaluate_candidate_internal(
            candidate,
            boundary_condition=boundary_condition,
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
            warm_start_oracle=warm_start_oracle if warm_start_depths else None,
        )
        rows.append(row)
        if warm_start_depths and row.get("valid") and oracle is not None:
            warm_start_oracle = oracle

    return rows


def _rows_to_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    """Write heterogeneous row dictionaries to a CSV file."""

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
    boundary_condition: str,
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
                boundary_condition=boundary_condition,
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
            boundary_condition=boundary_condition,
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
    boundary_condition: str,
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
    snap_warm_start_depths: bool,
) -> List[Dict[str, Any]]:
    if parameter == "snap_depth" and best_row["state_prep_method"] == "snap_d":
        rows = _evaluate_snap_depth_continuation(
            r_target=float(best_row["r_target"]),
            r_prime=float(best_row["r_prime"]),
            beta=float(best_row["beta"]),
            n_coeff=int(best_row["n_coeff"]),
            n_trotter_steps=int(best_row["n_trotter_steps"]),
            boundary_condition=boundary_condition,
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
            snap_depths=values,
            warm_start_depths=snap_warm_start_depths,
        )
        for row in rows:
            row["source"] = f"oat_{parameter}"
        return rows

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
            boundary_condition=boundary_condition,
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
    parser.add_argument(
        "--boundary-condition",
        choices=BOUNDARY_CONDITIONS,
        default="dirichlet",
        help="Heat-equation boundary condition used by the benchmark.",
    )
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
    parser.add_argument("--prep-method-grid", default="injection,snap_d,law_eberly")
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
    parser.add_argument(
        "--snap-warm-start-depths",
        action="store_true",
        help=(
            "For SNAP+D depth scans at a fixed kernel point, initialize each deeper "
            "depth from the best shallower oracle instead of cold-starting every depth."
        ),
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
        if method == "snap_d":
            rows = _evaluate_snap_depth_continuation(
                r_target=r_target,
                r_prime=r_prime,
                beta=beta,
                n_coeff=n_coeff,
                n_trotter_steps=n_trotter,
                boundary_condition=args.boundary_condition,
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
                snap_depths=snap_depth_grid,
                warm_start_depths=args.snap_warm_start_depths,
            )
            for row in rows:
                row["source"] = "grid"
                sweep_rows.append(row)
            continue

        candidate = SweepCandidate(
            r_target=r_target,
            r_prime=r_prime,
            beta=beta,
            n_coeff=n_coeff,
            n_trotter_steps=n_trotter,
            state_prep_method=method,
            snap_depth=0,
        )
        row = evaluate_candidate(
            candidate,
            boundary_condition=args.boundary_condition,
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
            boundary_condition=args.boundary_condition,
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
        "boundary_condition": args.boundary_condition,
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
            "boundary_condition": args.boundary_condition,
            "snap_restarts": args.snap_restarts,
            "snap_maxiter": args.snap_maxiter,
            "snap_warm_start_depths": args.snap_warm_start_depths,
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
                boundary_condition=args.boundary_condition,
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
                snap_warm_start_depths=args.snap_warm_start_depths,
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
