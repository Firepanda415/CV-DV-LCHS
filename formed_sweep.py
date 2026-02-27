#!/usr/bin/env python3
"""
Parameter sweep for CV-DV LCHS with two objectives:
  1) maximize circuit-vs-theory vector fidelity
  2) maximize post-selection probability

This script reuses the "formed" pipeline:
  - `formed_qutip.py` for LCHS coefficient generation and theory reference.
  - `formed_bosonic.py` for CV-DV circuit assembly and post-selection.

Rows are ranked by Pareto-front dominance on (fidelity, post_prob), then by a
balanced scalar score for deterministic ordering.

No CLI is used. Edit the parameter block in `__main__`.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

import numpy as np
from bosonic_qiskit import util as cv_util
from scipy.optimize import minimize

from formed_bosonic import (
    BosonicParams,
    build_bosonic_circuit,
    detect_statevector_layout,
    extract_postselected_dv,
)
from formed_qutip import (
    LCHSParams,
    ODESystemFromPauli,
    classical_map,
    fit_global_scale,
    lchs_seed_coefficients,
    state_fidelity,
)


@dataclass
class SweepConfig:
    """Shared sweep settings."""

    total_time: float
    max_fock_level: int
    n_quad: int
    coeff_method: str
    init_basis_index: int
    snap_depth: int
    snap_restarts: int
    snap_maxiter: int
    output_dir: str
    top_k: int


@dataclass
class LocalSearchConfig:
    """Continuous local refinement settings around top sweep points."""

    enabled: bool
    optimizers: Tuple[str, ...]
    n_seed_points: int
    maxiter: int
    maxfev: int
    r_target_bounds: Tuple[float, float]
    r_prime_bounds: Tuple[float, float]
    beta_bounds: Tuple[float, float]


def _basis_state(dim: int, index: int) -> np.ndarray:
    """Computational-basis vector |index> in C^dim."""
    out = np.zeros(dim, dtype=complex)
    out[index] = 1.0
    return out


def _prep_fidelity_from_meta(prep_meta: Dict[str, float]) -> float:
    """Best available scalar state-prep fidelity from method metadata."""
    for key in ("stateprep_seed_fidelity", "stateprep_analytic_fidelity", "preparation_fidelity"):
        if key in prep_meta:
            return float(prep_meta[key])
    return float("nan")


def _safe_fidelity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Numerically safe fidelity wrapper when the observed vector is near zero."""
    if np.linalg.norm(v1) < 1e-15 or np.linalg.norm(v2) < 1e-15:
        return 0.0
    return float(state_fidelity(v1, v2))


def _multi_objective_score(fidelity: float, post_prob: float) -> float:
    """
    Balanced scalarization for two objectives.

    Uses geometric mean so both terms must be high:
      score = sqrt(fidelity * post_prob)
    """
    f = max(0.0, float(fidelity))
    p = max(0.0, float(post_prob))
    return float(np.sqrt(f * p))


def expanded_bounds(
    values: Sequence[float],
    *,
    pad_low: float,
    pad_high: float,
    floor: float,
    ceil: float,
) -> Tuple[float, float]:
    """Expand a sweep range with margins, clipped to hard physical bounds."""
    if floor > ceil:
        raise ValueError(f"Invalid hard bounds: floor={floor} > ceil={ceil}")
    lo = max(floor, float(min(values)) - pad_low)
    hi = min(ceil, float(max(values)) + pad_high)
    if lo > hi:
        # Seed values can sit outside hard bounds; fall back to full hard interval.
        return (float(floor), float(ceil))
    return (lo, hi)


def evaluate_candidate(
    *,
    system: ODESystemFromPauli,
    cfg: SweepConfig,
    reference_vec: np.ndarray,
    detected_layout: str,
    coeff_cache: Dict[Tuple[float, float, float, int, int, str, int], np.ndarray],
    r_target: float,
    r_prime: float,
    beta: float,
    n_coeff: int,
    n_trotter_steps: int,
    state_prep_method: str,
    source: str = "grid",
) -> Dict[str, object]:
    """
    Evaluate one sweep point and return scalar metrics for ranking/export.

    Ranking uses multi-objective metrics:
      - fidelity
      - post_prob
    and stores a balanced scalar score for deterministic tie-breaking.
    """
    row: Dict[str, object] = {
        "source": source,
        "r_target": float(r_target),
        "r_prime": float(r_prime),
        "beta": float(beta),
        "n_coeff": int(n_coeff),
        "n_trotter_steps": int(n_trotter_steps),
        "state_prep_method": state_prep_method,
    }

    if r_target <= r_prime or r_prime < 0.0 or not (0.0 < beta < 1.0):
        row.update({"valid": False, "error": "invalid_lchs_constraints"})
        return row
    if n_coeff <= 0 or n_coeff > cfg.max_fock_level:
        row.update({"valid": False, "error": "invalid_n_coeff"})
        return row
    if n_trotter_steps <= 0:
        row.update({"valid": False, "error": "invalid_n_trotter_steps"})
        return row

    start = perf_counter()
    try:
        lchs_params = LCHSParams(
            total_time=cfg.total_time,
            n_fock=cfg.max_fock_level,
            n_coeff=n_coeff,
            r_target=r_target,
            r_prime=r_prime,
            beta=beta,
            n_quad=cfg.n_quad,
            coeff_method=cfg.coeff_method,
        )

        coeff_key = (
            float(r_target),
            float(r_prime),
            float(beta),
            int(n_coeff),
            int(cfg.n_quad),
            cfg.coeff_method,
            int(cfg.max_fock_level),
        )
        coeffs_seed = coeff_cache.get(coeff_key)
        if coeffs_seed is None:
            coeffs_seed = lchs_seed_coefficients(lchs_params)
            coeff_cache[coeff_key] = coeffs_seed

        bosonic_params = BosonicParams(
            max_fock_level=cfg.max_fock_level,
            n_trotter_steps=n_trotter_steps,
            state_prep_method=state_prep_method,
            snap_depth=cfg.snap_depth,
            snap_restarts=cfg.snap_restarts,
            snap_maxiter=cfg.snap_maxiter,
        )

        qc, prep_meta = build_bosonic_circuit(
            system=system,
            lchs_params=lchs_params,
            bosonic_params=bosonic_params,
            coeffs_seed=coeffs_seed,
            init_basis_index=cfg.init_basis_index,
        )

        state, _, _ = cv_util.simulate(
            qc,
            shots=1,
            return_fockcounts=False,
            add_save_statevector=True,
        )
        statevec = np.asarray(state.data, dtype=complex)
        dv_vec = extract_postselected_dv(
            statevec,
            layout=detected_layout,
            max_fock_level=cfg.max_fock_level,
            n_dv_qubits=system.n_qubits(),
        )

        eta, rel_error = fit_global_scale(dv_vec, reference_vec)
        post_prob = float(np.real(np.vdot(dv_vec, dv_vec)))
        fidelity = _safe_fidelity(dv_vec, reference_vec)

        elapsed = perf_counter() - start
        mo_score = _multi_objective_score(fidelity, post_prob)
        row.update(
            {
                "valid": True,
                "fidelity": float(fidelity),
                "post_prob": float(post_prob),
                "mo_score": float(mo_score),
                "rel_error": float(rel_error),
                "eta_real": float(np.real(eta)),
                "eta_imag": float(np.imag(eta)),
                "prep_fidelity": _prep_fidelity_from_meta(prep_meta),
                "circuit_depth": int(qc.depth()),
                "circuit_size": int(qc.size()),
                "runtime_sec": float(elapsed),
            }
        )
        return row
    except Exception as exc:  # noqa: BLE001 - keep sweep resilient
        row.update(
            {
                "valid": False,
                "error": type(exc).__name__,
                "runtime_sec": float(perf_counter() - start),
            }
        )
        return row


def _dominates(a: Dict[str, object], b: Dict[str, object]) -> bool:
    """True iff row a Pareto-dominates row b for maximizing (fidelity, post_prob)."""
    af, ap = float(a["fidelity"]), float(a["post_prob"])
    bf, bp = float(b["fidelity"]), float(b["post_prob"])
    return (af >= bf and ap >= bp) and (af > bf or ap > bp)


def rank_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """
    Multi-objective ranking:
      1) lower Pareto-front rank (front 1 is non-dominated set)
      2) higher scalar balanced score sqrt(fidelity * post_prob)
      3) higher fidelity
      4) higher post_prob
      5) lower relative error
    """
    valid_rows = [r for r in rows if bool(r.get("valid", False))]
    if not valid_rows:
        return []

    # Assign Pareto-front rank (O(N^2), acceptable for sweep sizes here).
    remaining = list(range(len(valid_rows)))
    rank = 1
    while remaining:
        front: List[int] = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                if _dominates(valid_rows[j], valid_rows[i]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        for i in front:
            valid_rows[i]["pareto_rank"] = int(rank)
        remaining = [i for i in remaining if i not in front]
        rank += 1

    # Ensure mo_score exists for all valid rows.
    for r in valid_rows:
        if "mo_score" not in r:
            r["mo_score"] = _multi_objective_score(float(r["fidelity"]), float(r["post_prob"]))

    valid_rows.sort(
        key=lambda r: (
            int(r.get("pareto_rank", 10**9)),
            -float(r["mo_score"]),
            -float(r["fidelity"]),
            -float(r["post_prob"]),
            float(r["rel_error"]),
        )
    )
    return valid_rows


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    """Write homogeneous CSV with a stable column order."""
    fieldnames = [
        "source",
        "valid",
        "error",
        "r_target",
        "r_prime",
        "beta",
        "n_coeff",
        "n_trotter_steps",
        "state_prep_method",
        "fidelity",
        "post_prob",
        "mo_score",
        "pareto_rank",
        "rel_error",
        "eta_real",
        "eta_imag",
        "prep_fidelity",
        "circuit_depth",
        "circuit_size",
        "runtime_sec",
        "optimizer",
        "seed_rank",
        "seed_method",
        "seed_n_coeff",
        "seed_n_trotter",
        "seed_fidelity",
        "seed_post_prob",
        "seed_mo_score",
        "opt_success",
        "opt_message",
        "opt_fun",
        "opt_nfev",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_default(obj: object) -> object:
    """JSON serializer for numpy scalar types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")


def run_sweep(
    *,
    system: ODESystemFromPauli,
    cfg: SweepConfig,
    reference_vec: np.ndarray,
    r_target_values: Sequence[float],
    r_prime_values: Sequence[float],
    beta_values: Sequence[float],
    n_coeff_values: Sequence[int],
    n_trotter_values: Sequence[int],
    state_prep_methods: Sequence[str],
) -> List[Dict[str, object]]:
    """Evaluate all sweep combinations and return raw rows (valid + invalid)."""
    n_cases = (
        len(r_target_values)
        * len(r_prime_values)
        * len(beta_values)
        * len(n_coeff_values)
        * len(n_trotter_values)
        * len(state_prep_methods)
    )
    print(f"Sweep cases: {n_cases}")

    layout = detect_statevector_layout(cfg.max_fock_level, system.n_qubits())
    print(f"Detected statevector layout: {layout}")

    coeff_cache: Dict[Tuple[float, float, float, int, int, str, int], np.ndarray] = {}
    rows: List[Dict[str, object]] = []

    for idx, (rt, rp, beta, n_coeff, n_trot, method) in enumerate(
        product(
            r_target_values,
            r_prime_values,
            beta_values,
            n_coeff_values,
            n_trotter_values,
            state_prep_methods,
        ),
        start=1,
    ):
        print(
            f"[{idx:4d}/{n_cases}] "
            f"r={rt:.3f}, r'={rp:.3f}, beta={beta:.3f}, "
            f"n_coeff={n_coeff}, trotter={n_trot}, prep={method}"
        )
        row = evaluate_candidate(
            system=system,
            cfg=cfg,
            reference_vec=reference_vec,
            detected_layout=layout,
            coeff_cache=coeff_cache,
            r_target=float(rt),
            r_prime=float(rp),
            beta=float(beta),
            n_coeff=int(n_coeff),
            n_trotter_steps=int(n_trot),
            state_prep_method=str(method),
        )
        rows.append(row)
        if row.get("valid", False):
            print(
                f"      fidelity={row['fidelity']:.8f}, post_prob={row['post_prob']:.3e}, "
                f"mo_score={row['mo_score']:.6f}, rel_err={row['rel_error']:.3e}"
            )
        else:
            print(f"      invalid: {row.get('error', 'unknown')}")
    return rows


def run_local_search(
    *,
    system: ODESystemFromPauli,
    cfg: SweepConfig,
    local_cfg: LocalSearchConfig,
    reference_vec: np.ndarray,
    seed_rows: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """
    Refine continuous parameters (r_target, r_prime, beta) from top seed points.

    Discrete choices (state_prep_method, n_coeff, n_trotter_steps) are fixed per seed.
    Objective is multi-objective scalarization: minimize 1 - sqrt(fidelity*post_prob).
    """
    if not local_cfg.enabled:
        return []
    if not seed_rows:
        return []

    valid_seeds = [r for r in seed_rows if bool(r.get("valid", False))]
    if not valid_seeds:
        return []
    seeds = valid_seeds[: local_cfg.n_seed_points]

    layout = detect_statevector_layout(cfg.max_fock_level, system.n_qubits())
    coeff_cache: Dict[Tuple[float, float, float, int, int, str, int], np.ndarray] = {}
    bounds = [local_cfg.r_target_bounds, local_cfg.r_prime_bounds, local_cfg.beta_bounds]
    bounds_arr = np.asarray(bounds, dtype=float)

    out_rows: List[Dict[str, object]] = []
    for seed_rank, seed in enumerate(seeds, start=1):
        x0 = np.array(
            [
                float(seed["r_target"]),
                float(seed["r_prime"]),
                float(seed["beta"]),
            ],
            dtype=float,
        )
        n_coeff = int(seed["n_coeff"])
        n_trotter = int(seed["n_trotter_steps"])
        method = str(seed["state_prep_method"])
        seed_fidelity = float(seed["fidelity"])
        seed_post_prob = float(seed["post_prob"])
        seed_mo_score = float(seed.get("mo_score", _multi_objective_score(seed_fidelity, seed_post_prob)))
        x0_bounded = np.clip(x0, bounds_arr[:, 0], bounds_arr[:, 1])

        for opt_name in local_cfg.optimizers:
            print(
                f"  Local seed #{seed_rank}: opt={opt_name}, "
                f"prep={method}, n_coeff={n_coeff}, trotter={n_trotter}, "
                f"x0=({x0[0]:.4f}, {x0[1]:.4f}, {x0[2]:.4f})"
            )

            def objective(x: np.ndarray) -> float:
                rt, rp, bt = float(x[0]), float(x[1]), float(x[2])

                # Keep unconstrained optimizers numerically stable with soft penalties.
                penalty = 0.0
                for val, (lo, hi) in zip((rt, rp, bt), bounds):
                    if val < lo:
                        penalty += (lo - val) ** 2
                    elif val > hi:
                        penalty += (val - hi) ** 2
                if penalty > 0.0:
                    return 10.0 + 100.0 * penalty

                row = evaluate_candidate(
                    system=system,
                    cfg=cfg,
                    reference_vec=reference_vec,
                    detected_layout=layout,
                    coeff_cache=coeff_cache,
                    r_target=rt,
                    r_prime=rp,
                    beta=bt,
                    n_coeff=n_coeff,
                    n_trotter_steps=n_trotter,
                    state_prep_method=method,
                    source="local_eval",
                )
                if not bool(row.get("valid", False)):
                    return 10.0
                return 1.0 - float(row["mo_score"])

            options = {"maxiter": int(local_cfg.maxiter), "maxfev": int(local_cfg.maxfev)}
            if opt_name in ("Powell", "L-BFGS-B", "TNC", "SLSQP"):
                res = minimize(objective, x0_bounded, method=opt_name, bounds=bounds, options=options)
            else:
                res = minimize(objective, x0, method=opt_name, options=options)

            rt_opt, rp_opt, bt_opt = (float(res.x[0]), float(res.x[1]), float(res.x[2]))
            row_opt = evaluate_candidate(
                system=system,
                cfg=cfg,
                reference_vec=reference_vec,
                detected_layout=layout,
                coeff_cache=coeff_cache,
                r_target=rt_opt,
                r_prime=rp_opt,
                beta=bt_opt,
                n_coeff=n_coeff,
                n_trotter_steps=n_trotter,
                state_prep_method=method,
                source="local_search",
            )
            row_opt.update(
                {
                    "optimizer": opt_name,
                    "seed_rank": seed_rank,
                    "seed_method": method,
                    "seed_n_coeff": n_coeff,
                    "seed_n_trotter": n_trotter,
                    "seed_fidelity": seed_fidelity,
                    "seed_post_prob": seed_post_prob,
                    "seed_mo_score": seed_mo_score,
                    "opt_success": bool(res.success),
                    "opt_message": str(res.message),
                    "opt_fun": float(res.fun),
                    "opt_nfev": int(res.nfev),
                }
            )
            out_rows.append(row_opt)

            if bool(row_opt.get("valid", False)):
                print(
                    f"      -> fidelity={row_opt['fidelity']:.8f}, post_prob={row_opt['post_prob']:.3e}, "
                    f"mo_score={row_opt['mo_score']:.6f}, rel_err={row_opt['rel_error']:.3e}"
                )
            else:
                print(f"      -> invalid: {row_opt.get('error', 'unknown')}")

    return out_rows


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # System model (same heat-equation setup as formed_qutip/formed_bosonic)
    # -----------------------------------------------------------------
    alpha = 1.0
    h_grid = 1.0
    s = alpha / (h_grid**2)

    system = ODESystemFromPauli(
        pauli_strings_l=["II", "IX", "XX", "YY"],
        coeffs_l=[2.0 * s, -1.0 * s, -0.5 * s, -0.5 * s],
        pauli_strings_h=["II"],
        coeffs_h=[0.0],
    )

    # -----------------------------------------------------------------
    # Sweep configuration
    # -----------------------------------------------------------------
    cfg = SweepConfig(
        total_time=1.0,
        max_fock_level=64,
        n_quad=300,
        coeff_method="explicit_overlap",
        init_basis_index=1,
        snap_depth=10,
        snap_restarts=2,
        snap_maxiter=100,
        output_dir="results_formed/sweep",
        top_k=15,
    )

    # User-requested widened search range:
    #   r_target in [0.5, 5], r_prime in [0.01, 4], beta in [0.3, 0.99]
    r_target_values = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    r_prime_values = [0.01, 0.5, 1.0, 2.0, 3.0, 4.0]
    beta_values = [0.30, 0.45, 0.60, 0.75, 0.90, 0.99]
    n_coeff_values = [24, 48]
    n_trotter_values = [100, 200]
    state_prep_methods = ["injection"]

    local_cfg = LocalSearchConfig(
        enabled=True,
        optimizers=("Nelder-Mead", "Powell"),
        n_seed_points=2,
        maxiter=18,
        maxfev=18,
        # Keep local search inside the same requested physical box.
        r_target_bounds=(0.5, 5.0),
        r_prime_bounds=(0.01, 4.0),
        beta_bounds=(0.3, 0.99),
    )

    # -----------------------------------------------------------------
    # Theoretical reference vector: u_ref(T) = exp(-A T) u_0
    # -----------------------------------------------------------------
    dv_dim = 2 ** system.n_qubits()
    u0 = _basis_state(dv_dim, cfg.init_basis_index)
    ref_map = classical_map(system.generator_matrix(), cfg.total_time)
    u_ref = ref_map @ u0

    # -----------------------------------------------------------------
    # Run sweep
    # -----------------------------------------------------------------
    rows = run_sweep(
        system=system,
        cfg=cfg,
        reference_vec=u_ref,
        r_target_values=r_target_values,
        r_prime_values=r_prime_values,
        beta_values=beta_values,
        n_coeff_values=n_coeff_values,
        n_trotter_values=n_trotter_values,
        state_prep_methods=state_prep_methods,
    )

    ranked_grid = rank_rows(rows)
    n_grid_valid = len(ranked_grid)
    n_grid_total = len(rows)
    print(f"\nValid grid points: {n_grid_valid}/{n_grid_total}")
    if n_grid_valid == 0:
        raise RuntimeError("No valid sweep points were produced.")
    best_grid = ranked_grid[0]
    print(
        f"Best grid (Pareto rank={best_grid['pareto_rank']}) "
        f"fidelity={best_grid['fidelity']:.10f}, post_prob={best_grid['post_prob']:.6e}, "
        f"mo_score={best_grid['mo_score']:.6f} "
        f"at r={best_grid['r_target']}, r'={best_grid['r_prime']}, beta={best_grid['beta']}"
    )

    local_rows = run_local_search(
        system=system,
        cfg=cfg,
        local_cfg=local_cfg,
        reference_vec=u_ref,
        seed_rows=ranked_grid,
    )

    rows_all = list(rows) + list(local_rows)
    ranked = rank_rows(rows_all)
    n_valid = len(ranked)
    n_total = len(rows_all)
    print(f"\nValid total points (grid + local): {n_valid}/{n_total}")
    if n_valid == 0:
        raise RuntimeError("No valid points after local search.")

    best = ranked[0]
    print("\n=== Best Multi-Objective Point ===")
    print(
        f"pareto_rank={best['pareto_rank']}, fidelity={best['fidelity']:.10f}, "
        f"post_prob={best['post_prob']:.6e}, mo_score={best['mo_score']:.6f}, "
        f"rel_error={best['rel_error']:.6e}"
    )
    print(
        f"r={best['r_target']}, r'={best['r_prime']}, beta={best['beta']}, "
        f"n_coeff={best['n_coeff']}, trotter={best['n_trotter_steps']}, "
        f"prep={best['state_prep_method']}"
    )
    if float(best["mo_score"]) > float(best_grid["mo_score"]):
        print(
            f"Improvement over best grid: +{float(best['mo_score']) - float(best_grid['mo_score']):.6e} mo_score"
        )
    else:
        print("Local search did not beat the best grid point on multi-objective score.")

    # -----------------------------------------------------------------
    # Save artifacts
    # -----------------------------------------------------------------
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_csv = out_dir / "sweep_grid.csv"
    local_csv = out_dir / "sweep_local.csv"
    all_csv = out_dir / "sweep_all.csv"
    top_csv = out_dir / "sweep_top.csv"
    best_json = out_dir / "sweep_best.json"

    _write_csv(grid_csv, rows)
    _write_csv(local_csv, local_rows)
    _write_csv(all_csv, rows_all)
    _write_csv(top_csv, ranked[: cfg.top_k])

    summary = {
        "best": best,
        "top_k": ranked[: cfg.top_k],
        "n_total": n_total,
        "n_valid": n_valid,
        "grid_best": best_grid,
        "n_grid_total": n_grid_total,
        "n_grid_valid": n_grid_valid,
        "n_local_total": len(local_rows),
        "n_local_valid": len([r for r in local_rows if bool(r.get("valid", False))]),
        "config": {
            "sweep": {
                "r_target_values": r_target_values,
                "r_prime_values": r_prime_values,
                "beta_values": beta_values,
                "n_coeff_values": n_coeff_values,
                "n_trotter_values": n_trotter_values,
                "state_prep_methods": state_prep_methods,
            },
            "local_search": {
                "enabled": local_cfg.enabled,
                "optimizers": list(local_cfg.optimizers),
                "n_seed_points": local_cfg.n_seed_points,
                "maxiter": local_cfg.maxiter,
                "maxfev": local_cfg.maxfev,
                "r_target_bounds": local_cfg.r_target_bounds,
                "r_prime_bounds": local_cfg.r_prime_bounds,
                "beta_bounds": local_cfg.beta_bounds,
            },
            "runtime": {
                "total_time": cfg.total_time,
                "max_fock_level": cfg.max_fock_level,
                "n_quad": cfg.n_quad,
                "coeff_method": cfg.coeff_method,
                "init_basis_index": cfg.init_basis_index,
            },
            "objective": {
                "type": "pareto_then_geometric_mean",
                "maximize": ["fidelity", "post_prob"],
                "scalar_score": "sqrt(fidelity*post_prob)",
            },
            "snap_d": {
                "snap_depth": cfg.snap_depth,
                "snap_restarts": cfg.snap_restarts,
                "snap_maxiter": cfg.snap_maxiter,
            },
        },
    }
    best_json.write_text(json.dumps(summary, indent=2, default=_json_default))

    print(f"\nWrote: {grid_csv}")
    print(f"Wrote: {local_csv}")
    print(f"\nWrote: {all_csv}")
    print(f"Wrote: {top_csv}")
    print(f"Wrote: {best_json}")
