#!/usr/bin/env python3
"""
Three-parameter optimization for squeezed-Fock CV-DV LCHS.

Optimizes (r_target, r_prime, beta) to jointly maximize:
  - shape fidelity  (direction accuracy)
  - 1 - map_error   (structural accuracy after best-scale fit)
  - postselection probability  (success rate)

Uses Nelder-Mead (zero-order, derivative-free) with a composite objective.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize, differential_evolution

from clacod_heat1d_qutip import (
    HeatLCHSConfig,
    classical_map,
    effective_lchs_map,
    fit_global_scale,
    heat_matrix,
    parse_initial_state,
    prepare_lchs_states,
    state_fidelity,
)


def evaluate(
    r_target: float,
    r_prime: float,
    beta: float,
    n_fock: int = 64,
    n_coeff: int = 64,
    n_quad: int = 300,
    position_convention: str = "sqrt2",
    init_state: str = "basis01",
) -> Dict[str, float]:
    """Evaluate LCHS metrics at a single parameter point."""
    # Enforce constraints
    if r_target <= 0 or r_prime <= 0:
        return {"fidelity": 0.0, "map_error": 1.0, "post_prob": 0.0,
                "eta_error": 1.0, "valid": False}
    if r_prime >= r_target:
        return {"fidelity": 0.0, "map_error": 1.0, "post_prob": 0.0,
                "eta_error": 1.0, "valid": False}
    gamma = np.exp(-2.0 * r_prime) - np.exp(-2.0 * r_target)
    if gamma <= 0:
        return {"fidelity": 0.0, "map_error": 1.0, "post_prob": 0.0,
                "eta_error": 1.0, "valid": False}
    if not (0.0 < beta < 1.0):
        return {"fidelity": 0.0, "map_error": 1.0, "post_prob": 0.0,
                "eta_error": 1.0, "valid": False}

    try:
        cfg = HeatLCHSConfig(
            n_fock=n_fock,
            n_coeff=n_coeff,
            r_target=r_target,
            r_prime=r_prime,
            beta=beta,
            n_quad=n_quad,
            init_state=init_state,
            position_convention=position_convention,
        )
        psi_osc, phi_post, coeffs = prepare_lchs_states(cfg)
        k_map = effective_lchs_map(cfg, psi_osc, phi_post)
        ref_map = classical_map(cfg)
        _, map_err = fit_global_scale(k_map, ref_map)

        u0 = parse_initial_state(cfg.init_state)
        u_cv = k_map @ u0
        u_ref = ref_map @ u0
        fid = state_fidelity(u_cv, u_ref)
        post_prob = float(np.real(np.vdot(u_cv, u_cv)))

        eta = complex(phi_post.overlap(psi_osc))
        eta_err = float("nan")
        if not np.isclose(np.abs(eta), 0.0):
            eta_err = float(
                np.linalg.norm((u_cv / eta) - u_ref)
                / max(np.linalg.norm(u_ref), 1e-15)
            )

        if np.isnan(map_err) or np.isnan(fid):
            return {"fidelity": 0.0, "map_error": 1.0, "post_prob": 0.0,
                    "eta_error": 1.0, "valid": False}

        return {
            "fidelity": fid,
            "map_error": map_err,
            "post_prob": post_prob,
            "eta_error": eta_err if not np.isnan(eta_err) else 1.0,
            "valid": True,
        }
    except Exception:
        return {"fidelity": 0.0, "map_error": 1.0, "post_prob": 0.0,
                "eta_error": 1.0, "valid": False}


def composite_objective(
    params: np.ndarray,
    w_fid: float,
    w_map: float,
    w_post: float,
    n_fock: int,
    n_coeff: int,
    position_convention: str,
) -> float:
    """
    Composite objective to MINIMIZE.

    obj = w_fid * (1 - fidelity) + w_map * map_error + w_post * (1 - post_prob)

    All three terms are in [0, 1] at optimum, so weights are directly comparable.
    """
    r_target, r_prime, beta = params
    m = evaluate(
        r_target=r_target,
        r_prime=r_prime,
        beta=beta,
        n_fock=n_fock,
        n_coeff=n_coeff,
        position_convention=position_convention,
    )
    if not m["valid"]:
        return 10.0  # penalty

    obj = (
        w_fid * (1.0 - m["fidelity"])
        + w_map * m["map_error"]
        + w_post * (1.0 - m["post_prob"])
    )
    return obj


def grid_search(
    r_target_range: np.ndarray,
    r_prime_range: np.ndarray,
    beta_range: np.ndarray,
    n_fock: int,
    n_coeff: int,
    position_convention: str,
) -> List[Dict]:
    """Exhaustive grid search for initial survey."""
    rows = []
    total = len(r_target_range) * len(r_prime_range) * len(beta_range)
    count = 0
    for rt in r_target_range:
        for rp in r_prime_range:
            if rp >= rt:
                continue
            for b in beta_range:
                count += 1
                m = evaluate(
                    r_target=rt, r_prime=rp, beta=b,
                    n_fock=n_fock, n_coeff=n_coeff,
                    position_convention=position_convention,
                )
                if m["valid"]:
                    row = {"r_target": rt, "r_prime": rp, "beta": b}
                    row.update(m)
                    rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="3-param optimization for squeezed-Fock LCHS"
    )
    p.add_argument("--n-fock", type=int, default=64)
    p.add_argument("--n-coeff", type=int, default=64)
    p.add_argument("--position-convention", choices=["half", "sqrt2"], default="sqrt2")

    p.add_argument("--w-fid", type=float, default=1.0,
                    help="Weight for (1 - fidelity) in objective")
    p.add_argument("--w-map", type=float, default=1.0,
                    help="Weight for map_error in objective")
    p.add_argument("--w-post", type=float, default=0.5,
                    help="Weight for (1 - post_prob) in objective")

    p.add_argument("--method", choices=["nelder-mead", "de", "grid+nm"],
                    default="grid+nm",
                    help="Optimization method: nelder-mead, de (differential evolution), "
                         "or grid+nm (grid search then Nelder-Mead refinement)")

    # Grid search resolution
    p.add_argument("--n-grid-r", type=int, default=12,
                    help="Grid points for r_target")
    p.add_argument("--n-grid-rp", type=int, default=10,
                    help="Grid points for r_prime")
    p.add_argument("--n-grid-beta", type=int, default=8,
                    help="Grid points for beta")

    # Search bounds
    p.add_argument("--r-min", type=float, default=0.001)
    p.add_argument("--r-max", type=float, default=2.0)
    p.add_argument("--rp-min", type=float, default=0.0005)
    p.add_argument("--rp-max", type=float, default=1.5)
    p.add_argument("--beta-min", type=float, default=0.2)
    p.add_argument("--beta-max", type=float, default=0.99)

    p.add_argument("--output-dir", type=str, default="results_bosonic/sweep")
    p.add_argument("--top-k", type=int, default=20,
                    help="Number of top results to display")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bounds = [
        (args.r_min, args.r_max),
        (args.rp_min, args.rp_max),
        (args.beta_min, args.beta_max),
    ]

    obj_kwargs = dict(
        w_fid=args.w_fid,
        w_map=args.w_map,
        w_post=args.w_post,
        n_fock=args.n_fock,
        n_coeff=args.n_coeff,
        position_convention=args.position_convention,
    )

    print(f"=== Squeezed-Fock 3-Parameter Optimization ===")
    print(f"n_fock={args.n_fock}, n_coeff={args.n_coeff}, "
          f"convention={args.position_convention}")
    print(f"Objective weights: w_fid={args.w_fid}, w_map={args.w_map}, "
          f"w_post={args.w_post}")
    print(f"Bounds: r=[{args.r_min}, {args.r_max}], "
          f"r'=[{args.rp_min}, {args.rp_max}], "
          f"beta=[{args.beta_min}, {args.beta_max}]")
    print(f"Method: {args.method}")
    print()

    best_params = None
    all_rows = []

    if args.method == "grid+nm":
        # Phase 1: Grid search
        r_grid = np.linspace(args.r_min, args.r_max, args.n_grid_r)
        rp_grid = np.linspace(args.rp_min, args.rp_max, args.n_grid_rp)
        beta_grid = np.linspace(args.beta_min, args.beta_max, args.n_grid_beta)

        print(f"Phase 1: Grid search ({args.n_grid_r} x {args.n_grid_rp} "
              f"x {args.n_grid_beta})...")
        all_rows = grid_search(
            r_grid, rp_grid, beta_grid,
            args.n_fock, args.n_coeff, args.position_convention,
        )
        print(f"  Evaluated {len(all_rows)} valid points.")

        if not all_rows:
            print("ERROR: No valid parameter points found.")
            return

        # Rank by composite objective
        for row in all_rows:
            row["objective"] = (
                args.w_fid * (1.0 - row["fidelity"])
                + args.w_map * row["map_error"]
                + args.w_post * (1.0 - row["post_prob"])
            )
        all_rows.sort(key=lambda r: r["objective"])

        best_grid = all_rows[0]
        print(f"  Best grid point: r={best_grid['r_target']:.4f}, "
              f"r'={best_grid['r_prime']:.4f}, beta={best_grid['beta']:.4f}")
        print(f"    fid={best_grid['fidelity']:.4f}, "
              f"map_err={best_grid['map_error']:.4f}, "
              f"post={best_grid['post_prob']:.4f}, "
              f"obj={best_grid['objective']:.6f}")

        # Phase 2: Nelder-Mead refinement from top-5 grid points
        print("\nPhase 2: Nelder-Mead refinement from top-5 grid seeds...")
        best_obj = float("inf")
        for i, seed in enumerate(all_rows[:5]):
            x0 = np.array([seed["r_target"], seed["r_prime"], seed["beta"]])
            res = minimize(
                composite_objective,
                x0,
                args=(args.w_fid, args.w_map, args.w_post,
                      args.n_fock, args.n_coeff, args.position_convention),
                method="Nelder-Mead",
                options={"maxiter": 500, "xatol": 1e-5, "fatol": 1e-7},
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
            rt, rp, b = res.x
            m = evaluate(rt, rp, b, args.n_fock, args.n_coeff,
                         position_convention=args.position_convention)
            print(f"  Seed {i+1}: r={rt:.6f}, r'={rp:.6f}, beta={b:.6f} -> "
                  f"fid={m['fidelity']:.4f}, map_err={m['map_error']:.4f}, "
                  f"post={m['post_prob']:.4f}, obj={res.fun:.6f}")

    elif args.method == "de":
        print("Running differential evolution...")
        res = differential_evolution(
            composite_objective,
            bounds=bounds,
            args=(args.w_fid, args.w_map, args.w_post,
                  args.n_fock, args.n_coeff, args.position_convention),
            maxiter=200,
            seed=42,
            tol=1e-7,
            polish=True,
        )
        best_params = res.x
        print(f"  DE converged: {res.message}")

    elif args.method == "nelder-mead":
        x0 = np.array([0.003, 0.001, 0.95])
        print(f"Running Nelder-Mead from x0={x0}...")
        res = minimize(
            composite_objective,
            x0,
            args=(args.w_fid, args.w_map, args.w_post,
                  args.n_fock, args.n_coeff, args.position_convention),
            method="Nelder-Mead",
            options={"maxiter": 1000, "xatol": 1e-6, "fatol": 1e-8},
        )
        best_params = res.x
        print(f"  Converged: {res.message}")

    # Final evaluation at best point
    if best_params is not None:
        rt, rp, b = best_params
        m = evaluate(rt, rp, b, args.n_fock, args.n_coeff,
                     position_convention=args.position_convention)
        print(f"\n=== OPTIMUM ===")
        print(f"  r_target = {rt:.6f}")
        print(f"  r_prime  = {rp:.6f}")
        print(f"  beta     = {b:.6f}")
        print(f"  gamma    = {np.exp(-2*rp) - np.exp(-2*rt):.6e}")
        print(f"  <n>_post = {np.sinh(rt)**2:.4f}")
        print(f"  sigma    = {np.exp(rt):.4f}")
        print(f"  ---")
        print(f"  fidelity       = {m['fidelity']:.6f}")
        print(f"  map_error      = {m['map_error']:.6f}")
        print(f"  post_prob      = {m['post_prob']:.6f}")
        print(f"  eta_error      = {m['eta_error']:.6f}")

        result = {
            "optimum": {
                "r_target": float(rt),
                "r_prime": float(rp),
                "beta": float(b),
                "gamma": float(np.exp(-2*rp) - np.exp(-2*rt)),
                "mean_n_post": float(np.sinh(rt)**2),
                "sigma": float(np.exp(rt)),
            },
            "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else v
                        for k, v in m.items()},
            "config": {
                "n_fock": args.n_fock,
                "n_coeff": args.n_coeff,
                "position_convention": args.position_convention,
                "weights": {"w_fid": args.w_fid, "w_map": args.w_map,
                            "w_post": args.w_post},
                "method": args.method,
            },
        }
        json_path = out_dir / "optimization_result.json"
        json_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"\n  Wrote: {json_path}")

    # Save grid results if we have them
    if all_rows:
        csv_path = out_dir / "grid_search.csv"
        fields = ["r_target", "r_prime", "beta", "fidelity", "map_error",
                   "post_prob", "eta_error", "objective"]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"  Wrote: {csv_path}")

        # Print top-k
        print(f"\n=== Top {args.top_k} Grid Points ===")
        print(f"{'r':>8s} {'r_prime':>8s} {'beta':>6s} "
              f"{'fid':>7s} {'map_err':>8s} {'post':>7s} {'eta_err':>8s} {'obj':>8s}")
        for row in all_rows[:args.top_k]:
            print(f"{row['r_target']:8.4f} {row['r_prime']:8.4f} {row['beta']:6.3f} "
                  f"{row['fidelity']:7.4f} {row['map_error']:8.4f} "
                  f"{row['post_prob']:7.4f} {row['eta_error']:8.4f} "
                  f"{row['objective']:8.5f}")


if __name__ == "__main__":
    main()
