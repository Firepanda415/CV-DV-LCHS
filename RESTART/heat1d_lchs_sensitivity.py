#!/usr/bin/env python3
"""
Theorem-guided sensitivity/optimization for RESTART heat-equation CV-DV LCHS.

Design goals:
- PDE error is the primary optimization target.
- Search domain is constrained by derivation-backed conditions (not random search).
- Expensive circuit calls are amortized by precomputing Fock-component tables per (r, r').
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import heat1d_lchs_cv_dv as core


def deterministic_grid(lo: float, hi: float, n: int) -> np.ndarray:
    if n <= 1:
        return np.array([0.5 * (lo + hi)], dtype=float)
    return np.linspace(lo, hi, n, dtype=float)


def to_key(x: float, ndigits: int = 12) -> float:
    return float(np.round(float(x), ndigits))


def write_csv(rows: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def theorem_constraints(
    r_target: float,
    r_prime: float,
    beta: float,
    gamma_min: float,
    min_gap: float,
) -> Tuple[bool, Dict[str, float], str]:
    gamma = core.coefficient_gamma(r_target, r_prime)
    reasons = []

    if not (0.0 < beta < 1.0):
        reasons.append("beta_out_of_open_interval")

    if not (r_prime < r_target - min_gap):
        reasons.append("invalid_squeeze_gap")

    if gamma <= gamma_min:
        reasons.append("gamma_too_small")

    ok = len(reasons) == 0
    return ok, {"gamma": float(gamma)}, ";".join(reasons)


def objective_value(
    pde_error: float,
    post_prob: float,
    min_post_prob: float,
    post_penalty: float,
    post_reward: float,
) -> float:
    # PDE-priority objective with soft feasibility penalty and small post-selection reward.
    penalty = post_penalty * max(0.0, min_post_prob - post_prob) / max(min_post_prob, 1e-15)
    return float(pde_error + penalty - post_reward * post_prob)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RESTART: theorem-guided sensitivity for 1D heat CV-DV LCHS")

    p.add_argument("--output-dir", type=str, default="RESTART/results")

    # Heat-equation run settings
    p.add_argument("--total-time", type=float, default=1.0)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--h", type=float, default=1.0)
    p.add_argument("--init-q0", type=int, default=0, choices=[0, 1])
    p.add_argument("--init-q1", type=int, default=1, choices=[0, 1])
    p.add_argument("--n-dim", type=int, default=32)
    p.add_argument("--max-fock-level", type=int, default=32)
    p.add_argument("--fock-weight-cutoff", type=float, default=1e-8)

    # Deterministic theorem-constrained search region
    p.add_argument("--r-target-min", type=float, default=0.3)
    p.add_argument("--r-target-max", type=float, default=1.4)
    p.add_argument("--r-prime-min", type=float, default=0.03)
    p.add_argument("--r-prime-max", type=float, default=0.40)
    p.add_argument("--kernel-beta-min", type=float, default=0.35)
    p.add_argument("--kernel-beta-max", type=float, default=0.95)

    # Practical (narrow) range from prior numerical evidence: beta around 0.7-0.8.
    # Historical broad exploratory range (for other examples):
    #   r_target in [0.15, 1.4], r_prime in [0.03, 0.9], beta in [0.0, 1.0].

    p.add_argument("--n-r-target", type=int, default=5)
    p.add_argument("--n-r-prime", type=int, default=5)
    p.add_argument("--n-beta", type=int, default=7)

    # Theorem/proxy constraints
    p.add_argument("--gamma-min", type=float, default=0.03)
    p.add_argument("--min-gap", type=float, default=0.02)
    p.add_argument("--tail-mass-max", type=float, default=1e-4)
    p.add_argument("--n-eff-max", type=float, default=16.0)

    # PDE-priority ranking
    p.add_argument("--min-post-prob", type=float, default=1e-3)
    p.add_argument("--post-penalty", type=float, default=100.0)
    p.add_argument("--post-reward", type=float, default=1e-3)

    # Deterministic local refinement around top seeds
    p.add_argument("--refine-top-k", type=int, default=3)
    p.add_argument("--refine-points", type=int, default=3)
    p.add_argument("--refine-dr", type=float, default=0.08)
    p.add_argument("--refine-drp", type=float, default=0.05)
    p.add_argument("--refine-db", type=float, default=0.08)

    p.add_argument("--disp-phase", type=float, default=0.0)
    p.add_argument("--calibrate-phase", action="store_true")

    return p


def run() -> None:
    args = build_parser().parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = core.HeatEquationConfig(
        total_time=args.total_time,
        n_steps=args.n_steps,
        alpha=args.alpha,
        h=args.h,
        init_qubits=(args.init_q0, args.init_q1),
        n_dim=args.n_dim,
        max_fock_level=args.max_fock_level,
        fock_weight_cutoff=args.fock_weight_cutoff,
    )
    solver = core.Heat1DLCHSSolver(cfg)

    phase = float(args.disp_phase)
    if args.calibrate_phase:
        mid = core.CVLCHSParams(
            r_target=0.5 * (args.r_target_min + args.r_target_max),
            r_prime=0.5 * (args.r_prime_min + args.r_prime_max),
            kernel_beta=0.5 * (args.kernel_beta_min + args.kernel_beta_max),
            disp_phase=phase,
        )
        best_phase, table = core.phase_calibration(solver, mid, [0.0, -np.pi / 2.0, np.pi / 2.0])
        print("Phase calibration:", flush=True)
        for ph, met in table.items():
            print(
                f"  phase={ph:+.6f} pde_error={met['pde_error']:.6e} "
                f"post_prob={met['post_prob']:.6e} fidelity={met['fidelity']:.6e}",
                flush=True,
            )
        phase = float(best_phase)
        print(f"Selected phase: {phase:+.6f}", flush=True)

    print("Theory-guided search model:", flush=True)
    print("  Primary objective: minimize pde_error", flush=True)
    print("  Secondary objective: maximize post_prob", flush=True)
    print("  Constraints: r_prime < r_target, gamma >= gamma_min, beta in (0,1),", flush=True)
    print("               tail_mass <= tail_mass_max, n_eff <= n_eff_max", flush=True)

    grid_r = deterministic_grid(args.r_target_min, args.r_target_max, args.n_r_target)
    grid_rp = deterministic_grid(args.r_prime_min, args.r_prime_max, args.n_r_prime)
    grid_b = deterministic_grid(args.kernel_beta_min, args.kernel_beta_max, args.n_beta)

    rows: List[Dict[str, float]] = []
    seen = set()

    component_cache: Dict[Tuple[float, float], Dict[str, np.ndarray]] = {}

    def get_components(r_target: float, r_prime: float) -> Dict[str, np.ndarray]:
        key = (to_key(r_target), to_key(r_prime))
        if key not in component_cache:
            component_cache[key] = solver.collect_component_table(
                r_target=r_target,
                r_prime=r_prime,
                disp_phase=phase,
                levels=range(cfg.n_dim),
            )
        return component_cache[key]

    def evaluate_point(r_target: float, r_prime: float, beta: float, stage: str) -> None:
        key = (to_key(r_target), to_key(r_prime), to_key(beta))
        if key in seen:
            return
        seen.add(key)

        ok, diag, reason = theorem_constraints(
            r_target=r_target,
            r_prime=r_prime,
            beta=beta,
            gamma_min=args.gamma_min,
            min_gap=args.min_gap,
        )

        base = {
            "stage": stage,
            "r_target": float(r_target),
            "r_prime": float(r_prime),
            "kernel_beta": float(beta),
            "disp_phase": float(phase),
            "theory_gamma": float(diag["gamma"]),
            "theory_ok": int(ok),
            "theory_reason": reason,
        }

        if not ok:
            base.update(
                {
                    "tail_mass_est": np.nan,
                    "n_eff": np.nan,
                    "post_prob": np.nan,
                    "pde_error": np.nan,
                    "fidelity": np.nan,
                    "objective": np.inf,
                    "feasible": 0,
                }
            )
            rows.append(base)
            return

        params = core.CVLCHSParams(
            r_target=float(r_target),
            r_prime=float(r_prime),
            kernel_beta=float(beta),
            disp_phase=float(phase),
        )

        comps = get_components(r_target, r_prime)
        metrics = solver.evaluate(params, component_table=comps)

        feasible = int(
            (metrics["post_prob"] >= args.min_post_prob)
            and (metrics["tail_mass_est"] <= args.tail_mass_max)
            and (metrics["n_eff"] <= args.n_eff_max)
        )

        obj = objective_value(
            pde_error=metrics["pde_error"],
            post_prob=metrics["post_prob"],
            min_post_prob=args.min_post_prob,
            post_penalty=args.post_penalty,
            post_reward=args.post_reward,
        )

        base.update(
            {
                "tail_mass_est": float(metrics["tail_mass_est"]),
                "n_eff": float(metrics["n_eff"]),
                "post_prob": float(metrics["post_prob"]),
                "pde_error": float(metrics["pde_error"]),
                "fidelity": float(metrics["fidelity"]),
                "rho_purity": float(metrics["rho_purity"]),
                "pauli_rmse": float(metrics["pauli_rmse"]),
                "objective": float(obj),
                "feasible": feasible,
            }
        )
        rows.append(base)

        print(
            f"{stage:>6} r={r_target:.4f} r'={r_prime:.4f} beta={beta:.4f} "
            f"-> pde={metrics['pde_error']:.5f} post={metrics['post_prob']:.3e} "
            f"tail={metrics['tail_mass_est']:.2e} n_eff={metrics['n_eff']:.2f} feas={feasible}",
            flush=True,
        )

    print("\nStage 1/2: deterministic coarse grid", flush=True)
    for r in grid_r:
        for rp in grid_rp:
            for b in grid_b:
                evaluate_point(float(r), float(rp), float(b), stage="coarse")

    numeric_rows = [r for r in rows if np.isfinite(r["objective"])]
    numeric_rows.sort(key=lambda x: (-(x["feasible"]), x["objective"], -x["post_prob"]))

    seed_rows = numeric_rows[: max(0, args.refine_top_k)]
    print("\nStage 2/2: deterministic local refinement", flush=True)
    for idx, seed in enumerate(seed_rows, start=1):
        r0 = seed["r_target"]
        rp0 = seed["r_prime"]
        b0 = seed["kernel_beta"]
        print(f"  seed {idx}: r={r0:.4f} r'={rp0:.4f} beta={b0:.4f}", flush=True)

        r_vals = deterministic_grid(max(args.r_target_min, r0 - args.refine_dr), min(args.r_target_max, r0 + args.refine_dr), args.refine_points)
        rp_vals = deterministic_grid(max(args.r_prime_min, rp0 - args.refine_drp), min(args.r_prime_max, rp0 + args.refine_drp), args.refine_points)
        b_vals = deterministic_grid(max(args.kernel_beta_min, b0 - args.refine_db), min(args.kernel_beta_max, b0 + args.refine_db), args.refine_points)

        for r in r_vals:
            for rp in rp_vals:
                for b in b_vals:
                    evaluate_point(float(r), float(rp), float(b), stage="refine")

    # Rank final results
    numeric_rows = [r for r in rows if np.isfinite(r["objective"])]
    numeric_rows.sort(key=lambda x: (-(x["feasible"]), x["objective"], x["pde_error"], -x["post_prob"]))

    top = numeric_rows[:20]

    write_csv(rows, out_dir / "sensitivity_all.csv")
    write_csv(top, out_dir / "sensitivity_top.csv")

    summary = {
        "config": asdict(cfg),
        "search": {
            "r_target_min": args.r_target_min,
            "r_target_max": args.r_target_max,
            "r_prime_min": args.r_prime_min,
            "r_prime_max": args.r_prime_max,
            "kernel_beta_min": args.kernel_beta_min,
            "kernel_beta_max": args.kernel_beta_max,
            "n_r_target": args.n_r_target,
            "n_r_prime": args.n_r_prime,
            "n_beta": args.n_beta,
            "gamma_min": args.gamma_min,
            "min_gap": args.min_gap,
            "tail_mass_max": args.tail_mass_max,
            "n_eff_max": args.n_eff_max,
            "min_post_prob": args.min_post_prob,
            "post_penalty": args.post_penalty,
            "post_reward": args.post_reward,
            "disp_phase": phase,
        },
        "best": top[0] if top else None,
    }
    (out_dir / "sensitivity_summary.json").write_text(json.dumps(summary, indent=2))

    if not top:
        print("No valid points evaluated.", flush=True)
        return

    best = top[0]
    print("\nBest candidate (PDE-priority):", flush=True)
    print(json.dumps(best, indent=2), flush=True)

    # ---- Plots ----
    arr = np.array(numeric_rows, dtype=object)

    # 1D: beta slice at best (r, r')
    beta_slice = [
        r for r in numeric_rows if abs(r["r_target"] - best["r_target"]) < 1e-12 and abs(r["r_prime"] - best["r_prime"]) < 1e-12
    ]
    if beta_slice:
        beta_slice = sorted(beta_slice, key=lambda x: x["kernel_beta"])
        xs = [r["kernel_beta"] for r in beta_slice]
        ys = [r["pde_error"] for r in beta_slice]
        ps = [r["post_prob"] for r in beta_slice]
        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, "o-", label="pde_error")
        plt.plot(xs, ps, "s-", label="post_prob")
        plt.xlabel("kernel_beta")
        plt.ylabel("metric")
        plt.title("Beta sensitivity at best (r, r')")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "beta_slice_metrics.png", dpi=180)
        plt.close()

    # 2D heatmap for best beta
    best_beta = best["kernel_beta"]
    near_beta = [r for r in numeric_rows if abs(r["kernel_beta"] - best_beta) < 1e-12]
    if near_beta:
        rs = sorted({r["r_target"] for r in near_beta})
        rps = sorted({r["r_prime"] for r in near_beta})
        if len(rs) >= 2 and len(rps) >= 2:
            z = np.full((len(rps), len(rs)), np.nan, dtype=float)
            for row in near_beta:
                i = rps.index(row["r_prime"])
                j = rs.index(row["r_target"])
                z[i, j] = row["pde_error"]

            plt.figure(figsize=(6.5, 4.8))
            im = plt.imshow(
                z,
                origin="lower",
                aspect="auto",
                extent=[min(rs), max(rs), min(rps), max(rps)],
            )
            plt.colorbar(im, label="pde_error")
            plt.xlabel("r_target")
            plt.ylabel("r_prime")
            plt.title(f"PDE error heatmap at beta={best_beta:.4f}")
            plt.tight_layout()
            plt.savefig(out_dir / "rrp_heatmap_pde_error.png", dpi=180)
            plt.close()

    # Global scatter: PDE error vs post probability
    xs = [r["post_prob"] for r in numeric_rows]
    ys = [r["pde_error"] for r in numeric_rows]
    cs = [r["kernel_beta"] for r in numeric_rows]
    plt.figure(figsize=(6.2, 4.2))
    sc = plt.scatter(xs, ys, c=cs, s=26, cmap="viridis")
    plt.colorbar(sc, label="kernel_beta")
    plt.xlabel("post_prob")
    plt.ylabel("pde_error")
    plt.title("Tradeoff: PDE error vs post-selection probability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "tradeoff_pde_vs_post.png", dpi=180)
    plt.close()

    print(f"\nOutputs written to: {out_dir}", flush=True)


if __name__ == "__main__":
    run()
