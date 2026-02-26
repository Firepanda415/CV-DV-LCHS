#!/usr/bin/env python3
"""
Deterministic theorem-guided sensitivity sweep for bosonic-qiskit PDE solver.
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

import heat1d_bosonic_pde_solver as core


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
    # Keep each term explicit for readability:
    # objective = pde_error + feasibility_penalty - tiny_post_prob_reward
    penalty = post_penalty * max(0.0, min_post_prob - post_prob) / max(min_post_prob, 1e-15)
    return float(pde_error + penalty - post_reward * post_prob)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bosonic-qiskit sensitivity run")

    p.add_argument("--output-dir", type=str, default="results_bosonic/sensitivity")

    p.add_argument("--total-time", type=float, default=1.0)
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--h", type=float, default=1.0)
    p.add_argument("--init-q0", type=int, default=0, choices=[0, 1])
    p.add_argument("--init-q1", type=int, default=1, choices=[0, 1])
    p.add_argument("--n-dim", type=int, default=32)
    p.add_argument("--max-fock-level", type=int, default=32)
    p.add_argument("--fock-weight-cutoff", type=float, default=1e-8)

    p.add_argument("--r-target-min", type=float, default=0.5)
    p.add_argument("--r-target-max", type=float, default=2)
    p.add_argument("--r-prime-min", type=float, default=0.03)
    p.add_argument("--r-prime-max", type=float, default=1)
    p.add_argument("--kernel-beta-min", type=float, default=0.3)
    p.add_argument("--kernel-beta-max", type=float, default=0.95)

    p.add_argument("--n-r-target", type=int, default=5)
    p.add_argument("--n-r-prime", type=int, default=5)
    p.add_argument("--n-beta", type=int, default=7)

    p.add_argument("--gamma-min", type=float, default=0.03)
    p.add_argument("--min-gap", type=float, default=0.02)
    p.add_argument("--tail-mass-max", type=float, default=1e-4)
    p.add_argument("--n-eff-max", type=float, default=16.0)

    p.add_argument("--min-post-prob", type=float, default=1e-3)
    p.add_argument("--post-penalty", type=float, default=100.0)
    p.add_argument("--post-reward", type=float, default=1e-3)

    p.add_argument("--refine-top-k", type=int, default=3)
    p.add_argument("--refine-points", type=int, default=3)
    p.add_argument("--refine-dr", type=float, default=0.08)
    p.add_argument("--refine-drp", type=float, default=0.05)
    p.add_argument("--refine-db", type=float, default=0.08)

    p.add_argument("--disp-phase", type=float, default=0.0)
    p.add_argument("--calibrate-phase", action="store_true")
    p.add_argument(
        "--state-prep",
        type=str,
        default="classical-injection",
        choices=["classical-injection", "gate-based"],
    )

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
    solver = core.create_solver(cfg, state_prep=args.state_prep)

    phase = float(args.disp_phase)
    dx_base, dp_base = core.displacement_phase_shifts(1.0, phase)
    print(
        f"Phase audit (amplitude=1): phase={phase:+.6f} -> Delta x={dx_base:+.6f}, Delta p={dp_base:+.6f}",
        flush=True,
    )
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
            dx, dp = core.displacement_phase_shifts(1.0, ph)
            print(
                f"  phase={ph:+.6f} pde_error={met['pde_error']:.6e} "
                f"post_prob={met['post_prob']:.6e} fidelity={met['fidelity']:.6e} "
                f"(Delta x={dx:+.3f}, Delta p={dp:+.3f})",
                flush=True,
            )
        phase = float(best_phase)
        print(f"Selected phase: {phase:+.6f}", flush=True)

    rows: List[Dict[str, float]] = []
    seen = set()

    grid_r = deterministic_grid(args.r_target_min, args.r_target_max, args.n_r_target)
    grid_rp = deterministic_grid(args.r_prime_min, args.r_prime_max, args.n_r_prime)
    grid_b = deterministic_grid(args.kernel_beta_min, args.kernel_beta_max, args.n_beta)

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
            "state_prep": args.state_prep,
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
        metrics = solver.evaluate(params)

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

    print("Stage 1/2: coarse grid", flush=True)
    for r in grid_r:
        for rp in grid_rp:
            for b in grid_b:
                evaluate_point(float(r), float(rp), float(b), stage="coarse")

    numeric_rows = [r for r in rows if np.isfinite(r["objective"])]
    numeric_rows.sort(key=lambda x: (-(x["feasible"]), x["objective"], -x["post_prob"]))

    seed_rows = numeric_rows[: max(0, args.refine_top_k)]
    print("Stage 2/2: local refinement", flush=True)
    for idx, seed in enumerate(seed_rows, start=1):
        r0 = seed["r_target"]
        rp0 = seed["r_prime"]
        b0 = seed["kernel_beta"]
        print(f"  seed {idx}: r={r0:.4f} r'={rp0:.4f} beta={b0:.4f}", flush=True)

        r_vals = deterministic_grid(
            max(args.r_target_min, r0 - args.refine_dr),
            min(args.r_target_max, r0 + args.refine_dr),
            args.refine_points,
        )
        rp_vals = deterministic_grid(
            max(args.r_prime_min, rp0 - args.refine_drp),
            min(args.r_prime_max, rp0 + args.refine_drp),
            args.refine_points,
        )
        b_vals = deterministic_grid(
            max(args.kernel_beta_min, b0 - args.refine_db),
            min(args.kernel_beta_max, b0 + args.refine_db),
            args.refine_points,
        )

        for r in r_vals:
            for rp in rp_vals:
                for b in b_vals:
                    evaluate_point(float(r), float(rp), float(b), stage="refine")

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
            "state_prep": args.state_prep,
        },
        "best": top[0] if top else None,
    }
    (out_dir / "sensitivity_summary.json").write_text(json.dumps(summary, indent=2))

    if top:
        best = top[0]
        print("Best candidate:", flush=True)
        print(json.dumps(best, indent=2), flush=True)

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

    print(f"Outputs written to: {out_dir}", flush=True)


if __name__ == "__main__":
    run()
