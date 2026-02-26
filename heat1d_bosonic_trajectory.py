#!/usr/bin/env python3
"""
CHO-style trajectory validation for the bosonic heat-equation solver.

For each time point t_k, we run the same model with matching Trotter step size and
compare quantum vs classical DV vectors component-wise.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

import heat1d_bosonic_pde_solver as core


def write_csv(rows: List[Dict[str, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CHO-style trajectory validation for heat-equation LCHS")

    p.add_argument("--output-dir", type=str, default="results_bosonic/trajectory")

    p.add_argument("--total-time", type=float, default=1.0)
    p.add_argument("--n-steps", type=int, default=40)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--h", type=float, default=1.0)
    p.add_argument("--init-q0", type=int, default=0, choices=[0, 1])
    p.add_argument("--init-q1", type=int, default=1, choices=[0, 1])
    p.add_argument("--n-dim", type=int, default=32)
    p.add_argument("--max-fock-level", type=int, default=32)
    p.add_argument("--fock-weight-cutoff", type=float, default=1e-8)

    p.add_argument("--r-target", type=float, default=1.2)
    p.add_argument("--r-prime", type=float, default=0.3)
    p.add_argument("--kernel-beta", type=float, default=0.8)
    p.add_argument("--disp-phase", type=float, default=0.0)

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

    if args.n_steps <= 0:
        raise ValueError("n_steps must be positive for trajectory validation.")

    dt_ref = float(args.total_time) / float(args.n_steps)
    params = core.CVLCHSParams(
        r_target=float(args.r_target),
        r_prime=float(args.r_prime),
        kernel_beta=float(args.kernel_beta),
        disp_phase=float(args.disp_phase),
    )

    rows: List[Dict[str, float]] = []

    print("Running CHO-style trajectory sweep...", flush=True)
    print(f"Base dt = {dt_ref:.6e}", flush=True)

    for k in range(args.n_steps + 1):
        # Keep Trotter dt constant across all time points:
        #   dt_ref = T_total / N_total = t_k / k  (for k >= 1)
        # For k=0 we run a zero-time probe with one step.
        if k == 0:
            total_time_k = 0.0
            n_steps_k = 1
        else:
            total_time_k = dt_ref * k
            n_steps_k = k

        cfg_k = core.HeatEquationConfig(
            total_time=float(total_time_k),
            n_steps=int(n_steps_k),
            alpha=float(args.alpha),
            h=float(args.h),
            init_qubits=(int(args.init_q0), int(args.init_q1)),
            n_dim=int(args.n_dim),
            max_fock_level=int(args.max_fock_level),
            fock_weight_cutoff=float(args.fock_weight_cutoff),
        )

        solver_k = core.create_solver(cfg_k, state_prep=args.state_prep)
        met = solver_k.evaluate(params, return_vectors=True)

        row: Dict[str, float] = {
            "step": float(k),
            "time": float(total_time_k),
            "pde_error": float(met["pde_error"]),
            "post_prob": float(met["post_prob"]),
            "fidelity": float(met["fidelity"]),
            "trace_distance": float(met["trace_distance"]),
        }
        for j in range(4):
            row[f"u_target_real_{j}"] = float(met[f"u_target_real_{j}"])
            row[f"u_target_imag_{j}"] = float(met[f"u_target_imag_{j}"])
            row[f"u_quantum_real_{j}"] = float(met[f"u_quantum_real_{j}"])
            row[f"u_quantum_imag_{j}"] = float(met[f"u_quantum_imag_{j}"])

        rows.append(row)
        print(
            f"k={k:3d} t={total_time_k:.6f} pde_error={row['pde_error']:.6e} "
            f"post_prob={row['post_prob']:.6e}",
            flush=True,
        )

    write_csv(rows, out_dir / "trajectory.csv")

    # Plot metric trajectory
    times = np.array([r["time"] for r in rows], dtype=float)
    pde_err = np.array([r["pde_error"] for r in rows], dtype=float)
    post_prob = np.array([r["post_prob"] for r in rows], dtype=float)

    plt.figure(figsize=(7.0, 4.4))
    plt.plot(times, pde_err, "o-", label="pde_error")
    plt.plot(times, post_prob, "s-", label="post_prob")
    plt.xlabel("time")
    plt.ylabel("metric")
    plt.title("Trajectory metrics (CHO-style validation)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "trajectory_metrics.png", dpi=180)
    plt.close()

    # Plot component-wise real-part comparison (classical vs quantum)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    axes = axes.flatten()
    for j in range(4):
        u_ref = np.array([r[f"u_target_real_{j}"] for r in rows], dtype=float)
        u_q = np.array([r[f"u_quantum_real_{j}"] for r in rows], dtype=float)
        ax = axes[j]
        ax.plot(times, u_ref, "k--", label="classical")
        ax.plot(times, u_q, "b-", label="quantum")
        ax.set_title(f"Component {j} (real)")
        ax.grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    for ax in axes[2:]:
        ax.set_xlabel("time")
    plt.tight_layout()
    plt.savefig(out_dir / "trajectory_components_real.png", dpi=180)
    plt.close()

    summary = {
        "config_base": {
            "total_time": float(args.total_time),
            "n_steps": int(args.n_steps),
            "alpha": float(args.alpha),
            "h": float(args.h),
            "init_qubits": [int(args.init_q0), int(args.init_q1)],
            "n_dim": int(args.n_dim),
            "max_fock_level": int(args.max_fock_level),
            "fock_weight_cutoff": float(args.fock_weight_cutoff),
            "state_prep": str(args.state_prep),
        },
        "params": asdict(params),
        "final": rows[-1],
    }
    (out_dir / "trajectory_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Outputs written to: {out_dir}", flush=True)


if __name__ == "__main__":
    run()
