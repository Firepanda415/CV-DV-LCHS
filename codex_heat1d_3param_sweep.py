#!/usr/bin/env python3
"""
3-parameter sweep for codex_heat1d_qutip.py model:
  (r_target, r_prime, beta)

This script evaluates each grid point with fixed numerical settings
(n_fock, n_coeff, n_quad, etc.) and stores metrics to CSV/JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

import codex_heat1d_qutip as core


def linspace_vals(lo: float, hi: float, n: int) -> np.ndarray:
    if n <= 1:
        return np.array([float(lo)], dtype=float)
    return np.linspace(float(lo), float(hi), int(n))


def evaluate_cfg(cfg: core.HeatLCHSConfig) -> Dict[str, float]:
    psi, phi, _ = core.prepare_lchs_states(cfg)
    k_map = core.effective_lchs_map(cfg, psi, phi)
    ref_map = core.classical_map(cfg)

    u0 = core.parse_initial_state(cfg.init_state)
    u_cv = k_map @ u0
    u_ref = ref_map @ u0

    map_scale, map_err = core.fit_global_scale(k_map, ref_map)
    vec_scale, vec_err = core.fit_global_scale(u_cv, u_ref)
    fidelity = core.state_fidelity(u_cv, u_ref)

    eta = complex(phi.overlap(psi))
    if np.isclose(abs(eta), 0.0):
        eta_err = float("nan")
    else:
        eta_err = float(np.linalg.norm((u_cv / eta) - u_ref) / max(np.linalg.norm(u_ref), 1e-15))

    return {
        "map_err_best_scale": float(map_err),
        "vec_err_best_scale": float(vec_err),
        "fidelity": float(fidelity),
        "post_prob": float(np.real(np.vdot(u_cv, u_cv))),
        "eta_err": eta_err,
        "map_scale_real": float(np.real(map_scale)),
        "map_scale_imag": float(np.imag(map_scale)),
        "vec_scale_real": float(np.real(vec_scale)),
        "vec_scale_imag": float(np.imag(vec_scale)),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3-parameter sweep for codex heat1d CV-DV LCHS")

    # Fixed simulation settings.
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--h-grid", type=float, default=1.0)
    p.add_argument("--total-time", type=float, default=1.0)
    p.add_argument("--init-state", choices=["basis00", "basis01", "basis10", "basis11", "sine", "ones"], default="basis01")
    p.add_argument("--position-convention", choices=["half", "sqrt2"], default="half")
    p.add_argument("--position-scale", type=float, default=0.8)
    p.add_argument("--n-fock", type=int, default=64)
    p.add_argument("--n-coeff", type=int, default=48)
    p.add_argument("--n-quad", type=int, default=300)

    # Sweep ranges.
    p.add_argument("--r-target-min", type=float, default=6.0)
    p.add_argument("--r-target-max", type=float, default=10.0)
    p.add_argument("--n-r-target", type=int, default=9)
    p.add_argument("--r-prime-min", type=float, default=2.0)
    p.add_argument("--r-prime-max", type=float, default=6.0)
    p.add_argument("--n-r-prime", type=int, default=9)
    p.add_argument("--beta-min", type=float, default=0.7)
    p.add_argument("--beta-max", type=float, default=0.95)
    p.add_argument("--n-beta", type=int, default=11)

    # Feasibility filters.
    p.add_argument("--min-gap", type=float, default=0.2, help="Require r_target - r_prime >= min-gap.")
    p.add_argument("--max-points", type=int, default=0, help="If >0, stop after this many evaluated points.")

    # Output.
    p.add_argument("--output-dir", type=str, default="results_bosonic/sweep_3param")
    p.add_argument("--top-k", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Base config; swept parameters are injected per point.
    base = core.HeatLCHSConfig(
        alpha=args.alpha,
        h_grid=args.h_grid,
        total_time=args.total_time,
        n_fock=args.n_fock,
        n_coeff=args.n_coeff,
        r_target=8.0,
        r_prime=4.0,
        beta=0.9,
        n_quad=args.n_quad,
        init_state=args.init_state,
        position_convention=args.position_convention,
        position_scale=args.position_scale,
        trajectory_steps=0,
    )

    if base.n_coeff > base.n_fock:
        raise ValueError("n_coeff must be <= n_fock.")

    r_targets = linspace_vals(args.r_target_min, args.r_target_max, args.n_r_target)
    r_primes = linspace_vals(args.r_prime_min, args.r_prime_max, args.n_r_prime)
    betas = linspace_vals(args.beta_min, args.beta_max, args.n_beta)

    rows: List[Dict] = []
    total = 0
    evaluated = 0
    for r_t in r_targets:
        for r_p in r_primes:
            for beta in betas:
                total += 1
                row = {
                    "r_target": float(r_t),
                    "r_prime": float(r_p),
                    "beta": float(beta),
                }

                # Feasibility checks.
                if r_t - r_p < args.min_gap:
                    row.update({
                        "status": "skipped",
                        "reason": "min_gap",
                    })
                    rows.append(row)
                    continue
                if not (0.0 < beta < 1.0):
                    row.update({
                        "status": "skipped",
                        "reason": "beta_out_of_range",
                    })
                    rows.append(row)
                    continue

                cfg = core.HeatLCHSConfig(
                    alpha=base.alpha,
                    h_grid=base.h_grid,
                    total_time=base.total_time,
                    n_fock=base.n_fock,
                    n_coeff=base.n_coeff,
                    r_target=float(r_t),
                    r_prime=float(r_p),
                    beta=float(beta),
                    n_quad=base.n_quad,
                    init_state=base.init_state,
                    position_convention=base.position_convention,
                    position_scale=base.position_scale,
                    trajectory_steps=0,
                )

                try:
                    met = evaluate_cfg(cfg)
                    row.update({"status": "ok"})
                    row.update(met)
                    # Main objective for sorting:
                    # prioritize low vector error, then map error, then high fidelity.
                    row["objective"] = float(met["vec_err_best_scale"] + 0.5 * met["map_err_best_scale"] - 0.05 * met["fidelity"])
                except Exception as exc:
                    row.update({
                        "status": "error",
                        "reason": str(exc),
                    })
                rows.append(row)
                evaluated += 1

                if args.max_points > 0 and evaluated >= args.max_points:
                    break
            if args.max_points > 0 and evaluated >= args.max_points:
                break
        if args.max_points > 0 and evaluated >= args.max_points:
            break

    # Write full CSV.
    fieldnames = sorted({k for row in rows for k in row.keys()})
    csv_path = outdir / "sweep_all.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if r.get("status") == "ok" and np.isfinite(r.get("objective", np.inf))]
    ok_rows.sort(key=lambda r: (r["objective"], r["vec_err_best_scale"], r["map_err_best_scale"], -r["fidelity"]))

    top_k = ok_rows[: max(1, args.top_k)]
    top_path = outdir / "sweep_top.csv"
    if top_k:
        with top_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(top_k[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(top_k)

    summary = {
        "base_config": asdict(base),
        "sweep_grid": {
            "r_target": [float(args.r_target_min), float(args.r_target_max), int(args.n_r_target)],
            "r_prime": [float(args.r_prime_min), float(args.r_prime_max), int(args.n_r_prime)],
            "beta": [float(args.beta_min), float(args.beta_max), int(args.n_beta)],
            "min_gap": float(args.min_gap),
        },
        "counts": {
            "total_grid_points": int(total),
            "rows_written": int(len(rows)),
            "ok": int(sum(1 for r in rows if r.get("status") == "ok")),
            "error": int(sum(1 for r in rows if r.get("status") == "error")),
            "skipped": int(sum(1 for r in rows if r.get("status") == "skipped")),
        },
        "best": top_k[0] if top_k else None,
        "top_k": top_k,
        "files": {
            "sweep_all_csv": str(csv_path),
            "sweep_top_csv": str(top_path) if top_k else "",
        },
    }
    summary_path = outdir / "sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Optional simple scatter plot for quick visual triage.
    try:
        import matplotlib.pyplot as plt

        if ok_rows:
            x = [r["vec_err_best_scale"] for r in ok_rows]
            y = [r["fidelity"] for r in ok_rows]
            c = [r["beta"] for r in ok_rows]
            plt.figure(figsize=(7.0, 4.5))
            sc = plt.scatter(x, y, c=c, s=18, cmap="viridis")
            plt.colorbar(sc, label="beta")
            plt.xlabel("Vector error (best scale)")
            plt.ylabel("Fidelity")
            plt.title("3-parameter sweep trade-off")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(outdir / "tradeoff_vecerr_vs_fidelity.png", dpi=180)
            plt.close()
    except Exception:
        pass

    print("Sweep complete.")
    print(f"  Total grid points: {total}")
    print(f"  Rows written:      {len(rows)}")
    print(f"  OK points:         {sum(1 for r in rows if r.get('status') == 'ok')}")
    print(f"  CSV (all):         {csv_path}")
    print(f"  Summary JSON:      {summary_path}")
    if top_k:
        b = top_k[0]
        print("Best point:")
        print(
            f"  r_target={b['r_target']:.6f}, r_prime={b['r_prime']:.6f}, beta={b['beta']:.6f}, "
            f"vec_err={b['vec_err_best_scale']:.6e}, map_err={b['map_err_best_scale']:.6e}, "
            f"fidelity={b['fidelity']:.6f}, post_prob={b['post_prob']:.6e}"
        )


if __name__ == "__main__":
    main()

