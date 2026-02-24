"""
heat_eq_systematic_optimize.py

Human-oriented summary:
  This script searches CV state-preparation parameters for the hybrid CV-DV LCHS
  heat-equation solver, with PDE correctness as the primary objective.

What is optimized:
  theta = (r_target, r_prime, kernel_beta)
  subject to r_prime < r_target and r_target - r_prime >= min_gap.

Why these parameters:
  - r_target controls the post-selection squeezed target profile.
  - r_prime controls the prep-basis squeezing and mode-weight spread.
  - kernel_beta controls the Gaussian-kernel weighting in the coefficient model.

Objective model (PDE-priority):
  J(theta) = pde_error
           + post_penalty * [p_min - post_prob]_+ / p_min
           + gamma_penalty * [gamma_min - gamma(theta)]_+ / gamma_min
           + neff_penalty * [n_eff(theta) - n_eff_max]_+ / n_eff_max
           - 1e-3 * post_prob
  with gamma(theta) = exp(-2*r_prime) - exp(-2*r_target).

Search strategy:
  1) Global Latin-hypercube sampling to map the landscape.
  2) Multi-start local Nelder-Mead from best global candidates.
  3) PDE-priority ranking and Pareto export (pde_error vs post_prob).

Range policy:
  Defaults below are intentionally narrowed for faster, more stable runs in the
  empirically good basin. You can always override via CLI flags.
  Original broad example ranges (kept here for reference):
    r_target in [0.15, 1.4]
    r_prime  in [0.03, 0.9]
    kernel_beta in [0.0, 1.0]
"""

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from heat_eq_sensitivity_refine import Evaluator, Settings


@dataclass(frozen=True)
class SearchConfig:
    # Fast, practical defaults for PDE-priority runs.
    # Original broad examples for reference:
    #   r_target in [0.15, 1.4]
    #   r_prime  in [0.03, 0.9]
    # Override with CLI flags when you want wider exploration.
    r_target_min: float = 0.25
    r_target_max: float = 1.2
    r_prime_min: float = 0.05
    r_prime_max: float = 0.45
    kernel_beta_min: float = 0.0
    kernel_beta_max: float = 1.0
    min_gap: float = 0.02
    min_post_prob: float = 1e-3
    gamma_min: float = 0.03
    neff_max: float = 16.0
    post_penalty: float = 100.0
    gamma_penalty: float = 0.2
    neff_penalty: float = 0.05
    global_samples: int = 30
    n_starts: int = 4
    local_maxiter: int = 30
    progress_every: int = 8
    checkpoint_every: int = 10
    seed: int = 7


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {}
            for k, v in raw.items():
                if v is None or v == "":
                    row[k] = np.nan
                    continue
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            rows.append(row)
    return rows


def _pareto_front(rows: list[dict]):
    valid = [
        r
        for r in rows
        if np.isfinite(r.get("pde_error", np.nan))
        and np.isfinite(r.get("post_prob", np.nan))
        and float(r.get("post_prob", 0.0)) > 0.0
    ]
    valid.sort(key=lambda r: float(r["pde_error"]))
    front = []
    best_post = -np.inf
    for r in valid:
        post = float(r["post_prob"])
        if post > best_post + 1e-15:
            front.append(r)
            best_post = post
    return front


def _plot_tradeoff(path: Path, rows: list[dict], pareto_rows: list[dict]):
    valid = [
        r
        for r in rows
        if np.isfinite(r.get("pde_error", np.nan))
        and np.isfinite(r.get("post_prob", np.nan))
        and float(r.get("post_prob", 0.0)) > 0.0
    ]
    if not valid:
        return
    x = np.array([float(r["post_prob"]) for r in valid], dtype=float)
    y = np.array([float(r["pde_error"]) for r in valid], dtype=float)

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.scatter(x, y, c="tab:blue", s=22, alpha=0.55, label="Evaluated points")
    if pareto_rows:
        px = np.array([float(r["post_prob"]) for r in pareto_rows], dtype=float)
        py = np.array([float(r["pde_error"]) for r in pareto_rows], dtype=float)
        order = np.argsort(px)
        ax.plot(px[order], py[order], "-", color="tab:red", lw=1.7, label="Pareto front")
        ax.scatter(px, py, c="tab:red", s=36, alpha=0.95)
    ax.set_xscale("log")
    ax.set_xlabel("Post-selection probability (log scale)")
    ax.set_ylabel("Relative PDE-vector error")
    ax.set_title("Systematic optimization trade-off")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_landscape(path: Path, rows: list[dict]):
    valid = [
        r
        for r in rows
        if np.isfinite(r.get("pde_error", np.nan))
        and np.isfinite(r.get("r_target", np.nan))
        and np.isfinite(r.get("r_prime", np.nan))
    ]
    if not valid:
        return
    x = np.array([float(r["r_target"]) for r in valid], dtype=float)
    y = np.array([float(r["r_prime"]) for r in valid], dtype=float)
    z = np.array([float(r["pde_error"]) for r in valid], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    sc = ax.scatter(x, y, c=z, cmap="viridis", s=28, alpha=0.8)
    fig.colorbar(sc, ax=ax, label="Relative PDE-vector error")
    ax.set_xlabel("r_target")
    ax.set_ylabel("r_prime")
    ax.set_title("Parameter landscape (colored by PDE error)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _rank_key_pde_priority(row: dict, min_post_prob: float):
    pde = row.get("pde_error", np.nan)
    post = row.get("post_prob", np.nan)
    pde_key = float(pde) if np.isfinite(pde) else float("inf")
    post_val = float(post) if np.isfinite(post) else -np.inf
    infeasible = 0 if post_val >= min_post_prob else 1
    return (infeasible, pde_key, -post_val if np.isfinite(post_val) else float("inf"))


def _validate_theta(theta: np.ndarray, cfg: SearchConfig):
    r_target, r_prime, kernel_beta = [float(v) for v in theta]
    if r_target < cfg.r_target_min or r_target > cfg.r_target_max:
        return False
    if r_prime < cfg.r_prime_min or r_prime > cfg.r_prime_max:
        return False
    if kernel_beta < cfg.kernel_beta_min or kernel_beta > cfg.kernel_beta_max:
        return False
    if r_prime >= r_target - cfg.min_gap:
        return False
    return True


def _theory_features(ev: Evaluator, base: Settings, r_target: float, r_prime: float, kernel_beta: float):
    """
    Theory-guided descriptors:
      gamma = exp(-2 r') - exp(-2 r) > 0
      n_eff = 1 / sum_n w_n^2 with w_n = |C_n|^2
    """
    gamma = float(np.exp(-2.0 * r_prime) - np.exp(-2.0 * r_target))
    coeffs = ev.get_coeffs(r_target, r_prime, base.n_dim, kernel_beta, base.n_quad_points)
    weights = np.abs(coeffs) ** 2
    neff = float(1.0 / np.sum(weights**2))
    return gamma, neff


def _evaluate_theta(ev: Evaluator, base: Settings, cfg: SearchConfig, theta: np.ndarray):
    r_target, r_prime, kernel_beta = [float(v) for v in theta]
    if not _validate_theta(theta, cfg):
        return {
            "r_target": r_target,
            "r_prime": r_prime,
            "kernel_beta": kernel_beta,
            "pde_error": np.nan,
            "post_prob": np.nan,
            "purity": np.nan,
            "fidelity": np.nan,
            "gamma": np.nan,
            "n_eff": np.nan,
            "post_shortfall": np.nan,
            "gamma_shortfall": np.nan,
            "n_eff_excess": np.nan,
            "objective": np.inf,
            "is_feasible": 0,
        }

    gamma, neff = _theory_features(ev, base, r_target, r_prime, kernel_beta)
    metrics = ev.evaluate(replace(base, r_target=r_target, r_prime=r_prime, kernel_beta=kernel_beta))
    pde_error = float(metrics["pde_error"])
    post_prob = float(metrics["post_prob"])
    purity = float(metrics["purity"])
    fidelity = float(metrics["fidelity"])

    if not np.isfinite(pde_error) or not np.isfinite(post_prob):
        objective = np.inf
        post_shortfall = np.nan
        gamma_shortfall = np.nan
        neff_excess = np.nan
        feasible = 0
    else:
        post_shortfall = max(0.0, cfg.min_post_prob - post_prob) / cfg.min_post_prob
        gamma_shortfall = max(0.0, cfg.gamma_min - gamma) / cfg.gamma_min
        neff_excess = max(0.0, neff - cfg.neff_max) / cfg.neff_max
        objective = (
            pde_error
            + cfg.post_penalty * post_shortfall
            + cfg.gamma_penalty * gamma_shortfall
            + cfg.neff_penalty * neff_excess
            - 1e-3 * post_prob
        )
        feasible = int(post_prob >= cfg.min_post_prob)

    return {
        "r_target": r_target,
        "r_prime": r_prime,
        "kernel_beta": kernel_beta,
        "pde_error": pde_error,
        "post_prob": post_prob,
        "purity": purity,
        "fidelity": fidelity,
        "gamma": gamma,
        "n_eff": neff,
        "post_shortfall": post_shortfall,
        "gamma_shortfall": gamma_shortfall,
        "n_eff_excess": neff_excess,
        "objective": float(objective),
        "is_feasible": feasible,
    }


def _lhs_global_samples(cfg: SearchConfig):
    sampler = qmc.LatinHypercube(d=3, seed=cfg.seed)
    u = sampler.random(n=cfg.global_samples)
    r_target = cfg.r_target_min + u[:, 0] * (cfg.r_target_max - cfg.r_target_min)
    r_prime = cfg.r_prime_min + u[:, 1] * (cfg.r_prime_max - cfg.r_prime_min)
    kernel_beta = cfg.kernel_beta_min + u[:, 2] * (cfg.kernel_beta_max - cfg.kernel_beta_min)
    candidates = np.column_stack([r_target, r_prime, kernel_beta])

    # Filter invalid points and backfill with random draws if needed.
    valid = [row for row in candidates if _validate_theta(row, cfg)]
    rng = np.random.default_rng(cfg.seed + 101)
    while len(valid) < cfg.global_samples:
        row = np.array(
            [
                rng.uniform(cfg.r_target_min, cfg.r_target_max),
                rng.uniform(cfg.r_prime_min, cfg.r_prime_max),
                rng.uniform(cfg.kernel_beta_min, cfg.kernel_beta_max),
            ],
            dtype=float,
        )
        if _validate_theta(row, cfg):
            valid.append(row)
    return np.array(valid[: cfg.global_samples], dtype=float)


def run_systematic_optimize(output_dir: Path, cfg: SearchConfig, resume: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)
    base = Settings()
    ev = Evaluator()

    print("Theory model (PDE-priority):")
    print("  theta = (r_target, r_prime, kernel_beta), with r_prime < r_target.")
    print("  gamma(theta) = exp(-2 r_prime) - exp(-2 r_target) > 0.")
    print("  n_eff(theta) = 1 / sum_n |C_n(theta)|^4.")
    print("  Objective:")
    print("    J(theta) = pde_error")
    print("             + post_penalty * [p_min - post_prob]_+ / p_min")
    print("             + gamma_penalty * [gamma_min - gamma]_+ / gamma_min")
    print("             + neff_penalty * [n_eff - n_eff_max]_+ / n_eff_max")
    print("             - 1e-3 * post_prob")
    print("  Optimization plan:")
    print("    Stage 1: global Latin-hypercube exploration")
    print("    Stage 2: multi-start local Nelder-Mead refinement")
    print("    Stage 3: PDE-priority ranking + Pareto front")
    print("")
    print("Search config:")
    print(f"  global_samples={cfg.global_samples}, n_starts={cfg.n_starts}, local_maxiter={cfg.local_maxiter}")
    print(f"  min_post_prob={cfg.min_post_prob}, gamma_min={cfg.gamma_min}, neff_max={cfg.neff_max}")
    print(f"  penalties: post={cfg.post_penalty}, gamma={cfg.gamma_penalty}, neff={cfg.neff_penalty}")
    print(f"  bounds: r_target[{cfg.r_target_min}, {cfg.r_target_max}], r_prime[{cfg.r_prime_min}, {cfg.r_prime_max}], kb[{cfg.kernel_beta_min}, {cfg.kernel_beta_max}]")

    # Stage 1: global exploration (supports resume).
    global_csv = output_dir / "systematic_global.csv"
    if resume and global_csv.exists():
        global_rows = _read_csv(global_csv)
        print(f"\nStage 1/3: loaded existing global exploration from {global_csv}")
    else:
        global_thetas = _lhs_global_samples(cfg)
        global_rows = []
        print("\nStage 1/3: global exploration")
        for i, theta in enumerate(global_thetas, start=1):
            row = _evaluate_theta(ev, base, cfg, theta)
            row["stage"] = "global"
            global_rows.append(row)
            print(
                f"  [{i:03d}/{len(global_thetas):03d}] "
                f"r={row['r_target']:.4f}, r'={row['r_prime']:.4f}, kb={row['kernel_beta']:.4f} "
                f"-> pde={row['pde_error']:.5f}, post={row['post_prob']:.3e}, J={row['objective']:.5f}"
            )
        _write_csv(global_csv, global_rows)

    ranked_global = [r for r in global_rows if np.isfinite(r.get("objective", np.nan))]
    ranked_global.sort(key=lambda r: (r["objective"], _rank_key_pde_priority(r, cfg.min_post_prob)))
    if not ranked_global:
        raise RuntimeError("No valid global candidates. Relax bounds/constraints and retry.")
    starts = ranked_global[: max(1, cfg.n_starts)]

    # Stage 2: multi-start local refinement.
    print("\nStage 2/3: local refinement")
    local_csv = output_dir / "systematic_local.csv"
    local_rows = _read_csv(local_csv) if (resume and local_csv.exists()) else []
    completed_starts = set()
    for r in local_rows:
        if r.get("stage") == "local_best" and np.isfinite(r.get("start_index", np.nan)):
            completed_starts.add(int(r["start_index"]))
    if completed_starts:
        print(f"  resuming: skipping completed starts {sorted(completed_starts)}")

    for i, s in enumerate(starts, start=1):
        if i in completed_starts:
            continue
        x0 = np.array([s["r_target"], s["r_prime"], s["kernel_beta"]], dtype=float)
        print(
            f"  start {i}/{len(starts)}: "
            f"r={x0[0]:.4f}, r'={x0[1]:.4f}, kb={x0[2]:.4f}, J0={s['objective']:.5f}"
        )
        eval_count = 0
        best_j = np.inf

        def objective_fn(x):
            nonlocal eval_count, best_j
            row = _evaluate_theta(ev, base, cfg, np.asarray(x, dtype=float))
            row.update({"stage": "local_eval", "start_index": i})
            local_rows.append(row)
            eval_count += 1
            best_j = min(best_j, float(row["objective"]))
            if eval_count % cfg.progress_every == 0:
                print(
                    f"    eval {eval_count:03d}: r={row['r_target']:.4f}, r'={row['r_prime']:.4f}, "
                    f"kb={row['kernel_beta']:.4f}, pde={row['pde_error']:.5f}, post={row['post_prob']:.3e}, "
                    f"J={row['objective']:.5f}, bestJ={best_j:.5f}"
                )
            if eval_count % cfg.checkpoint_every == 0:
                _write_csv(local_csv, local_rows)
            return float(row["objective"])

        result = minimize(
            objective_fn,
            x0,
            method="Nelder-Mead",
            options={"maxiter": cfg.local_maxiter, "xatol": 0.003, "fatol": 1e-5, "adaptive": True},
        )
        best_row = _evaluate_theta(ev, base, cfg, result.x)
        best_row.update(
            {
                "stage": "local_best",
                "start_index": i,
                "success": int(result.success),
                "message": str(result.message),
                "nfev": int(result.nfev),
            }
        )
        local_rows.append(best_row)
        _write_csv(local_csv, local_rows)
        print(
            f"    -> best r={best_row['r_target']:.4f}, r'={best_row['r_prime']:.4f}, kb={best_row['kernel_beta']:.4f} "
            f"pde={best_row['pde_error']:.5f}, post={best_row['post_prob']:.3e}, J={best_row['objective']:.5f}"
        )

    # Stage 3: aggregate and report.
    print("\nStage 3/3: ranking and export")
    all_rows = global_rows + local_rows
    ranked = [
        r
        for r in all_rows
        if np.isfinite(r.get("objective", np.nan))
        and np.isfinite(r.get("pde_error", np.nan))
        and np.isfinite(r.get("post_prob", np.nan))
    ]
    ranked.sort(key=lambda r: _rank_key_pde_priority(r, cfg.min_post_prob))
    top = ranked[:20]
    pareto = _pareto_front(ranked)

    _write_csv(local_csv, local_rows)
    _write_csv(output_dir / "systematic_all.csv", all_rows)
    _write_csv(output_dir / "systematic_top.csv", top)
    _write_csv(output_dir / "systematic_pareto_front.csv", pareto)
    _plot_tradeoff(output_dir / "systematic_tradeoff_pde_vs_post.png", ranked, pareto)
    _plot_landscape(output_dir / "systematic_landscape_rtarget_rprime.png", ranked)

    feasible = [r for r in ranked if int(r.get("is_feasible", 0)) == 1]
    print(f"  feasible points: {len(feasible)}/{len(ranked)} (post_prob >= {cfg.min_post_prob:.2e})")
    if top:
        b = top[0]
        print("Best PDE-priority candidate:")
        print(
            f"  r_target={b['r_target']:.6f}, r_prime={b['r_prime']:.6f}, kernel_beta={b['kernel_beta']:.6f}\n"
            f"  pde_error={b['pde_error']:.6e}, post_prob={b['post_prob']:.6e}, purity={b['purity']:.6e}\n"
            f"  gamma={b['gamma']:.6e}, n_eff={b['n_eff']:.6e}, objective={b['objective']:.6e}, feasible={b['is_feasible']}"
        )
    print(f"\nSaved outputs to: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Systematic, theory-guided parameter optimization for PDE-priority CV-DV LCHS."
    )
    parser.add_argument("--output-dir", default="systematic_opt_results", help="Output directory.")
    parser.add_argument("--global-samples", type=int, default=30, help="Number of global LHS samples.")
    parser.add_argument("--n-starts", type=int, default=4, help="Number of local starts from best global points.")
    parser.add_argument("--local-maxiter", type=int, default=30, help="Max iterations per local optimization.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--min-post-prob", type=float, default=1e-3, help="Post-selection feasibility threshold.")
    parser.add_argument("--gamma-min", type=float, default=0.03, help="Theory prior: minimum gamma.")
    parser.add_argument("--neff-max", type=float, default=16.0, help="Theory prior: maximum effective mode count.")
    parser.add_argument("--post-penalty", type=float, default=100.0, help="Penalty for post_prob below threshold.")
    parser.add_argument("--gamma-penalty", type=float, default=0.2, help="Penalty for gamma below threshold.")
    parser.add_argument("--neff-penalty", type=float, default=0.05, help="Penalty for n_eff above threshold.")
    # Narrow defaults for speed; broad examples:
    #   --r-target-min 0.15 --r-target-max 1.4
    #   --r-prime-min 0.03 --r-prime-max 0.9
    parser.add_argument("--r-target-min", type=float, default=0.25)
    parser.add_argument("--r-target-max", type=float, default=1.2)
    parser.add_argument("--r-prime-min", type=float, default=0.05)
    parser.add_argument("--r-prime-max", type=float, default=0.45)
    parser.add_argument("--kernel-beta-min", type=float, default=0.0)
    parser.add_argument("--kernel-beta-max", type=float, default=1.0)
    parser.add_argument("--min-gap", type=float, default=0.02, help="Constraint: r_target - r_prime >= min_gap.")
    parser.add_argument("--progress-every", type=int, default=8, help="Print progress every N local evaluations.")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Checkpoint local CSV every N local evaluations.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing CSVs in output directory.")
    args = parser.parse_args()

    cfg = SearchConfig(
        r_target_min=args.r_target_min,
        r_target_max=args.r_target_max,
        r_prime_min=args.r_prime_min,
        r_prime_max=args.r_prime_max,
        kernel_beta_min=args.kernel_beta_min,
        kernel_beta_max=args.kernel_beta_max,
        min_gap=args.min_gap,
        min_post_prob=args.min_post_prob,
        gamma_min=args.gamma_min,
        neff_max=args.neff_max,
        post_penalty=args.post_penalty,
        gamma_penalty=args.gamma_penalty,
        neff_penalty=args.neff_penalty,
        global_samples=args.global_samples,
        n_starts=args.n_starts,
        local_maxiter=args.local_maxiter,
        progress_every=max(1, args.progress_every),
        checkpoint_every=max(1, args.checkpoint_every),
        seed=args.seed,
    )
    run_systematic_optimize(Path(args.output_dir), cfg, resume=args.resume)


if __name__ == "__main__":
    main()
