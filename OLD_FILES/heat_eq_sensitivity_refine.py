import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

import heat_eq_postselect as hep


@dataclass(frozen=True)
class Settings:
    total_time: float = 1.0
    n_steps: int = 100
    init_qubits: tuple[int, int] = (0, 1)
    n_dim: int = 32
    r_target: float = 1.2
    r_prime: float = 0.3
    kernel_beta: float = 0.8
    fock_expansion_cutoff: float = 1e-8
    n_quad_points: int = 220


def _safe_numeric(value):
    """Convert to int if value is integer-like, else float."""
    try:
        f = float(value)
        if f == int(f) and not np.isnan(f):
            return int(f)
        return f
    except Exception:
        return np.nan


def _rank_key_pde_priority(row: dict, min_post_prob: float):
    """PDE-priority ranking with feasibility gate on post-selection probability."""
    pde = row.get("pde_error", np.nan)
    post = row.get("post_prob", np.nan)
    pde_key = float(pde) if np.isfinite(pde) else float("inf")
    post_val = float(post) if np.isfinite(post) else -np.inf
    infeasible = 0 if post_val >= min_post_prob else 1
    post_key = -post_val if np.isfinite(post_val) else float("inf")
    return (infeasible, pde_key, post_key)


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_xy(path: Path, x, y, xlabel: str, ylabel: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(x, y, "o-", color="tab:blue")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_surface(path: Path, x, y, z, zlabel: str, title: str):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if np.count_nonzero(mask) < 3:
        return

    x = x[mask]
    y = y[mask]
    z = z[mask]

    fig = plt.figure(figsize=(8.2, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(x, y, z, cmap="viridis", linewidth=0.2, antialiased=True)
    ax.scatter(x, y, z, c="k", s=10, alpha=0.45)
    ax.set_xlabel("r_target")
    ax.set_ylabel("r_prime")
    ax.set_zlabel(zlabel)
    fig.colorbar(surf, ax=ax, shrink=0.72, pad=0.1, label=zlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _pareto_front(rows: list[dict]):
    """
    Return non-dominated points for objective: minimize pde_error, maximize post_prob.
    """
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
    ax.scatter(x, y, c="tab:blue", s=20, alpha=0.6, label="Candidates")
    if pareto_rows:
        px = np.array([float(r["post_prob"]) for r in pareto_rows], dtype=float)
        py = np.array([float(r["pde_error"]) for r in pareto_rows], dtype=float)
        order = np.argsort(px)
        ax.plot(px[order], py[order], "-", color="tab:red", lw=1.7, label="Pareto front")
        ax.scatter(px, py, c="tab:red", s=34, alpha=0.95)
    ax.set_xscale("log")
    ax.set_xlabel("Post-selection probability (log scale)")
    ax.set_ylabel("Relative PDE-vector error")
    ax.set_title("PDE error vs post-selection probability")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


class Evaluator:
    def __init__(self):
        self.coeff_cache: dict[tuple, np.ndarray] = {}

    def get_coeffs(self, r_target: float, r_prime: float, n_dim: int, kernel_beta: float, n_quad_points: int):
        key = (
            round(float(r_target), 8),
            round(float(r_prime), 8),
            int(n_dim),
            round(float(kernel_beta), 8),
            int(n_quad_points),
        )
        coeffs = self.coeff_cache.get(key)
        if coeffs is None:
            coeffs = hep.lchs_coefficients(r_target, r_prime, n_dim, kernel_beta=kernel_beta, n_quad_points=n_quad_points)
            self.coeff_cache[key] = coeffs
        return coeffs

    def evaluate(self, cfg: Settings) -> dict:
        if cfg.r_prime >= cfg.r_target:
            return {
                "fidelity": np.nan,
                "post_prob": np.nan,
                "purity": np.nan,
                "pde_error": np.nan,
                "used_fock_terms": 0,
            }

        coeffs = self.get_coeffs(cfg.r_target, cfg.r_prime, cfg.n_dim, cfg.kernel_beta, cfg.n_quad_points)
        weights = np.abs(coeffs) ** 2
        post_prob = 0.0
        pauli_accum = np.zeros(len(hep.PAULI_COMBOS), dtype=float)
        used = 0

        for n, w in enumerate(weights):
            if w < cfg.fock_expansion_cutoff:
                continue
            res = hep.cvdv_heat_postselect_fock_component(
                fock_n=n,
                total_time=cfg.total_time,
                n_steps=cfg.n_steps,
                init_qubits=cfg.init_qubits,
                r_target=cfg.r_target,
                r_prime=cfg.r_prime,
            )
            post_prob += float(w) * float(res[0])
            pauli_accum += float(w) * np.array(res[1:], dtype=float)
            used += 1

        if np.isclose(post_prob, 0.0):
            fidelity = np.nan
            purity = np.nan
            pde_error = np.nan
        else:
            rho_raw = hep.rebuild_density_from_paulis(pauli_accum) / post_prob
            try:
                rho_post, _ = hep.sanitize_density_matrix(rho_raw)
            except Exception:
                fidelity = np.nan
                purity = np.nan
                pde_error = np.nan
                return {
                    "fidelity": fidelity,
                    "post_prob": float(post_prob),
                    "purity": purity,
                    "pde_error": pde_error,
                    "used_fock_terms": int(used),
                }

            purity = float(np.real(np.trace(rho_post @ rho_post)))
            dv_gen = hep.dv_generator_matrix()
            u_theory = expm(-hep.alpha * cfg.total_time * dv_gen) @ hep.initial_dv_state(cfg.init_qubits)
            norm_theory = np.linalg.norm(u_theory)
            if np.isclose(norm_theory, 0.0):
                fidelity = np.nan
                pde_error = np.nan
            else:
                u_theory_norm = u_theory / norm_theory
                fidelity = float(np.real(np.vdot(u_theory_norm, rho_post @ u_theory_norm)))
                fidelity = float(np.clip(fidelity, 0.0, 1.0))
                # PDE vector error: ||u_theory - sqrt(p_post)*psi_principal|| / ||u_theory||
                psi_principal = hep.principal_statevector(rho_post)
                overlap = np.vdot(u_theory_norm, psi_principal)
                if np.abs(overlap) > 0:
                    psi_principal = psi_principal * np.exp(-1j * np.angle(overlap))
                u_cvdv = np.sqrt(post_prob) * psi_principal
                pde_error = float(np.linalg.norm(u_theory - u_cvdv) / norm_theory)

        return {
            "fidelity": fidelity,
            "post_prob": float(post_prob),
            "purity": purity,
            "pde_error": pde_error,
            "used_fock_terms": int(used),
        }


def _sweep_1d(ev: Evaluator, base: Settings, name: str, values: np.ndarray):
    rows = []
    for v in values:
        cfg = replace(base, **{name: _safe_numeric(v)})
        m = ev.evaluate(cfg)
        row = {
            "value": float(v),
            "fidelity": m["fidelity"],
            "post_prob": m["post_prob"],
            "purity": m["purity"],
            "pde_error": m["pde_error"],
            "used_fock_terms": m["used_fock_terms"],
        }
        rows.append(row)
        print(
            f"{name}={float(v):.6g}  pde_err={m['pde_error']:.6e}  "
            f"post_prob={m['post_prob']:.6e}  purity={m['purity']:.6e}  terms={m['used_fock_terms']}"
        )
    return rows


def run(profile: str, output_dir: Path, min_post_prob: float, include_numerics_tests: bool):
    output_dir.mkdir(parents=True, exist_ok=True)
    base = Settings()
    ev = Evaluator()

    if profile == "quick":
        sweeps = {
            "r_target": np.linspace(0.8, 1.2, 6),
            "r_prime": np.array([0.03, 0.05, 0.08, 0.12, 0.16, 0.20]),
            "kernel_beta": np.linspace(0.4, 0.9, 6),
            "n_steps": np.array([70, 100, 130]),
        }
        if include_numerics_tests:
            sweeps["fock_expansion_cutoff"] = np.array([1e-6, 1e-8, 1e-10])
            sweeps["n_quad_points"] = np.array([140, 220, 320])
        r_target_grid = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        r_prime_grid = np.array([0.03, 0.06, 0.10, 0.14, 0.18])
    elif profile == "full":
        sweeps = {
            "r_target": np.linspace(0.7, 1.4, 12),
            "r_prime": np.linspace(0.03, 0.35, 12),
            "kernel_beta": np.linspace(0.3, 1.0, 12),
            "n_steps": np.array([60, 80, 100, 120, 150, 180]),
        }
        if include_numerics_tests:
            sweeps["fock_expansion_cutoff"] = np.array([1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
            sweeps["n_quad_points"] = np.array([120, 160, 220, 280, 360, 440])
        r_target_grid = np.linspace(0.7, 1.4, 10)
        r_prime_grid = np.linspace(0.03, 0.35, 10)
    else:
        sweeps = {
            "r_target": np.linspace(0.75, 1.3, 9),
            "r_prime": np.linspace(0.03, 0.25, 9),
            "kernel_beta": np.linspace(0.35, 0.95, 9),
            "n_steps": np.array([70, 90, 100, 110, 130, 150]),
        }
        if include_numerics_tests:
            sweeps["fock_expansion_cutoff"] = np.array([1e-6, 1e-7, 1e-8, 1e-9, 1e-10])
            sweeps["n_quad_points"] = np.array([140, 180, 220, 280, 360])
        r_target_grid = np.linspace(0.75, 1.3, 8)
        r_prime_grid = np.linspace(0.03, 0.25, 8)

    all_candidates = []
    print(
        f"PDE-priority mode: rank by pde_error first, then post_prob; "
        f"post_prob feasibility threshold={min_post_prob:.2e}"
    )
    core = ("r_target", "r_prime", "kernel_beta", "n_steps")
    active_sweeps = ", ".join(sweeps.keys())
    print(f"Active sweeps: {active_sweeps}")
    print(
        "Sweep set: "
        + ("core 4 tests (+ numerics)" if include_numerics_tests else "core 4 tests only")
        + f" [{', '.join(core)}]"
    )

    print("Running 1D sweeps...")
    for name, values in sweeps.items():
        print(f"\n--- Sweep: {name} ---")
        rows = _sweep_1d(ev, base, name, values)
        _write_csv(output_dir / f"refine_{name}.csv", rows)
        _plot_xy(
            output_dir / f"refine_{name}_pde_error.png",
            [r["value"] for r in rows],
            [r["pde_error"] for r in rows],
            name,
            "Relative PDE-vector error",
            f"PDE-vector error vs {name}",
        )
        _plot_xy(
            output_dir / f"refine_{name}_post_prob.png",
            [r["value"] for r in rows],
            [r["post_prob"] for r in rows],
            name,
            "Post-selection probability",
            f"Post-selection probability vs {name}",
        )
        for r in rows:
            candidate = dict(base.__dict__)
            candidate[name] = float(r["value"])
            candidate.update(
                {
                    "fidelity": r["fidelity"],
                    "post_prob": r["post_prob"],
                    "purity": r["purity"],
                    "pde_error": r["pde_error"],
                    "used_fock_terms": r["used_fock_terms"],
                    "is_feasible": int(np.isfinite(r["post_prob"]) and (r["post_prob"] >= min_post_prob)),
                }
            )
            all_candidates.append(candidate)

    print("\n--- Coupled sweep: r_target vs r_prime ---")
    surf_rows = []
    for r_target in r_target_grid:
        for r_prime in r_prime_grid:
            if r_prime >= r_target:
                continue
            cfg = replace(base, r_target=float(r_target), r_prime=float(r_prime))
            m = ev.evaluate(cfg)
            row = {
                "r_target": float(r_target),
                "r_prime": float(r_prime),
                "fidelity": m["fidelity"],
                "post_prob": m["post_prob"],
                "purity": m["purity"],
                "pde_error": m["pde_error"],
                "used_fock_terms": m["used_fock_terms"],
                "is_feasible": int(np.isfinite(m["post_prob"]) and (m["post_prob"] >= min_post_prob)),
            }
            surf_rows.append(row)
            all_candidates.append({**base.__dict__, **row})
            print(
                f"r_target={r_target:.4f}, r_prime={r_prime:.4f}  "
                f"pde_err={m['pde_error']:.6e}  post_prob={m['post_prob']:.6e}"
            )

    _write_csv(output_dir / "refine_rtarget_rprime_surface.csv", surf_rows)
    _plot_surface(
        output_dir / "refine_rtarget_rprime_surface_pde_error_3d.png",
        [r["r_target"] for r in surf_rows],
        [r["r_prime"] for r in surf_rows],
        [r["pde_error"] for r in surf_rows],
        "Relative PDE-vector error",
        "PDE-vector error surface: r_target vs r_prime",
    )

    ranked = [
        c
        for c in all_candidates
        if np.isfinite(c.get("pde_error", np.nan)) and np.isfinite(c.get("post_prob", np.nan))
    ]
    ranked.sort(key=lambda r: _rank_key_pde_priority(r, min_post_prob))
    top = ranked[:20]
    _write_csv(output_dir / "refine_top_candidates.csv", top)
    pareto_rows = _pareto_front(ranked)
    _write_csv(output_dir / "refine_pareto_front.csv", pareto_rows)
    _plot_tradeoff(output_dir / "refine_tradeoff_pde_vs_post.png", ranked, pareto_rows)

    feasible_count = sum(int(c.get("is_feasible", 0)) for c in ranked)
    print(f"\nFeasible candidates (post_prob >= {min_post_prob:.2e}): {feasible_count}/{len(ranked)}")
    print("Top candidates (PDE-priority rank):")
    for i, c in enumerate(top[:10], start=1):
        pde_str = f"{c['pde_error']:.4f}" if np.isfinite(c.get("pde_error", np.nan)) else "N/A"
        feasibility = "Y" if int(c.get("is_feasible", 0)) else "N"
        print(
            f"{i:2d}. feasible={feasibility}  pde_err={pde_str}, post_prob={c['post_prob']:.6e}, "
            f"purity={c['purity']:.6e}, "
            f"r_target={c['r_target']:.6g}, r_prime={c['r_prime']:.6g}, "
            f"kernel_beta={c.get('kernel_beta', 'N/A')}"
        )

    print(f"\nSaved outputs to: {output_dir.resolve()}")


def run_optimize(
    output_dir: Path,
    maxiter: int = 200,
    min_post_prob: float = 1e-3,
    low_post_penalty: float = 100.0,
):
    """Joint Nelder-Mead optimization over (r_target, r_prime, kernel_beta)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    base = Settings()
    ev = Evaluator()
    eval_log: list[dict] = []

    # Primary objective: minimize PDE-vector error.
    # Secondary objective: maximize post_prob.
    # Feasibility: penalize post_prob below min_post_prob.
    post_tiebreak_weight = 1e-3

    def objective(x):
        r_target, r_prime, kernel_beta = x
        # Enforce constraints
        if r_prime >= r_target or r_prime <= 0 or r_target <= 0 or kernel_beta < 0:
            return 1.0
        try:
            cfg = replace(
                base,
                r_target=float(r_target),
                r_prime=float(r_prime),
                kernel_beta=float(kernel_beta),
            )
            m = ev.evaluate(cfg)
        except Exception as exc:
            print(f"  eval error: {exc}")
            return 1.0
        if not np.isfinite(m.get("pde_error", np.nan)):
            return 1.0
        post = float(m["post_prob"])
        low_post_violation = max(0.0, min_post_prob - post)
        score = float(m["pde_error"]) + low_post_penalty * low_post_violation - post_tiebreak_weight * post
        feasible = int(post >= min_post_prob)
        eval_log.append({
            "r_target": float(r_target),
            "r_prime": float(r_prime),
            "kernel_beta": float(kernel_beta),
            "fidelity": m["fidelity"],
            "pde_error": m["pde_error"],
            "post_prob": m["post_prob"],
            "purity": m["purity"],
            "is_feasible": feasible,
            "low_post_violation": low_post_violation,
            "objective": score,
        })
        pde_str = f"{m['pde_error']:.4f}" if np.isfinite(m.get("pde_error", np.nan)) else "N/A"
        print(
            f"  r={r_target:.4f}, r'={r_prime:.4f}, kb={kernel_beta:.4f}  "
            f"-> pde_err={pde_str}, post_prob={post:.6e}, feasible={feasible}, objective={score:.6f}"
        )
        return score

    x0 = [base.r_target, base.r_prime, base.kernel_beta]
    print(f"Starting Nelder-Mead optimization from: {x0}")
    print(f"Max iterations: {maxiter}")

    result = minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 0.005, "fatol": 1e-5, "adaptive": True},
    )

    print(f"\nOptimization {'converged' if result.success else 'stopped'}: {result.message}")
    print(f"Evaluations: {result.nfev}")
    best_x = result.x
    print(f"Best parameters:")
    print(f"  r_target     = {best_x[0]:.6f}")
    print(f"  r_prime      = {best_x[1]:.6f}")
    print(f"  kernel_beta  = {best_x[2]:.6f}")
    print(f"  objective    = {result.fun:.6f}")

    _write_csv(output_dir / "optimize_log.csv", eval_log)

    # Sort by project priorities: pde_error first, then post_prob.
    valid = [
        r
        for r in eval_log
        if np.isfinite(r.get("pde_error", np.nan)) and np.isfinite(r.get("post_prob", np.nan))
    ]
    valid.sort(key=lambda r: _rank_key_pde_priority(r, min_post_prob))
    _write_csv(output_dir / "optimize_top.csv", valid[:20])

    if valid:
        best = valid[0]
        pde_str = f"{best['pde_error']:.4f}" if np.isfinite(best.get("pde_error", np.nan)) else "N/A"
        feasibility = "Y" if int(best.get("is_feasible", 0)) else "N"
        print("\nBest from all evaluations (pde_error priority):")
        print(
            f"  feasible={feasibility}, pde_error={pde_str}, post_prob={best['post_prob']:.6e}, "
            f"purity={best['purity']:.6f}"
        )

    print(f"\nSaved optimization log to: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Refinement sensitivity script for heat_eq_postselect settings."
    )
    parser.add_argument(
        "--profile",
        choices=["quick", "default", "full"],
        default="default",
        help="Sweep size profile.",
    )
    parser.add_argument(
        "--output-dir",
        default="sensitivity_refine_results",
        help="Directory for CSV/plot outputs.",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Run joint Nelder-Mead optimization instead of grid sweeps.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=200,
        help="Max iterations for optimizer (default: 200).",
    )
    parser.add_argument(
        "--min-post-prob",
        type=float,
        default=1e-3,
        help="Feasibility threshold for post-selection probability in PDE-priority ranking.",
    )
    parser.add_argument(
        "--low-post-penalty",
        type=float,
        default=100.0,
        help="Objective penalty weight for post_prob below --min-post-prob (optimizer only).",
    )
    parser.add_argument(
        "--include-numerics-tests",
        action="store_true",
        help=(
            "Include extra numerical sweeps (fock_expansion_cutoff, n_quad_points). "
            "Default keeps only 4 core PDE-priority tests to reduce runtime."
        ),
    )
    args = parser.parse_args()
    if args.optimize:
        run_optimize(
            Path(args.output_dir),
            maxiter=args.maxiter,
            min_post_prob=args.min_post_prob,
            low_post_penalty=args.low_post_penalty,
        )
    else:
        run(
            args.profile,
            Path(args.output_dir),
            min_post_prob=args.min_post_prob,
            include_numerics_tests=args.include_numerics_tests,
        )


if __name__ == "__main__":
    main()
