import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm

import heat_eq_postselect as hep


@dataclass(frozen=True)
class Settings:
    total_time: float = 1.0
    n_steps: int = 100
    init_qubits: tuple[int, int] = (0, 1)
    n_dim: int = 64
    r_target: float = 1.2
    r_prime: float = 0.3
    beta: float = 0.4
    use_displacement: bool = True
    fock_expansion_cutoff: float = 1e-8
    n_quad_points: int = 220


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


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


def _plot_surface(path: Path, x, y, z):
    fig = plt.figure(figsize=(8.2, 5.6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(x, y, z, cmap="viridis", linewidth=0.2, antialiased=True)
    ax.scatter(x, y, z, c="k", s=10, alpha=0.45)
    ax.set_xlabel("r_target")
    ax.set_ylabel("r_prime")
    ax.set_zlabel("Fidelity")
    ax.set_zlim(0.0, 1.0)
    fig.colorbar(surf, ax=ax, shrink=0.72, pad=0.1, label="Fidelity")
    ax.set_title("Fidelity Surface: r_target vs r_prime")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


class Evaluator:
    def __init__(self):
        self.coeff_cache: dict[tuple[float, float, int, int], np.ndarray] = {}

    def get_coeffs(self, r_target: float, r_prime: float, n_dim: int, n_quad_points: int):
        key = (
            round(float(r_target), 8),
            round(float(r_prime), 8),
            int(n_dim),
            int(n_quad_points),
        )
        coeffs = self.coeff_cache.get(key)
        if coeffs is None:
            coeffs = hep.lchs_coefficients(r_target, r_prime, n_dim, n_quad_points=n_quad_points)
            self.coeff_cache[key] = coeffs
        return coeffs

    def evaluate(self, cfg: Settings) -> dict:
        if cfg.r_prime >= cfg.r_target:
            return {
                "fidelity": np.nan,
                "post_prob": np.nan,
                "purity": np.nan,
                "used_fock_terms": 0,
            }

        coeffs = self.get_coeffs(cfg.r_target, cfg.r_prime, cfg.n_dim, cfg.n_quad_points)
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
        else:
            rho_post = hep.rebuild_density_from_paulis(pauli_accum) / post_prob
            purity = float(np.real(np.trace(rho_post @ rho_post)))
            u_theory = expm(-hep.alpha * cfg.total_time * hep.dv_generator_matrix()) @ hep.initial_dv_state(cfg.init_qubits)
            norm_theory = np.linalg.norm(u_theory)
            if np.isclose(norm_theory, 0.0):
                fidelity = np.nan
            else:
                u_theory = u_theory / norm_theory
                fidelity = float(np.real(np.vdot(u_theory, rho_post @ u_theory)))
                fidelity = float(np.clip(fidelity, 0.0, 1.0))

        return {
            "fidelity": fidelity,
            "post_prob": float(post_prob),
            "purity": purity,
            "used_fock_terms": int(used),
        }

    def cv_prep_fidelity(self, cfg: Settings, beta: float) -> float:
        psi_lchs, _ = hep.get_lchs_states(
            cfg.r_target, cfg.r_prime, cfg.n_dim, n_quad_points=cfg.n_quad_points
        )
        psi_gauss = hep.gaussian_cv_state(
            cfg.n_dim, cfg.r_prime, beta, use_displacement=cfg.use_displacement
        )
        return float(np.abs(np.vdot(psi_lchs, psi_gauss)) ** 2)


def _sweep_1d(ev: Evaluator, base: Settings, name: str, values: np.ndarray):
    rows = []
    for v in values:
        cfg = replace(base, **{name: _safe_float(v)})
        m = ev.evaluate(cfg)
        row = {
            "value": float(v),
            "fidelity": m["fidelity"],
            "post_prob": m["post_prob"],
            "purity": m["purity"],
            "used_fock_terms": m["used_fock_terms"],
        }
        rows.append(row)
        print(
            f"{name}={float(v):.6g}  fidelity={m['fidelity']:.6e}  "
            f"post_prob={m['post_prob']:.6e}  terms={m['used_fock_terms']}"
        )
    return rows


def run(profile: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    base = Settings()
    ev = Evaluator()

    if profile == "quick":
        sweeps = {
            "r_target": np.linspace(1.0, 1.5, 6),
            "r_prime": np.linspace(0.15, 0.55, 5),
            "n_steps": np.array([70, 100, 130]),
            "fock_expansion_cutoff": np.array([1e-6, 1e-8, 1e-10]),
            "n_quad_points": np.array([140, 220, 320]),
        }
        r_target_grid = np.array([1.0, 1.2, 1.4, 1.6])
        r_prime_grid = np.array([0.15, 0.3, 0.45, 0.6])
        beta_grid = np.linspace(0.1, 1.2, 9)
    elif profile == "full":
        sweeps = {
            "r_target": np.linspace(0.9, 1.8, 12),
            "r_prime": np.linspace(0.1, 0.8, 12),
            "n_steps": np.array([60, 80, 100, 120, 150, 180]),
            "fock_expansion_cutoff": np.array([1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]),
            "n_quad_points": np.array([120, 160, 220, 280, 360, 440]),
        }
        r_target_grid = np.linspace(0.9, 1.8, 10)
        r_prime_grid = np.linspace(0.1, 0.8, 10)
        beta_grid = np.linspace(0.1, 1.5, 15)
    else:
        sweeps = {
            "r_target": np.linspace(0.95, 1.7, 9),
            "r_prime": np.linspace(0.12, 0.72, 9),
            "n_steps": np.array([70, 90, 100, 110, 130, 150]),
            "fock_expansion_cutoff": np.array([1e-6, 1e-7, 1e-8, 1e-9, 1e-10]),
            "n_quad_points": np.array([140, 180, 220, 280, 360]),
        }
        r_target_grid = np.linspace(0.95, 1.7, 8)
        r_prime_grid = np.linspace(0.12, 0.72, 8)
        beta_grid = np.linspace(0.1, 1.3, 11)

    all_candidates = []

    print("Running 1D sweeps...")
    for name, values in sweeps.items():
        print(f"\n--- Sweep: {name} ---")
        rows = _sweep_1d(ev, base, name, values)
        _write_csv(output_dir / f"refine_{name}.csv", rows)
        _plot_xy(
            output_dir / f"refine_{name}_fidelity.png",
            [r["value"] for r in rows],
            [r["fidelity"] for r in rows],
            name,
            "Fidelity",
            f"Fidelity vs {name}",
        )
        for r in rows:
            candidate = dict(base.__dict__)
            candidate[name] = float(r["value"])
            candidate.update(
                {
                    "fidelity": r["fidelity"],
                    "post_prob": r["post_prob"],
                    "purity": r["purity"],
                    "used_fock_terms": r["used_fock_terms"],
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
                "used_fock_terms": m["used_fock_terms"],
            }
            surf_rows.append(row)
            all_candidates.append({**base.__dict__, **row})
            print(
                f"r_target={r_target:.4f}, r_prime={r_prime:.4f}  fidelity={m['fidelity']:.6e}"
            )

    _write_csv(output_dir / "refine_rtarget_rprime_surface.csv", surf_rows)
    _plot_surface(
        output_dir / "refine_rtarget_rprime_surface_fidelity_3d.png",
        [r["r_target"] for r in surf_rows],
        [r["r_prime"] for r in surf_rows],
        [r["fidelity"] for r in surf_rows],
    )

    print("\n--- CV prep diagnostic: beta vs |<psi_lchs|psi_gauss>|^2 ---")
    beta_rows = []
    for beta in beta_grid:
        cv_fid = ev.cv_prep_fidelity(base, float(beta))
        row = {"beta": float(beta), "cv_prep_fidelity": cv_fid}
        beta_rows.append(row)
        print(f"beta={beta:.4f}  cv_prep_fidelity={cv_fid:.6e}")
    _write_csv(output_dir / "refine_beta_cvprep.csv", beta_rows)
    _plot_xy(
        output_dir / "refine_beta_cvprep_fidelity.png",
        [r["beta"] for r in beta_rows],
        [r["cv_prep_fidelity"] for r in beta_rows],
        "beta",
        "|<psi_lchs|psi_gauss>|^2",
        "CV prep overlap vs beta (diagnostic)",
    )

    ranked = [
        c for c in all_candidates if np.isfinite(c.get("fidelity", np.nan))
    ]
    ranked.sort(key=lambda x: x["fidelity"], reverse=True)
    top = ranked[:20]
    _write_csv(output_dir / "refine_top_candidates.csv", top)

    print("\nTop candidates (by fidelity):")
    for i, c in enumerate(top[:10], start=1):
        print(
            f"{i:2d}. fidelity={c['fidelity']:.6e}, post_prob={c['post_prob']:.6e}, "
            f"r_target={c['r_target']:.6g}, r_prime={c['r_prime']:.6g}, "
            f"n_steps={int(c['n_steps'])}, cutoff={c['fock_expansion_cutoff']:.1e}, "
            f"n_quad={int(c['n_quad_points'])}"
        )

    print(f"\nSaved outputs to: {output_dir.resolve()}")


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
    args = parser.parse_args()
    run(args.profile, Path(args.output_dir))


if __name__ == "__main__":
    main()
