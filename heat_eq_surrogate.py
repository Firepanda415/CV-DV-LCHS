"""
Surrogate model for CV-DV LCHS heat equation optimization.

Theoretical basis
-----------------
The PDE output depends on three parameters (r_target, r_prime, kernel_beta) through:

    C_n(r, r', β_k)  -- LCHS coefficients (cheap, analytical)
    (p_n, ρ_n)(r, r') -- per-Fock-component circuit outputs (expensive)

The mixture aggregation is:
    p_post = Σ_n |C_n|² p_n
    ρ_post = (1/p_post) Σ_n |C_n|² p_n ρ_n

Key insight: kernel_beta only affects C_n, not the circuit.
So for a fixed (r_target, r_prime) grid, we precompute {p_n, pauli_n} once,
then sweep kernel_beta at zero additional circuit cost.

For the full 3-parameter landscape, we precompute on a (r_target, r_prime) grid,
store per-Fock results, and reconstruct metrics for any kernel_beta analytically.

Usage
-----
Phase 1 -- precompute (slow, one-time):
    python heat_eq_surrogate.py --precompute --output-dir surrogate_data

Phase 2 -- sweep kernel_beta at each grid point (fast):
    python heat_eq_surrogate.py --sweep --output-dir surrogate_data

Phase 3 -- fine optimize around best region (fast):
    python heat_eq_surrogate.py --optimize --output-dir surrogate_data
"""

import argparse
import csv
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

import heat_eq_postselect as hep
from heat_eq_sensitivity_refine import Evaluator, Settings


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SurrogateConfig:
    # grid for (r_target, r_prime)
    r_target_values: tuple = (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2)
    r_prime_values: tuple = (0.03, 0.06, 0.10, 0.15, 0.20, 0.25, 0.30)
    # kernel_beta sweep (no circuit cost)
    kernel_beta_values: tuple = tuple(np.round(np.linspace(0.3, 1.2, 19), 4))
    # circuit settings (fixed during precompute)
    n_dim: int = 32
    n_steps: int = 100
    total_time: float = 1.0
    init_qubits: tuple = (0, 1)
    n_quad_points: int = 220
    fock_expansion_cutoff: float = 1e-8
    # optimizer
    optimize_maxiter: int = 200


def _config_manifest(cfg: SurrogateConfig):
    """Serialize config in a stable JSON-friendly structure."""
    return {
        "r_target_values": [float(v) for v in cfg.r_target_values],
        "r_prime_values": [float(v) for v in cfg.r_prime_values],
        "kernel_beta_values": [float(v) for v in cfg.kernel_beta_values],
        "n_dim": int(cfg.n_dim),
        "n_steps": int(cfg.n_steps),
        "total_time": float(cfg.total_time),
        "init_qubits": [int(v) for v in cfg.init_qubits],
        "n_quad_points": int(cfg.n_quad_points),
        "fock_expansion_cutoff": float(cfg.fock_expansion_cutoff),
        "max_fock_level": int(hep.MAX_FOCK_LEVEL),
    }


def _save_manifest(path: Path, cfg: SurrogateConfig):
    with open(path, "w") as f:
        json.dump(_config_manifest(cfg), f, indent=2)


def _load_manifest(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _validate_loaded_config(saved_cfg: dict | None, cfg: SurrogateConfig):
    """
    Enforce compatibility between precompute-time settings and current run settings.
    """
    if saved_cfg is None:
        raise FileNotFoundError(
            "Missing surrogate_config.json next to fock_components.json. "
            "Re-run --precompute to generate consistent surrogate artifacts."
        )

    current = _config_manifest(cfg)
    critical_fields = ("n_dim", "n_steps", "init_qubits", "max_fock_level")
    mismatches = []
    for name in critical_fields:
        if saved_cfg.get(name) != current.get(name):
            mismatches.append((name, saved_cfg.get(name), current.get(name)))
    if not np.isclose(
        float(saved_cfg.get("total_time", np.nan)),
        float(current.get("total_time", np.nan)),
        atol=1e-12,
        rtol=0.0,
    ):
        mismatches.append(("total_time", saved_cfg.get("total_time"), current.get("total_time")))
    if mismatches:
        mismatch_lines = ", ".join(
            [f"{name}: saved={old} current={new}" for name, old, new in mismatches]
        )
        raise ValueError(
            "Precomputed data is incompatible with current settings. "
            f"Mismatches: {mismatch_lines}. "
            "Re-run --precompute with the same settings."
        )

    # Informational warnings for non-critical differences.
    for name in ("n_quad_points", "fock_expansion_cutoff", "r_target_values", "r_prime_values"):
        if saved_cfg.get(name) != current.get(name):
            print(
                f"Warning: using current {name}={current.get(name)} with precomputed data "
                f"generated under {name}={saved_cfg.get(name)}."
            )


# ---------------------------------------------------------------------------
# PDE reference (computed once)
# ---------------------------------------------------------------------------
def _pde_reference(cfg: SurrogateConfig):
    """Return u_theory and its norm."""
    dv_gen = hep.dv_generator_matrix()
    u0 = hep.initial_dv_state(cfg.init_qubits)
    u_theory = expm(-hep.alpha * cfg.total_time * dv_gen) @ u0
    return u_theory, float(np.linalg.norm(u_theory))


# ---------------------------------------------------------------------------
# Phase 1: precompute per-Fock circuit results on (r_target, r_prime) grid
# ---------------------------------------------------------------------------
def precompute_fock_components(output_dir: Path, cfg: SurrogateConfig):
    """Run circuit for each (r_target, r_prime, fock_n) and store results."""
    if cfg.n_dim > hep.MAX_FOCK_LEVEL:
        raise ValueError(
            f"n_dim={cfg.n_dim} exceeds backend MAX_FOCK_LEVEL={hep.MAX_FOCK_LEVEL}. "
            "Lower --n-dim or increase backend cutoff in heat_eq_postselect.py."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    n_r = len(cfg.r_target_values)
    n_rp = len(cfg.r_prime_values)
    total_grid = 0
    for rt in cfg.r_target_values:
        for rp in cfg.r_prime_values:
            if rp < rt:
                total_grid += 1

    print(f"Precomputing Fock components on {total_grid} valid (r_target, r_prime) grid points")
    print(f"  r_target: {cfg.r_target_values}")
    print(f"  r_prime:  {cfg.r_prime_values}")
    print(f"  n_dim={cfg.n_dim}, n_steps={cfg.n_steps}")
    print(f"  Max circuit evals: {total_grid * cfg.n_dim}")

    results = {}
    eval_count = 0
    t0 = time.time()

    for i_rt, r_target in enumerate(cfg.r_target_values):
        for i_rp, r_prime in enumerate(cfg.r_prime_values):
            if r_prime >= r_target:
                continue

            # Collision-safe key (index-based, not rounded float string).
            key = f"irt{i_rt:03d}_irp{i_rp:03d}"
            fock_data = {}

            for n in range(cfg.n_dim):
                try:
                    res = hep.cvdv_heat_postselect_fock_component(
                        fock_n=n,
                        total_time=cfg.total_time,
                        n_steps=cfg.n_steps,
                        init_qubits=cfg.init_qubits,
                        r_target=r_target,
                        r_prime=r_prime,
                    )
                    p_n = float(res[0])
                    paulis_n = [float(x) for x in res[1:]]
                    fock_data[str(n)] = {"p_n": p_n, "paulis": paulis_n}
                except Exception as exc:
                    fock_data[str(n)] = {"p_n": float("nan"), "paulis": [], "error": str(exc)}

                eval_count += 1
                if eval_count % 50 == 0:
                    elapsed = time.time() - t0
                    rate = eval_count / elapsed
                    print(f"  {eval_count} evals done ({elapsed:.0f}s, {rate:.1f} eval/s)")

            results[key] = {
                "r_target": r_target,
                "r_prime": r_prime,
                "fock_components": fock_data,
            }

    elapsed = time.time() - t0
    print(f"Precompute done: {eval_count} evals in {elapsed:.1f}s")

    # Save
    out_path = output_dir / "fock_components.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"Saved to {out_path}")

    # Also save config manifest for compatibility checks.
    config_path = output_dir / "surrogate_config.json"
    _save_manifest(config_path, cfg)


def load_fock_components(output_dir: Path, cfg: SurrogateConfig):
    """Load precomputed Fock component data and validate config compatibility."""
    path = output_dir / "fock_components.json"
    with open(path) as f:
        data = json.load(f)
    saved_cfg = _load_manifest(output_dir / "surrogate_config.json")
    _validate_loaded_config(saved_cfg, cfg)
    return data


# ---------------------------------------------------------------------------
# Phase 2: reconstruct metrics from precomputed data + arbitrary kernel_beta
# ---------------------------------------------------------------------------
def reconstruct_metrics(
    fock_data: dict,
    r_target: float,
    r_prime: float,
    kernel_beta: float,
    cfg: SurrogateConfig,
    u_theory: np.ndarray,
    norm_theory: float,
):
    """
    Given precomputed per-Fock circuit results and a kernel_beta,
    compute C_n and reconstruct all metrics analytically.
    """
    # Compute coefficients (cheap)
    try:
        coeffs = hep.lchs_coefficients(
            r_target, r_prime, cfg.n_dim,
            kernel_beta=kernel_beta, n_quad_points=cfg.n_quad_points,
        )
    except Exception:
        return {"pde_error": np.nan, "post_prob": np.nan, "fidelity": np.nan,
                "purity": np.nan, "used_fock_terms": 0, "n_eff": np.nan}

    weights = np.abs(coeffs) ** 2
    n_eff = 1.0 / np.sum(weights ** 2) if np.sum(weights ** 2) > 0 else np.nan

    # Aggregate using precomputed circuit results
    post_prob = 0.0
    pauli_accum = np.zeros(len(hep.PAULI_COMBOS), dtype=float)
    used = 0

    for n in range(cfg.n_dim):
        w = weights[n]
        if w < cfg.fock_expansion_cutoff:
            continue
        n_str = str(n)
        if n_str not in fock_data or "error" in fock_data[n_str]:
            continue
        p_n = fock_data[n_str]["p_n"]
        paulis_n = np.array(fock_data[n_str]["paulis"], dtype=float)
        if not np.isfinite(p_n) or len(paulis_n) != len(hep.PAULI_COMBOS):
            continue
        post_prob += float(w) * p_n
        pauli_accum += float(w) * paulis_n
        used += 1

    if np.isclose(post_prob, 0.0) or not np.isfinite(post_prob):
        return {"pde_error": np.nan, "post_prob": float(post_prob), "fidelity": np.nan,
                "purity": np.nan, "used_fock_terms": used, "n_eff": n_eff}

    # Reconstruct density matrix (same as Evaluator.evaluate)
    rho_raw = hep.rebuild_density_from_paulis(pauli_accum) / post_prob
    try:
        rho_post, _ = hep.sanitize_density_matrix(rho_raw)
    except Exception:
        return {"pde_error": np.nan, "post_prob": float(post_prob), "fidelity": np.nan,
                "purity": np.nan, "used_fock_terms": used, "n_eff": n_eff}

    purity = float(np.real(np.trace(rho_post @ rho_post)))

    u_theory_norm = u_theory / norm_theory
    fidelity = float(np.real(np.vdot(u_theory_norm, rho_post @ u_theory_norm)))
    fidelity = float(np.clip(fidelity, 0.0, 1.0))

    psi_principal = hep.principal_statevector(rho_post)
    overlap = np.vdot(u_theory_norm, psi_principal)
    if np.abs(overlap) > 0:
        psi_principal = psi_principal * np.exp(-1j * np.angle(overlap))
    u_cvdv = np.sqrt(post_prob) * psi_principal
    pde_error = float(np.linalg.norm(u_theory - u_cvdv) / norm_theory)

    return {
        "pde_error": pde_error,
        "post_prob": float(post_prob),
        "fidelity": fidelity,
        "purity": purity,
        "used_fock_terms": used,
        "n_eff": n_eff,
    }


# ---------------------------------------------------------------------------
# Phase 2: sweep kernel_beta at each grid point
# ---------------------------------------------------------------------------
def sweep_kernel_beta(output_dir: Path, cfg: SurrogateConfig):
    """For each precomputed (r_target, r_prime), sweep kernel_beta and record metrics."""
    if cfg.n_dim > hep.MAX_FOCK_LEVEL:
        raise ValueError(
            f"n_dim={cfg.n_dim} exceeds backend MAX_FOCK_LEVEL={hep.MAX_FOCK_LEVEL}."
        )
    fock_db = load_fock_components(output_dir, cfg)
    u_theory, norm_theory = _pde_reference(cfg)

    rows = []
    print(f"Sweeping kernel_beta at {len(fock_db)} grid points x {len(cfg.kernel_beta_values)} beta values")
    t0 = time.time()

    for key, entry in fock_db.items():
        r_target = entry["r_target"]
        r_prime = entry["r_prime"]
        fock_data = entry["fock_components"]

        for kb in cfg.kernel_beta_values:
            m = reconstruct_metrics(
                fock_data, r_target, r_prime, float(kb), cfg, u_theory, norm_theory
            )
            rows.append({
                "r_target": r_target,
                "r_prime": r_prime,
                "kernel_beta": float(kb),
                **m,
            })

    elapsed = time.time() - t0
    print(f"Sweep done: {len(rows)} evaluations in {elapsed:.1f}s")

    # Save CSV
    csv_path = output_dir / "surrogate_sweep.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {csv_path}")

    # Find and print best
    valid = [r for r in rows if np.isfinite(r["pde_error"])]
    if valid:
        valid.sort(key=lambda r: r["pde_error"])
        print("\nTop 10 by pde_error:")
        print(f"  {'r_target':>8s}  {'r_prime':>8s}  {'kb':>8s}  {'pde_err':>8s}  {'post_p':>8s}  {'fidelity':>8s}  {'n_eff':>6s}")
        for r in valid[:10]:
            print(f"  {r['r_target']:8.4f}  {r['r_prime']:8.4f}  {r['kernel_beta']:8.4f}  "
                  f"{r['pde_error']:8.5f}  {r['post_prob']:8.4f}  {r['fidelity']:8.5f}  {r['n_eff']:6.2f}")

    _plot_sweep(rows, output_dir)
    return rows


def _plot_sweep(rows, output_dir: Path):
    """Generate landscape plots from sweep data."""
    valid = [r for r in rows if np.isfinite(r["pde_error"])]
    if not valid:
        return

    # 1) pde_error vs kernel_beta, colored by r_target
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Group by (r_target, r_prime)
    groups = {}
    for r in valid:
        gk = (r["r_target"], r["r_prime"])
        groups.setdefault(gk, []).append(r)

    ax = axes[0]
    for (rt, rp), pts in sorted(groups.items()):
        kb = [p["kernel_beta"] for p in pts]
        pe = [p["pde_error"] for p in pts]
        ax.plot(kb, pe, ".-", label=f"r={rt:.2f}, r'={rp:.2f}", markersize=3, linewidth=0.8)
    ax.set_xlabel("kernel_beta")
    ax.set_ylabel("pde_error")
    ax.set_title("PDE error vs kernel_beta")
    if len(groups) <= 12:
        ax.legend(fontsize=6)

    # 2) pde_error vs post_prob tradeoff
    ax = axes[1]
    pe_all = [r["pde_error"] for r in valid]
    pp_all = [r["post_prob"] for r in valid]
    kb_all = [r["kernel_beta"] for r in valid]
    sc = ax.scatter(pp_all, pe_all, c=kb_all, cmap="viridis", s=8, alpha=0.7)
    plt.colorbar(sc, ax=ax, label="kernel_beta")
    ax.set_xlabel("post_prob")
    ax.set_ylabel("pde_error")
    ax.set_title("PDE error vs post-selection prob")

    # 3) Best pde_error at each (r_target, r_prime) — heatmap
    ax = axes[2]
    best_by_grid = {}
    for r in valid:
        gk = (r["r_target"], r["r_prime"])
        if gk not in best_by_grid or r["pde_error"] < best_by_grid[gk]["pde_error"]:
            best_by_grid[gk] = r

    rt_vals = sorted(set(r["r_target"] for r in valid))
    rp_vals = sorted(set(r["r_prime"] for r in valid))
    grid = np.full((len(rp_vals), len(rt_vals)), np.nan)
    for (rt, rp), best in best_by_grid.items():
        if rt in rt_vals and rp in rp_vals:
            grid[rp_vals.index(rp), rt_vals.index(rt)] = best["pde_error"]

    im = ax.imshow(grid, origin="lower", aspect="auto",
                   extent=[min(rt_vals), max(rt_vals), min(rp_vals), max(rp_vals)],
                   cmap="viridis_r")
    plt.colorbar(im, ax=ax, label="best pde_error")
    ax.set_xlabel("r_target")
    ax.set_ylabel("r_prime")
    ax.set_title("Best PDE error (over kernel_beta)")

    fig.tight_layout()
    fig.savefig(output_dir / "surrogate_landscape.png", dpi=180)
    plt.close(fig)
    print(f"Saved {output_dir / 'surrogate_landscape.png'}")


# ---------------------------------------------------------------------------
# Phase 3: fine optimization using surrogate
# ---------------------------------------------------------------------------
def optimize_surrogate(output_dir: Path, cfg: SurrogateConfig):
    """
    Bounded 1D optimization using the surrogate.
    For each precomputed (r_target, r_prime), optimize kernel_beta only.
    Then report global best across the grid.
    """
    if cfg.n_dim > hep.MAX_FOCK_LEVEL:
        raise ValueError(
            f"n_dim={cfg.n_dim} exceeds backend MAX_FOCK_LEVEL={hep.MAX_FOCK_LEVEL}."
        )
    fock_db = load_fock_components(output_dir, cfg)
    u_theory, norm_theory = _pde_reference(cfg)
    kb_min = float(min(cfg.kernel_beta_values))
    kb_max = float(max(cfg.kernel_beta_values))

    # Step 1: for each grid point, optimize kernel_beta (1D, very cheap)
    print("Phase 3a: optimizing kernel_beta at each (r_target, r_prime) grid point")
    grid_bests = []
    for key, entry in fock_db.items():
        r_target = entry["r_target"]
        r_prime = entry["r_prime"]
        fock_data = entry["fock_components"]

        def neg_obj_scalar(kb):
            kb = float(kb)
            if kb < kb_min or kb > kb_max:
                return 1e6 + abs(kb - np.clip(kb, kb_min, kb_max))
            m = reconstruct_metrics(fock_data, r_target, r_prime, kb, cfg, u_theory, norm_theory)
            pe = m["pde_error"]
            if not np.isfinite(pe):
                return 1e6
            return pe

        # Bounded 1D optimization: kernel_beta stays in configured domain.
        best_kb = None
        try:
            res = minimize_scalar(
                neg_obj_scalar,
                bounds=(kb_min, kb_max),
                method="bounded",
                options={"maxiter": int(cfg.optimize_maxiter), "xatol": 1e-4},
            )
            if np.isfinite(res.fun):
                best_kb = float(np.clip(res.x, kb_min, kb_max))
        except Exception:
            best_kb = None

        if best_kb is not None:
            m = reconstruct_metrics(fock_data, r_target, r_prime, best_kb, cfg, u_theory, norm_theory)
            grid_bests.append({
                "r_target": r_target,
                "r_prime": r_prime,
                "kernel_beta": best_kb,
                **m,
            })

    grid_bests.sort(key=lambda r: r.get("pde_error", np.inf))

    print(f"\nGrid-optimal results ({len(grid_bests)} points):")
    print(f"  {'r_target':>8s}  {'r_prime':>8s}  {'kb':>8s}  {'pde_err':>8s}  {'post_p':>8s}  {'fidelity':>8s}")
    for r in grid_bests[:10]:
        print(f"  {r['r_target']:8.4f}  {r['r_prime']:8.4f}  {r['kernel_beta']:8.4f}  "
              f"{r['pde_error']:8.5f}  {r['post_prob']:8.4f}  {r['fidelity']:8.5f}")

    # Save
    csv_path = output_dir / "surrogate_optimized.csv"
    if grid_bests:
        fieldnames = list(grid_bests[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(grid_bests)
        print(f"\nSaved {csv_path}")

    if grid_bests:
        best = grid_bests[0]
        print(f"\n=== BEST SURROGATE RESULT ===")
        print(f"  r_target    = {best['r_target']:.6f}")
        print(f"  r_prime     = {best['r_prime']:.6f}")
        print(f"  kernel_beta = {best['kernel_beta']:.6f}")
        print(f"  pde_error   = {best['pde_error']:.6f}")
        print(f"  post_prob   = {best['post_prob']:.6f}")
        print(f"  fidelity    = {best['fidelity']:.6f}")
        print(f"  purity      = {best['purity']:.6f}")
        print(f"  n_eff       = {best['n_eff']:.2f}")

    return grid_bests


# ---------------------------------------------------------------------------
# Validation: surrogate vs direct evaluator spot checks
# ---------------------------------------------------------------------------
def validate_surrogate(output_dir: Path, cfg: SurrogateConfig, n_samples: int = 8, seed: int = 7):
    """
    Spot-check surrogate reconstruction against direct circuit evaluation.
    """
    if cfg.n_dim > hep.MAX_FOCK_LEVEL:
        raise ValueError(
            f"n_dim={cfg.n_dim} exceeds backend MAX_FOCK_LEVEL={hep.MAX_FOCK_LEVEL}."
        )

    fock_db = load_fock_components(output_dir, cfg)
    entries = list(fock_db.values())
    if not entries:
        print("No precomputed entries found for validation.")
        return []

    rng = np.random.default_rng(seed)
    n_pick = min(max(1, int(n_samples)), len(entries))
    pick_idx = rng.choice(len(entries), size=n_pick, replace=False)
    kb_min = float(min(cfg.kernel_beta_values))
    kb_max = float(max(cfg.kernel_beta_values))
    u_theory, norm_theory = _pde_reference(cfg)

    ev = Evaluator()
    base = Settings(
        total_time=cfg.total_time,
        n_steps=cfg.n_steps,
        init_qubits=tuple(cfg.init_qubits),
        n_dim=cfg.n_dim,
        fock_expansion_cutoff=cfg.fock_expansion_cutoff,
        n_quad_points=cfg.n_quad_points,
    )

    def _abs_diff(a, b):
        if not (np.isfinite(a) and np.isfinite(b)):
            return np.nan
        return float(abs(float(a) - float(b)))

    rows = []
    print(f"Validating surrogate on {n_pick} random points (seed={seed})...")
    for i, idx in enumerate(pick_idx, 1):
        entry = entries[int(idx)]
        r_target = float(entry["r_target"])
        r_prime = float(entry["r_prime"])
        kb = float(rng.uniform(kb_min, kb_max))

        surrogate = reconstruct_metrics(
            entry["fock_components"], r_target, r_prime, kb, cfg, u_theory, norm_theory
        )
        direct = ev.evaluate(replace(base, r_target=r_target, r_prime=r_prime, kernel_beta=kb))

        row = {
            "r_target": r_target,
            "r_prime": r_prime,
            "kernel_beta": kb,
            "sur_pde_error": surrogate.get("pde_error", np.nan),
            "dir_pde_error": direct.get("pde_error", np.nan),
            "abs_diff_pde_error": _abs_diff(surrogate.get("pde_error", np.nan), direct.get("pde_error", np.nan)),
            "sur_post_prob": surrogate.get("post_prob", np.nan),
            "dir_post_prob": direct.get("post_prob", np.nan),
            "abs_diff_post_prob": _abs_diff(surrogate.get("post_prob", np.nan), direct.get("post_prob", np.nan)),
            "sur_fidelity": surrogate.get("fidelity", np.nan),
            "dir_fidelity": direct.get("fidelity", np.nan),
            "abs_diff_fidelity": _abs_diff(surrogate.get("fidelity", np.nan), direct.get("fidelity", np.nan)),
            "sur_purity": surrogate.get("purity", np.nan),
            "dir_purity": direct.get("purity", np.nan),
            "abs_diff_purity": _abs_diff(surrogate.get("purity", np.nan), direct.get("purity", np.nan)),
        }
        rows.append(row)
        print(
            f"  [{i:02d}/{n_pick}] r={r_target:.4f}, r'={r_prime:.4f}, kb={kb:.4f} "
            f"-> |Δpde|={row['abs_diff_pde_error']:.3e}, |Δpost|={row['abs_diff_post_prob']:.3e}"
        )

    csv_path = output_dir / "surrogate_validation.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {csv_path}")

    pde_diffs = np.array([r["abs_diff_pde_error"] for r in rows], dtype=float)
    post_diffs = np.array([r["abs_diff_post_prob"] for r in rows], dtype=float)
    finite_pde = pde_diffs[np.isfinite(pde_diffs)]
    finite_post = post_diffs[np.isfinite(post_diffs)]
    if finite_pde.size and finite_post.size:
        print(
            "Validation summary (surrogate vs direct): "
            f"mean|Δpde|={np.mean(finite_pde):.3e}, max|Δpde|={np.max(finite_pde):.3e}; "
            f"mean|Δpost|={np.mean(finite_post):.3e}, max|Δpost|={np.max(finite_post):.3e}"
        )
    else:
        print("Validation summary unavailable: no finite paired differences.")
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Surrogate model for CV-DV LCHS optimization")
    parser.add_argument("--output-dir", type=str, default="surrogate_data",
                        help="Directory for precomputed data and results")
    parser.add_argument("--precompute", action="store_true",
                        help="Phase 1: run circuits for Fock components on grid")
    parser.add_argument("--sweep", action="store_true",
                        help="Phase 2: sweep kernel_beta using precomputed data")
    parser.add_argument("--optimize", action="store_true",
                        help="Phase 3: optimize kernel_beta at each grid point")
    parser.add_argument("--validate-surrogate", action="store_true",
                        help="Spot-check surrogate reconstruction against direct circuit evaluation")
    parser.add_argument("--validate-samples", type=int, default=8,
                        help="Number of random points for --validate-surrogate")
    parser.add_argument("--validate-seed", type=int, default=7,
                        help="Random seed for --validate-surrogate point selection")
    parser.add_argument("--all", action="store_true",
                        help="Run all phases sequentially")

    # Grid configuration
    parser.add_argument("--r-target-grid", type=str, default=None,
                        help="Comma-separated r_target grid values")
    parser.add_argument("--r-prime-grid", type=str, default=None,
                        help="Comma-separated r_prime grid values")
    parser.add_argument("--kb-sweep", type=str, default=None,
                        help="Comma-separated kernel_beta sweep values")
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--n-dim", type=int, default=32)

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # Build config
    rt_vals = tuple(float(x) for x in args.r_target_grid.split(",")) if args.r_target_grid else None
    rp_vals = tuple(float(x) for x in args.r_prime_grid.split(",")) if args.r_prime_grid else None
    kb_vals = tuple(float(x) for x in args.kb_sweep.split(",")) if args.kb_sweep else None

    overrides = {}
    if rt_vals:
        overrides["r_target_values"] = rt_vals
    if rp_vals:
        overrides["r_prime_values"] = rp_vals
    if kb_vals:
        overrides["kernel_beta_values"] = kb_vals
    overrides["n_steps"] = args.n_steps
    overrides["n_dim"] = args.n_dim

    cfg = SurrogateConfig(**overrides)
    if cfg.n_dim > hep.MAX_FOCK_LEVEL:
        raise ValueError(
            f"n_dim={cfg.n_dim} exceeds backend MAX_FOCK_LEVEL={hep.MAX_FOCK_LEVEL}. "
            "Use --n-dim <= 32 or increase MAX_FOCK_LEVEL in heat_eq_postselect.py."
        )

    if args.all:
        args.precompute = True
        args.sweep = True
        args.optimize = True
        args.validate_surrogate = True

    if not (args.precompute or args.sweep or args.optimize or args.validate_surrogate):
        parser.print_help()
        return

    if args.precompute:
        precompute_fock_components(output_dir, cfg)

    if args.sweep:
        sweep_kernel_beta(output_dir, cfg)

    if args.optimize:
        optimize_surrogate(output_dir, cfg)

    if args.validate_surrogate:
        validate_surrogate(
            output_dir,
            cfg,
            n_samples=args.validate_samples,
            seed=args.validate_seed,
        )


if __name__ == "__main__":
    main()
