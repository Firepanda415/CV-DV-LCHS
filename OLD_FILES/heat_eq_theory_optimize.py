#!/usr/bin/env python3
"""
heat_eq_theory_optimize.py
==========================
Theory-driven parameter optimization for the CV-DV LCHS heat-equation solver.

Error-Budget Model
------------------
The total PDE-vector error decomposes as:

    ε_total  ≤  ε_trotter  +  ε_lchs  +  ε_trunc  +  ε_mixture

1. Trotter error (analytically bounded)
   --------------------------------------------------
   First-order Lie-Trotter with k terms H_j:
       ε_trotter ≤ (α·t)² / (2·n_steps) · C_comm
   where C_comm = Σ_{j<k} ||[H_j, H_k]||.

   For T = 2(I⊗I) − (I⊗X) − ½[(X⊗X) + (Y⊗Y)] the only non-zero
   commutator is  [−(I⊗X), −½(Y⊗Y)] = i(Y⊗Z), giving C_comm = 1.

   => ε_trotter ≤ (α·t)² / (2·n_steps)

   With α=t=1 and n_steps=100 this gives ε_trotter ≤ 0.005,
   which is small compared to the observed ~0.27.

2. Fock truncation error (exactly computable)
   --------------------------------------------------
       ε_trunc = Σ_{n : |C_n|² < cutoff} |C_n|²

3. Quadrature error (exponentially small)
   --------------------------------------------------
   Gauss-Hermite with N points converges as O(exp(−c·N)) for smooth
   integrands.  With N ≥ 200 the error is negligible (~1e-14).

4. Mixture decoherence error
   --------------------------------------------------
   The incoherent |C_n|² approximation discards off-diagonal coherences
   in the Fock basis.  The participation ratio (effective mode count)
       η = (Σ |C_n|²)² / Σ |C_n|⁴  =  n_eff
   quantifies the spread.  The lost coherence fraction is 1 − 1/η.
   Larger η → more decoherence → larger ε_mixture.

5. LCHS integral error (residual)
   --------------------------------------------------
   The remainder ε_lchs ≈ ε_total − ε_trotter − ε_trunc captures
   the spectral-coverage quality of the squeezed-kernel integral.
   It depends on γ = e^{−2r'} − e^{−2r} and kernel_beta.

Spectral analysis of T
-----------------------
T is the standard N=4 tridiagonal Laplacian.
Eigenvalues:  λ_k = 2 − 2 cos(kπ/5),  k = 1…4
    = {0.382, 1.382, 2.618, 3.618}
Spectral norm  ||T|| = 3.618
Condition number  κ = λ_max/λ_min = 9.472
||e^{−αtT} u(0)|| ≈ 0.422   (for α=t=1, u(0)=[0,1,0,0])

Optimization protocol
---------------------
Phase 0 : Spectral analysis + theoretical bounds (no simulation)
Phase 1 : Convergence of numerical params (n_steps, n_quad, cutoff)
Phase 2 : Theory-guided LHS search over (r_target, r_prime, kernel_beta)
Phase 3 : Multi-start Nelder-Mead local refinement
"""

import argparse
import csv
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize
from scipy.stats import qmc

import heat_eq_postselect as hep
from heat_eq_sensitivity_refine import Evaluator, Settings


# ============================================================
#  Phase 0 — Theory computations (no simulation)
# ============================================================

def spectral_analysis():
    """Eigenvalues, commutator norms, and derived bounds for T."""
    T = hep.dv_generator_matrix()
    eigvals = np.sort(np.real(np.linalg.eigvals(T)))
    lam_min, lam_max = float(eigvals[0]), float(eigvals[-1])
    kappa = lam_max / lam_min if lam_min > 0 else np.inf

    # Commutator analysis for Trotter bound
    I, X, Y, Z = hep.PAULI_I, hep.PAULI_X, hep.PAULI_Y, hep.PAULI_Z
    terms = [
        ("H_II", hep.lam_I * np.kron(I, I)),
        ("H_IX", hep.lam_X1 * np.kron(I, X)),
        ("H_XX", hep.lam_XX * np.kron(X, X)),
        ("H_YY", hep.lam_YY * np.kron(Y, Y)),
    ]
    comm_norms = {}
    c_comm = 0.0
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            Hi_name, Hi = terms[i]
            Hj_name, Hj = terms[j]
            comm = Hi @ Hj - Hj @ Hi
            norm = float(np.linalg.norm(comm, ord=2))
            key = f"[{Hi_name},{Hj_name}]"
            comm_norms[key] = norm
            c_comm += norm

    # ||u(t)|| for the default initial state
    u0 = hep.initial_dv_state((0, 1))
    u_theory = expm(-hep.alpha * 1.0 * T) @ u0
    norm_u = float(np.linalg.norm(u_theory))

    return {
        "T": T,
        "eigenvalues": eigvals,
        "lam_min": lam_min,
        "lam_max": lam_max,
        "kappa": kappa,
        "comm_norms": comm_norms,
        "c_comm": c_comm,
        "norm_u_theory": norm_u,
    }


def trotter_bound(alpha, t, n_steps, c_comm):
    """Operator-norm upper bound on first-order Trotter error."""
    return (alpha * t) ** 2 / (2.0 * n_steps) * c_comm


def trotter_bound_pde(alpha, t, n_steps, c_comm, norm_u):
    """Trotter contribution to relative PDE-vector error."""
    return trotter_bound(alpha, t, n_steps, c_comm) / norm_u


def fock_coefficient_analysis(r_target, r_prime, n_dim, kernel_beta, n_quad, cutoff):
    """Theory diagnostics computed purely from LCHS coefficients."""
    coeffs = hep.lchs_coefficients(
        r_target, r_prime, n_dim, kernel_beta=kernel_beta, n_quad_points=n_quad
    )
    weights = np.abs(coeffs) ** 2

    gamma = float(np.exp(-2 * r_prime) - np.exp(-2 * r_target))
    n_eff = float(1.0 / np.sum(weights ** 2))
    mean_n = float(np.sum(np.arange(len(weights)) * weights))
    var_n = float(np.sum((np.arange(len(weights)) - mean_n) ** 2 * weights))

    mask = weights >= cutoff
    trunc_error = float(1.0 - np.sum(weights[mask]))
    active_terms = int(np.sum(mask))
    coherence_frac = 1.0 - 1.0 / n_eff

    cum = np.cumsum(weights)
    n_99 = int(np.searchsorted(cum, 0.99)) + 1
    n_999 = int(np.searchsorted(cum, 0.999)) + 1

    return {
        "gamma": gamma,
        "n_eff": n_eff,
        "mean_n": mean_n,
        "std_n": float(np.sqrt(var_n)),
        "trunc_error": trunc_error,
        "active_terms": active_terms,
        "coherence_frac": coherence_frac,
        "n_99pct": n_99,
        "n_999pct": n_999,
    }


def print_phase0(spec, base):
    """Print the full Phase-0 theory report."""
    print("=" * 64)
    print("Phase 0: Spectral Analysis & Theoretical Bounds")
    print("=" * 64)

    print(f"\n  T eigenvalues: {np.array2string(spec['eigenvalues'], precision=4)}")
    print(f"  ||T|| = {spec['lam_max']:.4f}")
    print(f"  κ(T) = λ_max/λ_min = {spec['kappa']:.3f}")
    print(f"  ||e^{{-αtT}} u(0)|| = {spec['norm_u_theory']:.4f}  (α=t=1, u(0)=[0,1,0,0])")

    print("\n  Commutator norms (Trotter error budget):")
    for name, val in spec["comm_norms"].items():
        if val > 1e-12:
            print(f"    ||{name}|| = {val:.6f}")
        else:
            print(f"    ||{name}|| ≈ 0")
    print(f"  C_comm = Σ ||[H_j,H_k]|| = {spec['c_comm']:.6f}")

    # Trotter predictions
    print("\n  Trotter error predictions (ε_trotter ≤ (αt)²/(2n) · C_comm):")
    for ns in [10, 50, 100, 200, 500]:
        bnd = trotter_bound(hep.alpha, base.total_time, ns, spec["c_comm"])
        bnd_pde = trotter_bound_pde(
            hep.alpha, base.total_time, ns, spec["c_comm"], spec["norm_u_theory"]
        )
        print(f"    n_steps={ns:>4d}:  operator bound={bnd:.6f},  PDE-error bound={bnd_pde:.4f}")

    # Fock coefficient analysis at baseline
    fock = fock_coefficient_analysis(
        base.r_target, base.r_prime, base.n_dim, base.kernel_beta,
        base.n_quad_points, base.fock_expansion_cutoff,
    )
    print(f"\n  Fock coefficient diagnostics (baseline r={base.r_target}, r'={base.r_prime}, β_k={base.kernel_beta}):")
    print(f"    γ = e^{{-2r'}} - e^{{-2r}} = {fock['gamma']:.6f}")
    print(f"    n_eff (participation ratio) = {fock['n_eff']:.2f}")
    print(f"    ⟨n⟩ = {fock['mean_n']:.2f},  σ_n = {fock['std_n']:.2f}")
    print(f"    Coherence fraction = {fock['coherence_frac']:.4f}")
    print(f"    Truncation error (cutoff={base.fock_expansion_cutoff:.0e}): {fock['trunc_error']:.2e}")
    print(f"    Active Fock terms: {fock['active_terms']}")
    print(f"    Fock n for 99% / 99.9% weight: {fock['n_99pct']} / {fock['n_999pct']}")

    return fock


# ============================================================
#  Phase 1 — Convergence verification
# ============================================================

def convergence_sweep(ev, base, spec, output_dir):
    """Sweep numerical params to verify convergence, guided by theory."""
    print("\n" + "=" * 64)
    print("Phase 1: Convergence Verification")
    print("=" * 64)
    results = {}

    # ---- n_steps ----
    n_steps_vals = np.array([10, 20, 40, 70, 100, 150, 200])
    print("\n  1a. n_steps sweep (Trotter convergence):")
    rows_ns = []
    for ns in n_steps_vals:
        cfg = replace(base, n_steps=int(ns))
        m = ev.evaluate(cfg)
        bnd = trotter_bound(hep.alpha, base.total_time, int(ns), spec["c_comm"])
        bnd_pde = trotter_bound_pde(
            hep.alpha, base.total_time, int(ns), spec["c_comm"], spec["norm_u_theory"]
        )
        row = {
            "n_steps": int(ns),
            "pde_error": m["pde_error"],
            "post_prob": m["post_prob"],
            "purity": m["purity"],
            "trotter_bound_op": bnd,
            "trotter_bound_pde": bnd_pde,
        }
        rows_ns.append(row)
        pde = f"{m['pde_error']:.5f}" if np.isfinite(m["pde_error"]) else "N/A"
        print(
            f"    n_steps={ns:>4d}  pde={pde}  "
            f"trotter_bnd={bnd_pde:.5f}  post={m['post_prob']:.4e}"
        )
    results["n_steps"] = rows_ns
    _write_csv(output_dir / "phase1_n_steps.csv", rows_ns)
    _plot_convergence_trotter(
        output_dir / "phase1_n_steps.png",
        [r["n_steps"] for r in rows_ns],
        [r["pde_error"] for r in rows_ns],
        [r["trotter_bound_pde"] for r in rows_ns],
    )

    # ---- n_quad_points ----
    nq_vals = np.array([60, 100, 140, 180, 220, 300, 400])
    print("\n  1b. n_quad_points sweep (quadrature convergence):")
    rows_nq = []
    for nq in nq_vals:
        cfg = replace(base, n_quad_points=int(nq))
        m = ev.evaluate(cfg)
        row = {
            "n_quad_points": int(nq),
            "pde_error": m["pde_error"],
            "post_prob": m["post_prob"],
        }
        rows_nq.append(row)
        pde = f"{m['pde_error']:.5f}" if np.isfinite(m["pde_error"]) else "N/A"
        print(f"    n_quad={nq:>4d}  pde={pde}  post={m['post_prob']:.4e}")
    results["n_quad"] = rows_nq
    _write_csv(output_dir / "phase1_n_quad.csv", rows_nq)

    # ---- fock_cutoff ----
    cut_vals = np.array([1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10, 1e-12])
    print("\n  1c. fock_cutoff sweep (truncation convergence):")
    rows_cut = []
    for cut in cut_vals:
        cfg = replace(base, fock_expansion_cutoff=float(cut))
        fock = fock_coefficient_analysis(
            cfg.r_target, cfg.r_prime, cfg.n_dim, cfg.kernel_beta,
            cfg.n_quad_points, float(cut),
        )
        m = ev.evaluate(cfg)
        row = {
            "cutoff": float(cut),
            "pde_error": m["pde_error"],
            "post_prob": m["post_prob"],
            "trunc_error": fock["trunc_error"],
            "active_terms": fock["active_terms"],
        }
        rows_cut.append(row)
        pde = f"{m['pde_error']:.5f}" if np.isfinite(m["pde_error"]) else "N/A"
        print(
            f"    cutoff={cut:.0e}  pde={pde}  "
            f"trunc_err={fock['trunc_error']:.2e}  terms={fock['active_terms']}"
        )
    results["cutoff"] = rows_cut
    _write_csv(output_dir / "phase1_cutoff.csv", rows_cut)

    # Summary: identify converged values
    print("\n  Phase 1 summary:")
    converged_ns = _find_converged(rows_ns, "n_steps", "pde_error", rtol=0.01)
    converged_nq = _find_converged(rows_nq, "n_quad_points", "pde_error", rtol=0.005)
    converged_cut = _find_converged_decreasing(rows_cut, "cutoff", "pde_error", rtol=0.005)
    print(f"    Converged n_steps ≥ {converged_ns} (within 1% of asymptote)")
    print(f"    Converged n_quad ≥ {converged_nq} (within 0.5%)")
    print(f"    Converged cutoff ≤ {converged_cut:.0e} (within 0.5%)")

    return results


def _find_converged(rows, param_key, metric_key, rtol=0.01):
    """Find smallest param value where metric is within rtol of the last (finest) value."""
    vals = [r[metric_key] for r in rows if np.isfinite(r[metric_key])]
    if not vals:
        return rows[-1][param_key]
    ref = vals[-1]
    if np.isclose(ref, 0):
        return rows[0][param_key]
    for r in rows:
        v = r[metric_key]
        if np.isfinite(v) and abs(v - ref) / abs(ref) <= rtol:
            return r[param_key]
    return rows[-1][param_key]


def _find_converged_decreasing(rows, param_key, metric_key, rtol=0.005):
    """For decreasing param (like cutoff), find largest param value that's converged."""
    vals = [r[metric_key] for r in rows if np.isfinite(r[metric_key])]
    if not vals:
        return rows[0][param_key]
    ref = vals[-1]
    if np.isclose(ref, 0):
        return rows[0][param_key]
    for r in rows:
        v = r[metric_key]
        if np.isfinite(v) and abs(v - ref) / abs(ref) <= rtol:
            return r[param_key]
    return rows[-1][param_key]


# ============================================================
#  Phase 2 — Theory-guided LCHS parameter search
# ============================================================

@dataclass(frozen=True)
class SearchBounds:
    # Shifted toward empirically good PDE-error regions from refine optimizer logs.
    r_target_lo: float = 0.25
    r_target_hi: float = 1.2
    r_prime_lo: float = 0.05
    r_prime_hi: float = 0.45
    kb_lo: float = 0.0
    kb_hi: float = 1.0
    min_gap: float = 0.02


_BOUNDS = SearchBounds()


def _in_bounds(theta, bnd: SearchBounds):
    r, rp, kb = [float(v) for v in theta]
    return (
        bnd.r_target_lo <= r <= bnd.r_target_hi
        and bnd.r_prime_lo <= rp <= bnd.r_prime_hi
        and bnd.kb_lo <= kb <= bnd.kb_hi
        and rp < r - bnd.min_gap
    )


def _eval_point(ev, base, spec, theta, min_post_prob):
    """Evaluate parameter point with full theory diagnostics."""
    r_target, r_prime, kernel_beta = [float(v) for v in theta]

    if not _in_bounds(theta, _BOUNDS):
        return _nan_row(r_target, r_prime, kernel_beta)

    # Theory quantities (cheap, no simulation)
    fock = fock_coefficient_analysis(
        r_target, r_prime, base.n_dim, kernel_beta,
        base.n_quad_points, base.fock_expansion_cutoff,
    )

    # Simulation (expensive)
    cfg = replace(base, r_target=r_target, r_prime=r_prime, kernel_beta=kernel_beta)
    m = ev.evaluate(cfg)

    pde = float(m.get("pde_error", np.nan))
    post = float(m.get("post_prob", np.nan))
    purity = float(m.get("purity", np.nan))
    fidelity = float(m.get("fidelity", np.nan))

    trotter_bnd = trotter_bound_pde(
        hep.alpha, base.total_time, base.n_steps, spec["c_comm"], spec["norm_u_theory"]
    )

    # Error budget decomposition
    if np.isfinite(pde):
        eps_trunc = fock["trunc_error"]
        eps_trotter = min(trotter_bnd, pde)
        eps_lchs_resid = max(0.0, pde - eps_trunc - eps_trotter)
    else:
        eps_trunc = fock["trunc_error"]
        eps_trotter = trotter_bnd
        eps_lchs_resid = np.nan

    feasible = int(np.isfinite(post) and post >= min_post_prob)

    return {
        "r_target": r_target,
        "r_prime": r_prime,
        "kernel_beta": kernel_beta,
        "pde_error": pde,
        "post_prob": post,
        "purity": purity,
        "fidelity": fidelity,
        # Theory diagnostics
        "gamma": fock["gamma"],
        "n_eff": fock["n_eff"],
        "mean_n": fock["mean_n"],
        "coherence_frac": fock["coherence_frac"],
        # Error budget
        "eps_trunc": eps_trunc,
        "eps_trotter_bnd": eps_trotter,
        "eps_lchs_resid": eps_lchs_resid,
        "active_terms": fock["active_terms"],
        "is_feasible": feasible,
    }


def _nan_row(r_target, r_prime, kernel_beta):
    return {
        "r_target": r_target, "r_prime": r_prime, "kernel_beta": kernel_beta,
        "pde_error": np.nan, "post_prob": np.nan, "purity": np.nan, "fidelity": np.nan,
        "gamma": np.nan, "n_eff": np.nan, "mean_n": np.nan, "coherence_frac": np.nan,
        "eps_trunc": np.nan, "eps_trotter_bnd": np.nan, "eps_lchs_resid": np.nan,
        "active_terms": 0, "is_feasible": 0,
    }


def _objective(row, min_post_prob, post_penalty=100.0):
    """Scalarized objective: minimize PDE error with feasibility penalty."""
    pde = row.get("pde_error", np.nan)
    post = row.get("post_prob", np.nan)
    if not np.isfinite(pde):
        return float("inf")
    shortfall = max(0.0, min_post_prob - post) / min_post_prob if np.isfinite(post) else 1.0
    return float(pde) + post_penalty * shortfall - 1e-3 * (post if np.isfinite(post) else 0)


def _rank_key(row, min_post_prob):
    pde = row.get("pde_error", np.nan)
    post = row.get("post_prob", np.nan)
    pde_k = float(pde) if np.isfinite(pde) else float("inf")
    post_v = float(post) if np.isfinite(post) else -np.inf
    infeasible = 0 if post_v >= min_post_prob else 1
    return (infeasible, pde_k, -post_v if np.isfinite(post_v) else float("inf"))


def global_search(ev, base, spec, bnd, n_samples, min_post_prob, seed, output_dir, resume=False):
    """Latin-Hypercube global search with theory diagnostics."""
    global_csv = output_dir / "phase2_global.csv"
    if resume and global_csv.exists():
        rows = _read_csv(global_csv)
        print(f"\n  Loaded {len(rows)} existing global results from {global_csv}")
        return rows

    print(f"\n  Generating {n_samples} LHS samples in "
          f"r∈[{bnd.r_target_lo},{bnd.r_target_hi}], "
          f"r'∈[{bnd.r_prime_lo},{bnd.r_prime_hi}], "
          f"β_k∈[{bnd.kb_lo},{bnd.kb_hi}]")

    sampler = qmc.LatinHypercube(d=3, seed=seed)
    u = sampler.random(n=n_samples * 3)
    r_target = bnd.r_target_lo + u[:, 0] * (bnd.r_target_hi - bnd.r_target_lo)
    r_prime = bnd.r_prime_lo + u[:, 1] * (bnd.r_prime_hi - bnd.r_prime_lo)
    kb = bnd.kb_lo + u[:, 2] * (bnd.kb_hi - bnd.kb_lo)
    all_pts = np.column_stack([r_target, r_prime, kb])
    valid = [p for p in all_pts if _in_bounds(p, bnd)][:n_samples]

    rng = np.random.default_rng(seed + 42)
    while len(valid) < n_samples:
        p = np.array([
            rng.uniform(bnd.r_target_lo, bnd.r_target_hi),
            rng.uniform(bnd.r_prime_lo, bnd.r_prime_hi),
            rng.uniform(bnd.kb_lo, bnd.kb_hi),
        ])
        if _in_bounds(p, bnd):
            valid.append(p)

    rows = []
    for i, theta in enumerate(valid, 1):
        row = _eval_point(ev, base, spec, theta, min_post_prob)
        row["stage"] = "global"
        rows.append(row)
        pde_s = f"{row['pde_error']:.4f}" if np.isfinite(row["pde_error"]) else "  N/A"
        print(
            f"  [{i:03d}/{n_samples}] "
            f"r={row['r_target']:.3f} r'={row['r_prime']:.3f} β_k={row['kernel_beta']:.3f}  "
            f"pde={pde_s}  post={row['post_prob']:.3e}  "
            f"γ={row['gamma']:.3f}  n_eff={row['n_eff']:.1f}"
        )

    _write_csv(global_csv, rows)
    return rows


def local_refinement(ev, base, spec, starts, bnd, min_post_prob, maxiter, output_dir,
                     resume=False, progress_every=8, checkpoint_every=10):
    """Multi-start Nelder-Mead from best global points."""
    local_csv = output_dir / "phase3_local.csv"
    all_evals = _read_csv(local_csv) if (resume and local_csv.exists()) else []
    bests = []

    completed_starts = set()
    for r in all_evals:
        if r.get("stage") == "local_best" and np.isfinite(r.get("start_idx", np.nan)):
            completed_starts.add(int(r["start_idx"]))
    if completed_starts:
        print(f"  Resuming: skipping completed starts {sorted(completed_starts)}")

    for i, s in enumerate(starts, 1):
        if i in completed_starts:
            continue
        x0 = np.array([s["r_target"], s["r_prime"], s["kernel_beta"]])
        eval_count = 0
        best_obj = np.inf

        def obj(x):
            nonlocal eval_count, best_obj
            if not _in_bounds(x, bnd):
                return 10.0
            row = _eval_point(ev, base, spec, x, min_post_prob)
            row.update({"stage": "local", "start_idx": i})
            all_evals.append(row)
            eval_count += 1
            cur_obj = _objective(row, min_post_prob)
            best_obj = min(best_obj, cur_obj)
            if eval_count % progress_every == 0:
                pde_s = f"{row['pde_error']:.5f}" if np.isfinite(row["pde_error"]) else "N/A"
                print(
                    f"    eval {eval_count:03d}: r={row['r_target']:.4f} r'={row['r_prime']:.4f} "
                    f"β_k={row['kernel_beta']:.4f}  pde={pde_s}  post={row['post_prob']:.3e}  "
                    f"bestJ={best_obj:.5f}"
                )
            if eval_count % checkpoint_every == 0:
                _write_csv(local_csv, all_evals)
            return cur_obj

        pde0 = f"{s['pde_error']:.4f}" if np.isfinite(s["pde_error"]) else "N/A"
        print(
            f"\n  Start {i}/{len(starts)}: "
            f"r={x0[0]:.4f} r'={x0[1]:.4f} β_k={x0[2]:.4f}  pde0={pde0}"
        )
        result = minimize(
            obj, x0, method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 0.003, "fatol": 1e-5, "adaptive": True},
        )

        best = _eval_point(ev, base, spec, result.x, min_post_prob)
        best.update({"stage": "local_best", "start_idx": i, "nfev": result.nfev})
        all_evals.append(best)
        bests.append(best)
        _write_csv(local_csv, all_evals)

        pde_b = f"{best['pde_error']:.4f}" if np.isfinite(best["pde_error"]) else "N/A"
        print(
            f"    → r={best['r_target']:.4f} r'={best['r_prime']:.4f} "
            f"β_k={best['kernel_beta']:.4f}  pde={pde_b}  "
            f"post={best['post_prob']:.3e}  ({result.nfev} evals)"
        )

    return bests, all_evals


# ============================================================
#  I/O and plotting helpers
# ============================================================

def _write_csv(path, rows):
    if not rows:
        return
    keys = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _read_csv(path):
    path = Path(path)
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


def _plot_convergence_trotter(path, x_steps, pde_errors, trotter_bounds):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x_steps, pde_errors, "o-", color="tab:blue", label="Measured PDE error")
    ax.plot(x_steps, trotter_bounds, "s--", color="tab:red", alpha=0.7, label="Trotter bound (PDE-scaled)")
    ax.set_xlabel("n_steps")
    ax.set_ylabel("Relative PDE-vector error")
    ax.set_title("Trotter convergence: theory bound vs measured error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_error_budget(path, rows):
    valid = sorted(
        [r for r in rows if np.isfinite(r.get("pde_error", np.nan))],
        key=lambda r: r["pde_error"],
    )[:20]
    if not valid:
        return

    labels = [f"r={r['r_target']:.2f}\nr'={r['r_prime']:.2f}" for r in valid]
    pde = np.array([r["pde_error"] for r in valid])
    trotter = np.array([r["eps_trotter_bnd"] for r in valid])
    trunc = np.array([r["eps_trunc"] for r in valid])
    lchs = np.array([r.get("eps_lchs_resid", 0) for r in valid])

    x = np.arange(len(valid))
    w = 0.7
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, trotter, w, label="ε_trotter (bound)", color="tab:red", alpha=0.7)
    ax.bar(x, trunc, w, bottom=trotter, label="ε_trunc", color="tab:green", alpha=0.7)
    ax.bar(x, lchs, w, bottom=trotter + trunc, label="ε_lchs (residual)", color="tab:orange", alpha=0.7)
    ax.plot(x, pde, "kD", ms=5, label="Total PDE error")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Error")
    ax.set_title("Error budget decomposition (top candidates by PDE error)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_landscape(path, rows, color_key="pde_error"):
    valid = [r for r in rows if np.isfinite(r.get(color_key, np.nan))]
    if not valid:
        return
    x = np.array([r["r_target"] for r in valid])
    y = np.array([r["r_prime"] for r in valid])
    z = np.array([r[color_key] for r in valid])
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(x, y, c=z, cmap="viridis", s=30, alpha=0.8)
    fig.colorbar(sc, ax=ax, label=color_key)
    ax.set_xlabel("r_target")
    ax.set_ylabel("r_prime")
    ax.set_title(f"Parameter landscape ({color_key})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_theory_correlations(path, rows):
    valid = [r for r in rows if np.isfinite(r.get("pde_error", np.nan))]
    if len(valid) < 3:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    panels = [
        (axes[0, 0], "gamma", "γ = e^{-2r'} - e^{-2r}"),
        (axes[0, 1], "n_eff", "n_eff (participation ratio)"),
        (axes[0, 2], "coherence_frac", "Coherence fraction (1-1/η)"),
        (axes[1, 0], "post_prob", "Post-selection probability"),
        (axes[1, 1], "mean_n", "Mean Fock number ⟨n⟩"),
        (axes[1, 2], "eps_lchs_resid", "LCHS residual error (est.)"),
    ]
    for ax, key, label in panels:
        x = np.array([r.get(key, np.nan) for r in valid], dtype=float)
        y = np.array([r["pde_error"] for r in valid], dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        if np.sum(mask) < 2:
            ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
            continue
        ax.scatter(x[mask], y[mask], s=16, alpha=0.55)
        ax.set_xlabel(label)
        ax.set_ylabel("PDE error")
        ax.grid(True, alpha=0.3)
        corr = np.corrcoef(x[mask], y[mask])[0, 1]
        ax.set_title(f"ρ = {corr:.3f}")

    fig.suptitle("Theory diagnostics vs PDE error", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_tradeoff(path, rows):
    valid = [
        r for r in rows
        if np.isfinite(r.get("pde_error", np.nan))
        and float(r.get("post_prob", 0)) > 0
    ]
    if not valid:
        return
    x = np.array([r["post_prob"] for r in valid])
    y = np.array([r["pde_error"] for r in valid])

    # Pareto front
    order = np.argsort(y)
    px, py = [], []
    best_post = -np.inf
    for idx in order:
        if x[idx] > best_post + 1e-15:
            px.append(x[idx])
            py.append(y[idx])
            best_post = x[idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, s=20, alpha=0.5, c="tab:blue", label="Evaluated points")
    if px:
        pa_x, pa_y = np.array(px), np.array(py)
        o = np.argsort(pa_x)
        ax.plot(pa_x[o], pa_y[o], "-o", color="tab:red", lw=1.5, ms=5, label="Pareto front")
    ax.set_xscale("log")
    ax.set_xlabel("Post-selection probability")
    ax.set_ylabel("Relative PDE-vector error")
    ax.set_title("PDE error vs post_prob trade-off")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ============================================================
#  Main driver
# ============================================================

def run(output_dir, n_global, n_starts, local_maxiter, min_post_prob, seed,
        skip_convergence, resume=False, progress_every=8, checkpoint_every=10):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = Settings()
    ev = Evaluator()
    spec = spectral_analysis()

    # Phase 0
    fock_base = print_phase0(spec, base)

    # Phase 1
    if not skip_convergence:
        convergence_sweep(ev, base, spec, output_dir)
    else:
        print("\n(Skipping Phase 1 — use without --skip-convergence to run)")

    # Phase 2
    print("\n" + "=" * 64)
    print(f"Phase 2: Theory-Guided Global Search ({n_global} LHS samples)")
    print("=" * 64)
    bnd = _BOUNDS
    global_rows = global_search(
        ev, base, spec, bnd, n_global, min_post_prob, seed, output_dir, resume=resume
    )

    ranked = [r for r in global_rows if np.isfinite(r.get("pde_error", np.nan))]
    ranked.sort(key=lambda r: _rank_key(r, min_post_prob))
    if not ranked:
        raise RuntimeError("No valid global candidates. Relax bounds/constraints and retry.")
    starts = ranked[:n_starts]

    # Phase 3
    print("\n" + "=" * 64)
    print(f"Phase 3: Local Refinement ({n_starts} starts, maxiter={local_maxiter})")
    print("=" * 64)
    local_bests, local_all = local_refinement(
        ev, base, spec, starts, bnd, min_post_prob, local_maxiter, output_dir,
        resume=resume, progress_every=progress_every, checkpoint_every=checkpoint_every,
    )

    # Final report
    print("\n" + "=" * 64)
    print("Final Report")
    print("=" * 64)

    all_rows = global_rows + local_all
    all_ranked = [r for r in all_rows if np.isfinite(r.get("pde_error", np.nan))]
    all_ranked.sort(key=lambda r: _rank_key(r, min_post_prob))
    top = all_ranked[:20]

    _write_csv(output_dir / "final_top.csv", top)
    _write_csv(output_dir / "final_all.csv", all_rows)

    # Plots
    _plot_landscape(output_dir / "landscape_pde.png", all_rows, "pde_error")
    _plot_landscape(output_dir / "landscape_gamma.png", all_rows, "gamma")
    _plot_landscape(output_dir / "landscape_n_eff.png", all_rows, "n_eff")
    _plot_theory_correlations(output_dir / "theory_correlations.png", all_rows)
    _plot_tradeoff(output_dir / "tradeoff_pde_vs_post.png", all_rows)
    _plot_error_budget(output_dir / "error_budget.png", all_rows)

    feasible_count = sum(1 for r in all_ranked if r.get("is_feasible", 0))
    print(f"\n  Total evaluations: {len(all_rows)}")
    print(f"  Feasible (post_prob ≥ {min_post_prob:.0e}): {feasible_count}/{len(all_ranked)}")

    print("\n  Top 10 candidates (PDE-priority):")
    hdr = (
        f"  {'#':>2}  {'r_tgt':>6} {'r_prm':>6} {'β_k':>5}  "
        f"{'pde':>7} {'post':>9}  "
        f"{'γ':>6} {'n_eff':>5} {'coh%':>5} "
        f"{'ε_trot':>6} {'ε_trn':>7} {'ε_lchs':>6} {'F':>1}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for i, r in enumerate(top[:10], 1):
        pde_s = f"{r['pde_error']:.4f}" if np.isfinite(r["pde_error"]) else "  N/A"
        f_s = "Y" if r.get("is_feasible", 0) else "N"
        trn_s = f"{r['eps_trunc']:.1e}" if np.isfinite(r.get("eps_trunc", np.nan)) else "  N/A"
        lchs_s = f"{r.get('eps_lchs_resid', np.nan):.4f}" if np.isfinite(r.get("eps_lchs_resid", np.nan)) else "  N/A"
        print(
            f"  {i:2d}. {r['r_target']:6.3f} {r['r_prime']:6.3f} {r['kernel_beta']:5.3f}  "
            f"{pde_s:>7} {r['post_prob']:9.3e}  "
            f"{r['gamma']:6.3f} {r['n_eff']:5.1f} {r['coherence_frac']*100:5.1f} "
            f"{r['eps_trotter_bnd']:6.4f} {trn_s:>7} {lchs_s:>6} {f_s}"
        )

    if top:
        best = top[0]
        print(f"\n  Best candidate — error budget decomposition:")
        print(f"    Parameters:  r_target={best['r_target']:.4f}, r_prime={best['r_prime']:.4f}, kernel_beta={best['kernel_beta']:.4f}")
        print(f"    Total PDE error:      {best['pde_error']:.6f}")
        print(f"    ├─ ε_trotter (bound): {best['eps_trotter_bnd']:.6f}")
        print(f"    ├─ ε_trunc (exact):   {best['eps_trunc']:.2e}")
        print(f"    └─ ε_lchs (residual): {best.get('eps_lchs_resid', np.nan):.6f}")
        print(f"    Theory diagnostics:")
        print(f"      γ = {best['gamma']:.6f}")
        print(f"      n_eff = {best['n_eff']:.2f}")
        print(f"      Coherence frac = {best['coherence_frac']:.4f}")
        print(f"      Fidelity = {best['fidelity']:.4f}")
        print(f"      Post-prob = {best['post_prob']:.6e}")
        print(f"      Purity = {best['purity']:.4f}")

    print(f"\n  Outputs saved to: {output_dir.resolve()}")


def main():
    p = argparse.ArgumentParser(
        description="Theory-guided parameter optimization for CV-DV LCHS heat equation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Error budget model:
              ε_total ≤ ε_trotter + ε_lchs + ε_trunc + ε_mixture
              ε_trotter ≤ (αt)²/(2·n_steps)         [C_comm = 1 for this T]
              ε_trunc = Σ_{n<cutoff} |C_n|²          [exactly computable]
              ε_mixture ~ 1 - 1/n_eff                [coherence loss indicator]
              ε_lchs = ε_total - ε_trotter - ε_trunc  [residual]

            Optimization: Phase 0 (theory) → Phase 1 (convergence) →
              Phase 2 (global LHS search) → Phase 3 (local Nelder-Mead)
        """),
    )
    p.add_argument("--output-dir", default="theory_opt_results")
    p.add_argument("--n-global", type=int, default=30, help="LHS global samples (Phase 2)")
    p.add_argument("--n-starts", type=int, default=4, help="Local refinement starts (Phase 3)")
    p.add_argument("--local-maxiter", type=int, default=30, help="Max iter per local optimizer")
    p.add_argument("--min-post-prob", type=float, default=1e-3, help="Feasibility threshold")
    p.add_argument("--seed", type=int, default=7, help="Random seed")
    p.add_argument("--skip-convergence", action="store_true", help="Skip Phase 1 convergence sweeps")
    p.add_argument("--resume", action="store_true", help="Resume from existing CSVs in output dir")
    p.add_argument("--progress-every", type=int, default=8, help="Print progress every N local evals")
    p.add_argument("--checkpoint-every", type=int, default=10, help="Checkpoint local CSV every N evals")
    args = p.parse_args()

    run(
        output_dir=args.output_dir,
        n_global=args.n_global,
        n_starts=args.n_starts,
        local_maxiter=args.local_maxiter,
        min_post_prob=args.min_post_prob,
        seed=args.seed,
        skip_convergence=args.skip_convergence,
        resume=args.resume,
        progress_every=max(1, args.progress_every),
        checkpoint_every=max(1, args.checkpoint_every),
    )


if __name__ == "__main__":
    main()
