#!/usr/bin/env python3
"""
1D heat equation solver via CV-DV LCHS (bosonic-qiskit).
Paper-faithful implementation derived directly from:
  - Hybrid_CV_DV_LCHS.pdf (Theorem 1, Eq. 7-9, Section 3.5.2)
  - Near-optimal kernel paper (kernel g_beta, Eq. 32-33)

Key physics:
  D(alpha) = exp(alpha a† - alpha* a)
  For alpha = -i*lambda/2:  D(-i*lam/2) = exp(-i*lam*x_hat)
  CD(-i*lam/2) = exp(-i*lam*x_hat ⊗ sigma_z)
  S(r) = exp(r/2 (a†² - a²))
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from bosonic_qiskit import CVCircuit, QumodeRegister
from bosonic_qiskit import util as cv_util
from numpy.polynomial.hermite import hermgauss
from qiskit import QuantumRegister
from scipy.linalg import expm
from scipy.special import eval_hermite, factorial

# ---------------------------------------------------------------------------
# Section A: Math utilities
# ---------------------------------------------------------------------------

PAULI_I = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_MAP = {"I": PAULI_I, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}


def heat_matrix(alpha: float = 1.0, h: float = 1.0) -> np.ndarray:
    """4x4 tridiagonal Laplacian: A = (alpha/h²) T."""
    T = np.array(
        [[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]],
        dtype=float,
    )
    return alpha * T / (h ** 2)


def pauli_decomposition(alpha: float = 1.0, h: float = 1.0) -> List[Tuple[float, str]]:
    """
    A = (alpha/h²)(2·II - IX - ½XX - ½YY).
    Returns [(coefficient, pauli_label), ...].
    """
    s = alpha / (h ** 2)
    return [
        (2.0 * s, "II"),
        (-1.0 * s, "IX"),
        (-0.5 * s, "XX"),
        (-0.5 * s, "YY"),
    ]


def commutator_norm_bound(alpha: float = 1.0, h: float = 1.0) -> float:
    """Sum of ||[H_i, H_j]|| for Trotter error bound."""
    terms = pauli_decomposition(alpha, h)
    matrices = []
    for coeff, label in terms:
        P = np.eye(1, dtype=complex)
        for ch in label:
            P = np.kron(P, PAULI_MAP[ch])
        matrices.append(coeff * P)
    total = 0.0
    for i in range(len(matrices)):
        for j in range(i + 1, len(matrices)):
            comm = matrices[i] @ matrices[j] - matrices[j] @ matrices[i]
            total += float(np.linalg.norm(comm, ord=2))
    return total


def kernel_g_beta(k: np.ndarray, beta: float) -> np.ndarray:
    """
    Improved LCHS kernel: g_beta(k) = exp(-(1+ik)^beta) / (C_beta * (1-ik))
    C_beta = 2*pi*exp(-2^beta).
    """
    c_beta = 2.0 * np.pi * np.exp(-(2.0 ** beta))
    return np.exp(-((1.0 + 1j * k) ** beta)) / (c_beta * (1.0 - 1j * k))


def lchs_gamma(r: float, r_prime: float) -> float:
    """gamma = exp(-2r') - exp(-2r). Positive when r' < r."""
    return float(np.exp(-2.0 * r_prime) - np.exp(-2.0 * r))


def lchs_coefficients(
    r: float,
    r_prime: float,
    beta: float,
    n_max: int,
    n_quad: int = 220,
) -> np.ndarray:
    """
    Compute LCHS expansion coefficients C_n for the squeezed-Fock basis.

    C_n ∝ ∫ g_beta(k) exp(-gamma k²) H_n(k/e^{r'}) / sqrt(√π e^{r'} 2^n n!) dk

    Uses Gauss-Hermite quadrature with substitution k = root * √2 * e^{r'}.
    Returns unit-normalized coefficient vector.
    """
    gamma = lchs_gamma(r, r_prime)
    if gamma <= 0:
        raise ValueError(f"Need r' < r for gamma > 0 (got gamma={gamma:.4f}).")
    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must be in (0,1), got {beta}.")

    width = np.exp(r_prime)
    scale = np.sqrt(2.0) * width
    roots, weights = hermgauss(n_quad)
    k_pts = roots * scale

    envelope = np.exp(-gamma * k_pts ** 2) * kernel_g_beta(k_pts, beta)
    sqrt_pi = np.sqrt(np.pi)
    basis_pre = 1.0 / np.sqrt(sqrt_pi * width)

    coeffs = np.zeros(n_max, dtype=complex)
    for n in range(n_max):
        fock_norm = 1.0 / np.sqrt(float(2 ** n) * float(factorial(n)))
        herm = eval_hermite(n, k_pts / width)
        integrand = envelope * basis_pre * fock_norm * herm
        coeffs[n] = np.sum(weights * integrand) * scale

    norm = float(np.linalg.norm(coeffs))
    if np.isclose(norm, 0.0):
        raise RuntimeError("Coefficient vector has near-zero norm.")
    return coeffs / norm


def estimate_tail_mass(
    r: float, r_prime: float, beta: float, n_max: int, n_ref: int = 96
) -> float:
    """Estimate sum_{n>=n_max} |C_n|² using a larger reference expansion."""
    if n_ref <= n_max:
        return 0.0
    ref = lchs_coefficients(r, r_prime, beta, n_ref, n_quad=260)
    return max(0.0, 1.0 - float(np.sum(np.abs(ref[:n_max]) ** 2)))


# ---------------------------------------------------------------------------
# Section B: Circuit builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LCHSConfig:
    # PDE parameters
    alpha: float = 1.0
    h: float = 1.0
    total_time: float = 1.0
    init_q0: int = 0
    init_q1: int = 1
    # LCHS parameters
    r: float = 1.2
    r_prime: float = 0.3
    beta: float = 0.75
    # Numerics
    n_steps: int = 100
    n_fock: int = 32
    n_coeff: int = 32


def _validate(cfg: LCHSConfig) -> None:
    if cfg.n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if cfg.n_fock <= 0 or (cfg.n_fock & (cfg.n_fock - 1)) != 0:
        raise ValueError("n_fock must be a positive power of two.")
    if cfg.n_coeff > cfg.n_fock:
        raise ValueError("n_coeff must not exceed n_fock.")
    if cfg.r_prime >= cfg.r:
        raise ValueError("Need r_prime < r.")


def _apply_trotter_term(
    qc: CVCircuit,
    mode,
    qbr: QuantumRegister,
    lam: float,
    label: str,
) -> None:
    """
    Apply exp(-i lam x_hat ⊗ P_label).

    Displacement parameter: alpha = -i * lam / 2
    because D(-i*lam/2) = exp(-i*lam*x_hat).
    """
    alpha = -1j * lam / 2.0

    if label == "II":
        qc.cv_d(alpha, mode)

    elif label == "IX":
        qc.h(qbr[1])
        qc.cv_c_d(alpha, mode, qbr[1])
        qc.h(qbr[1])

    elif label == "XX":
        qc.h(qbr[0])
        qc.h(qbr[1])
        qc.cx(qbr[0], qbr[1])
        qc.cv_c_d(alpha, mode, qbr[1])
        qc.cx(qbr[0], qbr[1])
        qc.h(qbr[0])
        qc.h(qbr[1])

    elif label == "YY":
        # Diagonalize YY → ZZ via S†H on each qubit, then CNOT parity
        qc.rz(np.pi / 2, qbr[0])
        qc.rz(np.pi / 2, qbr[1])
        qc.h(qbr[0])
        qc.h(qbr[1])
        qc.cx(qbr[0], qbr[1])
        qc.cv_c_d(alpha, mode, qbr[1])
        qc.cx(qbr[0], qbr[1])
        qc.h(qbr[0])
        qc.h(qbr[1])
        qc.rz(-np.pi / 2, qbr[0])
        qc.rz(-np.pi / 2, qbr[1])

    else:
        raise ValueError(f"Unknown Pauli label: {label}")


def build_circuit(cfg: LCHSConfig) -> Tuple[CVCircuit, np.ndarray]:
    """
    Build the full LCHS circuit:
      1. cv_initialize(C_n)          — Fock superposition
      2. cv_sq(+r')                  — S(r') prep squeeze
      3. X gates                     — |u0> on qubits
      4. Trotter loop                — exp(-it x̂⊗A)
      5. cv_sq(-r)                   — S†(r) postselection squeeze

    Returns (circuit, coefficients).
    """
    _validate(cfg)
    coeffs = lchs_coefficients(cfg.r, cfg.r_prime, cfg.beta, cfg.n_coeff)

    n_mode_qubits = int(np.log2(cfg.n_fock))
    qmr = QumodeRegister(num_qumodes=1, num_qubits_per_qumode=n_mode_qubits)
    qbr = QuantumRegister(2, "q")
    qc = CVCircuit(qbr, qmr)
    mode = qmr[0]

    # Step 1: Fock superposition |psi_fock> = sum C_n |n>
    inject = np.zeros(cfg.n_fock, dtype=complex)
    inject[: cfg.n_coeff] = coeffs
    qc.cv_initialize(inject, mode)

    # Step 2: S(r') — paper Eq. 12
    if not np.isclose(cfg.r_prime, 0.0):
        qc.cv_sq(float(cfg.r_prime), mode)

    # Step 3: qubit initial state
    if cfg.init_q0 == 1:
        qc.x(qbr[0])
    if cfg.init_q1 == 1:
        qc.x(qbr[1])

    # Step 4: Trotter evolution exp(-it x̂ ⊗ A)
    terms = pauli_decomposition(cfg.alpha, cfg.h)
    dt = cfg.total_time / cfg.n_steps
    for _ in range(cfg.n_steps):
        for coeff, label in terms:
            _apply_trotter_term(qc, mode, qbr, coeff * dt, label)

    # Step 5: S†(r) = S(-r) for postselection
    if not np.isclose(cfg.r, 0.0):
        qc.cv_sq(-float(cfg.r), mode)

    return qc, coeffs


# ---------------------------------------------------------------------------
# Section C: Simulation and evaluation
# ---------------------------------------------------------------------------


def _detect_layout(n_fock: int) -> str:
    """Probe bosonic-qiskit statevector ordering."""
    n_mode_qubits = int(np.log2(n_fock))
    qmr = QumodeRegister(num_qumodes=1, num_qubits_per_qumode=n_mode_qubits)
    qbr = QuantumRegister(2, "q")
    qc = CVCircuit(qbr, qmr)
    probe = np.zeros(n_fock, dtype=complex)
    probe[1] = 1.0
    qc.cv_initialize(probe, qmr[0])
    state, _, _ = cv_util.simulate(qc, shots=1, return_fockcounts=False, add_save_statevector=True)
    if state is None:
        raise RuntimeError("Layout detection failed: no statevector.")
    vec = np.asarray(state.data, dtype=complex)
    nz = np.where(np.abs(vec) > 1e-12)[0]
    if len(nz) != 1:
        raise RuntimeError(f"Layout detection: expected 1 nonzero, got {len(nz)}.")
    idx = int(nz[0])
    if idx == 4:
        return "fock_major"
    if idx == 1:
        return "qubit_major"
    raise RuntimeError(f"Unknown layout probe index {idx}.")


def _extract_dv(vec: np.ndarray, n_fock: int, layout: str) -> np.ndarray:
    """Extract 4-component DV state from vacuum postselection, reorder to standard."""
    if layout == "fock_major":
        dv_qiskit = vec[:4]
    else:
        dv_qiskit = vec[0::n_fock][:4]
    # Qiskit |q1 q0> → standard |q0 q1>: swap indices 1↔2
    return dv_qiskit[[0, 2, 1, 3]]


def _initial_dv(q0: int, q1: int) -> np.ndarray:
    vec = np.zeros(4, dtype=complex)
    vec[q0 * 2 + q1] = 1.0
    return vec


def _classical_target(cfg: LCHSConfig) -> np.ndarray:
    A = heat_matrix(cfg.alpha, cfg.h)
    u0 = _initial_dv(cfg.init_q0, cfg.init_q1)
    return expm(-cfg.alpha * cfg.total_time * A) @ u0


def evaluate(cfg: LCHSConfig, _layout: Optional[str] = None) -> Dict[str, float]:
    """Run full LCHS pipeline and return all metrics."""
    _validate(cfg)
    if _layout is None:
        _layout = _detect_layout(cfg.n_fock)

    qc, coeffs = build_circuit(cfg)
    state, _, _ = cv_util.simulate(qc, shots=1, return_fockcounts=False, add_save_statevector=True)
    if state is None:
        raise RuntimeError("Simulation returned no statevector.")

    vec = np.asarray(state.data, dtype=complex)
    expected = 4 * cfg.n_fock
    if vec.size != expected:
        raise RuntimeError(f"Statevector size {vec.size}, expected {expected}.")

    dv_unnorm = _extract_dv(vec, cfg.n_fock, _layout)
    post_prob = float(np.real(np.vdot(dv_unnorm, dv_unnorm)))
    if np.isclose(post_prob, 0.0):
        raise RuntimeError("Zero post-selection probability.")

    # Density matrix (pure state from coherent injection)
    rho = np.outer(dv_unnorm, dv_unnorm.conj()) / post_prob
    purity = float(np.real(np.trace(rho @ rho)))

    # Classical reference
    u_ref = _classical_target(cfg)
    norm_ref = float(np.linalg.norm(u_ref))
    u_hat = u_ref / norm_ref

    # Fidelity
    fidelity = float(np.clip(np.real(np.vdot(u_hat, rho @ u_hat)), 0.0, 1.0))

    # PDE error
    eigvals, eigvecs = np.linalg.eigh(rho)
    psi_principal = eigvecs[:, np.argmax(eigvals)]
    psi_principal /= np.linalg.norm(psi_principal)
    overlap = np.vdot(u_hat, psi_principal)
    if np.abs(overlap) > 0.0:
        psi_principal *= np.exp(-1j * np.angle(overlap))
    u_cvdv = np.sqrt(post_prob) * psi_principal
    pde_error = float(np.linalg.norm(u_ref - u_cvdv) / norm_ref)

    # Pauli RMSE
    rho_target = np.outer(u_hat, u_hat.conj())
    pauli_labels = [(a, b) for a in "IXYZ" for b in "IXYZ"]
    pauli_err = []
    for la, lb in pauli_labels:
        P = np.kron(PAULI_MAP[la], PAULI_MAP[lb])
        pauli_err.append(
            float(np.real(np.trace(rho @ P))) - float(np.real(np.trace(rho_target @ P)))
        )
    pauli_rmse = float(np.sqrt(np.mean(np.array(pauli_err) ** 2)))

    # Diagnostics
    weights = np.abs(coeffs) ** 2
    weights /= weights.sum()
    gamma = lchs_gamma(cfg.r, cfg.r_prime)
    n_eff = float(1.0 / np.sum(weights ** 2))
    tail = estimate_tail_mass(cfg.r, cfg.r_prime, cfg.beta, cfg.n_coeff)
    comm = commutator_norm_bound(cfg.alpha, cfg.h)
    trotter_bound = (cfg.alpha * cfg.total_time) ** 2 * comm / (2.0 * cfg.n_steps)

    return {
        "post_prob": post_prob,
        "pde_error": pde_error,
        "fidelity": fidelity,
        "purity": purity,
        "pauli_rmse": pauli_rmse,
        "gamma": gamma,
        "n_eff": n_eff,
        "tail_mass": tail,
        "trotter_bound": trotter_bound,
        "active_coeffs": int(np.sum(np.abs(coeffs) > 1e-8)),
    }


# ---------------------------------------------------------------------------
# Section D: Sensitivity sweep and CLI
# ---------------------------------------------------------------------------


def _grid(lo: float, hi: float, n: int) -> np.ndarray:
    return np.linspace(lo, hi, max(n, 1))


def _to_key(x: float) -> float:
    return float(np.round(x, 12))


def sensitivity_sweep(
    base: LCHSConfig,
    r_range: Tuple[float, float, int],
    rp_range: Tuple[float, float, int],
    beta_range: Tuple[float, float, int],
    refine_top_k: int = 3,
    refine_pts: int = 3,
    refine_dr: float = 0.08,
    refine_drp: float = 0.05,
    refine_db: float = 0.08,
    gamma_min: float = 0.03,
    min_gap: float = 0.02,
    min_post_prob: float = 1e-3,
    post_penalty: float = 100.0,
    output_dir: Optional[str] = None,
) -> List[Dict]:
    """Two-stage sensitivity sweep: coarse grid + local refinement."""
    layout = _detect_layout(base.n_fock)

    grid_r = _grid(*r_range)
    grid_rp = _grid(*rp_range)
    grid_b = _grid(*beta_range)

    rows: List[Dict] = []
    seen = set()

    def run_point(r: float, rp: float, b: float, stage: str) -> None:
        key = (_to_key(r), _to_key(rp), _to_key(b))
        if key in seen:
            return
        seen.add(key)

        gamma = lchs_gamma(r, rp)
        if not (0.0 < b < 1.0) or rp >= r - min_gap or gamma <= gamma_min:
            rows.append({"stage": stage, "r": r, "r_prime": rp, "beta": b,
                         "pde_error": np.nan, "post_prob": np.nan, "feasible": 0,
                         "objective": np.inf})
            return

        cfg = LCHSConfig(
            alpha=base.alpha, h=base.h, total_time=base.total_time,
            init_q0=base.init_q0, init_q1=base.init_q1,
            r=r, r_prime=rp, beta=b,
            n_steps=base.n_steps, n_fock=base.n_fock, n_coeff=base.n_coeff,
        )
        try:
            met = evaluate(cfg, _layout=layout)
        except Exception as exc:
            rows.append({"stage": stage, "r": r, "r_prime": rp, "beta": b,
                         "pde_error": np.nan, "post_prob": np.nan, "feasible": 0,
                         "objective": np.inf, "error": str(exc)})
            return

        feasible = int(met["post_prob"] >= min_post_prob and met["tail_mass"] <= 1e-4)
        penalty = post_penalty * max(0.0, min_post_prob - met["post_prob"]) / max(min_post_prob, 1e-15)
        obj = met["pde_error"] + penalty - 1e-3 * met["post_prob"]

        row = {"stage": stage, "r": r, "r_prime": rp, "beta": b,
               "feasible": feasible, "objective": obj}
        row.update(met)
        rows.append(row)
        print(
            f"{stage:>6} r={r:.4f} r'={rp:.4f} beta={b:.4f} "
            f"-> pde={met['pde_error']:.5f} post={met['post_prob']:.3e} "
            f"fid={met['fidelity']:.4f} feas={feasible}",
            flush=True,
        )

    print("Stage 1/2: coarse grid", flush=True)
    for r in grid_r:
        for rp in grid_rp:
            for b in grid_b:
                run_point(float(r), float(rp), float(b), "coarse")

    # Select top seeds
    numeric = [r for r in rows if np.isfinite(r.get("objective", np.inf))]
    numeric.sort(key=lambda x: (-x.get("feasible", 0), x["objective"]))
    seeds = numeric[:refine_top_k]

    print("Stage 2/2: local refinement", flush=True)
    for i, seed in enumerate(seeds, 1):
        r0, rp0, b0 = seed["r"], seed["r_prime"], seed["beta"]
        print(f"  seed {i}: r={r0:.4f} r'={rp0:.4f} beta={b0:.4f}", flush=True)
        for r in _grid(max(r_range[0], r0 - refine_dr), min(r_range[1], r0 + refine_dr), refine_pts):
            for rp in _grid(max(rp_range[0], rp0 - refine_drp), min(rp_range[1], rp0 + refine_drp), refine_pts):
                for b in _grid(max(beta_range[0], b0 - refine_db), min(beta_range[1], b0 + refine_db), refine_pts):
                    run_point(float(r), float(rp), float(b), "refine")

    # Sort final
    numeric = [r for r in rows if np.isfinite(r.get("objective", np.inf))]
    numeric.sort(key=lambda x: (-x.get("feasible", 0), x["objective"]))

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        # CSV
        if rows:
            fields = list(rows[0].keys())
            with (out / "sweep_all.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
        # Summary JSON
        summary = {"config": asdict(base), "best": numeric[0] if numeric else None}
        (out / "sweep_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        # Top 20
        if numeric:
            with (out / "sweep_top.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(numeric[0].keys()), extrasaction="ignore")
                w.writeheader()
                w.writerows(numeric[:20])
        # Plot
        try:
            import matplotlib.pyplot as plt

            xs = [r["post_prob"] for r in numeric if not np.isnan(r.get("post_prob", np.nan))]
            ys = [r["pde_error"] for r in numeric if not np.isnan(r.get("pde_error", np.nan))]
            cs = [r["beta"] for r in numeric if not np.isnan(r.get("pde_error", np.nan))]
            if xs:
                plt.figure(figsize=(6, 4))
                sc = plt.scatter(xs, ys, c=cs, s=20, cmap="viridis")
                plt.colorbar(sc, label="beta")
                plt.xlabel("post_prob")
                plt.ylabel("pde_error")
                plt.title("PDE error vs success probability")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(out / "tradeoff.png", dpi=180)
                plt.close()
        except ImportError:
            pass
        print(f"Outputs: {out}", flush=True)

    if numeric:
        print("Best:", json.dumps(numeric[0], indent=2, default=str), flush=True)

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CV-DV LCHS heat equation solver")
    p.add_argument("--mode", choices=["single", "sweep"], default="single")

    # PDE
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--h-grid", type=float, default=1.0)
    p.add_argument("--total-time", type=float, default=1.0)
    p.add_argument("--init-q0", type=int, default=0, choices=[0, 1])
    p.add_argument("--init-q1", type=int, default=1, choices=[0, 1])

    # LCHS
    p.add_argument("--r", type=float, default=1.2)
    p.add_argument("--r-prime", type=float, default=0.3)
    p.add_argument("--beta", type=float, default=0.75)

    # Numerics
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--n-fock", type=int, default=32)
    p.add_argument("--n-coeff", type=int, default=32)

    # Output
    p.add_argument("--output", type=str, default="")

    # Sweep parameters
    p.add_argument("--r-min", type=float, default=0.5)
    p.add_argument("--r-max", type=float, default=2.5)
    p.add_argument("--n-r", type=int, default=7)
    p.add_argument("--rp-min", type=float, default=0.03)
    p.add_argument("--rp-max", type=float, default=1.0)
    p.add_argument("--n-rp", type=int, default=7)
    p.add_argument("--beta-min", type=float, default=0.5)
    p.add_argument("--beta-max", type=float, default=0.95)
    p.add_argument("--n-beta", type=int, default=9)
    p.add_argument("--output-dir", type=str, default="results/sweep")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = LCHSConfig(
        alpha=args.alpha,
        h=args.h_grid,
        total_time=args.total_time,
        init_q0=args.init_q0,
        init_q1=args.init_q1,
        r=args.r,
        r_prime=args.r_prime,
        beta=args.beta,
        n_steps=args.n_steps,
        n_fock=args.n_fock,
        n_coeff=args.n_coeff,
    )

    if args.mode == "sweep":
        sensitivity_sweep(
            cfg,
            r_range=(args.r_min, args.r_max, args.n_r),
            rp_range=(args.rp_min, args.rp_max, args.n_rp),
            beta_range=(args.beta_min, args.beta_max, args.n_beta),
            output_dir=args.output_dir,
        )
        return

    # Single run
    print("=== CV-DV LCHS Heat Equation Solver ===")
    print(json.dumps(asdict(cfg), indent=2))

    metrics = evaluate(cfg)
    print("=== Metrics ===")
    for k in ("post_prob", "pde_error", "fidelity", "purity", "pauli_rmse",
              "tail_mass", "n_eff", "trotter_bound"):
        print(f"{k}: {metrics[k]:.12e}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"config": asdict(cfg), "metrics": metrics}, indent=2))
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
