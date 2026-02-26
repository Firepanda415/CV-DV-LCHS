#!/usr/bin/env python3
r"""
CV-DV LCHS example for the 1D heat equation using SciPy + QuTiP only.

Model:
  du/dt = -A u,   A = (alpha / h^2) T,
  T = [[ 2,-1, 0, 0],
       [-1, 2,-1, 0],
       [ 0,-1, 2,-1],
       [ 0, 0,-1, 2]].

Hybrid CV-DV LCHS form:
  K(t) = <phi_r| exp(-i t x_hat \otimes A) |psi_{r,r',beta}>
  with
    |phi_r> = S(r)|0>,
    |psi_{r,r',beta}> = S(r') sum_n C_n |n>,
  and C_n from Gauss-Hermite projection of the improved kernel branch.

We report:
  1) operator-level error: K(t) vs exp(-A t), up to one global complex scale,
  2) vector-level error for a chosen initial state.

Example CLI usage:
  # Default run (basis01 initial state)
  python clacod_heat1d_qutip.py

  # Custom LCHS parameters
  python clacod_heat1d_qutip.py --r-target 0.85 --r-prime 0.003 --beta 0.35

  # Different initial state, save results
  python clacod_heat1d_qutip.py --init-state basis10 --output-json results/qutip_b10.json

  # Time trajectory with CSV output
  python clacod_heat1d_qutip.py --trajectory-steps 20 --trajectory-csv results/traj.csv

  # Larger Fock space, finer grid
  python clacod_heat1d_qutip.py --n-fock 128 --n-coeff 64 --h-grid 0.2 --alpha 1.0
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from numpy.polynomial.hermite import hermgauss
from qutip import Qobj, basis, destroy, qeye, squeeze, tensor
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.special import eval_hermite, gammaln


COEFF_METHODS = ("legacy_gh", "explicit_overlap")


@dataclass(frozen=True)
class HeatLCHSConfig:
    alpha: float = 1.0
    h_grid: float = 1.0
    total_time: float = 1.0
    n_fock: int = 64
    n_coeff: int = 64
    r_target: float = 0.003
    r_prime: float = 0.001
    beta: float = 0.95
    n_quad: int = 300
    coeff_method: str = "legacy_gh"
    init_state: str = "basis01"
    position_convention: str = "sqrt2"  # "half" -> (a+a^dag)/2, "sqrt2" -> (a+a^dag)/sqrt(2)
    trajectory_steps: int = 0


def basis_label_2qubit(index: int) -> str:
    if index < 0 or index > 3:
        raise ValueError(f"Component index out of range: {index}")
    return format(index, "02b")


def heat_matrix(alpha: float, h_grid: float) -> np.ndarray:
    t = np.array(
        [[2.0, -1.0, 0.0, 0.0], [-1.0, 2.0, -1.0, 0.0], [0.0, -1.0, 2.0, -1.0], [0.0, 0.0, -1.0, 2.0]],
        dtype=float,
    )
    return (alpha / (h_grid**2)) * t


def heat_matrix_from_paulis(alpha: float, h_grid: float) -> np.ndarray:
    s = alpha / (h_grid**2)
    i2 = np.eye(2, dtype=complex)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    a = (
        2.0 * s * np.kron(i2, i2)
        - 1.0 * s * np.kron(i2, x)
        - 0.5 * s * np.kron(x, x)
        - 0.5 * s * np.kron(y, y)
    )
    return np.real_if_close(a).astype(float)


def kernel_g_beta(k_points: np.ndarray, beta: float) -> np.ndarray:
    # Near-optimal kernel branch from arXiv:2312.03916 (Eq. 32-33):
    # g_beta(k) = exp(-(1+ik)^beta) / (C_beta * (1-ik)),
    # C_beta = 2*pi*exp(-2^beta), beta in (0,1).
    c_beta = 2.0 * np.pi * np.exp(-(2.0**beta))
    return np.exp(-((1.0 + 1.0j * k_points) ** beta)) / (c_beta * (1.0 - 1.0j * k_points))


def _validate_coefficient_inputs(
    r_target: float,
    r_prime: float,
    beta: float,
    n_fock: int,
    n_quad: int,
) -> float:
    if n_fock <= 0:
        raise ValueError("n_fock must be positive.")
    gamma = np.exp(-2.0 * r_prime) - np.exp(-2.0 * r_target)
    if gamma <= 0:
        raise ValueError("Need r_prime < r_target so gamma = exp(-2r') - exp(-2r) > 0.")
    if not (0.0 < beta < 1.0):
        raise ValueError("beta must be in (0,1).")
    if n_quad <= 0:
        raise ValueError("n_quad must be positive.")
    return float(gamma)


def _fock_prefactor(n: int, width: float) -> float:
    # 1/sqrt(2^n n!) in log-space for numerical stability.
    fock_norm = np.exp(-0.5 * (n * np.log(2.0) + gammaln(n + 1.0)))
    return float((1.0 / np.sqrt(np.sqrt(np.pi) * width)) * fock_norm)


def lchs_coefficients_legacy_gh(
    r_target: float,
    r_prime: float,
    beta: float,
    n_fock: int,
    n_quad: int,
) -> np.ndarray:
    """Legacy Gauss-Hermite backend.

    This computes the hbar=1 overlap integral using the identity
    x = sqrt(2) * sigma' * xi, where Gauss-Hermite contributes exp(-xi^2).
    """
    gamma = _validate_coefficient_inputs(r_target, r_prime, beta, n_fock, n_quad)

    width = np.exp(r_prime)
    scale = np.sqrt(2.0) * width
    roots, weights = hermgauss(n_quad)
    k_points = roots * scale

    # Effective integrand: H_n(k/sigma') * g(k) * exp(-gamma k^2) * exp(-k^2/(2 sigma'^2))
    envelope = np.exp(-gamma * k_points**2) * kernel_g_beta(k_points, beta)

    coeffs = np.zeros(n_fock, dtype=complex)
    for n in range(n_fock):
        pref = _fock_prefactor(n, width)
        herm = eval_hermite(n, k_points / width)
        coeffs[n] = np.sum(weights * envelope * pref * herm) * scale

    norm = np.linalg.norm(coeffs)
    if np.isclose(norm, 0.0):
        raise RuntimeError("Computed coefficient vector has near-zero norm.")
    return coeffs / norm


def lchs_coefficients_explicit_overlap(
    r_target: float,
    r_prime: float,
    beta: float,
    n_fock: int,
    n_quad: int,
) -> np.ndarray:
    """Explicit-overlap backend.

    Directly evaluates the same hbar=1 overlap integral as the legacy backend
    in physical coordinates:

      C_n ∝ ∫ H_n(k/sigma') g(k) exp(-gamma k^2) exp(-k^2/(2 sigma'^2)) dk

    using adaptive quadrature over a finite interval chosen from Gaussian tails.
    """
    gamma = _validate_coefficient_inputs(r_target, r_prime, beta, n_fock, n_quad)
    width = np.exp(r_prime)
    gauss_rate = gamma + 1.0 / (2.0 * width * width)
    k_tail = np.sqrt(max(-np.log(1e-14) / max(gauss_rate, 1e-15), 1.0))
    k_max = max(8.0 * width, 1.15 * k_tail, 10.0)
    c_beta = 2.0 * np.pi * np.exp(-(2.0**beta))

    def kernel_scalar(k: float) -> complex:
        return np.exp(-((1.0 + 1.0j * k) ** beta)) / (c_beta * (1.0 - 1.0j * k))

    coeffs = np.zeros(n_fock, dtype=complex)
    for n in range(n_fock):
        pref = _fock_prefactor(n, width)

        def integrand_real(k: float) -> float:
            val = (
                pref
                * eval_hermite(n, k / width)
                * kernel_scalar(k)
                * np.exp(-gamma * (k**2))
                * np.exp(-(k**2) / (2.0 * width * width))
            )
            return float(np.real(val))

        def integrand_imag(k: float) -> float:
            val = (
                pref
                * eval_hermite(n, k / width)
                * kernel_scalar(k)
                * np.exp(-gamma * (k**2))
                * np.exp(-(k**2) / (2.0 * width * width))
            )
            return float(np.imag(val))

        re = quad(
            integrand_real,
            -k_max,
            k_max,
            limit=max(100, n_quad),
            epsabs=1e-10,
            epsrel=1e-9,
        )[0]
        im = quad(
            integrand_imag,
            -k_max,
            k_max,
            limit=max(100, n_quad),
            epsabs=1e-10,
            epsrel=1e-9,
        )[0]
        coeffs[n] = re + 1.0j * im

    norm = np.linalg.norm(coeffs)
    if np.isclose(norm, 0.0):
        raise RuntimeError("Computed coefficient vector has near-zero norm.")
    return coeffs / norm


def lchs_coefficients(
    r_target: float,
    r_prime: float,
    beta: float,
    n_fock: int,
    n_quad: int,
    method: str = "legacy_gh",
) -> np.ndarray:
    if method == "legacy_gh":
        return lchs_coefficients_legacy_gh(
            r_target=r_target,
            r_prime=r_prime,
            beta=beta,
            n_fock=n_fock,
            n_quad=n_quad,
        )
    if method == "explicit_overlap":
        return lchs_coefficients_explicit_overlap(
            r_target=r_target,
            r_prime=r_prime,
            beta=beta,
            n_fock=n_fock,
            n_quad=n_quad,
        )
    raise ValueError(f"Unknown coeff method '{method}'. Expected one of: {', '.join(COEFF_METHODS)}")


def coefficient_tail_mass(coeffs: np.ndarray, tail_levels: int = 4) -> float:
    if tail_levels <= 0:
        return 0.0
    weights = np.abs(coeffs) ** 2
    total = float(np.sum(weights))
    if np.isclose(total, 0.0):
        return float("nan")
    m = min(int(tail_levels), len(coeffs))
    return float(np.sum(weights[-m:]) / total)


def prepare_lchs_states(cfg: HeatLCHSConfig) -> Tuple[Qobj, Qobj, np.ndarray]:
    if cfg.n_coeff <= 0 or cfg.n_coeff > cfg.n_fock:
        raise ValueError("n_coeff must satisfy 1 <= n_coeff <= n_fock.")
    if cfg.coeff_method not in COEFF_METHODS:
        raise ValueError(f"Unknown coeff_method '{cfg.coeff_method}'.")
    coeffs = lchs_coefficients(
        r_target=cfg.r_target,
        r_prime=cfg.r_prime,
        beta=cfg.beta,
        n_fock=cfg.n_coeff,
        n_quad=cfg.n_quad,
        method=cfg.coeff_method,
    )
    coeffs_full = np.zeros(cfg.n_fock, dtype=complex)
    coeffs_full[: cfg.n_coeff] = coeffs
    psi_seed = Qobj(coeffs_full.reshape((-1, 1))).unit()
    psi_osc = (squeeze(cfg.n_fock, cfg.r_prime) * psi_seed).unit()
    phi_post = (squeeze(cfg.n_fock, cfg.r_target) * basis(cfg.n_fock, 0)).unit()
    return psi_osc, phi_post, coeffs_full


def position_operator(n_fock: int, convention: str) -> Qobj:
    a = destroy(n_fock)
    if convention == "half":
        x_hat = (a + a.dag()) / 2.0
    elif convention == "sqrt2":
        x_hat = (a + a.dag()) / np.sqrt(2.0)
    else:
        raise ValueError(f"Unknown position convention: {convention}")
    return x_hat


def parse_initial_state(name: str) -> np.ndarray:
    if name == "basis00":
        v = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    elif name == "basis01":
        v = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
    elif name == "basis10":
        v = np.array([0.0, 0.0, 1.0, 0.0], dtype=complex)
    elif name == "basis11":
        v = np.array([0.0, 0.0, 0.0, 1.0], dtype=complex)
    elif name == "sine":
        j = np.arange(1, 5, dtype=float)
        v = np.sin(np.pi * j / 5.0).astype(complex)
    elif name == "ones":
        v = np.ones(4, dtype=complex)
    else:
        raise ValueError(f"Unknown init_state '{name}'.")

    nrm = np.linalg.norm(v)
    if np.isclose(nrm, 0.0):
        raise RuntimeError("Initial state has zero norm.")
    return v / nrm


def classical_map(cfg: HeatLCHSConfig) -> np.ndarray:
    a = heat_matrix(cfg.alpha, cfg.h_grid)
    return expm(-a * cfg.total_time)


def effective_lchs_map(cfg: HeatLCHSConfig, psi_osc: Qobj, phi_post: Qobj) -> np.ndarray:
    a = heat_matrix(cfg.alpha, cfg.h_grid)
    a_qobj = Qobj(a, dims=[[4], [4]])
    x_hat = position_operator(cfg.n_fock, cfg.position_convention)
    u_joint = (-1.0j * cfg.total_time * tensor(x_hat, a_qobj)).expm()

    post_bra = tensor(phi_post.dag(), qeye(4))
    embed = tensor(psi_osc, qeye(4))
    k_map = post_bra * u_joint * embed
    return np.asarray(k_map.full(), dtype=complex)


def fit_global_scale(observed: np.ndarray, target: np.ndarray) -> Tuple[complex, float]:
    denom = np.vdot(target, target)
    if np.isclose(denom, 0.0):
        return 0.0 + 0.0j, float("nan")
    scale = np.vdot(target, observed) / denom
    rel_err = np.linalg.norm(observed - scale * target) / max(np.linalg.norm(scale * target), 1e-15)
    return complex(scale), float(rel_err)


def state_fidelity(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if np.isclose(n1, 0.0) or np.isclose(n2, 0.0):
        return float("nan")
    return float(np.abs(np.vdot(v1 / n1, v2 / n2)) ** 2)


def run_trajectory(
    cfg: HeatLCHSConfig,
    psi_osc: Qobj,
    phi_post: Qobj,
    u0: np.ndarray,
) -> List[Dict[str, float]]:
    steps = int(cfg.trajectory_steps)
    if steps <= 0:
        return []

    a = heat_matrix(cfg.alpha, cfg.h_grid)
    a_qobj = Qobj(a, dims=[[4], [4]])
    x_hat = position_operator(cfg.n_fock, cfg.position_convention)

    dt = cfg.total_time / steps
    u_dt = (-1.0j * dt * tensor(x_hat, a_qobj)).expm()
    post_bra = tensor(phi_post.dag(), qeye(4))
    joint = tensor(psi_osc, Qobj(u0.reshape((-1, 1)), dims=[[4], [1]]))

    eta = complex(phi_post.overlap(psi_osc))
    rows: List[Dict[str, float]] = []
    for k in range(steps + 1):
        t = k * dt
        u_cv = np.asarray((post_bra * joint).full(), dtype=complex).reshape(-1)
        u_ref = expm(-a * t) @ u0

        scale_vec, rel_err = fit_global_scale(u_cv, u_ref)
        fid = state_fidelity(u_cv, u_ref)

        eta_err = float("nan")
        if not np.isclose(np.abs(eta), 0.0):
            eta_err = float(np.linalg.norm((u_cv / eta) - u_ref) / max(np.linalg.norm(u_ref), 1e-15))

        rows.append(
            {
                "time": float(t),
                "post_prob": float(np.real(np.vdot(u_cv, u_cv))),
                "scale_real": float(np.real(scale_vec)),
                "scale_imag": float(np.imag(scale_vec)),
                "rel_error_best_scale": rel_err,
                "rel_error_eta_calibrated": eta_err,
                "shape_fidelity": fid,
            }
        )

        if k < steps:
            joint = u_dt * joint

    return rows


def run_component_trajectory(
    cfg: HeatLCHSConfig,
    psi_osc: Qobj,
    phi_post: Qobj,
    u0: np.ndarray,
    calibration_scale: complex,
) -> Dict[str, np.ndarray]:
    steps = int(cfg.trajectory_steps)
    if steps <= 0:
        return {}

    a = heat_matrix(cfg.alpha, cfg.h_grid)
    a_qobj = Qobj(a, dims=[[4], [4]])
    x_hat = position_operator(cfg.n_fock, cfg.position_convention)

    dt = cfg.total_time / steps
    u_dt = (-1.0j * dt * tensor(x_hat, a_qobj)).expm()
    post_bra = tensor(phi_post.dag(), qeye(4))
    joint = tensor(psi_osc, Qobj(u0.reshape((-1, 1)), dims=[[4], [1]]))

    times = np.linspace(0.0, cfg.total_time, steps + 1)
    cv_raw = np.zeros((steps + 1, 4), dtype=complex)
    cv_cal = np.zeros((steps + 1, 4), dtype=complex)
    ref = np.zeros((steps + 1, 4), dtype=complex)

    for k, t in enumerate(times):
        u_cv = np.asarray((post_bra * joint).full(), dtype=complex).reshape(-1)
        u_ref = expm(-a * t) @ u0
        cv_raw[k, :] = u_cv
        ref[k, :] = u_ref
        if np.isclose(np.abs(calibration_scale), 0.0):
            cv_cal[k, :] = u_cv
        else:
            cv_cal[k, :] = u_cv / calibration_scale
        if k < steps:
            joint = u_dt * joint

    return {"time": times, "cv_raw": cv_raw, "cv_cal": cv_cal, "ref": ref}


def plot_trajectory_cho_style(
    traj: Dict[str, np.ndarray],
    component_indices: Sequence[int],
    out_path: Path,
) -> None:
    if not traj:
        return
    if len(component_indices) == 0 or len(component_indices) > 4:
        raise ValueError("Need between 1 and 4 component indices.")

    import matplotlib.pyplot as plt

    time = traj["time"]
    cv_cal = traj["cv_cal"]
    ref = traj["ref"]

    colors = ["#1f4bf2", "#e21a1a", "#1a8f44", "#8a2be2"]
    n = len(component_indices)
    if n == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
        axes_list = list(axes.ravel())
    else:
        fig, axes = plt.subplots(n, 1, figsize=(12, max(3 * n, 4)), sharex=True)
        if n == 1:
            axes_list = [axes]
        else:
            axes_list = list(axes)
    fig.patch.set_facecolor("#dddddd")

    for row, comp in enumerate(component_indices):
        ax = axes_list[row]
        bits = basis_label_2qubit(int(comp))
        y_cv = np.real(cv_cal[:, comp])
        y_ref = np.real(ref[:, comp])
        ax.set_facecolor("#dddddd")
        ax.plot(
            time,
            y_cv,
            "o-",
            lw=2.0,
            ms=3.0,
            color=colors[row],
            label=f"LCHS (Comp |{bits}>)",
        )
        ax.plot(
            time,
            y_ref,
            "--",
            lw=1.6,
            color="dimgray",
            label="Classical reference",
        )
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Component |{bits}>")
        ax.set_xlabel("Time")
        ax.legend(loc="best")

    for ax in axes_list[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def parse_component_indices(text: str) -> Tuple[int, ...]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) == 0:
        raise ValueError("plot-components must contain one to four comma-separated indices.")
    vals = tuple(int(p) for p in parts)
    if len(vals) > 4:
        raise ValueError("At most four component indices are allowed.")
    if len(set(vals)) != len(vals):
        raise ValueError("Component indices must be unique.")
    for idx in vals:
        if idx < 0 or idx > 3:
            raise ValueError("Component indices must be between 0 and 3 for the 4D DV state.")
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CV-DV LCHS 1D heat equation demo (SciPy + QuTiP)")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--h-grid", type=float, default=1.0)
    p.add_argument("--total-time", type=float, default=1.0)

    p.add_argument("--n-fock", type=int, default=64)
    p.add_argument("--n-coeff", type=int, default=64)
    p.add_argument("--r-target", type=float, default=0.003)
    p.add_argument("--r-prime", type=float, default=0.001)
    p.add_argument("--beta", type=float, default=0.95)
    p.add_argument("--n-quad", type=int, default=300)
    p.add_argument(
        "--coeff-method",
        choices=list(COEFF_METHODS),
        default="legacy_gh",
        help="Coefficient backend: legacy Gauss-Hermite or explicit overlap integral.",
    )
    p.add_argument(
        "--coeff-crosscheck-n",
        type=int,
        default=0,
        help="If >0, cross-check first N coefficients between both backends.",
    )

    p.add_argument(
        "--init-state",
        choices=["basis00", "basis01", "basis10", "basis11", "sine", "ones"],
        default="basis01",
    )
    p.add_argument("--position-convention", choices=["half", "sqrt2"], default="sqrt2")

    p.add_argument("--trajectory-steps", type=int, default=0)
    p.add_argument("--output-json", type=str, default="")
    p.add_argument("--trajectory-csv", type=str, default="")
    p.add_argument("--component-plot", type=str, default="")
    p.add_argument("--plot-components", type=str, default="0,1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    n_coeff = args.n_fock if args.n_coeff <= 0 else args.n_coeff
    cfg = HeatLCHSConfig(
        alpha=args.alpha,
        h_grid=args.h_grid,
        total_time=args.total_time,
        n_fock=args.n_fock,
        n_coeff=n_coeff,
        r_target=args.r_target,
        r_prime=args.r_prime,
        beta=args.beta,
        n_quad=args.n_quad,
        coeff_method=args.coeff_method,
        init_state=args.init_state,
        position_convention=args.position_convention,
        trajectory_steps=args.trajectory_steps,
    )

    if cfg.n_fock <= 0:
        raise ValueError("n_fock must be positive.")

    a_direct = heat_matrix(cfg.alpha, cfg.h_grid)
    a_pauli = heat_matrix_from_paulis(cfg.alpha, cfg.h_grid)
    pauli_consistency = float(np.linalg.norm(a_direct - a_pauli))

    psi_osc, phi_post, coeffs = prepare_lchs_states(cfg)
    k_map = effective_lchs_map(cfg, psi_osc, phi_post)
    ref_map = classical_map(cfg)
    scale_map, map_rel_error = fit_global_scale(k_map, ref_map)

    u0 = parse_initial_state(cfg.init_state)
    u_cv = k_map @ u0
    u_ref = ref_map @ u0
    scale_vec, vec_rel_error = fit_global_scale(u_cv, u_ref)
    vec_fidelity = state_fidelity(u_cv, u_ref)

    eta = complex(phi_post.overlap(psi_osc))
    eta_rel_error = float("nan")
    if not np.isclose(np.abs(eta), 0.0):
        eta_rel_error = float(np.linalg.norm((u_cv / eta) - u_ref) / max(np.linalg.norm(u_ref), 1e-15))

    coeff_weights = np.abs(coeffs) ** 2
    coeff_weights /= max(np.sum(coeff_weights), 1e-15)
    n_eff = float(1.0 / np.sum(coeff_weights**2))
    tail_mass_last4 = coefficient_tail_mass(coeffs[: cfg.n_coeff], tail_levels=4)

    coeff_crosscheck: Dict[str, float | int] = {}
    if args.coeff_crosscheck_n > 0:
        n_chk = min(int(args.coeff_crosscheck_n), cfg.n_coeff)
        c_legacy = lchs_coefficients(
            r_target=cfg.r_target,
            r_prime=cfg.r_prime,
            beta=cfg.beta,
            n_fock=n_chk,
            n_quad=cfg.n_quad,
            method="legacy_gh",
        )
        c_explicit = lchs_coefficients(
            r_target=cfg.r_target,
            r_prime=cfg.r_prime,
            beta=cfg.beta,
            n_fock=n_chk,
            n_quad=cfg.n_quad,
            method="explicit_overlap",
        )
        _, chk_rel = fit_global_scale(c_legacy, c_explicit)
        coeff_crosscheck = {
            "n_checked": n_chk,
            "legacy_vs_explicit_rel_error": float(chk_rel),
        }

    metrics = {
        "config": asdict(cfg),
        "pauli_consistency_norm": pauli_consistency,
        "map_scale_real": float(np.real(scale_map)),
        "map_scale_imag": float(np.imag(scale_map)),
        "map_rel_error_best_scale": map_rel_error,
        "vector_scale_real": float(np.real(scale_vec)),
        "vector_scale_imag": float(np.imag(scale_vec)),
        "vector_rel_error_best_scale": vec_rel_error,
        "vector_shape_fidelity": vec_fidelity,
        "vector_rel_error_eta_calibrated": eta_rel_error,
        "post_prob_selected_input": float(np.real(np.vdot(u_cv, u_cv))),
        "eta_overlap_real": float(np.real(eta)),
        "eta_overlap_imag": float(np.imag(eta)),
        "coeff_active_1e-8": int(np.sum(np.abs(coeffs) > 1e-8)),
        "coeff_n_eff": n_eff,
        "coeff_tail_mass_last4": tail_mass_last4,
    }
    metrics.update(coeff_crosscheck)

    print("=== CV-DV LCHS Heat1D (SciPy + QuTiP) ===")
    print(
        "Run config: "
        f"alpha={cfg.alpha}, h_grid={cfg.h_grid}, total_time={cfg.total_time}, "
        f"init_state={cfg.init_state}"
    )
    print(
        "CV config: "
        f"n_fock={cfg.n_fock}, n_coeff={cfg.n_coeff}, n_quad={cfg.n_quad}, "
        f"r_target={cfg.r_target}, r_prime={cfg.r_prime}, beta={cfg.beta}, "
        f"coeff_method={cfg.coeff_method}"
    )
    print(f"Position convention: {cfg.position_convention}")
    print(f"A (direct) vs A (Pauli) consistency norm: {pauli_consistency:.3e}")
    print(f"Operator map relative error (best global scale): {map_rel_error:.6f}")
    print(f"Vector relative error for {cfg.init_state} (best global scale): {vec_rel_error:.6f}")
    print(f"Vector shape fidelity for {cfg.init_state}: {vec_fidelity:.6f}")
    print(f"Vector relative error after eta calibration: {eta_rel_error:.6f}")
    print(f"Postselection probability ({cfg.init_state}): {metrics['post_prob_selected_input']:.6e}")
    print(f"Best map scale = {scale_map.real:+.6e} {scale_map.imag:+.6e}j")
    print(f"Coefficient tail mass (last 4 levels): {tail_mass_last4:.6e}")
    if tail_mass_last4 > 0.25:
        print("WARNING: Large coefficient mass at truncation edge; increase n_coeff/n_fock.")
    if coeff_crosscheck:
        print(
            "Coefficient backend cross-check "
            f"(N={coeff_crosscheck['n_checked']}): "
            f"legacy-vs-explicit rel err = {coeff_crosscheck['legacy_vs_explicit_rel_error']:.3e}"
        )

    rows = run_trajectory(cfg, psi_osc, phi_post, u0)
    if rows:
        best = min(rows, key=lambda r: r["rel_error_best_scale"])
        worst = max(rows, key=lambda r: r["rel_error_best_scale"])
        print(
            "Trajectory rel_error_best_scale: "
            f"min={best['rel_error_best_scale']:.6f} at t={best['time']:.4f}, "
            f"max={worst['rel_error_best_scale']:.6f} at t={worst['time']:.4f}"
        )

    if args.trajectory_csv:
        out_csv = Path(args.trajectory_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "time",
                    "post_prob",
                    "scale_real",
                    "scale_imag",
                    "rel_error_best_scale",
                    "rel_error_eta_calibrated",
                    "shape_fidelity",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote trajectory CSV: {out_csv}")

    if args.component_plot:
        comps = parse_component_indices(args.plot_components)
        comp_traj = run_component_trajectory(
            cfg=cfg,
            psi_osc=psi_osc,
            phi_post=phi_post,
            u0=u0,
            calibration_scale=scale_map,
        )
        plot_path = Path(args.component_plot)
        plot_trajectory_cho_style(comp_traj, comps, plot_path)
        print(f"Wrote CHO-style component plot: {plot_path}")

    if args.output_json:
        out_json = Path(args.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"metrics": metrics, "trajectory": rows}
        out_json.write_text(json.dumps(payload, indent=2))
        print(f"Wrote JSON: {out_json}")


if __name__ == "__main__":
    main()
