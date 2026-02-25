#!/usr/bin/env python3
"""
1D heat equation via hybrid CV-DV LCHS.

This script is intentionally self-contained and mathematically annotated so the
circuit construction can be audited line-by-line against the working formulas.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import hybridlane as hqml
import numpy as np
import pennylane as qml
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import expm
from scipy.special import eval_hermite, factorial


# ---- Pauli basis for 2-qubit tomography ----
PAULI_I = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_LABELS = ("I", "X", "Y", "Z")
PAULI_MATRICES = {"I": PAULI_I, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}
PAULI_COMBOS = [(a, b) for a in PAULI_LABELS for b in PAULI_LABELS]


@dataclass(frozen=True)
class HeatEquationConfig:
    # PDE / ODE integration parameters
    total_time: float = 1.0
    n_steps: int = 100
    alpha: float = 1.0
    h: float = 1.0

    # Two-qubit encoding for 4 interior grid points
    init_qubits: Tuple[int, int] = (0, 1)

    # CV truncation and backend details
    n_dim: int = 32
    max_fock_level: int = 32
    hbar: float = 2.0
    fock_weight_cutoff: float = 1e-8


@dataclass(frozen=True)
class CVLCHSParams:
    r_target: float = 1.2
    r_prime: float = 0.3
    kernel_beta: float = 0.8

    # Displacement phase branch for the k-operator coupling.
    # We keep this explicit because phase choice was a historical failure mode.
    disp_phase: float = 0.0


def validate_config(cfg: HeatEquationConfig) -> None:
    if cfg.n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if cfg.max_fock_level <= 0 or (cfg.max_fock_level & (cfg.max_fock_level - 1)) != 0:
        raise ValueError("max_fock_level must be a positive power of two for the bosonic backend.")
    if cfg.n_dim <= 0:
        raise ValueError("n_dim must be positive.")
    if cfg.n_dim > cfg.max_fock_level:
        raise ValueError("n_dim must not exceed max_fock_level.")
    if len(cfg.init_qubits) != 2 or any(bit not in (0, 1) for bit in cfg.init_qubits):
        raise ValueError("init_qubits must be a 2-tuple with entries 0 or 1.")


def heat_matrix(h: float = 1.0) -> np.ndarray:
    # 4x4 second-difference matrix for 4 interior points.
    base = np.array(
        [[2.0, -1.0, 0.0, 0.0], [-1.0, 2.0, -1.0, 0.0], [0.0, -1.0, 2.0, -1.0], [0.0, 0.0, -1.0, 2.0]],
        dtype=float,
    )
    return base / (h**2)


def pauli_coefficients_for_heat(h: float = 1.0) -> Dict[str, float]:
    inv_h2 = 1.0 / (h**2)
    # T = 2(I⊗I) - (I⊗X) - 1/2(X⊗X + Y⊗Y)
    return {
        "I": 2.0 * inv_h2,
        "IX": -1.0 * inv_h2,
        "XX": -0.5 * inv_h2,
        "YY": -0.5 * inv_h2,
    }


def dv_generator_matrix(h: float = 1.0) -> np.ndarray:
    c = pauli_coefficients_for_heat(h)
    return (
        c["I"] * np.kron(PAULI_I, PAULI_I)
        + c["IX"] * np.kron(PAULI_I, PAULI_X)
        + c["XX"] * np.kron(PAULI_X, PAULI_X)
        + c["YY"] * np.kron(PAULI_Y, PAULI_Y)
    )


def commutator_constant(h: float = 1.0) -> float:
    c = pauli_coefficients_for_heat(h)
    terms = [
        c["I"] * np.kron(PAULI_I, PAULI_I),
        c["IX"] * np.kron(PAULI_I, PAULI_X),
        c["XX"] * np.kron(PAULI_X, PAULI_X),
        c["YY"] * np.kron(PAULI_Y, PAULI_Y),
    ]
    comm_sum = 0.0
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            comm = terms[i] @ terms[j] - terms[j] @ terms[i]
            comm_sum += float(np.linalg.norm(comm, ord=2))
    return comm_sum


def improved_kernel_function(k_points: np.ndarray, beta: float) -> np.ndarray:
    """
    Improved LCHS kernel branch used in the near-optimal differential-equation work:
        g(k) = exp(-(1 + i k)^beta) / (C_beta * (1 - i k)),
        C_beta = 2*pi*exp(-2*beta),  beta in (0, 1).
    """
    c_beta = 2.0 * np.pi * np.exp(-2.0 * beta)
    return np.exp(-((1.0 + 1.0j * k_points) ** beta)) / (c_beta * (1.0 - 1.0j * k_points))


def coefficient_gamma(r_target: float, r_prime: float) -> float:
    return float(np.exp(-2.0 * r_prime) - np.exp(-2.0 * r_target))


def displacement_phase_shifts(amplitude: float, phase: float) -> Tuple[float, float]:
    """
    Return (Delta x, Delta p) from hybridlane's own Heisenberg representation
    for D(a, phi), under the backend hbar=2 convention.
    """
    rep = hqml.Displacement._heisenberg_rep([float(amplitude), float(phase)])
    return float(rep[1, 0]), float(rep[2, 0])


def lchs_coefficients(
    r_target: float,
    r_prime: float,
    n_dim: int,
    kernel_beta: float,
    n_quad_points: int = 220,
) -> np.ndarray:
    r"""
    Gauss-Hermite projection for finite squeezed-Fock expansion coefficients C_n.

    This follows the project kernel branch used in prior experiments:
      C_n \propto \int H_n(k/e^{r'}) exp(-gamma k^2) g_beta(k) dk,
      gamma = e^{-2r'} - e^{-2r} > 0.
    """
    gamma = coefficient_gamma(r_target, r_prime)
    if gamma <= 0:
        raise ValueError("Invalid squeeze geometry: require r_prime < r_target so gamma > 0.")
    if not (0.0 < kernel_beta < 1.0):
        raise ValueError("kernel_beta must be in (0, 1) for the improved-kernel branch.")

    width_param = np.exp(r_prime)
    scale_factor = np.sqrt(2.0) * width_param

    roots, weights = hermgauss(n_quad_points)
    k_points = roots * scale_factor

    target_part = np.exp(-gamma * (k_points**2)) * improved_kernel_function(k_points, kernel_beta)

    sqrt_pi = np.sqrt(np.pi)
    basis_prefactor = 1.0 / np.sqrt(sqrt_pi * width_param)

    coeffs = np.zeros(n_dim, dtype=complex)
    for n in range(n_dim):
        fock_norm = 1.0 / np.sqrt((2**n) * factorial(n))
        herm = eval_hermite(n, k_points / width_param)
        integrand = target_part * (basis_prefactor * fock_norm * herm)
        coeffs[n] = np.sum(weights * integrand) * scale_factor

    norm = float(np.linalg.norm(coeffs))
    if np.isclose(norm, 0.0):
        raise RuntimeError("Computed coefficient vector has near-zero norm.")
    return coeffs / norm


def estimate_tail_mass(
    r_target: float,
    r_prime: float,
    kernel_beta: float,
    n_dim: int,
    ref_dim: int = 96,
    n_quad_points: int = 260,
) -> float:
    """
    Coefficient-space truncation diagnostic:
      tail_mass = 1 - sum_{n=0}^{n_dim-1} |C_n|^2,
    estimated from a higher-dimensional reference coefficient vector.
    """
    if ref_dim <= n_dim:
        return 0.0
    coeff_ref = lchs_coefficients(
        r_target=r_target,
        r_prime=r_prime,
        n_dim=ref_dim,
        kernel_beta=kernel_beta,
        n_quad_points=n_quad_points,
    )
    w_ref = np.abs(coeff_ref) ** 2
    in_mass = float(np.sum(w_ref[:n_dim]))
    return max(0.0, 1.0 - in_mass)


def pauli_observable(label: str, wire: int):
    if label == "X":
        return qml.PauliX(wires=wire)
    if label == "Y":
        return qml.PauliY(wires=wire)
    if label == "Z":
        return qml.PauliZ(wires=wire)
    raise ValueError(f"Unsupported non-identity label: {label}")


def rebuild_density_from_paulis(expectations: Sequence[float]) -> np.ndarray:
    rho = np.zeros((4, 4), dtype=complex)
    for value, (la, lb) in zip(expectations, PAULI_COMBOS):
        rho += float(value) * np.kron(PAULI_MATRICES[la], PAULI_MATRICES[lb])
    return rho / 4.0


def sanitize_density_matrix(rho: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, Dict[str, float]]:
    rho_h = 0.5 * (rho + rho.conj().T)
    tr_before = np.trace(rho_h)
    tr_before_real = float(np.real(tr_before))
    if np.isclose(tr_before_real, 0.0):
        raise RuntimeError("Reconstructed density matrix has near-zero trace.")

    rho_h = rho_h / tr_before_real
    eigvals, eigvecs = np.linalg.eigh(rho_h)
    eigvals_real = np.real(eigvals)

    min_eig_before = float(np.min(eigvals_real))
    clipped_weight = float(np.sum(np.clip(-eigvals_real, 0.0, None)))

    eigvals_clipped = np.clip(eigvals_real, 0.0, None)
    clipped_sum = float(np.sum(eigvals_clipped))
    if np.isclose(clipped_sum, 0.0):
        raise RuntimeError("Density matrix collapsed to zero after eigenvalue clipping.")

    rho_psd = eigvecs @ np.diag(eigvals_clipped / clipped_sum) @ eigvecs.conj().T
    rho_psd = 0.5 * (rho_psd + rho_psd.conj().T)

    info = {
        "trace_before": tr_before_real,
        "trace_after": float(np.real(np.trace(rho_psd))),
        "min_eig_before": min_eig_before,
        "clipped_weight": clipped_weight,
        "purity": float(np.real(np.trace(rho_psd @ rho_psd))),
    }
    if abs(np.imag(tr_before)) > eps:
        info["trace_imag_before"] = float(np.imag(tr_before))
    return rho_psd, info


def principal_statevector(rho: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(rho)
    idx = int(np.argmax(eigvals))
    vec = eigvecs[:, idx]
    return vec / np.linalg.norm(vec)


def initial_dv_state(init_qubits: Tuple[int, int]) -> np.ndarray:
    vec = np.zeros(4, dtype=complex)
    idx = init_qubits[0] * 2 + init_qubits[1]
    vec[idx] = 1.0
    return vec


class Heat1DLCHSSolver:
    """Circuit builder + evaluator for the 1D heat-equation experiment."""

    def __init__(self, cfg: HeatEquationConfig):
        validate_config(cfg)
        self.cfg = cfg
        self.coeffs = pauli_coefficients_for_heat(cfg.h)
        self.device = qml.device(
            "bosonicqiskit.hybrid",
            wires=[0, 1, "m0"],
            shots=None,
            max_fock_level=cfg.max_fock_level,
            hbar=cfg.hbar,
        )
        self.qnode = self._build_fock_component_qnode()

    def _build_fock_component_qnode(self):
        lam_I = self.coeffs["I"]
        lam_X1 = self.coeffs["IX"]
        lam_XX = self.coeffs["XX"]
        lam_YY = self.coeffs["YY"]

        @qml.qnode(self.device)
        def fock_component(
            fock_n: int,
            total_time: float,
            n_steps: int,
            init_q0: int,
            init_q1: int,
            r_target: float,
            r_prime: float,
            disp_phase: float,
        ):
            mode = "m0"

            # Prepare |n> via hybridlane FockState template (requires ancilla+mode wires).
            hqml.FockState(int(fock_n), wires=[0, mode])

            # Preparation squeezing for the basis state branch.
            if not np.isclose(r_prime, 0.0):
                qml.Squeezing(r_prime, 0.0, wires=mode)

            if int(init_q0) == 1:
                qml.PauliX(0)
            if int(init_q1) == 1:
                qml.PauliX(1)

            dt = total_time / n_steps
            for _ in range(n_steps):
                # D(a,phi) = exp(alpha a^\dagger - alpha* a), alpha=a e^{i phi}.
                # Verified by tests: Delta x = 2a cos(phi), Delta p = 2a sin(phi).
                hqml.Displacement(lam_I * dt, disp_phase, wires=mode)

                # exp(-i dt * k_hat * lam_X1 * (I \otimes X))
                qml.H(1)
                hqml.ConditionalDisplacement(lam_X1 * dt, disp_phase, wires=[1, mode])
                qml.H(1)

                # exp(-i dt * k_hat * lam_XX * (X \otimes X))
                qml.H(0)
                qml.H(1)
                qml.CNOT(wires=[0, 1])
                hqml.ConditionalDisplacement(lam_XX * dt, disp_phase, wires=[1, mode])
                qml.CNOT(wires=[0, 1])
                qml.H(0)
                qml.H(1)

                # exp(-i dt * k_hat * lam_YY * (Y \otimes Y)) via basis rotation to X \otimes X frame.
                qml.RZ(np.pi / 2, wires=0)
                qml.RZ(np.pi / 2, wires=1)
                qml.H(0)
                qml.H(1)
                qml.CNOT(wires=[0, 1])
                hqml.ConditionalDisplacement(lam_YY * dt, disp_phase, wires=[1, mode])
                qml.CNOT(wires=[0, 1])
                qml.H(0)
                qml.H(1)
                qml.RZ(-np.pi / 2, wires=0)
                qml.RZ(-np.pi / 2, wires=1)

            # Post-selection onto S(r_target)|0> is implemented by S(-r_target) then |0><0| projector.
            if not np.isclose(r_target, 0.0):
                qml.Squeezing(-r_target, 0.0, wires=mode)

            proj = hqml.FockStateProjector(np.array([0]), wires=mode)
            measurements = [qml.expval(proj)]

            for la, lb in PAULI_COMBOS:
                obs = proj
                if la != "I":
                    obs = obs @ pauli_observable(la, 0)
                if lb != "I":
                    obs = obs @ pauli_observable(lb, 1)
                measurements.append(qml.expval(obs))

            return tuple(measurements)

        return fock_component

    def collect_component_table(
        self,
        r_target: float,
        r_prime: float,
        disp_phase: float,
        levels: Optional[Iterable[int]] = None,
    ) -> Dict[str, np.ndarray]:
        n_dim = self.cfg.n_dim
        post_probs = np.zeros(n_dim, dtype=float)
        paulis = np.zeros((n_dim, len(PAULI_COMBOS)), dtype=float)

        if levels is None:
            level_iter = range(n_dim)
        else:
            level_iter = sorted(set(int(i) for i in levels if 0 <= int(i) < n_dim))

        for n in level_iter:
            res = self.qnode(
                fock_n=int(n),
                total_time=self.cfg.total_time,
                n_steps=self.cfg.n_steps,
                init_q0=int(self.cfg.init_qubits[0]),
                init_q1=int(self.cfg.init_qubits[1]),
                r_target=float(r_target),
                r_prime=float(r_prime),
                disp_phase=float(disp_phase),
            )
            post_probs[n] = float(res[0])
            paulis[n, :] = np.asarray(res[1:], dtype=float)

        return {"post_probs": post_probs, "paulis": paulis}

    def evaluate(self, params: CVLCHSParams, component_table: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
        coeffs = lchs_coefficients(
            r_target=params.r_target,
            r_prime=params.r_prime,
            n_dim=self.cfg.n_dim,
            kernel_beta=params.kernel_beta,
        )
        weights = np.abs(coeffs) ** 2
        weights = weights / float(np.sum(weights))

        if component_table is None:
            active_levels = np.where(weights >= self.cfg.fock_weight_cutoff)[0]
            component_table = self.collect_component_table(
                r_target=params.r_target,
                r_prime=params.r_prime,
                disp_phase=params.disp_phase,
                levels=active_levels,
            )

        post_probs_n = component_table["post_probs"]
        paulis_n = component_table["paulis"]

        post_prob = float(np.dot(weights, post_probs_n))
        if np.isclose(post_prob, 0.0):
            raise RuntimeError("Post-selection probability is zero; cannot normalize.")

        pauli_weighted = np.tensordot(weights, paulis_n, axes=(0, 0))
        rho_raw = rebuild_density_from_paulis(pauli_weighted) / post_prob
        rho_post, rho_info = sanitize_density_matrix(rho_raw)

        t_matrix = dv_generator_matrix(self.cfg.h)
        u0 = initial_dv_state(self.cfg.init_qubits)
        u_target = expm(-self.cfg.alpha * self.cfg.total_time * t_matrix) @ u0
        norm_target = float(np.linalg.norm(u_target))
        if np.isclose(norm_target, 0.0):
            raise RuntimeError("Classical target norm is zero.")

        u_hat = u_target / norm_target
        rho_target = np.outer(u_hat, u_hat.conj())

        fidelity = float(np.real(np.vdot(u_hat, rho_post @ u_hat)))
        fidelity = float(np.clip(fidelity, 0.0, 1.0))

        rho_delta = 0.5 * ((rho_post - rho_target) + (rho_post - rho_target).conj().T)
        eig_delta = np.linalg.eigvalsh(rho_delta)
        trace_distance = float(0.5 * np.sum(np.abs(eig_delta)))
        hs_distance = float(np.linalg.norm(rho_post - rho_target, ord="fro"))

        psi_principal = principal_statevector(rho_post)
        overlap = np.vdot(u_hat, psi_principal)
        if np.abs(overlap) > 0.0:
            psi_principal = psi_principal * np.exp(-1j * np.angle(overlap))

        u_cvdv = np.sqrt(post_prob) * psi_principal
        pde_error = float(np.linalg.norm(u_target - u_cvdv) / norm_target)

        pauli_conditional = np.asarray(pauli_weighted, dtype=float) / post_prob
        pauli_target = []
        for la, lb in PAULI_COMBOS:
            pmat = np.kron(PAULI_MATRICES[la], PAULI_MATRICES[lb])
            pauli_target.append(float(np.real(np.trace(rho_target @ pmat))))
        pauli_target = np.asarray(pauli_target, dtype=float)
        pauli_err = pauli_conditional - pauli_target
        pauli_rmse = float(np.sqrt(np.mean(pauli_err**2)))
        pauli_max_abs = float(np.max(np.abs(pauli_err)))

        gamma = coefficient_gamma(params.r_target, params.r_prime)
        n_eff = float(1.0 / np.sum(weights**2))
        tail_mass = estimate_tail_mass(
            r_target=params.r_target,
            r_prime=params.r_prime,
            kernel_beta=params.kernel_beta,
            n_dim=self.cfg.n_dim,
            ref_dim=max(96, 2 * self.cfg.n_dim),
        )

        c_comm = commutator_constant(self.cfg.h)
        trotter_bound = (self.cfg.alpha * self.cfg.total_time) ** 2 * c_comm / (2.0 * self.cfg.n_steps)

        return {
            "post_prob": post_prob,
            "pde_error": pde_error,
            "fidelity": fidelity,
            "infidelity": 1.0 - fidelity,
            "trace_distance": trace_distance,
            "hs_distance": hs_distance,
            "pauli_rmse": pauli_rmse,
            "pauli_max_abs": pauli_max_abs,
            "gamma": float(gamma),
            "n_eff": n_eff,
            "tail_mass_est": tail_mass,
            "trotter_bound_est": float(trotter_bound),
            "rho_trace_before": rho_info["trace_before"],
            "rho_trace_after": rho_info["trace_after"],
            "rho_min_eig_before": rho_info["min_eig_before"],
            "rho_purity": rho_info["purity"],
            "weights_l1": float(np.sum(np.abs(weights))),
            "weights_l2_sq": float(np.sum(weights**2)),
            "active_levels": float(np.sum(weights >= self.cfg.fock_weight_cutoff)),
        }


def phase_calibration(
    solver: Heat1DLCHSSolver,
    base_params: CVLCHSParams,
    candidates: Sequence[float],
) -> Tuple[float, Dict[float, Dict[str, float]]]:
    results: Dict[float, Dict[str, float]] = {}
    best_phase = float(candidates[0])
    best_err = float("inf")

    for ph in candidates:
        params = CVLCHSParams(
            r_target=base_params.r_target,
            r_prime=base_params.r_prime,
            kernel_beta=base_params.kernel_beta,
            disp_phase=float(ph),
        )
        metrics = solver.evaluate(params)
        results[float(ph)] = metrics
        if metrics["pde_error"] < best_err:
            best_err = metrics["pde_error"]
            best_phase = float(ph)

    return best_phase, results


def print_problem_statement(cfg: HeatEquationConfig) -> None:
    print("=== 1) Original problem statement and equation ===")
    print("Source baseline: res_base/Hybrid_CV_DV_LCHS (2).pdf, Sec. 3.5.2, Eq. (41)-(46)")
    print("Continuous PDE:")
    print("  d/dt u(x,t) = alpha * d^2/dx^2 u(x,t),  u(0,t)=u(L,t)=0")
    print("Finite-difference ODE (4 interior points):")
    print("  d/dt u(t) = -alpha * T * u(t),  with h=1")
    print(np.array2string(heat_matrix(cfg.h), precision=6, suppress_small=True))
    print("Pauli decomposition:")
    print("  T = 2(I⊗I) - (I⊗X) - 1/2[(X⊗X) + (Y⊗Y)]")


def print_phase_audit(phase: float) -> None:
    dx, dp = displacement_phase_shifts(1.0, phase)
    if abs(dx) >= abs(dp):
        dominant = "x-dominant"
    else:
        dominant = "p-dominant"
    print("\n=== 1b) Gate-convention audit ===")
    print("Using hybridlane Displacement Heisenberg map (amplitude=1):")
    print(f"  phase={phase:+.6f}  Delta x={dx:+.6f}  Delta p={dp:+.6f}  ({dominant})")
    print("Reference relation: Delta x = 2a cos(phi), Delta p = 2a sin(phi)")


def print_run_summary(cfg: HeatEquationConfig, params: CVLCHSParams, metrics: Dict[str, float]) -> None:
    print("\n=== 2) Settings ===")
    print(json.dumps({"config": asdict(cfg), "params": asdict(params)}, indent=2))

    print("\n=== 3) Correctness metrics (PDE-priority) ===")
    print(f"post_prob:        {metrics['post_prob']:.12e}")
    print(f"pde_error:        {metrics['pde_error']:.12e}")
    print(f"fidelity:         {metrics['fidelity']:.12e}")
    print(f"trace_distance:   {metrics['trace_distance']:.12e}")
    print(f"pauli_rmse:       {metrics['pauli_rmse']:.12e}")

    print("\n=== 4) Theory diagnostics ===")
    print(f"gamma=e^(-2r')-e^(-2r):   {metrics['gamma']:.12e}")
    print(f"n_eff=1/sum|C_n|^4:        {metrics['n_eff']:.12e}")
    print(f"tail_mass_est(N={cfg.n_dim}): {metrics['tail_mass_est']:.12e}")
    print(f"trotter_bound_est:         {metrics['trotter_bound_est']:.12e}")
    print(f"purity Tr(rho^2):          {metrics['rho_purity']:.12e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1D heat equation CV-DV LCHS run")

    parser.add_argument("--total-time", type=float, default=1.0)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--init-q0", type=int, default=0, choices=[0, 1])
    parser.add_argument("--init-q1", type=int, default=1, choices=[0, 1])

    parser.add_argument("--n-dim", type=int, default=32)
    parser.add_argument("--max-fock-level", type=int, default=32)
    parser.add_argument("--fock-weight-cutoff", type=float, default=1e-8)

    parser.add_argument("--r-target", type=float, default=1.2)
    parser.add_argument("--r-prime", type=float, default=0.3)
    parser.add_argument("--kernel-beta", type=float, default=0.8)
    parser.add_argument(
        "--disp-phase",
        type=float,
        default=0.0,
        help="Displacement phase phi. Tested mapping: Delta x=2a cos(phi), Delta p=2a sin(phi).",
    )

    parser.add_argument(
        "--calibrate-phase",
        action="store_true",
        help="Evaluate phase candidates {0, -pi/2, +pi/2} and choose lowest PDE error.",
    )
    parser.add_argument("--output-json", type=str, default="", help="Optional path to save metrics JSON.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = HeatEquationConfig(
        total_time=args.total_time,
        n_steps=args.n_steps,
        alpha=args.alpha,
        h=args.h,
        init_qubits=(args.init_q0, args.init_q1),
        n_dim=args.n_dim,
        max_fock_level=args.max_fock_level,
        fock_weight_cutoff=args.fock_weight_cutoff,
    )

    params = CVLCHSParams(
        r_target=args.r_target,
        r_prime=args.r_prime,
        kernel_beta=args.kernel_beta,
        disp_phase=args.disp_phase,
    )

    solver = Heat1DLCHSSolver(cfg)

    print_problem_statement(cfg)
    print_phase_audit(params.disp_phase)

    if args.calibrate_phase:
        print("\n=== Phase calibration (historical bug guard) ===")
        phase_candidates = [0.0, -np.pi / 2.0, np.pi / 2.0]
        best_phase, table = phase_calibration(solver, params, phase_candidates)
        for ph in phase_candidates:
            m = table[float(ph)]
            dx, dp = displacement_phase_shifts(1.0, ph)
            print(
                f"phase={ph:+.6f}  pde_error={m['pde_error']:.6e}  "
                f"post_prob={m['post_prob']:.6e}  fidelity={m['fidelity']:.6e}  "
                f"(Delta x={dx:+.3f}, Delta p={dp:+.3f})"
            )
        print(f"Selected phase by PDE error: {best_phase:+.6f}")
        params = CVLCHSParams(
            r_target=params.r_target,
            r_prime=params.r_prime,
            kernel_beta=params.kernel_beta,
            disp_phase=best_phase,
        )
        print_phase_audit(params.disp_phase)

    metrics = solver.evaluate(params)
    print_run_summary(cfg, params, metrics)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(cfg),
            "params": asdict(params),
            "metrics": metrics,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved JSON: {out_path}")


if __name__ == "__main__":
    main()
