#!/usr/bin/env python3
"""
1D heat equation via CV-DV LCHS (bosonic-qiskit only, injected CV state).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from bosonic_qiskit import CVCircuit, QumodeRegister
from bosonic_qiskit import util as cv_util
from numpy.polynomial.hermite import hermgauss
from qiskit import QuantumRegister
from scipy.linalg import expm
from scipy.special import eval_hermite, factorial

from heat1d_bosonic_state_prep import (
    CVStatePreparationStrategy,
    ClassicalInjectionStatePreparation,
    GateBasedStatePreparation,
    PreparedCVState,
)

PAULI_I = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_LABELS = ("I", "X", "Y", "Z")
PAULI_MATRICES = {"I": PAULI_I, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}
PAULI_COMBOS = [(a, b) for a in PAULI_LABELS for b in PAULI_LABELS]


@dataclass(frozen=True)
class HeatEquationConfig:
    total_time: float = 1.0
    n_steps: int = 100
    alpha: float = 1.0
    h: float = 1.0
    init_qubits: Tuple[int, int] = (0, 1)
    n_dim: int = 32
    max_fock_level: int = 32
    hbar: float = 2.0
    fock_weight_cutoff: float = 1e-8


@dataclass(frozen=True)
class CVLCHSParams:
    r_target: float = 1.2
    r_prime: float = 0.3
    kernel_beta: float = 0.8
    disp_phase: float = 0.0


def validate_config(cfg: HeatEquationConfig) -> None:
    if cfg.n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    if cfg.max_fock_level <= 0 or (cfg.max_fock_level & (cfg.max_fock_level - 1)) != 0:
        raise ValueError("max_fock_level must be a positive power of two.")
    if cfg.n_dim <= 0:
        raise ValueError("n_dim must be positive.")
    if cfg.n_dim > cfg.max_fock_level:
        raise ValueError("n_dim must not exceed max_fock_level.")
    if len(cfg.init_qubits) != 2 or any(bit not in (0, 1) for bit in cfg.init_qubits):
        raise ValueError("init_qubits must be a 2-tuple with entries 0 or 1.")


def heat_matrix(h: float = 1.0) -> np.ndarray:
    base = np.array(
        [[2.0, -1.0, 0.0, 0.0], [-1.0, 2.0, -1.0, 0.0], [0.0, -1.0, 2.0, -1.0], [0.0, 0.0, -1.0, 2.0]],
        dtype=float,
    )
    return base / (h**2)


def pauli_coefficients_for_heat(h: float = 1.0) -> Dict[str, float]:
    inv_h2 = 1.0 / (h**2)
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
    c_beta = 2.0 * np.pi * np.exp(-2.0 * beta)
    return np.exp(-((1.0 + 1.0j * k_points) ** beta)) / (c_beta * (1.0 - 1.0j * k_points))


def coefficient_gamma(r_target: float, r_prime: float) -> float:
    return float(np.exp(-2.0 * r_prime) - np.exp(-2.0 * r_target))


def displacement_phase_shifts(amplitude: float, phase: float) -> Tuple[float, float]:
    # hbar=2 map for alpha = a exp(i phi): dx=2a cos(phi), dp=2a sin(phi)
    amp = float(amplitude)
    phi = float(phase)
    return float(2.0 * amp * np.cos(phi)), float(2.0 * amp * np.sin(phi))


def lchs_coefficients(
    r_target: float,
    r_prime: float,
    n_dim: int,
    kernel_beta: float,
    n_quad_points: int = 220,
) -> np.ndarray:
    gamma = coefficient_gamma(r_target, r_prime)
    if gamma <= 0:
        raise ValueError("Invalid geometry: require r_prime < r_target so gamma > 0.")
    if not (0.0 < kernel_beta < 1.0):
        raise ValueError("kernel_beta must be in (0, 1).")

    width_param = np.exp(r_prime)
    scale_factor = np.sqrt(2.0) * width_param

    roots, weights = hermgauss(n_quad_points)
    k_points = roots * scale_factor

    # Keep multi-line expression split into explicit factors so the code mirrors
    # the math clearly for readers:
    # target_part = exp(-gamma k^2) * g_beta(k)
    # basis_part  = H_n(k/e^{r'}) / sqrt(pi^(1/2) e^{r'} 2^n n!)
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
    if ref_dim <= n_dim:
        return 0.0
    coeff_ref = lchs_coefficients(
        r_target=r_target,
        r_prime=r_prime,
        n_dim=ref_dim,
        kernel_beta=kernel_beta,
        n_quad_points=n_quad_points,
    )
    in_mass = float(np.sum(np.abs(coeff_ref[:n_dim]) ** 2))
    return max(0.0, 1.0 - in_mass)


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

    eigvals_clipped = np.clip(eigvals_real, 0.0, None)
    clipped_weight = float(np.sum(eigvals_clipped))
    if clipped_weight <= eps:
        raise RuntimeError("All eigenvalues were clipped; cannot build a physical state.")

    eigvals_clipped = eigvals_clipped / clipped_weight
    rho_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.conj().T
    rho_psd = rho_psd / float(np.real(np.trace(rho_psd)))

    return rho_psd, {
        "trace_before": tr_before_real,
        "trace_after": float(np.real(np.trace(rho_psd))),
        "min_eig_before": min_eig_before,
        "clipped_weight": clipped_weight,
        "purity": float(np.real(np.trace(rho_psd @ rho_psd))),
    }


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


class Heat1DBosonicSolver:
    def __init__(
        self,
        cfg: HeatEquationConfig,
        state_preparer: Optional[CVStatePreparationStrategy] = None,
    ):
        validate_config(cfg)
        self.cfg = cfg
        self.coeffs = pauli_coefficients_for_heat(cfg.h)
        self.num_mode_qubits = int(np.log2(cfg.max_fock_level))
        self._state_layout = self._detect_statevector_layout()
        if state_preparer is None:
            state_preparer = ClassicalInjectionStatePreparation(self._coefficients_for_prepare)
        self.state_preparer = state_preparer

    def _coefficients_for_prepare(
        self,
        r_target: float,
        r_prime: float,
        n_dim: int,
        kernel_beta: float,
    ) -> np.ndarray:
        return lchs_coefficients(r_target, r_prime, n_dim, kernel_beta)

    def _detect_statevector_layout(self) -> str:
        """
        Detect whether backend statevector indexing is:
        - fock-major: index = qubit_index + 4 * fock_index
        - qubit-major: index = fock_index + max_fock_level * qubit_index

        We prepare |fock=1> and |q0 q1>=|00>; the nonzero index is:
        - 4 for fock-major,
        - 1 for qubit-major.
        """
        qmr = QumodeRegister(num_qumodes=1, num_qubits_per_qumode=self.num_mode_qubits)
        qbr = QuantumRegister(2, "q")
        qc = CVCircuit(qbr, qmr)
        mode = qmr[0]

        probe = np.zeros(self.cfg.max_fock_level, dtype=complex)
        probe[1] = 1.0
        qc.cv_initialize(probe, mode)

        state, _, _ = cv_util.simulate(
            qc,
            shots=1,
            return_fockcounts=False,
            add_save_statevector=True,
        )
        if state is None:
            raise RuntimeError("Failed to detect statevector layout: simulation returned no statevector.")

        vec = np.asarray(state.data, dtype=complex)
        nz = np.where(np.abs(vec) > 1e-12)[0]
        if len(nz) != 1:
            raise RuntimeError(f"Failed to detect statevector layout: expected one nonzero, got {len(nz)}.")

        idx = int(nz[0])
        if idx == 4:
            return "fock_major"
        if idx == 1:
            return "qubit_major"
        raise RuntimeError(f"Unrecognized statevector layout probe index {idx}.")

    def _extract_fock0_dv_qiskit_order(self, vec: np.ndarray) -> np.ndarray:
        """
        Extract the DV amplitude block for postselected fock=0 in qiskit qubit order.
        """
        if self._state_layout == "fock_major":
            return vec[:4]
        # qubit-major layout: take fock=0 entry from each qubit basis block.
        return vec[0:: self.cfg.max_fock_level][:4]

    def build_circuit(self, params: CVLCHSParams) -> Tuple[CVCircuit, PreparedCVState]:
        qmr = QumodeRegister(num_qumodes=1, num_qubits_per_qumode=self.num_mode_qubits)
        qbr = QuantumRegister(2, "q")
        qc = CVCircuit(qbr, qmr)
        mode = qmr[0]

        prepared = self.state_preparer.prepare(
            qc,
            mode,
            r_target=float(params.r_target),
            r_prime=float(params.r_prime),
            kernel_beta=float(params.kernel_beta),
            n_dim=self.cfg.n_dim,
            max_fock_level=self.cfg.max_fock_level,
            fock_weight_cutoff=self.cfg.fock_weight_cutoff,
        )

        if int(self.cfg.init_qubits[0]) == 1:
            qc.x(qbr[0])
        if int(self.cfg.init_qubits[1]) == 1:
            qc.x(qbr[1])

        dt = float(self.cfg.total_time) / float(self.cfg.n_steps)
        phase_factor = np.exp(1j * float(params.disp_phase))
        lam_i = self.coeffs["I"] * dt
        lam_x1 = self.coeffs["IX"] * dt
        lam_xx = self.coeffs["XX"] * dt
        lam_yy = self.coeffs["YY"] * dt

        for _ in range(self.cfg.n_steps):
            qc.cv_d(lam_i * phase_factor, mode)

            qc.h(qbr[1])
            qc.cv_c_d(lam_x1 * phase_factor, mode, qbr[1])
            qc.h(qbr[1])

            qc.h(qbr[0])
            qc.h(qbr[1])
            qc.cx(qbr[0], qbr[1])
            qc.cv_c_d(lam_xx * phase_factor, mode, qbr[1])
            qc.cx(qbr[0], qbr[1])
            qc.h(qbr[0])
            qc.h(qbr[1])

            qc.rz(np.pi / 2, qbr[0])
            qc.rz(np.pi / 2, qbr[1])
            qc.h(qbr[0])
            qc.h(qbr[1])
            qc.cx(qbr[0], qbr[1])
            qc.cv_c_d(lam_yy * phase_factor, mode, qbr[1])
            qc.cx(qbr[0], qbr[1])
            qc.h(qbr[0])
            qc.h(qbr[1])
            qc.rz(-np.pi / 2, qbr[0])
            qc.rz(-np.pi / 2, qbr[1])

        if not np.isclose(float(params.r_target), 0.0):
            qc.cv_sq(float(params.r_target), mode)

        return qc, prepared

    def _simulate_postselected_dv(self, params: CVLCHSParams) -> Tuple[np.ndarray, PreparedCVState]:
        qc, prepared = self.build_circuit(params)
        state, _, _ = cv_util.simulate(qc, shots=1, return_fockcounts=False, add_save_statevector=True)
        if state is None:
            raise RuntimeError("Bosonic-Qiskit simulation did not return a statevector.")

        vec = np.asarray(state.data, dtype=complex)
        expected_size = 4 * self.cfg.max_fock_level
        if vec.size != expected_size:
            raise RuntimeError(f"Unexpected statevector size {vec.size}; expected {expected_size}.")

        dv_qiskit = self._extract_fock0_dv_qiskit_order(vec)
        dv_order = dv_qiskit[[0, 2, 1, 3]]
        return dv_order, prepared

    def evaluate(self, params: CVLCHSParams, return_vectors: bool = False) -> Dict[str, float]:
        dv_unnorm, prepared = self._simulate_postselected_dv(params)

        coeffs = np.asarray(prepared.coeffs, dtype=complex)
        weights = np.abs(coeffs) ** 2
        weights = weights / float(np.sum(weights))

        post_prob = float(np.real(np.vdot(dv_unnorm, dv_unnorm)))
        if np.isclose(post_prob, 0.0):
            raise RuntimeError("Post-selection probability is zero; cannot normalize.")

        rho_raw = np.outer(dv_unnorm, dv_unnorm.conj()) / post_prob
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

        # PDE proxy metric (vector form):
        # 1) extract principal eigenvector,
        # 2) remove global phase ambiguity,
        # 3) compare u_target with sqrt(p_post) * psi_principal.
        psi_principal = principal_statevector(rho_post)
        overlap = np.vdot(u_hat, psi_principal)
        if np.abs(overlap) > 0.0:
            psi_principal = psi_principal * np.exp(-1j * np.angle(overlap))
        u_cvdv = np.sqrt(post_prob) * psi_principal
        pde_error = float(np.linalg.norm(u_target - u_cvdv) / norm_target)

        pauli_target = []
        pauli_conditional = []
        for la, lb in PAULI_COMBOS:
            pmat = np.kron(PAULI_MATRICES[la], PAULI_MATRICES[lb])
            pauli_target.append(float(np.real(np.trace(rho_target @ pmat))))
            pauli_conditional.append(float(np.real(np.trace(rho_post @ pmat))))
        pauli_target = np.asarray(pauli_target, dtype=float)
        pauli_conditional = np.asarray(pauli_conditional, dtype=float)
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

        metrics: Dict[str, float] = {
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
            "active_levels": float(prepared.active_levels),
            "state_prep_method": prepared.method,
        }
        if return_vectors:
            # These are CHO-style validation vectors: direct classical target and
            # quantum proxy at the same time point, both in 4-component DV space.
            metrics["u_target_real_0"] = float(np.real(u_target[0]))
            metrics["u_target_real_1"] = float(np.real(u_target[1]))
            metrics["u_target_real_2"] = float(np.real(u_target[2]))
            metrics["u_target_real_3"] = float(np.real(u_target[3]))
            metrics["u_target_imag_0"] = float(np.imag(u_target[0]))
            metrics["u_target_imag_1"] = float(np.imag(u_target[1]))
            metrics["u_target_imag_2"] = float(np.imag(u_target[2]))
            metrics["u_target_imag_3"] = float(np.imag(u_target[3]))

            metrics["u_quantum_real_0"] = float(np.real(u_cvdv[0]))
            metrics["u_quantum_real_1"] = float(np.real(u_cvdv[1]))
            metrics["u_quantum_real_2"] = float(np.real(u_cvdv[2]))
            metrics["u_quantum_real_3"] = float(np.real(u_cvdv[3]))
            metrics["u_quantum_imag_0"] = float(np.imag(u_cvdv[0]))
            metrics["u_quantum_imag_1"] = float(np.imag(u_cvdv[1]))
            metrics["u_quantum_imag_2"] = float(np.imag(u_cvdv[2]))
            metrics["u_quantum_imag_3"] = float(np.imag(u_cvdv[3]))
        return metrics


def create_state_preparer(state_prep: str) -> CVStatePreparationStrategy:
    tag = str(state_prep).strip().lower().replace("_", "-")
    if tag in {"classical-injection", "classical", "injected"}:
        return ClassicalInjectionStatePreparation(lchs_coefficients)
    if tag in {"gate-based", "gate"}:
        return GateBasedStatePreparation()
    raise ValueError("Unsupported state preparation. Choose 'classical-injection' or 'gate-based'.")


def create_solver(
    cfg: HeatEquationConfig,
    state_prep: str = "classical-injection",
) -> Heat1DBosonicSolver:
    return Heat1DBosonicSolver(cfg, state_preparer=create_state_preparer(state_prep))


def phase_calibration(
    solver: Heat1DBosonicSolver,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1D heat equation solver (bosonic-qiskit, injected CV state)")

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
    parser.add_argument("--disp-phase", type=float, default=0.0)

    parser.add_argument(
        "--state-prep",
        type=str,
        default="classical-injection",
        choices=["classical-injection", "gate-based"],
    )
    parser.add_argument("--calibrate-phase", action="store_true")
    parser.add_argument("--output-json", type=str, default="")

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

    solver = create_solver(cfg, state_prep=args.state_prep)

    print("=== Bosonic-Qiskit PDE Solver (Injected CV State) ===")
    print(json.dumps({"config": asdict(cfg), "params": asdict(params)}, indent=2))

    dx, dp = displacement_phase_shifts(1.0, params.disp_phase)
    print(f"Phase audit: phase={params.disp_phase:+.6f} -> Delta x={dx:+.6f}, Delta p={dp:+.6f}")

    if args.calibrate_phase:
        print("Phase calibration over {0, -pi/2, +pi/2}")
        best_phase, table = phase_calibration(solver, params, [0.0, -np.pi / 2.0, np.pi / 2.0])
        for ph, met in table.items():
            print(
                f"  phase={ph:+.6f} pde_error={met['pde_error']:.6e} "
                f"post_prob={met['post_prob']:.6e} fidelity={met['fidelity']:.6e}"
            )
        params = CVLCHSParams(
            r_target=params.r_target,
            r_prime=params.r_prime,
            kernel_beta=params.kernel_beta,
            disp_phase=best_phase,
        )
        print(f"Selected phase={best_phase:+.6f}")

    metrics = solver.evaluate(params)
    print("=== Metrics ===")
    for k in (
        "post_prob",
        "pde_error",
        "fidelity",
        "trace_distance",
        "pauli_rmse",
        "tail_mass_est",
        "n_eff",
        "trotter_bound_est",
        "state_prep_method",
    ):
        v = metrics[k]
        if isinstance(v, str):
            print(f"{k}: {v}")
        else:
            print(f"{k}: {v:.12e}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "config": asdict(cfg),
                    "params": asdict(params),
                    "metrics": metrics,
                },
                indent=2,
            )
        )
        print(f"Saved JSON: {out_path}")


if __name__ == "__main__":
    main()
