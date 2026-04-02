#!/usr/bin/env python3
"""
Independent CV oracle preparation and readout helpers for clean CV-DV LCHS.

This module owns:
  - kernel coefficient access,
  - independent CV state-prep paths,
  - statevector and density-matrix readout utilities.

It does not import any repo-local implementation files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from clean_core import (
    KernelSpec,
    PauliSystemSpec,
    StatePrepSpec,
    compute_lchs_coefficients as core_compute_lchs_coefficients,
    normalize_vector,
    padded_seed_state,
    physics_to_qiskit_permutation,
    state_fidelity,
)


@dataclass(frozen=True)
class GivensRotation:
    level: int
    theta: float
    phi: float


@dataclass
class OraclePreparation:
    method: str
    target_state: np.ndarray
    prepared_state: np.ndarray
    apply_mode: str
    oracle_fidelity: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    snap_thetas_per_layer: Tuple[np.ndarray, ...] = ()
    snap_alphas_per_layer: Tuple[complex, ...] = ()
    givens_rotations: Tuple[GivensRotation, ...] = ()


def compute_lchs_coefficients(kernel: KernelSpec) -> np.ndarray:
    return core_compute_lchs_coefficients(kernel)


def annihilation_operator(n_fock: int) -> np.ndarray:
    op = np.zeros((n_fock, n_fock), dtype=complex)
    for n in range(1, n_fock):
        op[n - 1, n] = np.sqrt(n)
    return op


def displacement_matrix(alpha: complex, n_fock: int) -> np.ndarray:
    a = annihilation_operator(n_fock)
    adag = a.conj().T
    return expm(alpha * adag - np.conj(alpha) * a)


def simulate_snap_d_state(
    params: np.ndarray,
    *,
    n_fock: int,
    depth: int,
    n_snap: int,
) -> np.ndarray:
    state = np.zeros(n_fock, dtype=complex)
    state[0] = 1.0
    params_per_layer = n_snap + 2

    for layer in range(depth):
        offset = layer * params_per_layer
        thetas = params[offset : offset + n_snap]
        alpha = complex(params[offset + n_snap], params[offset + n_snap + 1])

        state[:n_snap] *= np.exp(1.0j * thetas)
        state = displacement_matrix(alpha, n_fock) @ state

    return normalize_vector(state)


def optimize_snap_d(
    target_state: np.ndarray,
    *,
    n_fock: int,
    depth: int,
    n_snap: int,
    n_restarts: int,
    maxiter: int,
) -> OraclePreparation:
    if depth <= 0:
        raise ValueError("snap_d requires snap_depth > 0.")

    target = padded_seed_state(target_state, n_fock)
    params_per_layer = n_snap + 2
    n_params = depth * params_per_layer

    def _cost(x: np.ndarray) -> float:
        prepared = simulate_snap_d_state(x, n_fock=n_fock, depth=depth, n_snap=n_snap)
        return 1.0 - abs(np.vdot(target, prepared)) ** 2

    best_x: Optional[np.ndarray] = None
    best_cost = float("inf")
    total_iterations = 0

    for _ in range(n_restarts):
        guess = np.zeros(n_params, dtype=float)
        for layer in range(depth):
            offset = layer * params_per_layer
            guess[offset : offset + n_snap] = np.random.uniform(-np.pi, np.pi, size=n_snap)
            guess[offset + n_snap] = np.random.uniform(-0.5, 0.5)
            guess[offset + n_snap + 1] = np.random.uniform(-0.5, 0.5)

        result = minimize(
            _cost,
            guess,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-10},
        )
        total_iterations += int(getattr(result, "nit", 0))
        if float(result.fun) < best_cost:
            best_cost = float(result.fun)
            best_x = np.asarray(result.x, dtype=float)

    if best_x is None:
        raise RuntimeError("SNAP+D optimization failed to produce a result.")

    prepared = simulate_snap_d_state(best_x, n_fock=n_fock, depth=depth, n_snap=n_snap)
    thetas: List[np.ndarray] = []
    alphas: List[complex] = []
    for layer in range(depth):
        offset = layer * params_per_layer
        thetas.append(best_x[offset : offset + n_snap].copy())
        alphas.append(complex(best_x[offset + n_snap], best_x[offset + n_snap + 1]))

    fidelity = abs(np.vdot(target, prepared)) ** 2
    return OraclePreparation(
        method="snap_d",
        target_state=target,
        prepared_state=prepared,
        apply_mode="snap_d_layers",
        oracle_fidelity=float(fidelity),
        metadata={
            "snap_depth": int(depth),
            "snap_restarts": int(n_restarts),
            "snap_maxiter": int(maxiter),
            "snap_n_snap": int(n_snap),
            "snap_total_iterations": int(total_iterations),
        },
        snap_thetas_per_layer=tuple(thetas),
        snap_alphas_per_layer=tuple(alphas),
    )


def givens_decomposition(target_state: np.ndarray, *, n_fock: int) -> OraclePreparation:
    target = padded_seed_state(target_state, n_fock)

    n_active = 0
    for idx in range(n_fock - 1, -1, -1):
        if abs(target[idx]) > 1e-15:
            n_active = idx + 1
            break
    if n_active == 0:
        raise ValueError("Target state has zero norm.")

    state = target.copy()
    inverse_rotations: List[GivensRotation] = []

    for k in range(n_active - 1, 0, -1):
        a = state[k - 1]
        b = state[k]
        r = np.sqrt(abs(a) ** 2 + abs(b) ** 2)
        if r < 1e-15:
            continue

        theta = float(np.arctan2(abs(b), abs(a)))
        phi = float(np.angle(a) - np.angle(b) + np.pi)
        c = np.cos(theta)
        s = np.sin(theta)

        new_km1 = c * state[k - 1] - np.exp(1.0j * phi) * s * state[k]
        new_k = np.exp(-1.0j * phi) * s * state[k - 1] + c * state[k]
        state[k - 1] = new_km1
        state[k] = new_k
        inverse_rotations.append(GivensRotation(level=k - 1, theta=theta, phi=phi))

    global_phase = float(np.angle(state[0]))
    prep_rotations = list(reversed(inverse_rotations))
    if prep_rotations and abs(global_phase) > 1e-15:
        first = prep_rotations[0]
        prep_rotations[0] = GivensRotation(
            level=first.level,
            theta=first.theta,
            phi=first.phi + global_phase,
        )

    prepared = np.zeros(n_fock, dtype=complex)
    prepared[0] = 1.0
    for rot in prep_rotations:
        a = prepared[rot.level]
        b = prepared[rot.level + 1]
        c = np.cos(rot.theta)
        s = np.sin(rot.theta)
        prepared[rot.level] = c * a + np.exp(1.0j * rot.phi) * s * b
        prepared[rot.level + 1] = -np.exp(-1.0j * rot.phi) * s * a + c * b

    prepared = normalize_vector(prepared)
    fidelity = abs(np.vdot(target, prepared)) ** 2
    return OraclePreparation(
        method="givens",
        target_state=target,
        prepared_state=prepared,
        apply_mode="direct_injection",
        oracle_fidelity=float(fidelity),
        metadata={
            "n_active_fock_levels": int(n_active),
            "n_jc_pulses": int(len(prep_rotations)),
            "n_qubit_rotations": int(len(prep_rotations)),
        },
        givens_rotations=tuple(prep_rotations),
    )


def prepare_cv_oracle(
    kernel: KernelSpec,
    prep: StatePrepSpec,
    *,
    coeffs: Optional[np.ndarray] = None,
) -> OraclePreparation:
    if coeffs is None:
        coeffs = compute_lchs_coefficients(kernel)
    target = padded_seed_state(coeffs, kernel.n_fock)

    if prep.method == "injection":
        return OraclePreparation(
            method="injection",
            target_state=target,
            prepared_state=target.copy(),
            apply_mode="direct_injection",
            oracle_fidelity=1.0,
            metadata={"stateprep_seed_fidelity": 1.0},
        )
    if prep.method == "givens":
        return givens_decomposition(target, n_fock=kernel.n_fock)

    n_snap = min(kernel.n_coeff, kernel.n_fock)
    return optimize_snap_d(
        target,
        n_fock=kernel.n_fock,
        depth=prep.snap_depth,
        n_snap=n_snap,
        n_restarts=prep.snap_restarts,
        maxiter=prep.snap_maxiter,
    )


def _qiskit_to_physics_permutation(n_dv_qubits: int) -> np.ndarray:
    return physics_to_qiskit_permutation(n_dv_qubits)


def detect_statevector_layout(max_fock_level: int, n_dv_qubits: int) -> str:
    try:
        from bosonic_qiskit import CVCircuit, QumodeRegister
        from bosonic_qiskit import util as cv_util
        from qiskit import QuantumRegister
    except ImportError as exc:
        raise ImportError(
            "detect_statevector_layout requires bosonic_qiskit and qiskit."
        ) from exc

    qmr = QumodeRegister(
        num_qumodes=1,
        num_qubits_per_qumode=int(np.log2(max_fock_level)),
    )
    qbr = QuantumRegister(n_dv_qubits, "q")
    qc = CVCircuit(qbr, qmr)

    probe = np.zeros(max_fock_level, dtype=complex)
    probe[1] = 1.0
    qc.cv_initialize(probe, qmr[0])

    state, _, _ = cv_util.simulate(
        qc,
        shots=1,
        return_fockcounts=False,
        add_save_statevector=True,
    )
    vec = np.asarray(state.data, dtype=complex)
    nonzero = np.where(np.abs(vec) > 1e-12)[0]
    if nonzero.size != 1:
        raise RuntimeError(f"Unexpected layout probe state: found {nonzero.size} nonzero entries.")
    idx = int(nonzero[0])
    dv_dim = 2**n_dv_qubits

    if idx == dv_dim:
        return "fock_major"
    if idx == 1:
        return "qubit_major"
    raise RuntimeError(f"Unrecognized statevector layout probe index {idx}.")


def extract_direct_dv_slice(
    statevector: np.ndarray,
    *,
    layout: str,
    max_fock_level: int,
    n_dv_qubits: int,
    fock_index: int = 0,
) -> np.ndarray:
    dv_dim = 2**n_dv_qubits
    vec = np.asarray(statevector, dtype=complex).reshape(-1)
    if layout == "fock_major":
        start = fock_index * dv_dim
        dv_qiskit = vec[start : start + dv_dim]
    elif layout == "qubit_major":
        dv_qiskit = vec[fock_index::max_fock_level][:dv_dim]
    else:
        raise ValueError(f"Unknown layout '{layout}'.")
    return dv_qiskit[_qiskit_to_physics_permutation(n_dv_qubits)]


def extract_fock_zero_dv_statevector(
    statevector: np.ndarray,
    *,
    layout: str,
    max_fock_level: int,
    n_dv_qubits: int,
) -> np.ndarray:
    return extract_direct_dv_slice(
        statevector,
        layout=layout,
        max_fock_level=max_fock_level,
        n_dv_qubits=n_dv_qubits,
        fock_index=0,
    )


def extract_fock_zero_dv_density_matrix(
    density_matrix: np.ndarray,
    *,
    layout: str,
    max_fock_level: int,
    n_dv_qubits: int,
) -> Tuple[np.ndarray, float]:
    dv_dim = 2**n_dv_qubits
    rho = np.asarray(density_matrix, dtype=complex)

    if layout == "fock_major":
        indices = np.arange(dv_dim)
    elif layout == "qubit_major":
        indices = np.arange(dv_dim) * max_fock_level
    else:
        raise ValueError(f"Unknown layout '{layout}'.")

    rho_qiskit = rho[np.ix_(indices, indices)]
    perm = _qiskit_to_physics_permutation(n_dv_qubits)
    rho_physics = rho_qiskit[np.ix_(perm, perm)]
    post_prob = float(np.real(np.trace(rho_physics)))
    return rho_physics, post_prob


def principal_statevector_from_density_matrix(rho: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(np.asarray(rho, dtype=complex))
    idx = int(np.argmax(np.real(vals)))
    eigval = max(float(np.real(vals[idx])), 0.0)
    if eigval < 1e-15:
        return np.zeros(rho.shape[0], dtype=complex)
    principal = vecs[:, idx] * np.sqrt(eigval)
    return np.asarray(principal, dtype=complex)


def fidelity_density_matrix_vs_pure(rho: np.ndarray, pure_state: np.ndarray) -> float:
    pure = normalize_vector(pure_state)
    rho_arr = np.asarray(rho, dtype=complex)
    trace = float(np.real(np.trace(rho_arr)))
    if trace < 1e-15:
        return 0.0
    return float(np.real(np.conj(pure) @ rho_arr @ pure) / trace)


def postselect_cv_output(
    raw_state: np.ndarray,
    *,
    readout_mode: str,
    layout: str,
    max_fock_level: int,
    n_dv_qubits: int,
) -> Dict[str, Any]:
    if readout_mode == "postselect_statevector":
        observed = extract_fock_zero_dv_statevector(
            raw_state,
            layout=layout,
            max_fock_level=max_fock_level,
            n_dv_qubits=n_dv_qubits,
        )
        return {
            "observed_vector": observed,
            "postselection_probability": float(np.linalg.norm(observed) ** 2),
        }

    if readout_mode == "direct_statevector":
        observed = extract_direct_dv_slice(
            raw_state,
            layout=layout,
            max_fock_level=max_fock_level,
            n_dv_qubits=n_dv_qubits,
            fock_index=0,
        )
        return {
            "observed_vector": observed,
            "postselection_probability": float(np.linalg.norm(observed) ** 2),
        }

    if readout_mode == "postselect_density_matrix":
        rho_post, post_prob = extract_fock_zero_dv_density_matrix(
            raw_state,
            layout=layout,
            max_fock_level=max_fock_level,
            n_dv_qubits=n_dv_qubits,
        )
        principal = principal_statevector_from_density_matrix(rho_post)
        return {
            "observed_vector": principal,
            "postselection_probability": float(post_prob),
            "postselected_density_matrix": rho_post,
        }

    raise ValueError(f"Unknown readout_mode '{readout_mode}'.")
