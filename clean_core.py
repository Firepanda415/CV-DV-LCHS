#!/usr/bin/env python3
r"""
Independent math and shared types for a clean CV-DV LCHS implementation.

This module is intentionally standalone and does not import any repo-local
implementation files. It owns:
  - shared dataclasses,
  - Pauli and matrix utilities,
  - exact matrix references,
  - independent LCHS coefficient generation,
  - Dirichlet heat-equation and generic Pauli-system builders.

All internal CV formulas use the hbar=1 convention:
  x_hat = (a + a^\dagger) / sqrt(2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.special import eval_hermite, gammaln


ArrayLike = Sequence[complex]


COEFF_BACKENDS = ("explicit_overlap", "gh_comp")
READOUT_MODES = ("postselect_statevector", "postselect_density_matrix", "direct_statevector")
STATE_PREP_METHODS = ("injection", "snap_d", "givens")


@dataclass(frozen=True)
class PauliTerm:
    label: str
    coeff: complex

    def __post_init__(self) -> None:
        if not self.label or any(ch not in "IXYZ" for ch in self.label):
            raise ValueError(f"Invalid Pauli label '{self.label}'.")


@dataclass(frozen=True)
class PauliSystemSpec:
    l_terms: Tuple[PauliTerm, ...]
    h_terms: Tuple[PauliTerm, ...]
    total_time: float
    init_state: np.ndarray
    label: str = "unnamed_system"

    def __post_init__(self) -> None:
        if self.total_time < 0.0:
            raise ValueError("total_time must be non-negative.")
        if not self.l_terms:
            raise ValueError("l_terms must not be empty.")

        n_qubits = len(self.l_terms[0].label)
        if any(len(term.label) != n_qubits for term in self.l_terms):
            raise ValueError("All L-term labels must have the same length.")
        if any(len(term.label) != n_qubits for term in self.h_terms):
            raise ValueError("All H-term labels must have the same length.")

        dv_dim = 2**n_qubits
        init = np.asarray(self.init_state, dtype=complex).reshape(-1)
        if init.size != dv_dim:
            raise ValueError(f"init_state has size {init.size}; expected {dv_dim}.")
        if np.linalg.norm(init) < 1e-15:
            raise ValueError("init_state must have nonzero norm.")

    @property
    def n_qubits(self) -> int:
        return len(self.l_terms[0].label)

    @property
    def dv_dim(self) -> int:
        return 2**self.n_qubits


@dataclass(frozen=True)
class KernelSpec:
    r_target: float
    r_prime: float
    beta: float
    n_coeff: int
    n_fock: int
    n_quad: int
    coeff_backend: str = "explicit_overlap"

    def __post_init__(self) -> None:
        if self.r_target <= self.r_prime:
            raise ValueError("Require r_target > r_prime for a valid kernel window.")
        if self.r_prime < 0.0:
            raise ValueError("r_prime must be non-negative.")
        if not (0.0 < self.beta < 1.0):
            raise ValueError("beta must lie in (0, 1).")
        if self.n_coeff <= 0:
            raise ValueError("n_coeff must be positive.")
        if self.n_fock < self.n_coeff:
            raise ValueError("n_fock must be >= n_coeff.")
        if self.n_quad <= 0:
            raise ValueError("n_quad must be positive.")
        if self.coeff_backend not in COEFF_BACKENDS:
            raise ValueError(
                f"Unknown coeff_backend '{self.coeff_backend}'. Expected one of {COEFF_BACKENDS}."
            )


@dataclass(frozen=True)
class StatePrepSpec:
    method: str = "injection"
    snap_depth: int = 4
    snap_restarts: int = 3
    snap_maxiter: int = 1000

    def __post_init__(self) -> None:
        if self.method not in STATE_PREP_METHODS:
            raise ValueError(
                f"Unknown state-prep method '{self.method}'. Expected one of {STATE_PREP_METHODS}."
            )
        if self.snap_depth < 0:
            raise ValueError("snap_depth must be non-negative.")
        if self.snap_restarts <= 0:
            raise ValueError("snap_restarts must be positive.")
        if self.snap_maxiter <= 0:
            raise ValueError("snap_maxiter must be positive.")


@dataclass(frozen=True)
class EvolutionSpec:
    n_trotter_steps: int
    readout_mode: str = "postselect_statevector"
    photon_loss_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.n_trotter_steps <= 0:
            raise ValueError("n_trotter_steps must be positive.")
        if self.readout_mode not in READOUT_MODES:
            raise ValueError(
                f"Unknown readout_mode '{self.readout_mode}'. Expected one of {READOUT_MODES}."
            )
        if self.photon_loss_rate < 0.0:
            raise ValueError("photon_loss_rate must be non-negative.")


@dataclass
class CleanRunResult:
    system_label: str
    coeff_backend: str
    state_prep_method: str
    readout_mode: str
    postselection_probability: float
    fidelity_vs_exact: float
    fidelity_vs_truncated: float
    rel_error_vs_exact: float
    rel_error_vs_truncated: float
    scale_vs_exact: complex
    scale_vs_truncated: complex
    coeff_backend_gap: float
    oracle_fidelity: float
    observed_vector: np.ndarray
    exact_reference_vector: np.ndarray
    exact_truncated_vector: np.ndarray
    exact_reference_map: np.ndarray
    exact_truncated_map: np.ndarray
    circuit_depth: int
    circuit_size: int
    count_ops: Mapping[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "system_label": self.system_label,
            "coeff_backend": self.coeff_backend,
            "state_prep_method": self.state_prep_method,
            "readout_mode": self.readout_mode,
            "postselection_probability": float(self.postselection_probability),
            "fidelity_vs_exact": float(self.fidelity_vs_exact),
            "fidelity_vs_truncated": float(self.fidelity_vs_truncated),
            "rel_error_vs_exact": float(self.rel_error_vs_exact),
            "rel_error_vs_truncated": float(self.rel_error_vs_truncated),
            "scale_vs_exact_real": float(np.real(self.scale_vs_exact)),
            "scale_vs_exact_imag": float(np.imag(self.scale_vs_exact)),
            "scale_vs_truncated_real": float(np.real(self.scale_vs_truncated)),
            "scale_vs_truncated_imag": float(np.imag(self.scale_vs_truncated)),
            "coeff_backend_gap": float(self.coeff_backend_gap),
            "oracle_fidelity": float(self.oracle_fidelity),
            "circuit_depth": int(self.circuit_depth),
            "circuit_size": int(self.circuit_size),
            "count_ops": dict(self.count_ops),
            "metadata": dict(self.metadata),
        }


def normalize_vector(vec: ArrayLike) -> np.ndarray:
    out = np.asarray(vec, dtype=complex).reshape(-1)
    norm = np.linalg.norm(out)
    if norm < 1e-15:
        raise ValueError("Cannot normalize a zero vector.")
    return out / norm


def basis_state(dim: int, index: int) -> np.ndarray:
    if not (0 <= index < dim):
        raise ValueError(f"Basis index {index} out of range for dim={dim}.")
    out = np.zeros(dim, dtype=complex)
    out[index] = 1.0
    return out


def physics_to_qiskit_permutation(n_qubits: int) -> np.ndarray:
    return np.array([int(f"{i:0{n_qubits}b}"[::-1], 2) for i in range(2**n_qubits)], dtype=int)


def reorder_physics_to_qiskit(vec: ArrayLike, n_qubits: int) -> np.ndarray:
    perm = physics_to_qiskit_permutation(n_qubits)
    out = np.asarray(vec, dtype=complex).reshape(-1)
    if out.size != 2**n_qubits:
        raise ValueError("Vector size does not match n_qubits.")
    qiskit_vec = np.zeros_like(out)
    qiskit_vec[perm] = out
    return qiskit_vec


def reorder_qiskit_to_physics(vec: ArrayLike, n_qubits: int) -> np.ndarray:
    perm = physics_to_qiskit_permutation(n_qubits)
    out = np.asarray(vec, dtype=complex).reshape(-1)
    if out.size != 2**n_qubits:
        raise ValueError("Vector size does not match n_qubits.")
    return out[perm]


def pauli_matrix(label: str) -> np.ndarray:
    if label == "I":
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    if label == "X":
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    if label == "Y":
        return np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    if label == "Z":
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    raise ValueError(f"Unsupported Pauli label '{label}'.")


def pauli_string_matrix(label: str) -> np.ndarray:
    out = np.array([[1.0]], dtype=complex)
    for ch in label:
        out = np.kron(out, pauli_matrix(ch))
    return out


def pauli_sum_matrix(terms: Sequence[PauliTerm]) -> np.ndarray:
    if not terms:
        raise ValueError("Cannot build a Pauli sum from an empty term list.")
    dim = 2 ** len(terms[0].label)
    out = np.zeros((dim, dim), dtype=complex)
    for term in terms:
        out = out + complex(term.coeff) * pauli_string_matrix(term.label)
    return out


def system_blocks(spec: PauliSystemSpec) -> Tuple[np.ndarray, np.ndarray]:
    l_block = pauli_sum_matrix(spec.l_terms)
    if spec.h_terms:
        h_block = pauli_sum_matrix(spec.h_terms)
    else:
        h_block = np.zeros_like(l_block)
    return l_block, h_block


def generator_matrix(spec: PauliSystemSpec) -> np.ndarray:
    l_block, h_block = system_blocks(spec)
    return l_block + 1.0j * h_block


def exact_reference_map(spec: PauliSystemSpec) -> np.ndarray:
    return expm(-generator_matrix(spec) * spec.total_time)


def fit_global_scale(observed: ArrayLike, target: ArrayLike) -> Tuple[complex, float]:
    obs = np.asarray(observed, dtype=complex).reshape(-1)
    tar = np.asarray(target, dtype=complex).reshape(-1)
    denom = np.vdot(tar, tar)
    if abs(denom) < 1e-15:
        return 0.0 + 0.0j, float("inf")
    eta = np.vdot(tar, obs) / denom
    rel_err = np.linalg.norm(obs - eta * tar) / max(np.linalg.norm(eta * tar), 1e-15)
    return complex(eta), float(rel_err)


def state_fidelity(v1: ArrayLike, v2: ArrayLike) -> float:
    a = np.asarray(v1, dtype=complex).reshape(-1)
    b = np.asarray(v2, dtype=complex).reshape(-1)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return float(np.abs(np.vdot(a / na, b / nb)) ** 2)


def decompose_matrix_to_pauli_terms(matrix: np.ndarray, tol: float = 1e-10) -> Tuple[PauliTerm, ...]:
    arr = np.asarray(matrix, dtype=complex)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("matrix must be square.")

    dim = arr.shape[0]
    n_qubits = int(np.log2(dim))
    if 2**n_qubits != dim:
        raise ValueError("matrix dimension must be a power of two.")

    terms: List[PauliTerm] = []
    basis = ["I", "X", "Y", "Z"]
    scale = 1.0 / dim

    def _labels(prefix: str, remaining: int) -> Iterable[str]:
        if remaining == 0:
            yield prefix
            return
        for ch in basis:
            yield from _labels(prefix + ch, remaining - 1)

    for label in _labels("", n_qubits):
        coeff = scale * np.trace(pauli_string_matrix(label).conj().T @ arr)
        if abs(coeff) > tol:
            terms.append(PauliTerm(label=label, coeff=complex(coeff)))

    return tuple(terms)


def kernel_g_beta(k_points: np.ndarray, beta: float) -> np.ndarray:
    c_beta = 2.0 * np.pi * np.exp(-(2.0**beta))
    return np.exp(-((1.0 + 1.0j * k_points) ** beta)) / (c_beta * (1.0 - 1.0j * k_points))


def gamma_hbar1(r_target: float, r_prime: float) -> float:
    return 0.5 * (np.exp(-2.0 * r_prime) - np.exp(-2.0 * r_target))


def _fock_prefactor(n: int, sigma_prime: float) -> float:
    log_pref = -0.5 * (n * np.log(2.0) + gammaln(n + 1.0))
    return float(np.exp(log_pref) / np.sqrt(np.sqrt(np.pi) * sigma_prime))


def _coefficients_explicit_overlap(kernel: KernelSpec) -> np.ndarray:
    sigma_prime = float(np.exp(kernel.r_prime))
    gamma = gamma_hbar1(kernel.r_target, kernel.r_prime)
    k_tail = np.sqrt(max(-np.log(1e-14) / max(gamma, 1e-15), 1.0))
    k_max = max(8.0 * sigma_prime, 1.15 * k_tail, 12.0)

    coeffs = np.zeros(kernel.n_coeff, dtype=complex)
    for n in range(kernel.n_coeff):
        pref = _fock_prefactor(n, sigma_prime)

        def re_fn(k: float) -> float:
            val = (
                pref
                * eval_hermite(n, k / sigma_prime)
                * kernel_g_beta(np.array([k]), kernel.beta)[0]
                * np.exp(-gamma * k * k)
            )
            return float(np.real(val))

        def im_fn(k: float) -> float:
            val = (
                pref
                * eval_hermite(n, k / sigma_prime)
                * kernel_g_beta(np.array([k]), kernel.beta)[0]
                * np.exp(-gamma * k * k)
            )
            return float(np.imag(val))

        re = quad(re_fn, -k_max, k_max, limit=max(100, kernel.n_quad), epsabs=1e-10, epsrel=1e-9)[0]
        im = quad(im_fn, -k_max, k_max, limit=max(100, kernel.n_quad), epsabs=1e-10, epsrel=1e-9)[0]
        coeffs[n] = re + 1.0j * im

    return normalize_vector(coeffs)


def _coefficients_gh_comp(kernel: KernelSpec) -> np.ndarray:
    sigma = float(np.exp(kernel.r_target))
    sigma_prime = float(np.exp(kernel.r_prime))
    ratio = (sigma_prime * sigma_prime) / (sigma * sigma)

    roots, weights = hermgauss(kernel.n_quad)
    k_points = np.sqrt(2.0) * sigma_prime * roots
    scale = np.sqrt(2.0) * sigma_prime

    coeffs = np.zeros(kernel.n_coeff, dtype=complex)
    for n in range(kernel.n_coeff):
        pref = _fock_prefactor(n, sigma_prime)
        herm = eval_hermite(n, np.sqrt(2.0) * roots)
        boost = ratio * (roots**2)
        logw = np.log(np.abs(weights) + 1e-300)
        shift = np.max(logw + boost)
        weighted = (
            np.sign(weights)
            * herm
            * kernel_g_beta(k_points, kernel.beta)
            * np.exp((logw + boost) - shift)
        )
        coeffs[n] = pref * np.exp(shift) * np.sum(weighted) * scale

    return normalize_vector(coeffs)


def compute_lchs_coefficients(kernel: KernelSpec) -> np.ndarray:
    if kernel.coeff_backend == "gh_comp":
        return _coefficients_gh_comp(kernel)
    return _coefficients_explicit_overlap(kernel)


def coefficient_backend_gap(kernel: KernelSpec) -> float:
    explicit_spec = KernelSpec(
        r_target=kernel.r_target,
        r_prime=kernel.r_prime,
        beta=kernel.beta,
        n_coeff=kernel.n_coeff,
        n_fock=kernel.n_fock,
        n_quad=kernel.n_quad,
        coeff_backend="explicit_overlap",
    )
    gh_spec = KernelSpec(
        r_target=kernel.r_target,
        r_prime=kernel.r_prime,
        beta=kernel.beta,
        n_coeff=kernel.n_coeff,
        n_fock=kernel.n_fock,
        n_quad=kernel.n_quad,
        coeff_backend="gh_comp",
    )
    coeffs_exp = compute_lchs_coefficients(explicit_spec)
    coeffs_gh = compute_lchs_coefficients(gh_spec)
    _, rel_err = fit_global_scale(coeffs_gh, coeffs_exp)
    return float(rel_err)


def annihilation_operator(n_fock: int) -> np.ndarray:
    op = np.zeros((n_fock, n_fock), dtype=complex)
    for n in range(1, n_fock):
        op[n - 1, n] = np.sqrt(n)
    return op


def position_operator(n_fock: int) -> np.ndarray:
    a = annihilation_operator(n_fock)
    return (a + a.conj().T) / np.sqrt(2.0)


def squeeze_operator(n_fock: int, r: float) -> np.ndarray:
    if abs(r) < 1e-15:
        return np.eye(n_fock, dtype=complex)
    a = annihilation_operator(n_fock)
    adag = a.conj().T
    generator = 0.5 * r * (a @ a - adag @ adag)
    return expm(generator)


def padded_seed_state(coeffs: ArrayLike, n_fock: int) -> np.ndarray:
    coeff_arr = np.asarray(coeffs, dtype=complex).reshape(-1)
    if coeff_arr.size > n_fock:
        raise ValueError(f"Need n_fock >= len(coeffs); got {n_fock} < {coeff_arr.size}.")
    out = np.zeros(n_fock, dtype=complex)
    out[: coeff_arr.size] = coeff_arr
    return normalize_vector(out)


def truncated_oscillator_states(
    kernel: KernelSpec,
    seed_state: ArrayLike,
) -> Tuple[np.ndarray, np.ndarray]:
    seed = padded_seed_state(seed_state, kernel.n_fock)
    vacuum = basis_state(kernel.n_fock, 0)
    psi_osc = normalize_vector(squeeze_operator(kernel.n_fock, kernel.r_prime) @ seed)
    phi_post = normalize_vector(squeeze_operator(kernel.n_fock, kernel.r_target) @ vacuum)
    return psi_osc, phi_post


def exact_truncated_cv_map(
    system: PauliSystemSpec,
    kernel: KernelSpec,
    seed_state: Optional[ArrayLike] = None,
) -> np.ndarray:
    l_block, h_block = system_blocks(system)
    dv_dim = system.dv_dim
    x_hat = position_operator(kernel.n_fock)
    if seed_state is None:
        seed_state = compute_lchs_coefficients(kernel)
    psi_osc, phi_post = truncated_oscillator_states(kernel, seed_state)

    joint_generator = np.kron(x_hat, l_block) + np.kron(np.eye(kernel.n_fock), h_block)
    u_joint = expm(-1.0j * system.total_time * joint_generator)
    embed = np.kron(psi_osc.reshape((-1, 1)), np.eye(dv_dim, dtype=complex))
    project = np.kron(phi_post.conj().reshape((1, -1)), np.eye(dv_dim, dtype=complex))
    return np.asarray(project @ u_joint @ embed, dtype=complex)


def build_pauli_system(
    *,
    l_terms: Sequence[Tuple[str, complex] | PauliTerm],
    h_terms: Sequence[Tuple[str, complex] | PauliTerm],
    total_time: float,
    init_state: ArrayLike,
    label: str = "generic_pauli_system",
) -> PauliSystemSpec:
    def _coerce(terms: Sequence[Tuple[str, complex] | PauliTerm]) -> Tuple[PauliTerm, ...]:
        out: List[PauliTerm] = []
        for term in terms:
            if isinstance(term, PauliTerm):
                out.append(term)
            else:
                out.append(PauliTerm(label=str(term[0]), coeff=complex(term[1])))
        return tuple(out)

    return PauliSystemSpec(
        l_terms=_coerce(l_terms),
        h_terms=_coerce(h_terms),
        total_time=float(total_time),
        init_state=normalize_vector(init_state),
        label=label,
    )


def build_dirichlet_heat_system(
    *,
    num_qubits: int,
    alpha: float,
    grid_spacing: float,
    total_time: float,
    init_state: Optional[ArrayLike] = None,
    init_basis_index: Optional[int] = 1,
    pauli_tol: float = 1e-10,
    label: Optional[str] = None,
) -> PauliSystemSpec:
    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive.")
    if grid_spacing <= 0.0:
        raise ValueError("grid_spacing must be positive.")

    dim = 2**num_qubits
    lap = 2.0 * np.eye(dim, dtype=complex)
    for idx in range(dim - 1):
        lap[idx, idx + 1] = -1.0
        lap[idx + 1, idx] = -1.0
    system_matrix_heat = (alpha / (grid_spacing**2)) * lap
    l_terms = decompose_matrix_to_pauli_terms(system_matrix_heat, tol=pauli_tol)

    if init_state is None:
        if init_basis_index is None:
            raise ValueError("Provide init_state or init_basis_index.")
        init_state = basis_state(dim, init_basis_index)

    return PauliSystemSpec(
        l_terms=l_terms,
        h_terms=(PauliTerm("I" * num_qubits, 0.0 + 0.0j),),
        total_time=float(total_time),
        init_state=normalize_vector(init_state),
        label=label or f"dirichlet_heat_{dim}d",
    )
