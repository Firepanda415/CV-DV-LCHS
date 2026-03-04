#!/usr/bin/env python3
"""
Readable QuTiP reference implementation for CV-DV LCHS.

Organization
1) ODE system from Pauli decomposition
2) LCHS kernel coefficients C_n
3) CV / DV state preparation
4) Effective map K(T) and comparisons

The code is written so the heat equation is just one example. To reuse for
another ODE/PDE, only replace the Pauli strings and coefficients in __main__.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from numpy.polynomial.hermite import hermgauss
from qutip import Qobj, basis, destroy, qeye, squeeze, tensor
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.special import eval_hermite, gammaln


# -----------------------------------------------------------------------------
# 1) ODE system: Pauli decomposition -> matrix
# -----------------------------------------------------------------------------


def pauli_1q(label: str) -> np.ndarray:
    """Single-qubit Pauli matrices."""
    if label == "I":
        return np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    if label == "X":
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    if label == "Y":
        return np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    if label == "Z":
        return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    raise ValueError(f"Unsupported Pauli label: {label}")


def pauli_string_matrix(pauli: str) -> np.ndarray:
    """Kronecker product matrix for a multi-qubit Pauli string, e.g. 'IXY'."""
    out = np.array([[1.0]], dtype=complex)
    for ch in pauli:
        out = np.kron(out, pauli_1q(ch))
    return out


def pauli_sum_matrix(pauli_strings: Sequence[str], coeffs: Sequence[float]) -> np.ndarray:
    r"""Build \sum_j c_j P_j for one operator block (L or H)."""
    dim = 2 ** len(pauli_strings[0])
    out = np.zeros((dim, dim), dtype=complex)
    for p, c in zip(pauli_strings, coeffs):
        out = out + complex(c) * pauli_string_matrix(p)
    return out


@dataclass
class ODESystemFromPauli:
    # L and H are both Hermitian operator blocks written in Pauli basis.
    pauli_strings_l: Sequence[str]
    coeffs_l: Sequence[float]
    pauli_strings_h: Sequence[str]
    coeffs_h: Sequence[float]

    def n_qubits(self) -> int:
        return len(self.pauli_strings_l[0])

    def generator_matrix(self) -> np.ndarray:
        # Internal model used in this file: du/dt = -A u, A = L + iH.
        l_mat = pauli_sum_matrix(self.pauli_strings_l, self.coeffs_l)
        h_mat = pauli_sum_matrix(self.pauli_strings_h, self.coeffs_h)
        return l_mat + 1j * h_mat


# -----------------------------------------------------------------------------
# 2) LCHS kernel coefficients C_n
# -----------------------------------------------------------------------------


def kernel_g_beta(k_points: np.ndarray, beta: float) -> np.ndarray:
    r"""
    Near-optimal kernel branch from arXiv:2312.03916 Eq. (32)-(33):
      g_beta(k) = exp(-(1 + i k)^beta) / (C_beta (1 - i k)),
      C_beta = 2 pi exp(-2^beta), beta in (0,1).
    """
    c_beta = 2.0 * np.pi * np.exp(-(2.0**beta))
    return np.exp(-((1.0 + 1.0j * k_points) ** beta)) / (c_beta * (1.0 - 1.0j * k_points))


def gamma_hbar1(r_target: float, r_prime: float) -> float:
    r"""For x = (a + a^dag)/sqrt(2): gamma = 0.5*(exp(-2r') - exp(-2r))."""
    return 0.5 * (np.exp(-2.0 * r_prime) - np.exp(-2.0 * r_target))


def fock_prefactor(n: int, sigma_prime: float) -> float:
    r"""Stable evaluation of 1/sqrt(2^n n!) and sigma' normalization."""
    return float(np.exp(-0.5 * (n * np.log(2.0) + gammaln(n + 1.0))) / np.sqrt(np.sqrt(np.pi) * sigma_prime))


def lchs_coefficients_explicit_overlap(
    r_target: float,
    r_prime: float,
    beta: float,
    n_coeff: int,
    n_quad: int,
) -> np.ndarray:
    r"""
    Correct hbar=1 overlap formula used in the codebase:

      C_n \propto \int H_n(k/sigma') g(k) exp(-gamma_hbar1 k^2) dk,
      gamma_hbar1 = 0.5 * (exp(-2r') - exp(-2r)).

    Numerical integration is done on [-k_max, k_max], where k_max comes from
    Gaussian tail scale and sigma'.
    """
    sigma_prime = float(np.exp(r_prime))
    gamma = gamma_hbar1(r_target, r_prime)
    k_tail = np.sqrt(max(-np.log(1e-14) / max(gamma, 1e-15), 1.0))
    k_max = max(8.0 * sigma_prime, 1.15 * k_tail, 12.0)

    coeffs = np.zeros(n_coeff, dtype=complex)
    for n in range(n_coeff):
        pref = fock_prefactor(n, sigma_prime)

        def re_fn(k: float) -> float:
            val = pref * eval_hermite(n, k / sigma_prime) * kernel_g_beta(np.array([k]), beta)[0] * np.exp(-gamma * k * k)
            return float(np.real(val))

        def im_fn(k: float) -> float:
            val = pref * eval_hermite(n, k / sigma_prime) * kernel_g_beta(np.array([k]), beta)[0] * np.exp(-gamma * k * k)
            return float(np.imag(val))

        re = quad(re_fn, -k_max, k_max, limit=max(100, n_quad), epsabs=1e-10, epsrel=1e-9)[0]
        im = quad(im_fn, -k_max, k_max, limit=max(100, n_quad), epsabs=1e-10, epsrel=1e-9)[0]
        coeffs[n] = re + 1.0j * im

    coeffs /= np.linalg.norm(coeffs)
    return coeffs


def lchs_coefficients_gh_comp(
    r_target: float,
    r_prime: float,
    beta: float,
    n_coeff: int,
    n_quad: int,
) -> np.ndarray:
    r"""
    Same corrected formula in Gauss-Hermite coordinates (compensated form):

      k = sqrt(2) sigma' xi,
      C_n \propto \sum_i w_i H_n(sqrt(2) xi_i) g(sqrt(2)sigma'xi_i)
                 exp((sigma'^2 / sigma^2) xi_i^2).
    """
    sigma = float(np.exp(r_target))
    sigma_prime = float(np.exp(r_prime))
    ratio = (sigma_prime * sigma_prime) / (sigma * sigma)

    roots, weights = hermgauss(n_quad)
    k_points = np.sqrt(2.0) * sigma_prime * roots
    scale = np.sqrt(2.0) * sigma_prime

    coeffs = np.zeros(n_coeff, dtype=complex)
    for n in range(n_coeff):
        pref = fock_prefactor(n, sigma_prime)
        herm = eval_hermite(n, np.sqrt(2.0) * roots)

        # log-rescaling keeps the positive exponential numerically stable
        b = ratio * (roots**2)
        logw = np.log(np.abs(weights) + 1e-300)
        shift = np.max(logw + b)
        weighted = np.sign(weights) * herm * kernel_g_beta(k_points, beta) * np.exp((logw + b) - shift)
        coeffs[n] = pref * np.exp(shift) * np.sum(weighted) * scale

    coeffs /= np.linalg.norm(coeffs)
    return coeffs


# -----------------------------------------------------------------------------
# 3) CV and DV state preparation
# -----------------------------------------------------------------------------


@dataclass
class LCHSParams:
    total_time: float
    n_fock: int
    n_coeff: int
    r_target: float
    r_prime: float
    beta: float
    n_quad: int
    coeff_method: str  # "explicit_overlap" or "gh_comp"


def lchs_seed_coefficients(params: LCHSParams) -> np.ndarray:
    if params.coeff_method == "gh_comp":
        return lchs_coefficients_gh_comp(
            r_target=params.r_target,
            r_prime=params.r_prime,
            beta=params.beta,
            n_coeff=params.n_coeff,
            n_quad=params.n_quad,
        )
    return lchs_coefficients_explicit_overlap(
        r_target=params.r_target,
        r_prime=params.r_prime,
        beta=params.beta,
        n_coeff=params.n_coeff,
        n_quad=params.n_quad,
    )


def prepare_cv_states_qutip(params: LCHSParams, coeffs_seed: np.ndarray) -> tuple[Qobj, Qobj, np.ndarray]:
    """
    |psi> = S(r') sum_n C_n |n>,  |phi_r> = S(r)|0>.
    The map uses <phi_r| as the CV post-selection bra.
    """
    full = np.zeros(params.n_fock, dtype=complex)
    full[: len(coeffs_seed)] = coeffs_seed

    psi_seed = Qobj(full.reshape((-1, 1))).unit()
    psi_osc = (squeeze(params.n_fock, params.r_prime) * psi_seed).unit()
    phi_post = (squeeze(params.n_fock, params.r_target) * basis(params.n_fock, 0)).unit()
    return psi_osc, phi_post, full


def position_operator_qutip(n_fock: int) -> Qobj:
    r"""x_hat = (a + a^dag) / sqrt(2), consistent with QuTiP hbar=1 convention."""
    a = destroy(n_fock)
    return (a + a.dag()) / np.sqrt(2.0)


# -----------------------------------------------------------------------------
# 4) Effective map K(T) and analysis
# -----------------------------------------------------------------------------


def effective_lchs_map_qutip(
    generator_matrix: np.ndarray,
    lchs_params: LCHSParams,
    psi_osc: Qobj,
    phi_post: Qobj,
) -> np.ndarray:
    """
    K(T) = <phi_r| exp(-i T x_hat ⊗ A) |psi>.
    Returns a DV matrix of size (2^n) x (2^n).
    """
    dv_dim = generator_matrix.shape[0]
    x_hat = position_operator_qutip(lchs_params.n_fock)
    a_qobj = Qobj(generator_matrix, dims=[[dv_dim], [dv_dim]])

    u_joint = (-1.0j * lchs_params.total_time * tensor(x_hat, a_qobj)).expm()
    post_bra = tensor(phi_post.dag(), qeye(dv_dim))
    embed = tensor(psi_osc, qeye(dv_dim))
    return np.asarray((post_bra * u_joint * embed).full(), dtype=complex)


def classical_map(generator_matrix: np.ndarray, total_time: float) -> np.ndarray:
    """Reference map for du/dt = -A u is exp(-A T)."""
    return expm(-generator_matrix * total_time)


def fit_global_scale(observed: np.ndarray, target: np.ndarray) -> tuple[complex, float]:
    """Best complex scale eta minimizing ||observed - eta*target||_2."""
    eta = np.vdot(target, observed) / np.vdot(target, target)
    rel_err = np.linalg.norm(observed - eta * target) / max(np.linalg.norm(eta * target), 1e-15)
    return complex(eta), float(rel_err)


def state_fidelity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Shape-only fidelity after vector normalization."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-15 or n2 < 1e-15:
        return 0.0
    return float(np.abs(np.vdot(v1 / n1, v2 / n2)) ** 2)


def run_qutip_lchs(
    system: ODESystemFromPauli,
    lchs_params: LCHSParams,
    init_state: np.ndarray,
) -> Dict[str, object]:
    generator = system.generator_matrix()
    coeffs_seed = lchs_seed_coefficients(lchs_params)
    psi_osc, phi_post, coeffs_full = prepare_cv_states_qutip(lchs_params, coeffs_seed)

    k_map = effective_lchs_map_qutip(generator, lchs_params, psi_osc, phi_post)
    ref_map = classical_map(generator, lchs_params.total_time)

    eta_map, map_rel_error = fit_global_scale(k_map, ref_map)

    u_cv = k_map @ init_state
    u_ref = ref_map @ init_state
    eta_vec, vec_rel_error = fit_global_scale(u_cv, u_ref)
    vec_fidelity = state_fidelity(u_cv, u_ref)

    return {
        "generator_matrix": generator,
        "coeffs_seed": coeffs_seed,
        "coeffs_full": coeffs_full,
        "k_map": k_map,
        "ref_map": ref_map,
        "u_cv": u_cv,
        "u_ref": u_ref,
        "map_scale": eta_map,
        "map_rel_error": map_rel_error,
        "vector_scale": eta_vec,
        "vector_rel_error": vec_rel_error,
        "vector_fidelity": vec_fidelity,
    }


# -----------------------------------------------------------------------------
# 5) Example run (no CLI): all parameters live here
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # ODE setup (swap these for any Pauli-decomposed ODE/PDE system)
    # -----------------------------------------------------------------
    # Heat equation example on 2 qubits (4D): A = (alpha/h^2)T.
    # Recommended range for heat-like tests: alpha > 0, h_grid in (0, 2].
    alpha = 1.0
    h_grid = 1.0
    s = alpha / (h_grid**2)

    # L block in Pauli basis: L = sum_j l_j P_j.
    pauli_strings_l = ["II", "IX", "XX", "YY"]
    coeffs_l = [2.0 * s, -1.0 * s, -0.5 * s, -0.5 * s]

    # H block in Pauli basis: H = sum_k h_k P_k.
    # For pure heat equation, H = 0.
    pauli_strings_h = ["II"]
    coeffs_h = [0.0]

    total_time = 1.0  # practical scan range: [0.1, 2.0]

    # Initial DV state u(0), dimension = 2^n_qubits.
    init_state = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)  # |01>

    # -----------------------------------------------------------------
    # CV-DV LCHS parameters
    # -----------------------------------------------------------------
    # Recommended ranges from current experiments:
    # r_target > r_prime >= 0, beta in (0, 1), n_coeff <= n_fock.
    # Low-tail default (current):
    #   r_target=1.50, r_prime=0.02, beta=0.75, n_coeff=24, n_quad=260
    # Optimized-for-map-error reference (higher tail pressure):
    #   r_target=0.80, r_prime=0.04, beta=0.95, n_coeff=24, n_quad=220
    lchs_params = LCHSParams(
        total_time=total_time,
        n_fock=64,
        n_coeff=24,
        r_target=0.79,
        r_prime=0.01,
        beta=0.95,
        n_quad=300,
        coeff_method="explicit_overlap",
    )

    # Optional coefficient back-end consistency check.
    coeffs_exp = lchs_coefficients_explicit_overlap(
        lchs_params.r_target,
        lchs_params.r_prime,
        lchs_params.beta,
        lchs_params.n_coeff,
        lchs_params.n_quad,
    )
    coeffs_gh = lchs_coefficients_gh_comp(
        lchs_params.r_target,
        lchs_params.r_prime,
        lchs_params.beta,
        lchs_params.n_coeff,
        lchs_params.n_quad,
    )
    _, coeff_backend_gap = fit_global_scale(coeffs_gh, coeffs_exp)

    # -----------------------------------------------------------------
    # Run QuTiP reference
    # -----------------------------------------------------------------
    system = ODESystemFromPauli(
        pauli_strings_l=pauli_strings_l,
        coeffs_l=coeffs_l,
        pauli_strings_h=pauli_strings_h,
        coeffs_h=coeffs_h,
    )

    result = run_qutip_lchs(system, lchs_params, init_state)

    print("=== formed_qutip.py: CV-DV LCHS Reference ===")
    print(f"n_qubits: {system.n_qubits()}, n_fock: {lchs_params.n_fock}, n_coeff: {lchs_params.n_coeff}")
    print(
        f"r={lchs_params.r_target}, r'={lchs_params.r_prime}, beta={lchs_params.beta}, "
        f"T={lchs_params.total_time}"
    )
    print(f"Coefficient backend gap (GH vs explicit): {coeff_backend_gap:.6e}")
    print()
    print(f"Map relative error (best scale): {result['map_rel_error']:.6e}")
    print(f"Vector relative error (best scale): {result['vector_rel_error']:.6e}")
    print(f"Vector fidelity:                  {result['vector_fidelity']:.8f}")
    print(
        f"Best map scale eta: {result['map_scale'].real:+.6e} "
        f"{result['map_scale'].imag:+.6e}j"
    )

    print("\nDV vector from LCHS:")
    print(result["u_cv"])
    print("DV vector from classical exp(-AT):")
    print(result["u_ref"])
