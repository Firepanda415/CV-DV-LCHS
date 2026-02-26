#!/usr/bin/env python3
r"""
CV-DV LCHS 1D heat equation: bosonic-qiskit implementation with CV state injection.

Translates the QuTiP reference (clacod_heat1d_qutip.py) into a bosonic-qiskit
circuit that uses:
  - cv_initialize for direct injection of the squeezed-Fock coefficients,
  - Trotterized conditional-displacement gates for exp(-it x_hat ⊗ A),
  - Fock-|0⟩ post-selection after inverse squeezing for ⟨φ_r|.

This bypasses gate-level state preparation (a hard problem) and focuses on
verifying the Hamiltonian evolution circuit on the bosonic simulator.

Convention notes:
  - Paper x̂ = (a+a†)/√2, matching [x̂,p̂]=i (ℏ=1 physics convention).
  - bosonic_qiskit D(α) = exp(α a† − α* a), hbar=2 internally.
  - exp(-iλ x̂) = D(−iλ/√2) for real λ.
  - cv_c_d(α, qubit): applies D(α) on |0⟩ and D(−α) on |1⟩.
  - cv_sq(r) = S(r) = exp(r/2 (a†² − a²)) for real r.

Example CLI usage:
  # Default run with CV state injection
  python clacod_heat1d_bosonic.py

  # Custom LCHS parameters
  python clacod_heat1d_bosonic.py --r-target 0.85 --r-prime 0.003 --beta 0.35

  # Gate-based state preparation (SNAP+D)
  python clacod_heat1d_bosonic.py --state-prep gate-based --stateprep-depth 8

  # Gate-based with more optimization effort
  python clacod_heat1d_bosonic.py --state-prep gate-based --stateprep-depth 12 \
      --stateprep-restarts 5 --stateprep-maxiter 5000

  # Different initial state, fewer Trotter steps
  python clacod_heat1d_bosonic.py --init-state basis10 --n-trotter-steps 100

  # Save results to JSON
  python clacod_heat1d_bosonic.py --state-prep gate-based --output-json results/bosonic_gb.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from bosonic_qiskit import CVCircuit, QumodeRegister
from bosonic_qiskit import util as cv_util
from qiskit import QuantumRegister
from scipy.linalg import expm

from clacod_heat1d_qutip import (
    HeatLCHSConfig,
    classical_map,
    fit_global_scale,
    heat_matrix,
    heat_matrix_from_paulis,
    lchs_coefficients,
    parse_initial_state,
    state_fidelity,
)


@dataclass(frozen=True)
class BosonicConfig:
    """Configuration for the bosonic-qiskit circuit."""

    max_fock_level: int = 64
    n_trotter_steps: int = 200
    alpha: float = 1.0
    h_grid: float = 1.0
    total_time: float = 1.0
    r_target: float = 0.003
    r_prime: float = 0.001
    beta: float = 0.95
    n_coeff: int = 64
    n_quad: int = 300
    init_state: str = "basis01"


def pauli_coefficients(alpha: float, h_grid: float) -> Dict[str, float]:
    """Pauli decomposition coefficients for A = (alpha/h²) T.

    A = c_II · II + c_IX · IX + c_XX · XX + c_YY · YY
    with T = 2·II - IX - ½(XX + YY)  (PDF Eq. 46).
    """
    s = alpha / (h_grid**2)
    return {
        "II": 2.0 * s,
        "IX": -1.0 * s,
        "XX": -0.5 * s,
        "YY": -0.5 * s,
    }


def _detect_statevector_layout(max_fock_level: int) -> str:
    """Detect whether bosonic-qiskit uses fock-major or qubit-major indexing."""
    num_mode_qubits = int(np.log2(max_fock_level))
    qmr = QumodeRegister(num_qumodes=1, num_qubits_per_qumode=num_mode_qubits)
    qbr = QuantumRegister(2, "q")
    qc = CVCircuit(qbr, qmr)
    mode = qmr[0]

    probe = np.zeros(max_fock_level, dtype=complex)
    probe[1] = 1.0
    qc.cv_initialize(probe, mode)

    state, _, _ = cv_util.simulate(
        qc, shots=1, return_fockcounts=False, add_save_statevector=True
    )
    if state is None:
        raise RuntimeError("Statevector layout detection failed.")

    vec = np.asarray(state.data, dtype=complex)
    nz = np.where(np.abs(vec) > 1e-12)[0]
    if len(nz) != 1:
        raise RuntimeError(f"Layout detection: expected 1 nonzero, got {len(nz)}.")

    idx = int(nz[0])
    if idx == 4:
        return "fock_major"
    if idx == 1:
        return "qubit_major"
    raise RuntimeError(f"Unrecognized layout probe index {idx}.")


def _extract_fock0_dv(
    vec: np.ndarray, layout: str, max_fock_level: int
) -> np.ndarray:
    """Extract the 4-component DV amplitude vector for Fock |0⟩ post-selection.

    Returns in physics ordering: |00⟩, |01⟩, |10⟩, |11⟩.
    """
    if layout == "fock_major":
        dv_qiskit = vec[:4]
    else:
        dv_qiskit = vec[0::max_fock_level][:4]
    # Qiskit qubit ordering is reversed from physics ordering:
    # qiskit |q1 q0⟩ → physics |q0 q1⟩, so swap indices 1↔2.
    return dv_qiskit[[0, 2, 1, 3]]


def build_circuit(
    cfg: BosonicConfig,
    coeffs: np.ndarray,
    *,
    state_prep: str = "injection",
    stateprep_depth: int = 8,
    stateprep_restarts: int = 3,
    stateprep_maxiter: int = 2000,
) -> CVCircuit:
    """Build the full CV-DV LCHS circuit.

    Steps:
      1. Prepare CV state |ψ_seed⟩ = Σ C_n |n⟩ then apply S(r').
         - "injection": direct cv_initialize (simulator-only)
         - "gate-based": SNAP+D optimized circuit (physical gates)
      2. Prepare DV initial state |u0⟩.
      3. Trotter evolution: exp(-it x̂ ⊗ A) via conditional displacements.
      4. Apply S†(r) = S(-r) for post-selection projection ⟨0|S†(r).
    """
    num_mode_qubits = int(np.log2(cfg.max_fock_level))
    qmr = QumodeRegister(num_qumodes=1, num_qubits_per_qumode=num_mode_qubits)
    qbr = QuantumRegister(2, "q")
    qc = CVCircuit(qbr, qmr)
    mode = qmr[0]

    # --- Step 1: CV state preparation ---
    if state_prep == "gate-based":
        from clacod_heat1d_stateprep import (
            apply_snap_d_circuit,
            optimize_snap_d_params,
        )

        print(f"  Gate-based state prep: optimizing SNAP+D (depth={stateprep_depth})...")
        snap_d_result = optimize_snap_d_params(
            coeffs,
            cfg.max_fock_level,
            stateprep_depth,
            n_restarts=stateprep_restarts,
            maxiter=stateprep_maxiter,
            verbose=True,
        )
        print(
            f"  State prep fidelity: {snap_d_result.fidelity:.8f} "
            f"(infidelity: {1-snap_d_result.fidelity:.2e})"
        )
        apply_snap_d_circuit(qc, mode, snap_d_result)
    else:
        # Default: direct injection (simulator-only)
        inject = np.zeros(cfg.max_fock_level, dtype=complex)
        inject[: len(coeffs)] = coeffs
        qc.cv_initialize(inject, mode)

    # Apply S(r') to get |ψ⟩ = S(r') Σ C_n |n⟩
    if not np.isclose(cfg.r_prime, 0.0):
        qc.cv_sq(cfg.r_prime, mode)

    # --- Step 2: DV initial state ---
    # NOTE: Only computational basis states (basis00, basis01, basis10, basis11)
    # are correctly prepared here.  Superposition inputs (sine, ones) are
    # silently reduced to the dominant basis state via argmax, because
    # preparing an arbitrary 2-qubit superposition requires a general
    # unitary decomposition (Ry + CNOT) that is not yet implemented.
    u0 = parse_initial_state(cfg.init_state)
    idx = int(np.argmax(np.abs(u0)))
    q0_bit = (idx >> 1) & 1
    q1_bit = idx & 1
    if q0_bit == 1:
        qc.x(qbr[0])
    if q1_bit == 1:
        qc.x(qbr[1])

    # --- Step 3: Trotterized evolution ---
    # exp(-it x̂ ⊗ A) with x̂ = (a+a†)/√2, A = Σ c_k P_k
    # Each Trotter step: ∏_k exp(-iδt c_k x̂ ⊗ P_k)
    # exp(-iλ x̂) = D(-iλ/√2) for real λ.
    #
    # For II term (unconditional): cv_d(-i δt c_II / √2)
    # For IX term: H(q1) → cv_c_d(-i δt c_IX / √2, q1) → H(q1)
    # For XX term: H(q0)H(q1) → CNOT(q0,q1) → cv_c_d(-i δt c_XX / √2, q1) → CNOT → H(q0)H(q1)
    # For YY term: Rz(π/2)(q0)Rz(π/2)(q1) → H(q0)H(q1) → CNOT → cv_c_d → CNOT → H → Rz(-π/2)
    #
    # cv_c_d(α, qubit): D(α)|0⟩ + D(-α)|1⟩
    # For exp(-iλ x̂ ⊗ Z): need D(-iλ/√2)|0⟩ + D(+iλ/√2)|1⟩
    # So α = -iλ/√2.

    pc = pauli_coefficients(cfg.alpha, cfg.h_grid)
    dt = cfg.total_time / cfg.n_trotter_steps
    sqrt2 = np.sqrt(2.0)

    # Displacement amplitudes (purely imaginary for real Pauli coefficients)
    alpha_ii = -1j * dt * pc["II"] / sqrt2
    alpha_ix = -1j * dt * pc["IX"] / sqrt2
    alpha_xx = -1j * dt * pc["XX"] / sqrt2
    alpha_yy = -1j * dt * pc["YY"] / sqrt2

    for _ in range(cfg.n_trotter_steps):
        # II term: unconditional displacement
        qc.cv_d(alpha_ii, mode)

        # IX term: basis change for X on qubit 1
        qc.h(qbr[1])
        qc.cv_c_d(alpha_ix, mode, qbr[1])
        qc.h(qbr[1])

        # XX term: diagonalize X⊗X = (H⊗H)(Z⊗Z)(H⊗H), Z⊗Z = CNOT(I⊗Z)CNOT
        qc.h(qbr[0])
        qc.h(qbr[1])
        qc.cx(qbr[0], qbr[1])
        qc.cv_c_d(alpha_xx, mode, qbr[1])
        qc.cx(qbr[0], qbr[1])
        qc.h(qbr[0])
        qc.h(qbr[1])

        # YY term: Rz(π/2)H diagonalizes Y to Z
        qc.rz(np.pi / 2, qbr[0])
        qc.rz(np.pi / 2, qbr[1])
        qc.h(qbr[0])
        qc.h(qbr[1])
        qc.cx(qbr[0], qbr[1])
        qc.cv_c_d(alpha_yy, mode, qbr[1])
        qc.cx(qbr[0], qbr[1])
        qc.h(qbr[0])
        qc.h(qbr[1])
        qc.rz(-np.pi / 2, qbr[0])
        qc.rz(-np.pi / 2, qbr[1])

    # --- Step 4: Post-selection ---
    # Project onto ⟨φ_r| = ⟨0|S†(r). Apply S†(r) = S(-r), then read Fock |0⟩.
    if not np.isclose(cfg.r_target, 0.0):
        qc.cv_sq(-cfg.r_target, mode)

    return qc


def circuit_resource_stats(
    cfg: BosonicConfig, *, state_prep: str = "injection", stateprep_depth: int = 0
) -> Dict[str, object]:
    """Compute gate counts and circuit resource statistics.

    Per Trotter step the circuit applies:
      II:  1 cv_d
      IX:  2 H, 1 cv_c_d
      XX:  4 H, 2 CNOT, 1 cv_c_d
      YY:  4 Rz, 4 H, 2 CNOT, 1 cv_c_d

    Plus state-prep / post-selection overhead.
    """
    n = cfg.n_trotter_steps

    # --- Per-step counts ---
    cv_d_per_step = 1       # II term
    cv_c_d_per_step = 3     # IX + XX + YY
    h_per_step = 2 + 4 + 4  # IX(2) + XX(4) + YY(4)
    cnot_per_step = 2 + 2   # XX(2) + YY(2)
    rz_per_step = 4         # YY only

    # --- Totals from Trotter ---
    cv_d_total = cv_d_per_step * n
    cv_c_d_total = cv_c_d_per_step * n
    h_total = h_per_step * n
    cnot_total = cnot_per_step * n
    rz_total = rz_per_step * n

    # --- State prep / post-selection overhead ---
    cv_sq_prep = 1 if not np.isclose(cfg.r_prime, 0.0) else 0
    cv_sq_post = 1 if not np.isclose(cfg.r_target, 0.0) else 0
    x_gates = 0     # depends on init_state; at most 2
    u0 = parse_initial_state(cfg.init_state)
    idx = int(np.argmax(np.abs(u0)))
    x_gates += ((idx >> 1) & 1) + (idx & 1)

    cv_sq_total = cv_sq_prep + cv_sq_post

    # --- State prep gates ---
    if state_prep == "gate-based":
        snap_gates = stateprep_depth
        snap_d_disp_gates = stateprep_depth
        cv_init = 0
        stateprep_physical = snap_gates + snap_d_disp_gates
    else:
        snap_gates = 0
        snap_d_disp_gates = 0
        cv_init = 1
        stateprep_physical = 0

    # --- Hybrid gate summary ---
    total_physical_cv_gates = cv_d_total + cv_c_d_total + cv_sq_total + snap_d_disp_gates + snap_gates
    total_dv_gates = h_total + cnot_total + rz_total + x_gates
    total_hybrid_gates = cv_c_d_total  # CD gates couple CV and DV

    # --- Circuit depth (sequential layers per Trotter step) ---
    layers_per_step = 22
    depth_estimate = layers_per_step * n + 2 + 2 * stateprep_depth  # +2 for squeeze

    total_physical = total_physical_cv_gates + total_dv_gates

    stats = {
        "n_trotter_steps": n,
        "n_pauli_terms": 4,
        "gates_per_step": {
            "cv_d (displacement)": cv_d_per_step,
            "cv_c_d (conditional displacement)": cv_c_d_per_step,
            "H (Hadamard)": h_per_step,
            "CNOT": cnot_per_step,
            "Rz": rz_per_step,
        },
        "total_gates": {
            "cv_d": cv_d_total,
            "cv_c_d": cv_c_d_total,
            "cv_sq (squeeze)": cv_sq_total,
            "H": h_total,
            "CNOT": cnot_total,
            "Rz": rz_total,
            "X": x_gates,
        },
        "state_preparation": {
            "method": state_prep,
            "cv_initialize": cv_init,
            "snap_gates": snap_gates,
            "snap_d_displacement_gates": snap_d_disp_gates,
            "stateprep_physical_gates": stateprep_physical,
            "note": (
                "Gate-based SNAP+D state preparation with physical gates."
                if state_prep == "gate-based"
                else "cv_initialize is a simulator instruction (direct Fock "
                "amplitude injection), not a physical gate."
            ),
        },
        "summary": {
            "total_physical_cv_gates": total_physical_cv_gates,
            "total_dv_gates": total_dv_gates,
            "total_hybrid_cd_gates": total_hybrid_gates,
            "total_physical_gates": total_physical,
            "circuit_depth_estimate": depth_estimate,
        },
        "system": {
            "dv_qubits": 2,
            "cv_qumodes": 1,
            "fock_dim": cfg.max_fock_level,
            "mode_qubits": int(np.log2(cfg.max_fock_level)),
            "total_qubits": 2 + int(np.log2(cfg.max_fock_level)),
        },
    }
    return stats


def run_bosonic_simulation(
    cfg: BosonicConfig,
    *,
    state_prep: str = "injection",
    stateprep_depth: int = 8,
    stateprep_restarts: int = 3,
    stateprep_maxiter: int = 2000,
) -> Dict[str, object]:
    """Run the full CV-DV LCHS circuit on the bosonic simulator and compare with reference."""

    # Compute LCHS coefficients using the correct kernel from clacod
    coeffs_raw = lchs_coefficients(
        r_target=cfg.r_target,
        r_prime=cfg.r_prime,
        beta=cfg.beta,
        n_fock=cfg.n_coeff,
        n_quad=cfg.n_quad,
    )
    coeffs = np.zeros(cfg.n_coeff, dtype=complex)
    coeffs[: len(coeffs_raw)] = coeffs_raw

    # Detect layout once
    layout = _detect_statevector_layout(cfg.max_fock_level)

    # Build and simulate
    qc = build_circuit(
        cfg,
        coeffs,
        state_prep=state_prep,
        stateprep_depth=stateprep_depth,
        stateprep_restarts=stateprep_restarts,
        stateprep_maxiter=stateprep_maxiter,
    )
    state, _, _ = cv_util.simulate(
        qc, shots=1, return_fockcounts=False, add_save_statevector=True
    )
    if state is None:
        raise RuntimeError("Bosonic-qiskit simulation returned no statevector.")

    vec = np.asarray(state.data, dtype=complex)
    expected_size = 4 * cfg.max_fock_level
    if vec.size != expected_size:
        raise RuntimeError(
            f"Unexpected statevector size {vec.size}; expected {expected_size}."
        )

    # Extract post-selected DV state (Fock |0⟩)
    dv_vec = _extract_fock0_dv(vec, layout, cfg.max_fock_level)

    # Classical reference
    a_mat = heat_matrix(cfg.alpha, cfg.h_grid)
    ref_map = expm(-a_mat * cfg.total_time)
    u0 = parse_initial_state(cfg.init_state)
    u_ref = ref_map @ u0

    # QuTiP reference (exact, no Trotter)
    qutip_cfg = HeatLCHSConfig(
        alpha=cfg.alpha,
        h_grid=cfg.h_grid,
        total_time=cfg.total_time,
        n_fock=cfg.max_fock_level,
        n_coeff=cfg.n_coeff,
        r_target=cfg.r_target,
        r_prime=cfg.r_prime,
        beta=cfg.beta,
        n_quad=cfg.n_quad,
        init_state=cfg.init_state,
        position_convention="sqrt2",
    )
    from clacod_heat1d_qutip import effective_lchs_map, prepare_lchs_states

    psi_osc, phi_post, _ = prepare_lchs_states(qutip_cfg)
    qutip_map = effective_lchs_map(qutip_cfg, psi_osc, phi_post)
    u_qutip = qutip_map @ u0

    # Metrics
    post_prob = float(np.real(np.vdot(dv_vec, dv_vec)))
    scale_vs_ref, err_vs_ref = fit_global_scale(dv_vec, u_ref)
    fid_vs_ref = state_fidelity(dv_vec, u_ref)
    scale_vs_qutip, err_vs_qutip = fit_global_scale(dv_vec, u_qutip)
    fid_vs_qutip = state_fidelity(dv_vec, u_qutip)

    # Pauli consistency check
    a_direct = heat_matrix(cfg.alpha, cfg.h_grid)
    a_pauli = heat_matrix_from_paulis(cfg.alpha, cfg.h_grid)
    pauli_consistency = float(np.linalg.norm(a_direct - a_pauli))

    # Resource statistics
    resources = circuit_resource_stats(
        cfg, state_prep=state_prep, stateprep_depth=stateprep_depth
    )

    metrics = {
        "pauli_consistency_norm": pauli_consistency,
        "post_prob": post_prob,
        "bosonic_vs_classical_rel_error": err_vs_ref,
        "bosonic_vs_classical_fidelity": fid_vs_ref,
        "bosonic_vs_classical_scale_real": float(np.real(scale_vs_ref)),
        "bosonic_vs_classical_scale_imag": float(np.imag(scale_vs_ref)),
        "bosonic_vs_qutip_rel_error": err_vs_qutip,
        "bosonic_vs_qutip_fidelity": fid_vs_qutip,
        "bosonic_vs_qutip_scale_real": float(np.real(scale_vs_qutip)),
        "bosonic_vs_qutip_scale_imag": float(np.imag(scale_vs_qutip)),
        "n_trotter_steps": cfg.n_trotter_steps,
        "max_fock_level": cfg.max_fock_level,
        "layout": layout,
    }

    vectors = {
        "dv_bosonic": dv_vec,
        "u_ref": u_ref,
        "u_qutip": u_qutip,
    }

    return {
        "metrics": metrics,
        "vectors": vectors,
        "config": asdict(cfg),
        "resources": resources,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CV-DV LCHS 1D heat equation (bosonic-qiskit, CV state injection)"
    )
    p.add_argument("--max-fock-level", type=int, default=64)
    p.add_argument("--n-trotter-steps", type=int, default=200)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--h-grid", type=float, default=1.0)
    p.add_argument("--total-time", type=float, default=1.0)
    p.add_argument("--r-target", type=float, default=0.003)
    p.add_argument("--r-prime", type=float, default=0.001)
    p.add_argument("--beta", type=float, default=0.95)
    p.add_argument("--n-coeff", type=int, default=64)
    p.add_argument("--n-quad", type=int, default=300)
    p.add_argument(
        "--init-state",
        choices=["basis00", "basis01", "basis10", "basis11", "sine", "ones"],
        default="basis01",
    )
    p.add_argument(
        "--state-prep",
        choices=["injection", "gate-based"],
        default="injection",
        help="CV state preparation method: injection (cv_initialize) or gate-based (SNAP+D)",
    )
    p.add_argument(
        "--stateprep-depth",
        type=int,
        default=8,
        help="Number of SNAP+D layers for gate-based state prep",
    )
    p.add_argument(
        "--stateprep-restarts",
        type=int,
        default=3,
        help="Number of random restarts for SNAP+D optimization",
    )
    p.add_argument(
        "--stateprep-maxiter",
        type=int,
        default=2000,
        help="Max iterations per SNAP+D optimization run",
    )
    p.add_argument("--output-json", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # max_fock_level must be a power of 2 for bosonic-qiskit
    fock = args.max_fock_level
    if fock <= 0 or (fock & (fock - 1)) != 0:
        raise ValueError("max-fock-level must be a positive power of two.")

    cfg = BosonicConfig(
        max_fock_level=fock,
        n_trotter_steps=args.n_trotter_steps,
        alpha=args.alpha,
        h_grid=args.h_grid,
        total_time=args.total_time,
        r_target=args.r_target,
        r_prime=args.r_prime,
        beta=args.beta,
        n_coeff=min(args.n_coeff, fock),
        n_quad=args.n_quad,
        init_state=args.init_state,
    )

    state_prep = args.state_prep
    prep_label = "SNAP+D gate-based" if state_prep == "gate-based" else "CV injection"
    print(f"=== CV-DV LCHS Heat1D (bosonic-qiskit, {prep_label}) ===")
    print(f"Fock dim: {cfg.max_fock_level}, Trotter steps: {cfg.n_trotter_steps}")
    print(
        f"CV params: r={cfg.r_target}, r'={cfg.r_prime}, beta={cfg.beta}, "
        f"n_coeff={cfg.n_coeff}"
    )
    print(f"PDE: alpha={cfg.alpha}, h={cfg.h_grid}, T={cfg.total_time}")
    print(f"Init state: {cfg.init_state}, State prep: {state_prep}")
    if state_prep == "gate-based":
        print(f"SNAP+D depth: {args.stateprep_depth}, restarts: {args.stateprep_restarts}")
    print()

    result = run_bosonic_simulation(
        cfg,
        state_prep=state_prep,
        stateprep_depth=args.stateprep_depth,
        stateprep_restarts=args.stateprep_restarts,
        stateprep_maxiter=args.stateprep_maxiter,
    )
    m = result["metrics"]
    v = result["vectors"]
    r = result["resources"]

    print(f"Statevector layout: {m['layout']}")
    print(f"Pauli consistency norm: {m['pauli_consistency_norm']:.3e}")
    print(f"Post-selection probability: {m['post_prob']:.6e}")
    print()
    print("--- Bosonic vs Classical Reference ---")
    print(f"  Relative error (best scale): {m['bosonic_vs_classical_rel_error']:.6f}")
    print(f"  Shape fidelity:              {m['bosonic_vs_classical_fidelity']:.6f}")
    print(
        f"  Best scale: {m['bosonic_vs_classical_scale_real']:+.6e}"
        f" {m['bosonic_vs_classical_scale_imag']:+.6e}j"
    )
    print()
    print("--- Bosonic vs QuTiP (exact LCHS, no Trotter) ---")
    print(f"  Relative error (best scale): {m['bosonic_vs_qutip_rel_error']:.6f}")
    print(f"  Shape fidelity:              {m['bosonic_vs_qutip_fidelity']:.6f}")
    print(
        f"  Best scale: {m['bosonic_vs_qutip_scale_real']:+.6e}"
        f" {m['bosonic_vs_qutip_scale_imag']:+.6e}j"
    )

    print()
    print("--- Vectors ---")
    for label, vec in [
        ("Bosonic DV", v["dv_bosonic"]),
        ("Classical", v["u_ref"]),
        ("QuTiP LCHS", v["u_qutip"]),
    ]:
        fmt = " ".join(f"{x.real:+.6f}{x.imag:+.6f}j" for x in vec)
        print(f"  {label:12s}: [{fmt}]")

    # --- Resource statistics ---
    sys_info = r["system"]
    gps = r["gates_per_step"]
    tot = r["total_gates"]
    summ = r["summary"]
    sp_info = r["state_preparation"]
    print()
    print("--- Circuit Resource Statistics ---")
    print(f"  System: {sys_info['dv_qubits']} DV qubits + {sys_info['cv_qumodes']} CV qumode "
          f"(Fock dim {sys_info['fock_dim']}, {sys_info['mode_qubits']} mode qubits, "
          f"{sys_info['total_qubits']} total qubits)")
    print(f"  Trotter steps: {r['n_trotter_steps']}, "
          f"Pauli terms: {r['n_pauli_terms']}")
    print()
    print(f"  Gates per Trotter step:")
    for name, count in gps.items():
        print(f"    {name:40s} {count:>4d}")
    print()
    print(f"  Total physical gate counts:")
    for name, count in tot.items():
        print(f"    {name:40s} {count:>6d}")
    print()
    print(f"  State preparation ({sp_info['method']}):")
    if sp_info['method'] == 'gate-based':
        print(f"    SNAP gates:                             {sp_info['snap_gates']}")
        print(f"    Displacement gates (state prep):        {sp_info['snap_d_displacement_gates']}")
        print(f"    Total state-prep physical gates:        {sp_info['stateprep_physical_gates']}")
    else:
        print(f"    cv_initialize (state injection):         {sp_info['cv_initialize']}")
    print(f"    Note: {sp_info['note']}")
    print()
    print(f"  Summary:")
    print(f"    Physical CV gates (D, CD, Sq):       {summ['total_physical_cv_gates']:>6d}")
    print(f"    DV gates (H, CNOT, Rz, X):          {summ['total_dv_gates']:>6d}")
    print(f"    Hybrid CD gates (CV-DV coupling):    {summ['total_hybrid_cd_gates']:>6d}")
    print(f"    Total physical gates:                {summ['total_physical_gates']:>6d}")
    print(f"    Circuit depth estimate:              {summ['circuit_depth_estimate']:>6d}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": result["config"],
            "metrics": {
                k: v for k, v in m.items() if isinstance(v, (int, float, str))
            },
            "resources": result["resources"],
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote JSON: {out}")


if __name__ == "__main__":
    main()
