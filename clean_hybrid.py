#!/usr/bin/env python3
"""
Independent backend-facing hybrid CV-DV LCHS circuit construction and execution.

This module owns:
  - bosonic-qiskit circuit assembly,
  - hybrid Pauli compilation,
  - Trotterized evolution,
  - end-to-end simulation and comparison against exact references.

All internal math follows the hbar=1 convention. Any bosonic-qiskit-specific
parameter/sign adaptations are localized here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from clean_core import (
    CleanRunResult,
    EvolutionSpec,
    KernelSpec,
    PauliSystemSpec,
    PauliTerm,
    StatePrepSpec,
    coefficient_backend_gap,
    exact_reference_map,
    exact_truncated_cv_map,
    fit_global_scale,
    generator_matrix,
    normalize_vector,
    reorder_physics_to_qiskit,
    state_fidelity,
    system_blocks,
)
from clean_oracles import (
    OraclePreparation,
    compute_lchs_coefficients,
    detect_statevector_layout,
    fidelity_density_matrix_vs_pure,
    postselect_cv_output,
    prepare_cv_oracle,
)


def _require_backend() -> Tuple[Any, Any, Any, Any, Any]:
    try:
        from bosonic_qiskit import CVCircuit, QumodeRegister
        from bosonic_qiskit import util as cv_util
        from qiskit import QuantumRegister
    except ImportError as exc:
        raise ImportError(
            "clean_hybrid requires bosonic_qiskit and qiskit for runtime execution."
        ) from exc

    try:
        from bosonic_qiskit.kraus import PhotonLossNoisePass
    except ImportError:
        PhotonLossNoisePass = None

    return CVCircuit, QumodeRegister, QuantumRegister, cv_util, PhotonLossNoisePass


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _real_coeff(coeff: complex, *, tol: float = 1e-10) -> float:
    if abs(np.imag(coeff)) > tol:
        raise ValueError(f"Expected a real Pauli coefficient, got {coeff}.")
    return float(np.real(coeff))


def _bosonic_sq_for_prepare(r: float) -> float:
    # bosonic_qiskit uses the opposite sign relative to the dense hbar=1 convention.
    return -float(r)


def _bosonic_sq_for_postselect(r: float) -> float:
    # Applying S^\dagger(r) before the Fock-|0> slice implements <0|S^\dagger(r).
    return float(r)


def _conditional_displacement_alpha(lam: float) -> complex:
    # exp(-i lam x_hat), x_hat=(a+a^\dagger)/sqrt(2), equals D(-i lam / sqrt(2)).
    return -1.0j * float(lam) / np.sqrt(2.0)


def _apply_oracle_to_circuit(qc: Any, mode: Any, oracle: OraclePreparation) -> None:
    if oracle.apply_mode == "direct_injection":
        qc.cv_initialize(np.asarray(oracle.prepared_state, dtype=complex), mode)
        return

    if oracle.apply_mode == "snap_d_layers":
        for thetas, alpha in zip(oracle.snap_thetas_per_layer, oracle.snap_alphas_per_layer):
            for level, theta in enumerate(thetas):
                if abs(float(theta)) > 1e-12:
                    qc.cv_snap(float(theta), level, mode)
            qc.cv_d(alpha, mode)
        return

    raise ValueError(f"Unknown oracle apply_mode '{oracle.apply_mode}'.")


def _initialize_dv_state(qc: Any, qreg: Any, init_state: np.ndarray) -> None:
    physics_state = normalize_vector(init_state)
    qiskit_state = reorder_physics_to_qiskit(physics_state, len(qreg))
    qc.initialize(qiskit_state, list(qreg))


def _apply_basis_change_forward(qc: Any, qreg: Any, pauli_label: str) -> None:
    for idx, ch in enumerate(pauli_label):
        if ch == "X":
            qc.h(qreg[idx])
        elif ch == "Y":
            qc.rz(np.pi / 2.0, qreg[idx])
            qc.h(qreg[idx])


def _apply_basis_change_inverse(qc: Any, qreg: Any, pauli_label: str) -> None:
    for idx in reversed(range(len(pauli_label))):
        ch = pauli_label[idx]
        if ch == "X":
            qc.h(qreg[idx])
        elif ch == "Y":
            qc.h(qreg[idx])
            qc.rz(-np.pi / 2.0, qreg[idx])


def _active_qubits(pauli_label: str) -> List[int]:
    return [idx for idx, ch in enumerate(pauli_label) if ch != "I"]


def _apply_hybrid_z_parity_block(qc: Any, mode: Any, qreg: Any, active: Sequence[int], alpha: complex) -> None:
    if not active:
        qc.cv_d(alpha, mode)
        return

    if len(active) == 1:
        qc.cv_c_d(alpha, mode, qreg[active[0]])
        return

    target = active[-1]
    for control in active[:-1]:
        qc.cx(qreg[control], qreg[target])
    qc.cv_c_d(alpha, mode, qreg[target])
    for control in reversed(active[:-1]):
        qc.cx(qreg[control], qreg[target])


def _apply_hybrid_pauli_factor(qc: Any, mode: Any, qreg: Any, pauli_term: PauliTerm, dt: float) -> None:
    coeff = _real_coeff(pauli_term.coeff)
    if abs(coeff) < 1e-15:
        return

    lam = coeff * dt
    alpha = _conditional_displacement_alpha(lam)
    active = _active_qubits(pauli_term.label)
    _apply_basis_change_forward(qc, qreg, pauli_term.label)
    _apply_hybrid_z_parity_block(qc, mode, qreg, active, alpha)
    _apply_basis_change_inverse(qc, qreg, pauli_term.label)


def _apply_dv_pauli_rotation(qc: Any, qreg: Any, pauli_term: PauliTerm, dt: float) -> None:
    coeff = _real_coeff(pauli_term.coeff)
    if abs(coeff) < 1e-15:
        return

    theta = coeff * dt
    active = _active_qubits(pauli_term.label)
    if not active:
        return

    _apply_basis_change_forward(qc, qreg, pauli_term.label)
    if len(active) == 1:
        qc.rz(2.0 * theta, qreg[active[0]])
    else:
        target = active[-1]
        for control in active[:-1]:
            qc.cx(qreg[control], qreg[target])
        qc.rz(2.0 * theta, qreg[target])
        for control in reversed(active[:-1]):
            qc.cx(qreg[control], qreg[target])
    _apply_basis_change_inverse(qc, qreg, pauli_term.label)


def _apply_first_order_trotter(
    qc: Any,
    mode: Any,
    qreg: Any,
    *,
    l_terms: Sequence[PauliTerm],
    h_terms: Sequence[PauliTerm],
    total_time: float,
    n_trotter_steps: int,
) -> None:
    dt = total_time / n_trotter_steps
    for _ in range(n_trotter_steps):
        for term in l_terms:
            _apply_hybrid_pauli_factor(qc, mode, qreg, term, dt)
        for term in h_terms:
            _apply_dv_pauli_rotation(qc, qreg, term, dt)


def circuit_resource_report(qc: Any) -> Dict[str, Any]:
    return {
        "depth": int(qc.depth()),
        "size": int(qc.size()),
        "count_ops": {str(k): int(v) for k, v in dict(qc.count_ops()).items()},
    }


def build_hybrid_circuit(
    system: PauliSystemSpec,
    kernel: KernelSpec,
    prep: StatePrepSpec,
    evolution: EvolutionSpec,
    *,
    coeffs: Optional[np.ndarray] = None,
    oracle: Optional[OraclePreparation] = None,
) -> Tuple[Any, OraclePreparation, Dict[str, Any]]:
    if not _is_power_of_two(kernel.n_fock):
        raise ValueError("kernel.n_fock must be a power of two for bosonic_qiskit.")

    CVCircuit, QumodeRegister, QuantumRegister, _, _ = _require_backend()

    if coeffs is None:
        coeffs = compute_lchs_coefficients(kernel)
    if oracle is None:
        oracle = prepare_cv_oracle(kernel, prep, coeffs=coeffs)

    qmr = QumodeRegister(
        num_qumodes=1,
        num_qubits_per_qumode=int(np.log2(kernel.n_fock)),
    )
    qbr = QuantumRegister(system.n_qubits, "q")
    qc = CVCircuit(qbr, qmr)
    mode = qmr[0]

    _apply_oracle_to_circuit(qc, mode, oracle)

    if abs(kernel.r_prime) > 1e-14:
        qc.cv_sq(_bosonic_sq_for_prepare(kernel.r_prime), mode)

    _initialize_dv_state(qc, qbr, system.init_state)

    filtered_l_terms = [term for term in system.l_terms if abs(term.coeff) > 1e-15]
    filtered_h_terms = [term for term in system.h_terms if abs(term.coeff) > 1e-15]
    _apply_first_order_trotter(
        qc,
        mode,
        qbr,
        l_terms=filtered_l_terms,
        h_terms=filtered_h_terms,
        total_time=system.total_time,
        n_trotter_steps=evolution.n_trotter_steps,
    )

    if abs(kernel.r_target) > 1e-14:
        qc.cv_sq(_bosonic_sq_for_postselect(kernel.r_target), mode)

    build_meta = {
        "n_dv_qubits": int(system.n_qubits),
        "n_fock": int(kernel.n_fock),
        "coeff_backend": kernel.coeff_backend,
        "state_prep_method": prep.method,
        "n_trotter_steps": int(evolution.n_trotter_steps),
    }
    return qc, oracle, build_meta


def _simulate_with_readout(
    qc: Any,
    *,
    evolution: EvolutionSpec,
    kernel: KernelSpec,
    system: PauliSystemSpec,
) -> Dict[str, Any]:
    _, _, _, cv_util, PhotonLossNoisePass = _require_backend()
    layout = detect_statevector_layout(kernel.n_fock, system.n_qubits)

    if evolution.readout_mode == "postselect_density_matrix":
        qc.save_density_matrix()
        noise_pass = None
        if evolution.photon_loss_rate > 0.0:
            if PhotonLossNoisePass is None:
                raise ImportError("PhotonLossNoisePass is unavailable in this bosonic_qiskit build.")
            noise_pass = PhotonLossNoisePass(
                photon_loss_rates=evolution.photon_loss_rate,
                circuit=qc,
                time_unit="s",
            )

        _, result, _ = cv_util.simulate(
            qc,
            shots=1,
            return_fockcounts=False,
            add_save_statevector=False,
            noise_passes=noise_pass,
        )
        density_matrix = np.asarray(result.data(0)["density_matrix"], dtype=complex)
        readout = postselect_cv_output(
            density_matrix,
            readout_mode=evolution.readout_mode,
            layout=layout,
            max_fock_level=kernel.n_fock,
            n_dv_qubits=system.n_qubits,
        )
        readout["layout"] = layout
        readout["density_matrix"] = density_matrix
        return readout

    state, _, _ = cv_util.simulate(
        qc,
        shots=1,
        return_fockcounts=False,
        add_save_statevector=True,
    )
    statevector = np.asarray(state.data, dtype=complex)
    readout = postselect_cv_output(
        statevector,
        readout_mode=evolution.readout_mode,
        layout=layout,
        max_fock_level=kernel.n_fock,
        n_dv_qubits=system.n_qubits,
    )
    readout["layout"] = layout
    readout["statevector"] = statevector
    return readout


def run_clean_lchs(
    system: PauliSystemSpec,
    kernel: KernelSpec,
    prep: StatePrepSpec,
    evolution: EvolutionSpec,
) -> CleanRunResult:
    coeffs = compute_lchs_coefficients(kernel)
    coeff_gap = coefficient_backend_gap(kernel)
    oracle = prepare_cv_oracle(kernel, prep, coeffs=coeffs)

    qc, oracle, build_meta = build_hybrid_circuit(
        system,
        kernel,
        prep,
        evolution,
        coeffs=coeffs,
        oracle=oracle,
    )
    readout = _simulate_with_readout(qc, evolution=evolution, kernel=kernel, system=system)

    ref_map = exact_reference_map(system)
    ref_vec = ref_map @ normalize_vector(system.init_state)

    truncated_oracle_map = exact_truncated_cv_map(system, kernel, seed_state=oracle.prepared_state)
    truncated_oracle_vec = truncated_oracle_map @ normalize_vector(system.init_state)

    truncated_ideal_map = exact_truncated_cv_map(system, kernel, seed_state=coeffs)
    truncated_ideal_vec = truncated_ideal_map @ normalize_vector(system.init_state)

    observed_vec = np.asarray(readout["observed_vector"], dtype=complex).reshape(-1)
    post_prob = float(readout["postselection_probability"])

    scale_exact, rel_exact = fit_global_scale(observed_vec, ref_vec)
    scale_trunc, rel_trunc = fit_global_scale(observed_vec, truncated_oracle_vec)

    if "postselected_density_matrix" in readout:
        rho_post = np.asarray(readout["postselected_density_matrix"], dtype=complex)
        fidelity_exact = fidelity_density_matrix_vs_pure(rho_post, ref_vec)
        fidelity_trunc = fidelity_density_matrix_vs_pure(rho_post, truncated_oracle_vec)
    else:
        fidelity_exact = state_fidelity(observed_vec, ref_vec)
        fidelity_trunc = state_fidelity(observed_vec, truncated_oracle_vec)

    resources = circuit_resource_report(qc)
    l_block, h_block = system_blocks(system)

    metadata: Dict[str, Any] = {
        **build_meta,
        "generator_matrix": generator_matrix(system),
        "l_block": l_block,
        "h_block": h_block,
        "oracle_metadata": dict(oracle.metadata),
        "oracle_apply_mode": oracle.apply_mode,
        "layout": readout["layout"],
        "ideal_truncated_vector": truncated_ideal_vec,
        "ideal_truncated_map": truncated_ideal_map,
        "ideal_vs_reference_fidelity": state_fidelity(truncated_ideal_vec, ref_vec),
        "oracle_vs_ideal_fidelity": state_fidelity(truncated_oracle_vec, truncated_ideal_vec),
        "statevector": readout.get("statevector"),
        "density_matrix": readout.get("density_matrix"),
        "postselected_density_matrix": readout.get("postselected_density_matrix"),
    }

    return CleanRunResult(
        system_label=system.label,
        coeff_backend=kernel.coeff_backend,
        state_prep_method=prep.method,
        readout_mode=evolution.readout_mode,
        postselection_probability=post_prob,
        fidelity_vs_exact=float(fidelity_exact),
        fidelity_vs_truncated=float(fidelity_trunc),
        rel_error_vs_exact=float(rel_exact),
        rel_error_vs_truncated=float(rel_trunc),
        scale_vs_exact=complex(scale_exact),
        scale_vs_truncated=complex(scale_trunc),
        coeff_backend_gap=float(coeff_gap),
        oracle_fidelity=float(oracle.oracle_fidelity),
        observed_vector=observed_vec,
        exact_reference_vector=ref_vec,
        exact_truncated_vector=truncated_oracle_vec,
        exact_reference_map=ref_map,
        exact_truncated_map=truncated_oracle_map,
        circuit_depth=int(resources["depth"]),
        circuit_size=int(resources["size"]),
        count_ops=resources["count_ops"],
        metadata=metadata,
    )
