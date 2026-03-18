#!/usr/bin/env python3
"""
Heat-equation CV-DV LCHS implementation with MCX-based Trotter blocks.

This file keeps the current CV preparation and postselection pipeline from the
formed_* codepath, but replaces the DV evolution with the paper-style
incrementer/decrementer construction synthesized via Qiskit's KG24 MCX plugins.

This implementation is intentionally scoped to the current heat-only pass:
  - 1D heat equation
  - H = 0
  - modular incrementer/decrementer blocks

Run this file inside the ``cvdvu`` environment, which should contain:
  - qiskit 2.1.2
  - qiskit-aer 0.17.0
  - bosonic-qiskit 15.1
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from bosonic_qiskit import CVCircuit, QumodeRegister
from bosonic_qiskit import util as cv_util
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler.passes.synthesis.high_level_synthesis import HLSConfig
from qiskit.transpiler.passes.synthesis.plugin import (
    high_level_synthesis_plugin_names,
)
from scipy.linalg import expm

from formed_bosonic import (
    apply_cv_state_preparation,
    circuit_resource_report,
    prepare_dv_basis_state,
)
from formed_qutip import (
    LCHSParams,
    ODESystemFromPauli,
    fit_global_scale,
    lchs_seed_coefficients,
    run_qutip_lchs,
    state_fidelity,
)


@dataclass
class MCXHeatParams:
    max_fock_level: int
    n_trotter_steps: int
    state_prep_method: str
    snap_depth: int
    snap_restarts: int
    snap_maxiter: int
    grid_qubits: int
    clean_ancillas: int = 2
    mcx_plugin: str = "2_clean_kg24"
    photon_loss_rate: float = 0.0


def ensure_kg24_support(plugin_name: str) -> None:
    plugins = high_level_synthesis_plugin_names("mcx")
    if plugin_name not in plugins:
        joined = ", ".join(plugins)
        raise RuntimeError(
            f"MCX plugin '{plugin_name}' is unavailable. Installed MCX plugins: {joined}. "
            "Use the 'cvdvu' environment with qiskit 2.1.2."
        )


def qiskit_to_physics_index_map(n_qubits: int) -> np.ndarray:
    return np.array(
        [int(f"{i:0{n_qubits}b}"[::-1], 2) for i in range(2**n_qubits)], dtype=int
    )


def detect_layout_with_ancillas(max_fock_level: int, n_total_qubits: int) -> str:
    qmr = QumodeRegister(
        num_qumodes=1, num_qubits_per_qumode=int(np.log2(max_fock_level))
    )
    qbr = QuantumRegister(n_total_qubits, "q")
    qc = CVCircuit(qbr, qmr)

    probe = np.zeros(max_fock_level, dtype=complex)
    probe[1] = 1.0
    qc.cv_initialize(probe, qmr[0])

    state, _, _ = cv_util.simulate(
        qc, shots=1, return_fockcounts=False, add_save_statevector=True
    )
    vec = np.asarray(state.data, dtype=complex)
    idx = int(np.where(np.abs(vec) > 1e-12)[0][0])

    return "fock_major" if idx == 2**n_total_qubits else "qubit_major"


def extract_postselected_dv_with_ancillas(
    statevector: np.ndarray,
    *,
    layout: str,
    max_fock_level: int,
    n_dv_qubits: int,
    n_ancillas: int,
) -> np.ndarray:
    dv_dim = 2**n_dv_qubits
    total_qubits = n_dv_qubits + n_ancillas
    total_dim = 2**total_qubits

    if layout == "fock_major":
        qubit_slice = statevector[:total_dim]
        dv_qiskit = qubit_slice[:dv_dim]
    else:
        qubit_slice = statevector[0 :: max_fock_level][:total_dim]
        dv_qiskit = qubit_slice[:dv_dim]

    return dv_qiskit[qiskit_to_physics_index_map(n_dv_qubits)]


def modular_shift_matrix(n_qubits: int) -> np.ndarray:
    dim = 2**n_qubits
    shift = np.zeros((dim, dim), dtype=complex)
    for x in range(dim):
        shift[(x + 1) % dim, x] = 1.0
    return shift


def implemented_binary_position_generator(
    n_qubits: int, alpha: float, h_grid: float
) -> np.ndarray:
    r"""
    Effective DV generator implied by the current CD_pos = Q assumption.

    With D_Q(lam) = exp(-i lam x_hat ⊗ Q) and one paper-style step

        D_Q(2 theta) S D_Q(theta) S^\dagger S^\dagger D_Q(theta) S,

    conjugation by the modular shift makes the net generator diagonal:

        G = 2Q + S Q S^\dagger + S^\dagger Q S.

    This is not the Dirichlet heat generator; it is the exact reference for the
    currently implemented binary-position model.
    """
    dim = 2**n_qubits
    q_op = np.diag(np.arange(dim, dtype=float)).astype(complex)
    shift = modular_shift_matrix(n_qubits)
    g_op = 2.0 * q_op + shift @ q_op @ shift.conj().T + shift.conj().T @ q_op @ shift
    return (alpha / (h_grid**2)) * g_op


def heat_dirichlet_system_2q(alpha: float, h_grid: float) -> ODESystemFromPauli:
    s = alpha / (h_grid**2)
    return ODESystemFromPauli(
        pauli_strings_l=["II", "IX", "XX", "YY"],
        coeffs_l=[2.0 * s, -1.0 * s, -0.5 * s, -0.5 * s],
        pauli_strings_h=["II"],
        coeffs_h=[0.0],
    )


def physics_bit_weights(n_qubits: int) -> List[int]:
    return [1 << (n_qubits - 1 - qidx) for qidx in range(n_qubits)]


def _append_controlled_x(
    qc: QuantumCircuit,
    controls: Sequence[int],
    target: int,
) -> None:
    if len(controls) == 1:
        qc.cx(controls[0], target)
    elif len(controls) == 2:
        qc.ccx(controls[0], controls[1], target)
    elif len(controls) > 2:
        qc.mcx(list(controls), target)


def build_incrementer_template(n_qubits: int, n_ancillas: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits + n_ancillas, name=f"S_{n_qubits}")
    if n_qubits == 1:
        qc.x(0)
        return qc

    for target in range(0, n_qubits - 1):
        controls = list(range(target + 1, n_qubits))
        _append_controlled_x(qc, controls, target)
    qc.x(n_qubits - 1)
    return qc


def build_decrementer_template(n_qubits: int, n_ancillas: int) -> QuantumCircuit:
    qc = build_incrementer_template(n_qubits, n_ancillas).inverse()
    qc.name = f"Sdg_{n_qubits}"
    return qc


def synthesize_mcx_block(
    template: QuantumCircuit, plugin_name: str
) -> QuantumCircuit:
    ensure_kg24_support(plugin_name)
    return transpile(
        template,
        optimization_level=0,
        hls_config=HLSConfig(mcx=[plugin_name]),
    )


def block_logical_summary(block: QuantumCircuit) -> Dict[str, int]:
    counts = dict(block.count_ops())
    return {
        "x": int(counts.get("x", 0)),
        "cx": int(counts.get("cx", 0)),
        "ccx": int(counts.get("ccx", 0)),
        "mcx": int(counts.get("mcx", 0)),
    }


def compose_qubit_block(
    qc: CVCircuit,
    block: QuantumCircuit,
    data_reg: QuantumRegister,
    ancilla_reg: QuantumRegister,
) -> None:
    mapping = list(data_reg) + list(ancilla_reg)
    qc.compose(block, qubits=mapping, inplace=True)


def apply_cd_pos_block(
    qc: CVCircuit,
    mode,
    data_reg: QuantumRegister,
    lam: float,
) -> None:
    weights = physics_bit_weights(len(data_reg))
    total_weight = sum(weights) / 2.0
    if abs(total_weight) > 1e-14:
        qc.cv_d(-1.0j * lam * total_weight / np.sqrt(2.0), mode)

    for qidx, weight in enumerate(weights):
        alpha = 1.0j * lam * weight / (2.0 * np.sqrt(2.0))
        qc.cv_c_d(alpha, mode, data_reg[qidx])


def apply_mcx_heat_trotter_step(
    qc: CVCircuit,
    mode,
    data_reg: QuantumRegister,
    ancilla_reg: QuantumRegister,
    theta: float,
    incrementer: QuantumCircuit,
    decrementer: QuantumCircuit,
) -> None:
    apply_cd_pos_block(qc, mode, data_reg, 2.0 * theta)
    compose_qubit_block(qc, incrementer, data_reg, ancilla_reg)
    apply_cd_pos_block(qc, mode, data_reg, theta)
    compose_qubit_block(qc, decrementer, data_reg, ancilla_reg)
    compose_qubit_block(qc, decrementer, data_reg, ancilla_reg)
    apply_cd_pos_block(qc, mode, data_reg, theta)
    compose_qubit_block(qc, incrementer, data_reg, ancilla_reg)


def build_mcx_heat_circuit(
    lchs_params: LCHSParams,
    mcx_params: MCXHeatParams,
    coeffs_seed: np.ndarray,
    *,
    init_basis_index: int,
    alpha: float,
    h_grid: float,
) -> Tuple[CVCircuit, Dict[str, float], Dict[str, object]]:
    if mcx_params.max_fock_level & (mcx_params.max_fock_level - 1):
        raise ValueError("max_fock_level must be a power of 2")
    if mcx_params.clean_ancillas < 2 and mcx_params.grid_qubits > 3:
        raise ValueError("2_clean_kg24 requires two clean ancillas for >=4 data qubits")

    qmr = QumodeRegister(
        num_qumodes=1,
        num_qubits_per_qumode=int(np.log2(mcx_params.max_fock_level)),
    )
    data_reg = QuantumRegister(mcx_params.grid_qubits, "q")
    ancilla_reg = QuantumRegister(mcx_params.clean_ancillas, "a")
    qc = CVCircuit(data_reg, ancilla_reg, qmr)
    mode = qmr[0]

    prep_meta = apply_cv_state_preparation(
        qc,
        mode,
        coeffs_seed,
        prep_qubit=data_reg[0],
        max_fock_level=mcx_params.max_fock_level,
        method=mcx_params.state_prep_method,
        snap_depth=mcx_params.snap_depth,
        snap_restarts=mcx_params.snap_restarts,
        snap_maxiter=mcx_params.snap_maxiter,
    )

    if abs(lchs_params.r_prime) > 1e-14:
        qc.cv_sq(-lchs_params.r_prime, mode)

    prepare_dv_basis_state(qc, data_reg, init_basis_index)

    inc_template = build_incrementer_template(
        mcx_params.grid_qubits, mcx_params.clean_ancillas
    )
    dec_template = build_decrementer_template(
        mcx_params.grid_qubits, mcx_params.clean_ancillas
    )
    inc_block = synthesize_mcx_block(inc_template, mcx_params.mcx_plugin)
    dec_block = synthesize_mcx_block(dec_template, mcx_params.mcx_plugin)

    theta = alpha * lchs_params.total_time / (
        mcx_params.n_trotter_steps * (h_grid**2)
    )
    for _ in range(mcx_params.n_trotter_steps):
        apply_mcx_heat_trotter_step(
            qc,
            mode,
            data_reg,
            ancilla_reg,
            theta,
            inc_block,
            dec_block,
        )

    if abs(lchs_params.r_target) > 1e-14:
        qc.cv_sq(+lchs_params.r_target, mode)

    logical_inc = block_logical_summary(inc_template)
    logical_dec = block_logical_summary(dec_template)
    synth_inc = dict(inc_block.count_ops())
    synth_dec = dict(dec_block.count_ops())

    logical_resources = {
        "kg24_plugin": mcx_params.mcx_plugin,
        "clean_ancillas": mcx_params.clean_ancillas,
        "n_trotter_steps": mcx_params.n_trotter_steps,
        "cd_pos_blocks_per_step": 3,
        "incrementer_blocks_per_step": 2,
        "decrementer_blocks_per_step": 2,
        "incrementer_logical": logical_inc,
        "decrementer_logical": logical_dec,
        "incrementer_synth_ops": synth_inc,
        "decrementer_synth_ops": synth_dec,
        "total_logical_mcx_count": mcx_params.n_trotter_steps
        * (2 * logical_inc["mcx"] + 2 * logical_dec["mcx"]),
        "total_cd_pos_blocks": 3 * mcx_params.n_trotter_steps,
        "total_incrementer_blocks": 2 * mcx_params.n_trotter_steps,
        "total_decrementer_blocks": 2 * mcx_params.n_trotter_steps,
    }

    return qc, prep_meta, logical_resources


def run_mcx_heat_method(
    lchs_params: LCHSParams,
    mcx_params: MCXHeatParams,
    coeffs_seed: np.ndarray,
    *,
    init_basis_index: int,
    alpha: float,
    h_grid: float,
    u_ref_dirichlet: np.ndarray | None = None,
    u_qutip_dirichlet: np.ndarray | None = None,
) -> Dict[str, object]:
    qc, prep_meta, logical_resources = build_mcx_heat_circuit(
        lchs_params=lchs_params,
        mcx_params=mcx_params,
        coeffs_seed=coeffs_seed,
        init_basis_index=init_basis_index,
        alpha=alpha,
        h_grid=h_grid,
    )

    state, _, _ = cv_util.simulate(
        qc, shots=1, return_fockcounts=False, add_save_statevector=True
    )
    vec = np.asarray(state.data, dtype=complex)

    total_qubits = mcx_params.grid_qubits + mcx_params.clean_ancillas
    layout = detect_layout_with_ancillas(mcx_params.max_fock_level, total_qubits)
    dv_vec = extract_postselected_dv_with_ancillas(
        vec,
        layout=layout,
        max_fock_level=mcx_params.max_fock_level,
        n_dv_qubits=mcx_params.grid_qubits,
        n_ancillas=mcx_params.clean_ancillas,
    )

    init_state = np.zeros(2**mcx_params.grid_qubits, dtype=complex)
    init_state[init_basis_index] = 1.0
    u_impl_ref = (
        expm(
            -implemented_binary_position_generator(
                mcx_params.grid_qubits, alpha, h_grid
            )
            * lchs_params.total_time
        )
        @ init_state
    )

    eta_impl, err_impl = fit_global_scale(dv_vec, u_impl_ref)
    result: Dict[str, object] = {
        "dv_mcx": dv_vec,
        "layout": layout,
        "post_prob": float(np.real(np.vdot(dv_vec, dv_vec))),
        "vs_implemented_model_rel_error": err_impl,
        "vs_implemented_model_fidelity": state_fidelity(dv_vec, u_impl_ref),
        "eta_vs_implemented_model": eta_impl,
        "u_implemented_model_ref": u_impl_ref,
        "resources": circuit_resource_report(qc),
        "logical_resources": logical_resources,
        "prep_meta": prep_meta,
    }

    if u_ref_dirichlet is not None:
        eta_ref, err_ref = fit_global_scale(dv_vec, u_ref_dirichlet)
        result.update(
            {
                "vs_dirichlet_classical_rel_error": err_ref,
                "vs_dirichlet_classical_fidelity": state_fidelity(
                    dv_vec, u_ref_dirichlet
                ),
                "eta_vs_dirichlet_classical": eta_ref,
            }
        )
    if u_qutip_dirichlet is not None:
        eta_q, err_q = fit_global_scale(dv_vec, u_qutip_dirichlet)
        result.update(
            {
                "vs_dirichlet_qutip_rel_error": err_q,
                "vs_dirichlet_qutip_fidelity": state_fidelity(
                    dv_vec, u_qutip_dirichlet
                ),
                "eta_vs_dirichlet_qutip": eta_q,
            }
        )

    return result


def _basis_state_with_ancillas(
    n_dv_qubits: int, n_ancillas: int, physics_index: int
) -> Statevector:
    qiskit_index = qiskit_to_physics_index_map(n_dv_qubits)[physics_index]
    total_index = qiskit_index
    vec = np.zeros(2 ** (n_dv_qubits + n_ancillas), dtype=complex)
    vec[total_index] = 1.0
    return Statevector(vec)


def validate_permutation_block(
    block: QuantumCircuit,
    *,
    n_dv_qubits: int,
    n_ancillas: int,
    direction: int,
) -> None:
    dim = 2**n_dv_qubits
    for x in range(dim):
        sv = _basis_state_with_ancillas(n_dv_qubits, n_ancillas, x).evolve(block)
        probs = np.abs(sv.data) ** 2
        out_index = int(np.argmax(probs))
        anc_index = out_index >> n_dv_qubits
        if anc_index != 0:
            raise AssertionError(
                f"Ancilla leakage detected for x={x}: ancilla basis index {anc_index}"
            )
        qiskit_dv = out_index & (dim - 1)
        physics_out = qiskit_to_physics_index_map(n_dv_qubits)[qiskit_dv]
        expected = (x + direction) % dim
        if physics_out != expected:
            raise AssertionError(
                f"Permutation mismatch for x={x}: expected {expected}, observed {physics_out}"
            )


if __name__ == "__main__":
    alpha = 1.0
    h_grid = 1.0
    init_basis_index = 1

    lchs_params = LCHSParams(
        total_time=1.0,
        n_fock=64,
        n_coeff=48,
        r_target=6.0,
        r_prime=4.0,
        beta=0.3,
        n_quad=300,
        coeff_method="explicit_overlap",
    )

    mcx_params = MCXHeatParams(
        max_fock_level=64,
        n_trotter_steps=10,
        state_prep_method="injection",
        snap_depth=10,
        snap_restarts=2,
        snap_maxiter=100,
        grid_qubits=2,
        clean_ancillas=2,
        mcx_plugin="2_clean_kg24",
    )

    system_2q = heat_dirichlet_system_2q(alpha, h_grid)
    init_state = np.zeros(2 ** system_2q.n_qubits(), dtype=complex)
    init_state[init_basis_index] = 1.0

    qutip_result = run_qutip_lchs(system_2q, lchs_params, init_state)
    coeffs_seed = qutip_result["coeffs_seed"]

    inc_block = synthesize_mcx_block(
        build_incrementer_template(mcx_params.grid_qubits, mcx_params.clean_ancillas),
        mcx_params.mcx_plugin,
    )
    dec_block = synthesize_mcx_block(
        build_decrementer_template(mcx_params.grid_qubits, mcx_params.clean_ancillas),
        mcx_params.mcx_plugin,
    )

    for n_qubits in (2, 3, 4):
        inc = synthesize_mcx_block(
            build_incrementer_template(n_qubits, mcx_params.clean_ancillas),
            mcx_params.mcx_plugin,
        )
        dec = synthesize_mcx_block(
            build_decrementer_template(n_qubits, mcx_params.clean_ancillas),
            mcx_params.mcx_plugin,
        )
        validate_permutation_block(
            inc,
            n_dv_qubits=n_qubits,
            n_ancillas=mcx_params.clean_ancillas,
            direction=+1,
        )
        validate_permutation_block(
            dec,
            n_dv_qubits=n_qubits,
            n_ancillas=mcx_params.clean_ancillas,
            direction=-1,
        )
        if "mcx" in inc.count_ops() or "mcx" in dec.count_ops():
            raise AssertionError(f"Unsynthesized mcx remained for n_qubits={n_qubits}")

    result = run_mcx_heat_method(
        lchs_params=lchs_params,
        mcx_params=mcx_params,
        coeffs_seed=coeffs_seed,
        init_basis_index=init_basis_index,
        alpha=alpha,
        h_grid=h_grid,
        u_ref_dirichlet=qutip_result["u_ref"],
        u_qutip_dirichlet=qutip_result["u_cv"],
    )

    print("\n=== Summary: formed_MCX.py ===")
    print(
        f"n_qubits={mcx_params.grid_qubits}, n_fock={lchs_params.n_fock}, "
        f"n_coeff={lchs_params.n_coeff}"
    )
    print(
        f"r={lchs_params.r_target}, r'={lchs_params.r_prime}, beta={lchs_params.beta}, "
        f"T={lchs_params.total_time}, trotter_steps={mcx_params.n_trotter_steps}"
    )
    print(
        f"kg24_plugin={result['logical_resources']['kg24_plugin']}, "
        f"clean_ancillas={result['logical_resources']['clean_ancillas']}"
    )
    print(f"post_prob={result['post_prob']:.6e}, layout={result['layout']}")
    print(
        "rel_err vs impl model: "
        f"{result['vs_implemented_model_rel_error']:.6e}"
    )
    print(
        "fidelity vs impl mod : "
        f"{result['vs_implemented_model_fidelity']:.8f}"
    )
    print(
        "rel_err vs dirichlet : "
        f"{result.get('vs_dirichlet_classical_rel_error', float('nan')):.6e}"
    )
    print(
        "fidelity vs dirichlet: "
        f"{result.get('vs_dirichlet_classical_fidelity', float('nan')):.8f}"
    )
    print(
        "rel_err vs qutip     : "
        f"{result.get('vs_dirichlet_qutip_rel_error', float('nan')):.6e}"
    )
    print(
        "fidelity vs qutip    : "
        f"{result.get('vs_dirichlet_qutip_fidelity', float('nan')):.8f}"
    )
    if result["prep_meta"]:
        print(f"state-prep metadata  : {result['prep_meta']}")

    resources = result["resources"]
    print(f"depth={resources['depth']}, size={resources['size']}")
    print(f"count_ops={resources['count_ops']}")
    print(f"logical_resources={result['logical_resources']}")
    print(
        f"inc_block_ops_2q={dict(inc_block.count_ops())}, "
        f"dec_block_ops_2q={dict(dec_block.count_ops())}"
    )
