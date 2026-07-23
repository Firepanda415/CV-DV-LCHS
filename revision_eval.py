#!/usr/bin/env python3
"""Revision evidence driver for the CV-DV LCHS study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import shlex
import signal
import subprocess
import sys
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.special import eval_hermite, gammaln

from clean_core import (
    EvolutionSpec,
    KernelSpec,
    PauliSystemSpec,
    PauliTerm,
    StatePrepSpec,
    basis_state,
    build_dirichlet_heat_system,
    build_neumann_heat_system,
    build_pauli_system,
    build_periodic_heat_system,
    code_l_terms,
    coefficient_backend_gap,
    compute_lchs_coefficients,
    compute_lchs_coefficients_explicit,
    decompose_matrix_to_pauli_terms,
    exact_reference_map,
    exact_truncated_cv_map,
    exact_truncated_cv_map_h0,
    gamma_hbar1,
    kernel_g_beta,
    normalize_vector,
    padded_seed_state,
    pauli_sum_matrix,
    position_operator,
    reorder_physics_to_qiskit,
    scaled_map_metrics,
    squeeze_operator,
    state_fidelity,
    system_blocks,
    truncated_oscillator_states,
    zeroth_moment_scale,
)
from clean_hybrid import (
    _bosonic_sq_for_prepare,
    _conditional_displacement_alpha,
    build_hybrid_circuit,
    circuit_resource_report,
    run_clean_lchs,
)
from clean_oracles import (
    OraclePreparation,
    align_oracle_global_phase,
    detect_statevector_layout,
    displacement_matrix,
    optimize_snap_d,
    oracle_from_snap_parameter_payload,
    prepare_cv_oracle,
    snap_d_initial_guess_from_oracle,
    snap_parameter_payload,
)


SQRT2 = np.sqrt(2.0)
SQUEEZE_CONVENTION = (
    "dense S(-r) broad-x; bosonic prepare +r / postselect -r; "
    "L_code = sqrt(2) L at coupling boundary only"
)


def to_paper_coordinate(q: Any) -> Any:
    return SQRT2 * q


def to_code_coordinate(x: Any) -> Any:
    return x / SQRT2


def to_paper_lambda(lam_code: Any) -> Any:
    return lam_code / SQRT2


def to_code_lambda(lam: Any) -> Any:
    return SQRT2 * lam


def to_paper_gamma(gamma_code: Any) -> Any:
    return gamma_code / 2.0


def to_code_gamma(gamma: Any) -> Any:
    return 2.0 * gamma


def x_norm_bound_paper(n_fock: int) -> float:
    return float(2.0 * np.sqrt(n_fock - 1))


def x_norm_bound_code(n_fock: int) -> float:
    return float(np.sqrt(2.0 * (n_fock - 1)))


def to_paper_l_norm(l_coupling_norm_code: Any) -> Any:
    return l_coupling_norm_code / SQRT2


def to_code_l_coupling_norm(l_norm_paper: Any) -> Any:
    return SQRT2 * l_norm_paper


def build_sidecar(command_args: argparse.Namespace, extra: dict) -> dict:
    """Build the small provenance record shared by revision artifacts."""

    root = Path(__file__).resolve().parent

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    parameters = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(command_args).items()
        if not key.startswith("_") and not callable(value)
    }
    extra_fields = dict(extra)
    input_artifacts = extra_fields.pop("input_artifacts", {})
    sidecar = {
        "git_head": git_head,
        "git_dirty": bool(git_status.strip()),
        "command": shlex.join([sys.executable, *sys.argv]),
        "parameters": parameters,
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {
            name: sha256(root / name)
            for name in (
                "revision_eval.py",
                "clean_core.py",
                "clean_hybrid.py",
                "clean_oracles.py",
            )
        },
        "input_artifact_sha256": {
            label: sha256(Path(path)) for label, path in input_artifacts.items()
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "qiskit": importlib.metadata.version("qiskit"),
            "bosonic_qiskit": importlib.metadata.version("bosonic-qiskit"),
        },
        "coordinate_conversion": (
            "paper hbar=2: x=sqrt(2) q_code, lambda=lambda_code/sqrt(2), "
            "gamma=gamma_code/2; L_code=sqrt(2) L"
        ),
        "squeeze_convention": SQUEEZE_CONVENTION,
        "physical_qumodes": 1,
    }
    sidecar.update(extra_fields)
    return sidecar


def write_artifact(output: Path, data: Any, sidecar: dict) -> None:
    """Write one JSON artifact or one CSV plus its metadata sidecar."""

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".json":
        document = dict(data)
        document.setdefault("hbar_convention", sidecar["hbar_convention"])
        document["sidecar"] = sidecar
        output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        return

    if output.suffix == ".csv":
        rows = [dict(row) for row in data]
        for row in rows:
            row.setdefault("hbar_convention", sidecar["hbar_convention"])
        with output.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(
                dict.fromkeys(key for row in rows for key in row)
            )
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        metadata = {
            "sidecar": sidecar,
            "gate_evaluation": sidecar.get("gate_evaluation"),
        }
        output.with_suffix(".meta.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        return

    raise ValueError("output must end in .json or .csv")


def _criterion(name: str, value: float, threshold: float, comparator: str) -> dict:
    passed = value <= threshold if comparator == "<=" else value >= threshold
    return {
        "name": name,
        "value": float(value),
        "threshold": float(threshold),
        "comparator": comparator,
        "pass": bool(passed),
    }


def _complex_fields(name: str, values: Any) -> dict:
    array = np.asarray(values)
    return {
        f"{name}_re": np.real(array).tolist(),
        f"{name}_im": np.imag(array).tolist(),
    }


def _analytic_squeezed_vacuum(n_fock: int, r: float) -> np.ndarray:
    state = np.zeros(n_fock, dtype=complex)
    for m in range(n_fock // 2):
        state[2 * m] = (
            math.sqrt(math.factorial(2 * m))
            / (2**m * math.factorial(m))
            * np.tanh(r) ** m
        )
    return state / np.linalg.norm(state)


def _bosonic_squeezed_vacuum(n_fock: int, r: float, layout: str) -> np.ndarray:
    from bosonic_qiskit import CVCircuit, QumodeRegister
    from bosonic_qiskit import util as cv_util
    from qiskit import QuantumRegister

    qmr = QumodeRegister(
        num_qumodes=1,
        num_qubits_per_qumode=int(np.log2(n_fock)),
    )
    probe = QuantumRegister(1, "probe")
    qc = CVCircuit(probe, qmr)
    qc.cv_sq(_bosonic_sq_for_prepare(r), qmr[0])
    state, _, _ = cv_util.simulate(
        qc,
        shots=1,
        return_fockcounts=False,
        add_save_statevector=True,
    )
    vector = np.asarray(state.data, dtype=complex)
    if layout == "fock_major":
        return vector[::2]
    return vector[:n_fock]


def _paper_coordinate_coefficients(kernel: KernelSpec) -> np.ndarray:
    """Direct paper-coordinate integral fixture for Gate A."""

    sigma_prime = float(np.exp(kernel.r_prime))
    gamma_code = gamma_hbar1(kernel.r_target, kernel.r_prime)
    gamma_paper = to_paper_gamma(gamma_code)
    q_tail = np.sqrt(max(-np.log(1e-14) / max(gamma_code, 1e-15), 1.0))
    q_max = max(8.0 * sigma_prime, 1.15 * q_tail, 12.0)
    x_max = to_paper_coordinate(q_max)
    limit = max(100, kernel.n_quad)
    coeffs = np.zeros(kernel.n_coeff, dtype=complex)

    for n in range(kernel.n_coeff):
        prefactor = np.exp(-0.5 * (n * np.log(2.0) + gammaln(n + 1.0)))

        def integrand(x: float) -> complex:
            return (
                prefactor
                * eval_hermite(n, x / (SQRT2 * sigma_prime))
                * kernel_g_beta(np.array([x]), kernel.beta)[0]
                * np.exp(-gamma_paper * x * x)
            )

        real = quad(
            lambda x: float(np.real(integrand(x))),
            -x_max,
            x_max,
            limit=limit,
            epsabs=1e-10,
            epsrel=1e-9,
        )[0]
        imag = quad(
            lambda x: float(np.imag(integrand(x))),
            -x_max,
            x_max,
            limit=limit,
            epsabs=1e-10,
            epsrel=1e-9,
        )[0]
        coeffs[n] = real + 1.0j * imag

    return coeffs / np.linalg.norm(coeffs)


def _scalar_map(
    x_operator: np.ndarray,
    l_coupling: np.ndarray,
    h_block: np.ndarray,
    psi: np.ndarray,
    phi: np.ndarray,
    total_time: float,
) -> np.ndarray:
    """Construct the direct scalar fixture without Pauli-system helpers."""

    joint = np.kron(x_operator, l_coupling) + np.kron(
        np.eye(x_operator.shape[0]), h_block
    )
    unitary = expm(-1.0j * total_time * joint)
    embed = np.kron(psi.reshape((-1, 1)), np.eye(1, dtype=complex))
    project = np.kron(phi.conj().reshape((1, -1)), np.eye(1, dtype=complex))
    return np.asarray(project @ unitary @ embed, dtype=complex)


def run_convention(args: argparse.Namespace) -> None:
    criteria = []
    raw_data = {}

    x_code_small = position_operator(8)
    x_paper_small = to_paper_coordinate(x_code_small)
    l_terms = (PauliTerm("I", 0.7), PauliTerm("X", 0.2), PauliTerm("Z", 0.3))
    l_block = pauli_sum_matrix(l_terms)
    h_block = np.array([[0.3, 0.0], [0.0, -0.3]], dtype=complex)
    paper_generator = np.kron(x_paper_small, l_block) + np.kron(
        np.eye(8), h_block
    )
    # The code side must go through the implementation's own conversion
    # (code_l_terms), so this criterion certifies the stored constant.
    code_generator = np.kron(
        x_code_small, pauli_sum_matrix(code_l_terms(l_terms))
    ) + np.kron(np.eye(8), h_block)
    generator_error = np.linalg.norm(code_generator - paper_generator, "fro")
    generator_error /= np.linalg.norm(paper_generator, "fro")
    criteria.append(
        _criterion("generator_pairing_relative_matrix_error", generator_error, 1e-12, "<=")
    )
    raw_data["generator_pairing"] = {
        **_complex_fields("paper_generator", paper_generator),
        **_complex_fields("code_generator", code_generator),
    }

    n_fock = 64
    layout = detect_statevector_layout(n_fock, 1)
    x_code = position_operator(n_fock)
    vacuum = basis_state(n_fock, 0)
    squeeze_raw = {}
    for r in (0.5, 1.0):
        dense = squeeze_operator(n_fock, -r) @ vacuum
        bosonic = _bosonic_squeezed_vacuum(n_fock, r, layout)
        analytic = _analytic_squeezed_vacuum(n_fock, r)
        expected_variance = np.exp(2.0 * r) / 2.0
        variance = float(np.real(np.vdot(dense, x_code @ x_code @ dense)))
        suffix = str(r).replace(".", "p")
        criteria.extend(
            (
                _criterion(
                    f"squeeze_dense_bosonic_fidelity_r_{suffix}",
                    state_fidelity(dense, bosonic),
                    1.0 - 1e-8,
                    ">=",
                ),
                _criterion(
                    f"squeeze_dense_analytic_fidelity_r_{suffix}",
                    state_fidelity(dense, analytic),
                    1.0 - 1e-8,
                    ">=",
                ),
                _criterion(
                    f"squeeze_x_variance_relative_error_r_{suffix}",
                    abs(variance - expected_variance) / expected_variance,
                    1e-6,
                    "<=",
                ),
            )
        )
        squeeze_raw[str(r)] = {
            **_complex_fields("dense_state", dense),
            **_complex_fields("bosonic_state", bosonic),
            **_complex_fields("analytic_projected_state", analytic),
            "expected_x_variance": float(expected_variance),
        }
    raw_data["squeeze_states"] = squeeze_raw

    lam_paper = 0.13
    lam_code = to_code_lambda(lam_paper)
    paper_displacement = expm(-1.0j * lam_paper * x_paper_small)
    # Route through the hybrid implementation's coupling-to-displacement map.
    code_displacement = displacement_matrix(_conditional_displacement_alpha(lam_code), 8)
    displacement_error = np.linalg.norm(
        paper_displacement - code_displacement, "fro"
    ) / np.linalg.norm(paper_displacement, "fro")
    criteria.append(
        _criterion("displacement_relative_operator_error", displacement_error, 1e-12, "<=")
    )
    raw_data["displacement"] = {
        "lambda_paper": lam_paper,
        "lambda_code": float(lam_code),
        **_complex_fields("paper_displacement", paper_displacement),
        **_complex_fields("code_displacement", code_displacement),
    }

    kernel = KernelSpec(
        r_target=7.9,
        r_prime=4.1,
        beta=0.5,
        n_coeff=48,
        n_fock=n_fock,
        n_quad=200,
        coeff_backend="explicit_overlap",
    )
    paper_coeffs = _paper_coordinate_coefficients(kernel)
    code_coeffs = compute_lchs_coefficients(kernel)
    overlap = np.vdot(paper_coeffs, code_coeffs)
    aligned_paper_coeffs = paper_coeffs * overlap / abs(overlap)
    coefficient_distance = np.linalg.norm(aligned_paper_coeffs - code_coeffs)
    alpha_paper = zeroth_moment_scale(kernel, aligned_paper_coeffs)
    alpha_code = zeroth_moment_scale(kernel, code_coeffs)
    alpha_relative_error = abs(alpha_paper - alpha_code) / abs(alpha_code)
    criteria.extend(
        (
            _criterion(
                "coefficient_coordinate_phase_aligned_l2_distance",
                coefficient_distance,
                1e-8,
                "<=",
            ),
            _criterion(
                "coefficient_coordinate_alpha_relative_error",
                alpha_relative_error,
                1e-8,
                "<=",
            ),
        )
    )
    coefficient_output = args.output.with_name(f"{args.output.stem}_coefficients.json")
    coefficient_sidecar = build_sidecar(
        args,
        {
            "evaluation": "exact_finite_map",
            "state_prep_method": "ideal_injection",
            "n_coeff": kernel.n_coeff,
            "n_fock": kernel.n_fock,
            "dv_dim": 2,
            "simulator_qubits_per_qumode": int(np.log2(kernel.n_fock)),
            "hbar_convention": "code_hbar1",
        },
    )
    write_artifact(
        coefficient_output,
        {
            "kernel": {
                "r_target": kernel.r_target,
                "r_prime": kernel.r_prime,
                "beta": kernel.beta,
                "n_coeff": kernel.n_coeff,
                "n_fock": kernel.n_fock,
                "n_quad": kernel.n_quad,
                "coeff_backend": kernel.coeff_backend,
            },
            **_complex_fields("paper_coefficients_common_gauge", aligned_paper_coeffs),
            **_complex_fields("code_coefficients", code_coeffs),
            **_complex_fields("alpha_paper", alpha_paper),
            **_complex_fields("alpha_code", alpha_code),
        },
        coefficient_sidecar,
    )

    psi, phi = truncated_oscillator_states(kernel, code_coeffs)
    scalar_l_paper = np.array([[1.0]], dtype=complex)
    scalar_h = np.array([[0.0]], dtype=complex)
    t0_map = _scalar_map(
        x_code,
        to_code_l_coupling_norm(scalar_l_paper),
        scalar_h,
        psi,
        phi,
        0.0,
    )
    t0_defect = np.linalg.norm(alpha_code * t0_map - np.eye(1), 2)
    paper_map = _scalar_map(
        to_paper_coordinate(x_code),
        scalar_l_paper,
        scalar_h,
        psi,
        phi,
        0.1,
    )
    code_map = _scalar_map(
        x_code,
        to_code_l_coupling_norm(scalar_l_paper),
        scalar_h,
        psi,
        phi,
        0.1,
    )
    initial = np.array([1.0], dtype=complex)
    scalar_error = np.linalg.norm(paper_map @ initial - code_map @ initial)
    scalar_error /= np.linalg.norm(paper_map @ initial)
    criteria.extend(
        (
            _criterion("scalar_t0_normalized_map_defect", t0_defect, 1e-12, "<="),
            _criterion(
                "scalar_coordinate_relative_output_error", scalar_error, 1e-8, "<="
            ),
        )
    )
    raw_data["scalar_maps"] = {
        **_complex_fields("initial", initial),
        **_complex_fields("t0_map", t0_map),
        **_complex_fields("paper_map", paper_map),
        **_complex_fields("code_map", code_map),
    }

    gate_evaluation = {
        "criteria": criteria,
        "all_pass": all(item["pass"] for item in criteria),
    }
    sidecar = build_sidecar(
        args,
        {
            "gate_evaluation": gate_evaluation,
            "evaluation": ["circuit_statevector", "exact_finite_map"],
            "state_prep_method": "ideal_injection",
            "n_coeff": kernel.n_coeff,
            "n_fock": kernel.n_fock,
            "dv_dim": 2,
            "physical_qumodes": 1,
            "simulator_qubits_per_qumode": int(np.log2(kernel.n_fock)),
            "hbar_convention": "code_hbar1",
            "input_artifacts": {"coefficient_fixture": coefficient_output},
        },
    )
    write_artifact(
        args.output,
        {
            "gate_evaluation": gate_evaluation,
            "coefficient_artifact": str(coefficient_output),
            "raw_data": raw_data,
        },
        sidecar,
    )
    print(f"[invariant] Gate A automated criteria all_pass={gate_evaluation['all_pass']}")
    print(f"[hbar=1] wrote {args.output}")


RETAINED_KERNELS = {
    "dirichlet": {"r_target": 7.9, "r_prime": 4.1, "beta": 0.5},
    "neumann": {"r_target": 7.9, "r_prime": 4.0, "beta": 0.3},
    "periodic": {"r_target": 8.1, "r_prime": 4.1, "beta": 0.3},
}


def _heat_system(
    boundary: str,
    total_time: float,
    init_state: np.ndarray | None = None,
    num_qubits: int = 2,
) -> PauliSystemSpec:
    builders = {
        "dirichlet": build_dirichlet_heat_system,
        "neumann": build_neumann_heat_system,
        "periodic": build_periodic_heat_system,
    }
    return builders[boundary](
        num_qubits=num_qubits,
        alpha=1.0,
        grid_spacing=1.0,
        total_time=total_time,
        init_state=init_state,
        init_basis_index=1 if init_state is None else None,
    )


def first_order_trotter_bound(
    system: PauliSystemSpec,
    n_fock: int,
    total_time: float,
    n_t: int,
) -> dict:
    """Return the code-convention first-order Lie--Trotter bound."""

    l_terms = code_l_terms(system.l_terms)
    h_terms = system.h_terms
    x_bound = x_norm_bound_code(n_fock)

    def commutator_norm(left: PauliTerm, right: PauliTerm) -> float:
        left_pauli = pauli_sum_matrix((PauliTerm(left.label, 1.0),))
        right_pauli = pauli_sum_matrix((PauliTerm(right.label, 1.0),))
        return float(np.linalg.norm(left_pauli @ right_pauli - right_pauli @ left_pauli, 2))

    ll = x_bound**2 * sum(
        abs(left.coeff * right.coeff) * commutator_norm(left, right)
        for index, left in enumerate(l_terms)
        for right in l_terms[index + 1 :]
    )
    lh = x_bound * sum(
        abs(left.coeff * right.coeff) * commutator_norm(left, right)
        for left in l_terms
        for right in h_terms
    )
    hh = sum(
        abs(left.coeff * right.coeff) * commutator_norm(left, right)
        for index, left in enumerate(h_terms)
        for right in h_terms[index + 1 :]
    )
    gamma = float(ll + lh + hh)
    return {
        "Gamma_1": gamma,
        "Gamma_1_LL": float(ll),
        "Gamma_1_LH": float(lh),
        "Gamma_1_HH": float(hh),
        "eps_t_bound": float(total_time**2 * gamma / (2.0 * n_t)),
    }


def reconstruct_circuit_map(
    system: PauliSystemSpec,
    kernel: KernelSpec,
    prep: StatePrepSpec,
    evolution: EvolutionSpec,
    coeffs: np.ndarray,
    oracle: OraclePreparation,
) -> np.ndarray:
    """Reconstruct a DV map while reusing one frozen oscillator oracle."""

    columns = []
    for j in range(system.dv_dim):
        column_system = replace(system, init_state=basis_state(system.dv_dim, j))
        result = run_clean_lchs(
            column_system,
            kernel,
            prep,
            evolution,
            coeffs=coeffs,
            oracle=oracle,
        )
        columns.append(result.observed_vector)
    return np.column_stack(columns)


def map_error_chain(
    system: PauliSystemSpec,
    kernel: KernelSpec,
    target_coeffs: np.ndarray,
    prepared_coeffs: np.ndarray,
    circuit_map: np.ndarray,
) -> dict:
    """Return target-scale error chains and prepared-scale diagnostics."""

    target_map = exact_truncated_cv_map(system, kernel, target_coeffs)
    prepared_map = exact_truncated_cv_map(system, kernel, prepared_coeffs)
    reference_map = exact_reference_map(system)
    alpha_target = zeroth_moment_scale(kernel, target_coeffs)
    alpha_prepared = zeroth_moment_scale(kernel, prepared_coeffs)
    return _map_error_chain_from_maps(
        target_map,
        prepared_map,
        circuit_map,
        reference_map,
        alpha_target,
        alpha_prepared,
    )


def _map_error_chain_from_maps(
    target_map: np.ndarray,
    prepared_map: np.ndarray,
    circuit_map: np.ndarray,
    reference_map: np.ndarray,
    alpha_target: complex,
    alpha_prepared: complex,
) -> dict:
    """Apply the shared error-chain accounting to precomputed maps."""

    circuit = np.asarray(circuit_map, dtype=complex)

    def one_norm(order: int | str) -> dict:
        norm = lambda matrix: float(np.linalg.norm(matrix, order))
        model = alpha_target * target_map - reference_map
        synth = prepared_map - target_map
        trotter = circuit - prepared_map
        total = alpha_target * circuit - reference_map
        values = {
            "eps_model": norm(model),
            "eps_synth": norm(synth),
            "eps_t": norm(trotter),
            "eps_tot": norm(total),
        }
        bound = values["eps_model"] + abs(alpha_target) * (
            values["eps_synth"] + values["eps_t"]
        )
        if values["eps_tot"] > bound + 1e-10:
            raise AssertionError("target-scale error chain violates the triangle inequality")

        prepared_terms = (
            model,
            alpha_target * synth,
            alpha_prepared * trotter,
            (alpha_prepared - alpha_target) * prepared_map,
        )
        prepared_total = alpha_prepared * circuit - reference_map
        prepared_sum = sum(prepared_terms, np.zeros_like(prepared_total))
        values["triangle_bound"] = float(bound)
        values["prepared_scale"] = {
            "eps_tot": norm(prepared_total),
            "model_term": norm(prepared_terms[0]),
            "synth_term": norm(prepared_terms[1]),
            "circuit_term": norm(prepared_terms[2]),
            "scale_change_term": norm(prepared_terms[3]),
            "decomposition_residual": norm(prepared_total - prepared_sum),
        }
        return values

    spectral = one_norm(2)
    return {
        **spectral,
        "frobenius": one_norm("fro"),
        "alpha_target": complex(alpha_target),
        "alpha_prepared": complex(alpha_prepared),
    }


def _add_counts(*counts: dict[str, int]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for item in counts:
        total.update(item)
    return dict(sorted((name, int(value)) for name, value in total.items() if value))


def _basis_change_counts(label: str) -> dict[str, int]:
    return _add_counts(
        {"h": 2 * sum(ch in "XY" for ch in label)},
        {"rz": 2 * label.count("Y")},
    )


def _resource_segments(
    oracle: OraclePreparation,
    evolution: EvolutionSpec,
    system: PauliSystemSpec,
    kernel: KernelSpec,
) -> dict[str, dict[str, int]]:
    osc: Counter[str] = Counter()
    if oracle.apply_mode == "direct_injection":
        osc["initialize"] += 1
    elif oracle.apply_mode == "snap_d_layers":
        osc["SNAP"] += sum(
            abs(float(theta)) > 1e-12
            for layer in oracle.snap_thetas_per_layer
            for theta in layer
        )
        osc["D"] += len(oracle.snap_alphas_per_layer)
    elif oracle.apply_mode == "law_eberly_pulses":
        osc["x"] += 1
        names = {"jc": "jc", "r": "r", "sqr": "SQR"}
        for pulse in oracle.law_eberly_pulses:
            if abs(float(pulse.theta)) > 1e-12:
                osc[names[pulse.kind]] += 1
    if abs(kernel.r_prime) > 1e-14:
        osc["S"] += 1

    qiskit_state = reorder_physics_to_qiskit(
        normalize_vector(system.init_state), system.n_qubits
    )
    support = np.where(np.abs(qiskit_state) > 1e-12)[0]
    if support.size == 1 and abs(abs(qiskit_state[int(support[0])]) - 1.0) <= 1e-12:
        input_counts = {"x": int(int(support[0]).bit_count())}
    else:
        input_counts = {"initialize": 1}

    step: Counter[str] = Counter()
    for term in code_l_terms(system.l_terms):
        if abs(term.coeff) <= 1e-15:
            continue
        step.update(_basis_change_counts(term.label))
        active = sum(ch != "I" for ch in term.label)
        step["D" if active == 0 else "cD"] += 1
        step["cx"] += 2 * max(active - 1, 0)
    for term in system.h_terms:
        if abs(term.coeff) <= 1e-15:
            continue
        active = sum(ch != "I" for ch in term.label)
        if active == 0:
            continue
        step.update(_basis_change_counts(term.label))
        step["rz"] += 1
        step["cx"] += 2 * max(active - 1, 0)

    return {
        "C_osc_prep": _add_counts(dict(osc)),
        "C_input": _add_counts(input_counts),
        "C_step": _add_counts(dict(step)),
        "n_t_C_step": _add_counts(
            {name: count * evolution.n_trotter_steps for name, count in step.items()}
        ),
        "C_post": {"S": 1} if abs(kernel.r_target) > 1e-14 else {},
    }


def resource_ledger(
    qc: Any,
    *,
    oracle: OraclePreparation,
    evolution: EvolutionSpec,
    system: PauliSystemSpec,
    kernel: KernelSpec,
) -> dict:
    """Build and reconcile the four circuit-resource segments."""

    segments = _resource_segments(oracle, evolution, system, kernel)
    predicted = _add_counts(
        segments["C_osc_prep"],
        segments["C_input"],
        segments["n_t_C_step"],
        segments["C_post"],
    )
    actual = circuit_resource_report(qc)["count_ops"]
    if predicted != actual:
        raise ValueError(f"resource ledger mismatch: predicted={predicted}, actual={actual}")
    return {
        **segments,
        "C_run": actual,
        "C_run_total": int(sum(actual.values())),
        "physical_ops": {"fock_zero_postselection": 1},
        "simulator_only": ["initialize"] if oracle.apply_mode == "direct_injection" else [],
        "static_footprint": {
            "physical_qumodes": 1,
            "simulator_qubits_per_qumode": int(np.log2(kernel.n_fock)),
            "n_dv_qubits": system.n_qubits,
            "le_aux_qubits": int(oracle.apply_mode == "law_eberly_pulses"),
        },
    }


def _ledger_with_probability(ledger: dict, probability: float) -> dict:
    inverse = None if probability <= 0.0 else float(1.0 / probability)
    return {
        **ledger,
        "p_succ": float(probability),
        "inv_p_succ": inverse,
        "E_N_run": inverse,
        "E_C_accept": None
        if inverse is None
        else {
            name: float(count / probability) for name, count in ledger["C_run"].items()
        },
    }


def _kernel_fields(kernel: KernelSpec) -> dict:
    return {
        "r_target": kernel.r_target,
        "r_prime": kernel.r_prime,
        "beta": kernel.beta,
        "n_coeff": kernel.n_coeff,
        "n_fock": kernel.n_fock,
        "n_quad": kernel.n_quad,
        "coeff_backend": kernel.coeff_backend,
    }


def _freeze_coefficients(
    boundary: str,
    kernel: KernelSpec,
    coeffs: np.ndarray,
    args: argparse.Namespace,
    *,
    source: str,
    path: Path | None = None,
) -> Path:
    if path is None:
        path = Path(f"results_revision/coefficients_{boundary}.json")
    data = {
        "boundary": boundary,
        "kernel": _kernel_fields(kernel),
        **_complex_fields("coefficients", coeffs),
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing.pop("sidecar", None)
        if existing == {**data, "hbar_convention": "code_hbar1"}:
            return path
        raise ValueError(f"coefficient artifact is write-once: {path}")
    sidecar = build_sidecar(
        args,
        {
            "evaluation": "exact_finite_map",
            "state_prep_method": "ideal_injection",
            "n_coeff": kernel.n_coeff,
            "n_fock": kernel.n_fock,
            "dv_dim": 4,
            "simulator_qubits_per_qumode": int(np.log2(kernel.n_fock)),
            "hbar_convention": "code_hbar1",
            "coefficient_source": source,
        },
    )
    write_artifact(
        path,
        data,
        sidecar,
    )
    return path


def _load_frozen_coefficients(path: Path, kernel: KernelSpec) -> np.ndarray | None:
    if not path.exists():
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    expected = _kernel_fields(kernel)
    if any(document["kernel"].get(key) != value for key, value in expected.items()):
        return None
    return np.asarray(document["coefficients_re"], dtype=float) + 1.0j * np.asarray(
        document["coefficients_im"], dtype=float
    )


def _coefficients_for_finite(
    boundary: str,
    kernel: KernelSpec,
    args: argparse.Namespace,
) -> tuple[np.ndarray, Path]:
    canonical = Path(f"results_revision/coefficients_{boundary}.json")
    coeffs = _load_frozen_coefficients(canonical, kernel)
    if coeffs is not None:
        return coeffs, canonical
    # Different kernel parameters get their own frozen file so the
    # coeff-backend artifact (and its recorded hash) is never overwritten.
    legacy_tag = (
        f"r{kernel.r_target}_rp{kernel.r_prime}_b{kernel.beta}_N{kernel.n_coeff}"
    ).replace(".", "p")
    legacy = Path(f"results_revision/coefficients_{boundary}_{legacy_tag}.json")
    for candidate in (legacy, legacy.with_name(f"{legacy.stem}_NF{kernel.n_fock}.json")):
        coeffs = _load_frozen_coefficients(candidate, kernel)
        if coeffs is not None:
            return coeffs, candidate
    tag = f"{legacy_tag}_nq{kernel.n_quad}"
    if kernel.coeff_backend != "explicit_overlap":
        tag += f"_cb{kernel.coeff_backend}"
    tagged = Path(f"results_revision/coefficients_{boundary}_{tag}.json")
    coeffs = _load_frozen_coefficients(tagged, kernel)
    if tagged.exists() and coeffs is None:
        tagged = tagged.with_name(f"{tagged.stem}_NF{kernel.n_fock}.json")
        coeffs = _load_frozen_coefficients(tagged, kernel)
    if coeffs is None:
        fresh = compute_lchs_coefficients_explicit(
            kernel, k_max_scale=1.0, epsabs=1e-12, epsrel=1e-11
        )
        _freeze_coefficients(
            boundary,
            kernel,
            fresh,
            args,
            source="finite_map_selected_point",
            path=tagged,
        )
        coeffs = _load_frozen_coefficients(tagged, kernel)
    assert coeffs is not None
    return coeffs, tagged


def run_coeff_backend(args: argparse.Namespace) -> None:
    rows = []
    criteria = []
    coefficient_paths = {}
    for boundary, retained in RETAINED_KERNELS.items():
        kernel = KernelSpec(
            **retained,
            n_coeff=48,
            n_fock=64,
            n_quad=240,
            coeff_backend="explicit_overlap",
        )
        system = _heat_system(boundary, 1.0)
        reference_map = exact_reference_map(system)
        by_scale = {
            scale: compute_lchs_coefficients_explicit(
                kernel,
                k_max_scale=scale,
                epsabs=1e-12,
                epsrel=1e-11,
            )
            for scale in (0.75, 1.0, 1.5)
        }
        reference_coeffs = by_scale[1.0]
        alpha_reference = zeroth_moment_scale(kernel, reference_coeffs)
        map_reference = exact_truncated_cv_map_h0(system, kernel, reference_coeffs)
        epsilon_reference = scaled_map_metrics(
            map_reference, reference_map, scale=alpha_reference
        )["rel_frobenius_error"]
        coefficient_changes = []
        alpha_changes = []
        epsilon_values = []
        for scale, candidate in by_scale.items():
            overlap = np.vdot(candidate, reference_coeffs)
            aligned = candidate * overlap / abs(overlap)
            alpha = zeroth_moment_scale(kernel, aligned)
            candidate_map = exact_truncated_cv_map_h0(system, kernel, aligned)
            epsilon = scaled_map_metrics(candidate_map, reference_map, scale=alpha)[
                "rel_frobenius_error"
            ]
            coefficient_changes.append(float(np.linalg.norm(aligned - reference_coeffs)))
            alpha_changes.append(float(abs(alpha - alpha_reference) / abs(alpha_reference)))
            epsilon_values.append(float(epsilon))

        values = {
            "coefficient_change": max(coefficient_changes),
            "alpha_relative_change": max(alpha_changes),
            "map_epsilon_change": max(abs(value - epsilon_reference) for value in epsilon_values),
        }
        boundary_criteria = (
            _criterion(f"{boundary}_coefficient_change", values["coefficient_change"], 1e-3, "<="),
            _criterion(f"{boundary}_alpha_relative_change", values["alpha_relative_change"], 0.005, "<="),
            _criterion(f"{boundary}_map_epsilon_change", values["map_epsilon_change"], 0.01, "<="),
        )
        criteria.extend(boundary_criteria)
        decision = "frozen" if all(item["pass"] for item in boundary_criteria) else "STOP"
        gap = coefficient_backend_gap(kernel)
        coefficient_paths[boundary] = _freeze_coefficients(
            boundary,
            kernel,
            reference_coeffs,
            args,
            source="coeff_backend_strict_scale_1",
        )
        rows.append(
            {
                "boundary": boundary,
                **_kernel_fields(kernel),
                **values,
                "epsilon_f_by_k_max_scale": json.dumps(dict(zip((0.75, 1.0, 1.5), epsilon_values))),
                "coefficient_change_pass": boundary_criteria[0]["pass"],
                "alpha_change_pass": boundary_criteria[1]["pass"],
                "map_change_pass": boundary_criteria[2]["pass"],
                "coefficient_backend_gap": float(gap),
                "explicit_overlap": "frozen" if decision == "frozen" else "not_frozen",
                "gh_comp": "rejected_diagnostic",
                "backend_decision": decision,
                "coefficient_artifact": str(coefficient_paths[boundary]),
            }
        )

    backend_decision = "frozen" if all(item["pass"] for item in criteria) else "STOP"
    gate_evaluation = {
        "criteria": criteria,
        "all_pass": backend_decision == "frozen",
        "backend_decision": backend_decision,
    }
    sidecar = build_sidecar(
        args,
        {
            "gate_evaluation": gate_evaluation,
            "backend_decision": backend_decision,
            "evaluation": "exact_finite_map",
            "state_prep_method": "ideal_injection",
            "n_coeff": 48,
            "n_fock": 64,
            "dv_dim": 4,
            "simulator_qubits_per_qumode": 6,
            "hbar_convention": "code_hbar1",
            "input_artifacts": {
                f"coefficients_{boundary}": path for boundary, path in coefficient_paths.items()
            },
        },
    )
    write_artifact(args.output, rows, sidecar)
    print(f"[backend] decision={backend_decision}")
    print(f"[hbar=1] wrote {args.output}")


def _coerce_kernel_kwargs(raw: dict) -> dict:
    return {
        "r_target": float(raw.get("r_target", raw.get("r"))),
        "r_prime": float(raw.get("r_prime", raw.get("r'"))),
        "beta": float(raw["beta"]),
        "n_coeff": int(raw.get("n_coeff", 48)),
        "n_fock": int(raw.get("n_fock", 64)),
        "n_quad": int(raw.get("n_quad", 240)),
        "coeff_backend": "explicit_overlap",
    }


def evaluate_exact_map_row(
    kernel_kwargs: dict,
    boundary: str,
    total_time: float,
    *,
    return_raw: bool = False,
):
    """Evaluate one ideal-injection finite map without building a circuit.

    With ``return_raw=True``, also return the raw arrays (observed map columns,
    reference map, input vectors, alpha) from which every reported metric is a
    pure post-processing step, so callers can freeze them to disk.
    """

    kernel = KernelSpec(**_coerce_kernel_kwargs(kernel_kwargs))
    system = _heat_system(boundary, total_time)
    coeffs = compute_lchs_coefficients_explicit(
        kernel, k_max_scale=1.0, epsabs=1e-12, epsrel=1e-11
    )
    finite_map = exact_truncated_cv_map_h0(system, kernel, coeffs)
    reference_map = exact_reference_map(system)
    alpha = zeroth_moment_scale(kernel, coeffs)
    metrics = scaled_map_metrics(finite_map, reference_map, scale=alpha)
    per_input = metrics["per_input"]
    row = {
        "boundary": boundary,
        **_kernel_fields(kernel),
        "total_time": float(total_time),
        "evaluation": "exact_finite_map",
        "state_prep_method": "ideal_injection",
        "oracle_fidelity": 1.0,
        "oracle_fidelity_basis": "by_construction",
        "rel_frobenius_error": metrics["rel_frobenius_error"],
        "rel_spectral_error": metrics["rel_spectral_error"],
        "worst_scaled_relative_error": max(item["scaled_relative_error"] for item in per_input),
        "worst_conditional_fidelity": min(item["conditional_fidelity"] for item in per_input),
        "p_min": min(item["success_probability"] for item in per_input),
        "p_max": max(item["success_probability"] for item in per_input),
        "alpha_target_re": float(np.real(alpha)),
        "alpha_target_im": float(np.imag(alpha)),
        "best_fit_scale_re": float(np.real(metrics["best_fit_scale"])),
        "best_fit_scale_im": float(np.imag(metrics["best_fit_scale"])),
        "best_fit_rel_frobenius_error": metrics["best_fit_rel_frobenius_error"],
        "per_input": json.dumps(per_input, sort_keys=True),
    }
    if not return_raw:
        return row
    raw = {
        "observed_vectors": finite_map,
        "reference_map": reference_map,
        "input_vectors": np.eye(system.dv_dim, dtype=complex),
        "alpha_target": np.asarray(alpha),
    }
    return row, raw


def _product_formula_column(
    system: PauliSystemSpec,
    kernel: KernelSpec,
    prepared_coeffs: np.ndarray,
    evolution: EvolutionSpec,
    input_state: np.ndarray,
) -> np.ndarray:
    """Apply the circuit's ordered first-order factors as dense matrices."""

    psi, phi = truncated_oscillator_states(kernel, prepared_coeffs)
    x_operator = position_operator(kernel.n_fock)
    identity_fock = np.eye(kernel.n_fock, dtype=complex)
    dt = system.total_time / evolution.n_trotter_steps
    factors = []
    for term in code_l_terms(system.l_terms):
        if abs(term.coeff) > 1e-15:
            factors.append(
                expm(-1.0j * dt * np.kron(x_operator, pauli_sum_matrix((term,))))
            )
    for term in system.h_terms:
        if abs(term.coeff) > 1e-15:
            factors.append(
                expm(-1.0j * dt * np.kron(identity_fock, pauli_sum_matrix((term,))))
            )
    state = np.kron(psi, normalize_vector(input_state))
    for _ in range(evolution.n_trotter_steps):
        for factor in factors:
            state = factor @ state
    return np.asarray(phi.conj() @ state.reshape(kernel.n_fock, system.dv_dim))


def _slow_mode_orthogonal(system: PauliSystemSpec) -> np.ndarray:
    l_block, _ = system_blocks(system)
    _, vectors = np.linalg.eigh(l_block)
    slow = vectors[:, 0]
    for index in range(system.dv_dim):
        candidate = basis_state(system.dv_dim, index) - slow * np.vdot(
            slow, basis_state(system.dv_dim, index)
        )
        if np.linalg.norm(candidate) > 1e-12:
            return normalize_vector(candidate)
    raise RuntimeError("failed to construct slow-mode-orthogonal input")


def _gate_level(metrics: dict) -> str:
    if (
        metrics["rel_frobenius_error"] <= 0.02
        and metrics["rel_spectral_error"] <= 0.02
        and metrics["worst_scaled_relative_error"] <= 0.05
        and metrics["worst_conditional_fidelity"] >= 0.99
    ):
        return "Green"
    if (
        metrics["rel_frobenius_error"] <= 0.05
        and metrics["rel_spectral_error"] <= 0.10
        and metrics["worst_scaled_relative_error"] <= 0.10
        and metrics["worst_conditional_fidelity"] >= 0.99
    ):
        return "Narrow"
    return "fail"


def _read_parameter_mapping(path: Path | None) -> dict:
    if path is None:
        return RETAINED_KERNELS
    document = json.loads(path.read_text(encoding="utf-8"))
    return document.get("boundaries", document)


def _run_retuning_grid(args: argparse.Namespace, output: Path) -> None:
    grid = _read_parameter_mapping(args.retuning_grid)
    rows = []
    for boundary in args.boundaries.split(","):
        boundary = boundary.strip()
        for index, point in enumerate(grid[boundary]):
            row = evaluate_exact_map_row(point, boundary, 1.0)
            row["grid_index"] = index
            row["is_best_fixed_alpha"] = False
            rows.append(row)
    for boundary in {row["boundary"] for row in rows}:
        candidates = [row for row in rows if row["boundary"] == boundary]
        min(candidates, key=lambda row: row["rel_frobenius_error"])[
            "is_best_fixed_alpha"
        ] = True
    best = {
        boundary: min(
            (row for row in rows if row["boundary"] == boundary),
            key=lambda row: row["rel_frobenius_error"],
        )["grid_index"]
        for boundary in {row["boundary"] for row in rows}
    }
    sidecar = build_sidecar(
        args,
        {
            "gate_evaluation": {"best_grid_index_by_boundary": best},
            "evaluation": "exact_finite_map",
            "state_prep_method": "ideal_injection",
            "dv_dim": 4,
            "physical_qumodes": 1,
            "hbar_convention": "code_hbar1",
            "input_artifacts": {"retuning_grid": args.retuning_grid},
        },
    )
    write_artifact(output, rows, sidecar)
    print(f"[retuning-grid] wrote {output}; circuit suite not started")


def run_finite_map(args: argparse.Namespace) -> None:
    output = args.output or Path(
        "results_revision/retuning_grid.csv"
        if args.retuning_grid
        else "results_revision/finite_map_m4.csv"
    )
    if args.retuning_grid:
        _run_retuning_grid(args, output)
        return

    parameters = _read_parameter_mapping(args.params_file)
    rows = []
    coefficient_paths = {}
    raw_paths = {}
    for boundary in (item.strip() for item in args.boundaries.split(",")):
        raw_kernel = {**RETAINED_KERNELS[boundary], **parameters.get(boundary, {})}
        kernel = KernelSpec(**_coerce_kernel_kwargs(raw_kernel))
        coeffs, coefficient_path = _coefficients_for_finite(boundary, kernel, args)
        coefficient_paths[boundary] = coefficient_path
        system = _heat_system(boundary, 1.0)
        target_map = exact_truncated_cv_map_h0(system, kernel, coeffs)
        reference_map = exact_reference_map(system)

        prep = StatePrepSpec(method="law_eberly")
        evolution = EvolutionSpec(n_trotter_steps=100)
        oracle = prepare_cv_oracle(kernel, prep, coeffs=coeffs)
        prepared_map = exact_truncated_cv_map_h0(
            system, kernel, oracle.prepared_state
        )

        first_system = replace(system, init_state=basis_state(system.dv_dim, 0))
        first_result = run_clean_lchs(
            first_system,
            kernel,
            prep,
            evolution,
            coeffs=coeffs,
            oracle=oracle,
        )
        pf_column = _product_formula_column(
            first_system,
            kernel,
            oracle.prepared_state,
            evolution,
            first_system.init_state,
        )
        wiring_error = float(
            np.linalg.norm(first_result.observed_vector - pf_column)
            / max(np.linalg.norm(pf_column), 1e-15)
        )
        if wiring_error > 1e-8:
            raise RuntimeError(
                f"{boundary} product-formula wiring crosscheck failed: {wiring_error:.3e}"
            )

        circuit_map = reconstruct_circuit_map(
            system, kernel, prep, evolution, coeffs, oracle
        )
        slow_input = _slow_mode_orthogonal(system)
        slow_system = replace(system, init_state=slow_input)
        slow_result = run_clean_lchs(
            slow_system,
            kernel,
            prep,
            evolution,
            coeffs=coeffs,
            oracle=oracle,
        )

        alpha_target = zeroth_moment_scale(kernel, coeffs)
        alpha_prepared = zeroth_moment_scale(kernel, oracle.prepared_state)
        metrics = scaled_map_metrics(circuit_map, reference_map, scale=alpha_target)
        per_input = []
        ledgers = []
        for index, item in enumerate(metrics["per_input"]):
            record = {**item, "name": f"basis_{index}"}
            per_input.append(record)
            column_system = replace(system, init_state=basis_state(system.dv_dim, index))
            qc, _, _ = build_hybrid_circuit(
                column_system,
                kernel,
                prep,
                evolution,
                coeffs=coeffs,
                oracle=oracle,
            )
            ledger = resource_ledger(
                qc,
                oracle=oracle,
                evolution=evolution,
                system=column_system,
                kernel=kernel,
            )
            ledgers.append(
                {"input": record["name"], **_ledger_with_probability(ledger, record["success_probability"])}
            )

        slow_observed = np.asarray(slow_result.observed_vector, dtype=complex)
        slow_target = reference_map @ slow_input
        slow_scaled = alpha_target * slow_observed
        slow_probability = float(np.linalg.norm(slow_observed) ** 2)
        slow_fidelity = state_fidelity(slow_observed, slow_target)
        per_input.append(
            {
                "index": 4,
                "name": "slow_mode_orthogonal",
                "scaled_relative_error": float(
                    np.linalg.norm(slow_scaled - slow_target)
                    / max(np.linalg.norm(slow_target), 1e-15)
                ),
                "conditional_fidelity": float(slow_fidelity),
                "success_probability": slow_probability,
            }
        )
        slow_qc, _, _ = build_hybrid_circuit(
            slow_system,
            kernel,
            prep,
            evolution,
            coeffs=coeffs,
            oracle=oracle,
        )
        slow_ledger = resource_ledger(
            slow_qc,
            oracle=oracle,
            evolution=evolution,
            system=slow_system,
            kernel=kernel,
        )
        ledgers.append(
            {
                "input": "slow_mode_orthogonal",
                **_ledger_with_probability(slow_ledger, slow_probability),
            }
        )

        summary_metrics = {
            "rel_frobenius_error": metrics["rel_frobenius_error"],
            "rel_spectral_error": metrics["rel_spectral_error"],
            "worst_scaled_relative_error": max(
                item["scaled_relative_error"] for item in per_input
            ),
            "worst_conditional_fidelity": min(
                item["conditional_fidelity"] for item in per_input
            ),
        }
        probabilities = [item["success_probability"] for item in per_input]
        chain = map_error_chain(
            system, kernel, coeffs, oracle.prepared_state, circuit_map
        )
        raw_path = Path(f"results_revision/finite_map_{boundary}_raw.npz")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        input_vectors = np.column_stack(
            [basis_state(system.dv_dim, index) for index in range(system.dv_dim)]
            + [slow_input]
        )
        observed_vectors = np.column_stack(
            [circuit_map[:, index] for index in range(system.dv_dim)]
            + [slow_observed]
        )
        np.savez(
            raw_path,
            target_map=target_map,
            prepared_map=prepared_map,
            circuit_map=circuit_map,
            input_vectors=input_vectors,
            observed_vectors=observed_vectors,
            reference_map=reference_map,
            alpha_target=np.asarray(alpha_target),
            alpha_prepared=np.asarray(alpha_prepared),
        )
        raw_paths[boundary] = raw_path
        benchmark_probability = per_input[1]["success_probability"]
        row = {
            "boundary": boundary,
            **_kernel_fields(kernel),
            "dv_dim": system.dv_dim,
            "n_trotter_steps": evolution.n_trotter_steps,
            "evaluation": json.dumps(["exact_finite_map", "circuit_statevector"]),
            "state_prep_method": "law_eberly",
            "target_state_prep_method": "ideal_injection",
            "oracle_fidelity": float(oracle.oracle_fidelity),
            **summary_metrics,
            "worst_input_name": max(
                per_input, key=lambda item: item["scaled_relative_error"]
            )["name"],
            "p_succ": benchmark_probability,
            # Manuscript definition: p_ref = ||exp(-At) u||^2 / |alpha_{N,r}|^2.
            "p_ref": float(
                np.linalg.norm(reference_map[:, 1]) ** 2 / abs(alpha_target) ** 2
            ),
            "p_target_model": float(np.linalg.norm(target_map[:, 1]) ** 2),
            "inv_p_succ": None
            if benchmark_probability <= 0.0
            else float(1.0 / benchmark_probability),
            "p_min": min(probabilities),
            "p_max": max(probabilities),
            "eps_model": chain["eps_model"],
            "eps_synth": chain["eps_synth"],
            "eps_t": chain["eps_t"],
            "eps_tot": chain["eps_tot"],
            "eps_model_fro": chain["frobenius"]["eps_model"],
            "eps_synth_fro": chain["frobenius"]["eps_synth"],
            "eps_t_fro": chain["frobenius"]["eps_t"],
            "eps_tot_fro": chain["frobenius"]["eps_tot"],
            "alpha_target_re": float(np.real(alpha_target)),
            "alpha_target_im": float(np.imag(alpha_target)),
            "alpha_prepared_re": float(np.real(alpha_prepared)),
            "alpha_prepared_im": float(np.imag(alpha_prepared)),
            "best_fit_scale_re": float(np.real(metrics["best_fit_scale"])),
            "best_fit_scale_im": float(np.imag(metrics["best_fit_scale"])),
            "best_fit_rel_frobenius_error": metrics[
                "best_fit_rel_frobenius_error"
            ],
            "wiring_crosscheck_relative_error": wiring_error,
            "slow_input_linearity_error": float(
                np.linalg.norm(slow_observed - circuit_map @ slow_input)
            ),
            "per_input": json.dumps(per_input, sort_keys=True),
            "resource_ledger": json.dumps(ledgers, sort_keys=True),
            "prepared_scale_diagnostic": json.dumps(
                {
                    "spectral": chain["prepared_scale"],
                    "frobenius": chain["frobenius"]["prepared_scale"],
                },
                sort_keys=True,
            ),
            "gate_evaluation": _gate_level(summary_metrics),
            "coefficient_artifact": str(coefficient_path),
            "raw_artifact": str(raw_path),
        }
        rows.append(row)

    gate_evaluation = {
        row["boundary"]: row["gate_evaluation"] for row in rows
    }
    input_artifacts = {
        **{f"coefficients_{name}": path for name, path in coefficient_paths.items()},
        **({"params_file": args.params_file} if args.params_file else {}),
        **{f"raw_{name}": path for name, path in raw_paths.items()},
    }
    sidecar = build_sidecar(
        args,
        {
            "gate_evaluation": gate_evaluation,
            "evaluation": ["exact_finite_map", "circuit_statevector"],
            "state_prep_method": "law_eberly",
            "dv_dim": 4,
            "physical_qumodes": 1,
            "hbar_convention": "code_hbar1",
            "input_artifacts": input_artifacts,
        },
    )
    write_artifact(output, rows, sidecar)
    print(f"[finite-map] levels={gate_evaluation}")
    print(f"[hbar=1] wrote {output}")


def _exact_case_maps(
    system: PauliSystemSpec,
    kernel: KernelSpec,
    coeffs: np.ndarray,
    oracle: OraclePreparation,
) -> tuple[np.ndarray, np.ndarray]:
    exact_map = (
        exact_truncated_cv_map_h0
        if all(abs(term.coeff) <= 1e-15 for term in system.h_terms)
        else exact_truncated_cv_map
    )
    return exact_map(system, kernel, coeffs), exact_map(
        system, kernel, oracle.prepared_state
    )


def _map_case_row(
    *,
    case: str,
    system: PauliSystemSpec,
    kernel: KernelSpec,
    coeffs: np.ndarray,
    oracle: OraclePreparation,
    evolution: EvolutionSpec,
    evaluation: list[str],
    coefficient_path: Path,
    raw_path: Path,
    circuit_map: np.ndarray | None = None,
    target_map: np.ndarray | None = None,
    prepared_map: np.ndarray | None = None,
    compile_resources: bool = True,
    extra_raw: dict[str, Any] | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict:
    """Build the shared breadth/scaling row and its replayable raw data."""

    if target_map is None or prepared_map is None:
        target_map, prepared_map = _exact_case_maps(system, kernel, coeffs, oracle)
    reference_map = exact_reference_map(system)
    observed_map = prepared_map if circuit_map is None else circuit_map
    alpha_target = zeroth_moment_scale(kernel, coeffs)
    alpha_prepared = zeroth_moment_scale(kernel, oracle.prepared_state)
    metrics = scaled_map_metrics(observed_map, reference_map, scale=alpha_target)
    per_input = [
        {**item, "name": f"basis_{index}"}
        for index, item in enumerate(metrics["per_input"])
    ]

    ledgers = []
    compiled = []
    if compile_resources:
        for index, item in enumerate(per_input):
            column_system = replace(
                system, init_state=basis_state(system.dv_dim, index)
            )
            qc, _, _ = build_hybrid_circuit(
                column_system,
                kernel,
                StatePrepSpec(method="law_eberly"),
                evolution,
                coeffs=coeffs,
                oracle=oracle,
            )
            ledger = resource_ledger(
                qc,
                oracle=oracle,
                evolution=evolution,
                system=column_system,
                kernel=kernel,
            )
            ledgers.append(
                {
                    "input": item["name"],
                    **_ledger_with_probability(
                        ledger, item["success_probability"]
                    ),
                }
            )
            compiled.append(
                {
                    "input": item["name"],
                    "width": int(qc.num_qubits),
                    **circuit_resource_report(qc),
                }
            )

    chain = _map_error_chain_from_maps(
        target_map,
        prepared_map,
        observed_map,
        reference_map,
        alpha_target,
        alpha_prepared,
    )
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw = {
        "target_map": target_map,
        "prepared_map": prepared_map,
        "observed_vectors": observed_map,
        "reference_map": reference_map,
        "input_vectors": np.eye(system.dv_dim, dtype=complex),
        "alpha_target": np.asarray(alpha_target),
        "alpha_prepared": np.asarray(alpha_prepared),
    }
    if circuit_map is not None:
        raw["circuit_map"] = circuit_map
    raw.update(extra_raw or {})
    np.savez(raw_path, **raw)

    benchmark = min(1, system.dv_dim - 1)
    probability = per_input[benchmark]["success_probability"]
    static_footprint = (
        ledgers[benchmark]["static_footprint"]
        if ledgers
        else {
            "physical_qumodes": 1,
            "simulator_qubits_per_qumode": int(np.log2(kernel.n_fock)),
            "n_dv_qubits": system.n_qubits,
            "le_aux_qubits": 1,
        }
    )
    bound = first_order_trotter_bound(
        system, kernel.n_fock, system.total_time, evolution.n_trotter_steps
    )
    row = {
        "case": case,
        "system_label": system.label,
        **_kernel_fields(kernel),
        "dv_dim": system.dv_dim,
        "total_time": system.total_time,
        "n_trotter_steps": evolution.n_trotter_steps,
        "evaluation": json.dumps(evaluation),
        "state_prep_method": "law_eberly",
        "target_state_prep_method": "ideal_injection",
        "oracle_fidelity": float(oracle.oracle_fidelity),
        "rel_frobenius_error": metrics["rel_frobenius_error"],
        "rel_spectral_error": metrics["rel_spectral_error"],
        "worst_scaled_relative_error": max(
            item["scaled_relative_error"] for item in per_input
        ),
        "worst_conditional_fidelity": min(
            item["conditional_fidelity"] for item in per_input
        ),
        "worst_input_name": max(
            per_input, key=lambda item: item["scaled_relative_error"]
        )["name"],
        "p_succ": probability,
        "p_ref": float(
            np.linalg.norm(reference_map[:, benchmark]) ** 2
            / abs(alpha_target) ** 2
        ),
        "p_target_model": float(np.linalg.norm(target_map[:, benchmark]) ** 2),
        "inv_p_succ": None if probability <= 0.0 else float(1.0 / probability),
        "p_min": min(item["success_probability"] for item in per_input),
        "p_max": max(item["success_probability"] for item in per_input),
        "eps_model": chain["eps_model"],
        "eps_synth": chain["eps_synth"],
        "eps_t": chain["eps_t"],
        "eps_tot": chain["eps_tot"],
        "eps_model_fro": chain["frobenius"]["eps_model"],
        "eps_synth_fro": chain["frobenius"]["eps_synth"],
        "eps_t_fro": chain["frobenius"]["eps_t"],
        "eps_tot_fro": chain["frobenius"]["eps_tot"],
        "alpha_target_re": float(np.real(alpha_target)),
        "alpha_target_im": float(np.imag(alpha_target)),
        "alpha_prepared_re": float(np.real(alpha_prepared)),
        "alpha_prepared_im": float(np.imag(alpha_prepared)),
        "best_fit_scale_re": float(np.real(metrics["best_fit_scale"])),
        "best_fit_scale_im": float(np.imag(metrics["best_fit_scale"])),
        "best_fit_rel_frobenius_error": metrics[
            "best_fit_rel_frobenius_error"
        ],
        "per_input": json.dumps(per_input, sort_keys=True),
        "static_footprint": json.dumps(static_footprint, sort_keys=True),
        "per_attempt_ops": json.dumps(
            ledgers[benchmark]["C_run"] if ledgers else {}, sort_keys=True
        ),
        "accepted_sample_ops": json.dumps(
            ledgers[benchmark]["E_C_accept"] if ledgers else {}, sort_keys=True
        ),
        "resource_ledger": json.dumps(ledgers, sort_keys=True),
        "compiled_width": max((item["width"] for item in compiled), default=""),
        "compiled_depth": max((item["depth"] for item in compiled), default=""),
        "compiled_count_ops": json.dumps(
            compiled[benchmark]["count_ops"] if compiled else {}, sort_keys=True
        ),
        "compiled_resources": json.dumps(compiled, sort_keys=True),
        **bound,
        "coefficient_artifact": str(coefficient_path),
        "coefficient_sha256": hashlib.sha256(
            coefficient_path.read_bytes()
        ).hexdigest(),
        "raw_artifact": str(raw_path),
    }
    row.update(extra_fields or {})
    return row


def _selected_kernel(
    boundary: str,
    *,
    n_coeff: int | None = None,
    n_fock: int | None = None,
) -> KernelSpec:
    selected = _read_parameter_mapping(Path("results_revision/selected_params.json"))[
        boundary
    ]
    return KernelSpec(
        **_coerce_kernel_kwargs(
            {
                **selected,
                **({"n_coeff": n_coeff} if n_coeff is not None else {}),
                **({"n_fock": n_fock} if n_fock is not None else {}),
            }
        )
    )


def _gate_b_anchor(boundary: str) -> float:
    gates = json.loads(Path("results_revision/gate_summary.json").read_text())
    gate_b = next(item for item in gates if item["gate"] == "B")
    return float(gate_b["gate_evaluation"]["per_boundary"][boundary]["eps_t"])


def run_breadth(args: argparse.Namespace) -> None:
    rows = []
    coefficient_paths: dict[str, Path] = {}
    raw_paths: dict[str, Path] = {}
    prep = StatePrepSpec(method="law_eberly")
    evolution = EvolutionSpec(n_trotter_steps=100)
    boundary = args.d8_boundary

    kernel = _selected_kernel(boundary)
    coeffs, coefficient_path = _coefficients_for_finite(boundary, kernel, args)
    coefficient_paths[boundary] = coefficient_path
    oracle = prepare_cv_oracle(kernel, prep, coeffs=coeffs)

    d8 = _heat_system(boundary, 1.0, num_qubits=3)
    d8_map = reconstruct_circuit_map(
        d8, kernel, prep, evolution, coeffs, oracle
    )
    d8_pf = _product_formula_column(
        replace(d8, init_state=basis_state(d8.dv_dim, 0)),
        kernel,
        oracle.prepared_state,
        evolution,
        basis_state(d8.dv_dim, 0),
    )
    d8_wiring = float(
        np.linalg.norm(d8_map[:, 0] - d8_pf)
        / max(np.linalg.norm(d8_pf), 1e-15)
    )
    if d8_wiring > 1e-8:
        raise RuntimeError(f"D=8 wiring crosscheck failed: {d8_wiring:.3e}")
    raw_path = Path("results_revision/breadth_d8_1d_raw.npz")
    rows.append(
        _map_case_row(
            case="d8_1d_heat",
            system=d8,
            kernel=kernel,
            coeffs=coeffs,
            oracle=oracle,
            evolution=evolution,
            evaluation=["exact_finite_map", "circuit_statevector"],
            coefficient_path=coefficient_path,
            raw_path=raw_path,
            circuit_map=d8_map,
            extra_raw={"product_formula_column_0": d8_pf},
            extra_fields={
                "boundary": boundary,
                "wiring_crosscheck_relative_error": d8_wiring,
            },
        )
    )
    raw_paths["d8_1d"] = raw_path

    d16 = _heat_system(boundary, 1.0, num_qubits=4)
    target_d16, prepared_d16 = _exact_case_maps(d16, kernel, coeffs, oracle)
    raw_path = Path("results_revision/breadth_d16_1d_raw.npz")
    d16_row = _map_case_row(
        case="d16_1d_heat",
        system=d16,
        kernel=kernel,
        coeffs=coeffs,
        oracle=oracle,
        evolution=evolution,
        evaluation=["exact_finite_map", "compiled_resources_only"],
        coefficient_path=coefficient_path,
        raw_path=raw_path,
        target_map=target_d16,
        prepared_map=prepared_d16,
        extra_fields={
            "boundary": boundary,
            "eps_t_kind": "first_order_upper_bound",
            "eps_t_anchor_d4": _gate_b_anchor(boundary),
            "eps_t_anchor_d8": rows[0]["eps_t"],
        },
    )
    d16_row["eps_t_measured"] = ""
    d16_row["eps_t"] = d16_row["eps_t_bound"]
    rows.append(d16_row)
    raw_paths["d16_1d"] = raw_path

    anchor_kernel = _selected_kernel("dirichlet")
    anchor_coeffs, anchor_coefficient_path = _coefficients_for_finite(
        "dirichlet", anchor_kernel, args
    )
    coefficient_paths["dirichlet"] = anchor_coefficient_path
    anchor_oracle = prepare_cv_oracle(
        anchor_kernel, prep, coeffs=anchor_coeffs
    )

    nonnormal = build_pauli_system(
        l_terms=[("II", 0.75), ("ZI", 0.25), ("IZ", 0.15)],
        h_terms=[("XI", 0.30), ("IX", 0.20)],
        total_time=1.0,
        init_state=np.eye(4, dtype=complex)[:, 1],
        label="qst_nonnormal_4d",
    )
    nonnormal_map = reconstruct_circuit_map(
        nonnormal,
        anchor_kernel,
        prep,
        evolution,
        anchor_coeffs,
        anchor_oracle,
    )
    nonnormal_pf = _product_formula_column(
        replace(nonnormal, init_state=basis_state(4, 0)),
        anchor_kernel,
        anchor_oracle.prepared_state,
        evolution,
        basis_state(4, 0),
    )
    nonnormal_wiring = float(
        np.linalg.norm(nonnormal_map[:, 0] - nonnormal_pf)
        / max(np.linalg.norm(nonnormal_pf), 1e-15)
    )
    if nonnormal_wiring > 1e-8:
        raise RuntimeError(
            f"non-normal wiring crosscheck failed: {nonnormal_wiring:.3e}"
        )
    raw_path = Path("results_revision/breadth_nonnormal_d4_raw.npz")
    rows.append(
        _map_case_row(
            case="nonnormal_d4",
            system=nonnormal,
            kernel=anchor_kernel,
            coeffs=anchor_coeffs,
            oracle=anchor_oracle,
            evolution=evolution,
            evaluation=["exact_finite_map", "circuit_statevector"],
            coefficient_path=anchor_coefficient_path,
            raw_path=raw_path,
            circuit_map=nonnormal_map,
            extra_raw={"product_formula_column_0": nonnormal_pf},
            extra_fields={
                "boundary": "nonnormal",
                "wiring_crosscheck_relative_error": nonnormal_wiring,
            },
        )
    )
    raw_paths["nonnormal_d4"] = raw_path

    lap4 = 2.0 * np.eye(4)
    lap4 += np.diag(-np.ones(3), 1) + np.diag(-np.ones(3), -1)
    lap2d = np.kron(lap4, np.eye(4)) + np.kron(np.eye(4), lap4)
    anchor_index = 5
    heat2d = build_pauli_system(
        l_terms=decompose_matrix_to_pauli_terms(lap2d),
        h_terms=[],
        total_time=1.0,
        init_state=basis_state(16, anchor_index),
        label="qst_heat_2d_4x4",
    )
    target_2d, prepared_2d = _exact_case_maps(
        heat2d, anchor_kernel, anchor_coeffs, anchor_oracle
    )
    def abort_2d_anchor(_signum: int, _frame: Any) -> None:
        raise TimeoutError("2D anchor exceeded its 30-minute wall-time budget")

    previous_alarm = signal.signal(signal.SIGALRM, abort_2d_anchor)
    signal.alarm(30 * 60)
    try:
        anchor = run_clean_lchs(
            heat2d,
            anchor_kernel,
            prep,
            evolution,
            coeffs=anchor_coeffs,
            oracle=anchor_oracle,
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
    anchor_pf = _product_formula_column(
        heat2d,
        anchor_kernel,
        anchor_oracle.prepared_state,
        evolution,
        heat2d.init_state,
    )
    anchor_observed = np.asarray(anchor.observed_vector)
    wiring_2d = float(
        np.linalg.norm(anchor_observed - anchor_pf)
        / max(np.linalg.norm(anchor_pf), 1e-15)
    )
    if wiring_2d > 1e-8:
        raise RuntimeError(f"2D wiring crosscheck failed: {wiring_2d:.3e}")
    eps_t_2d = float(
        np.linalg.norm(anchor_observed - prepared_2d[:, anchor_index])
    )
    relative_2d = float(
        eps_t_2d / max(np.linalg.norm(prepared_2d[:, anchor_index]), 1e-15)
    )
    raw_path = Path("results_revision/breadth_2d_4x4_raw.npz")
    row_2d = _map_case_row(
        case="heat_2d_4x4",
        system=heat2d,
        kernel=anchor_kernel,
        coeffs=anchor_coeffs,
        oracle=anchor_oracle,
        evolution=evolution,
        evaluation=[
            "exact_finite_map",
            "compiled_resources_only",
            "circuit_statevector",
        ],
        coefficient_path=anchor_coefficient_path,
        raw_path=raw_path,
        target_map=target_2d,
        prepared_map=prepared_2d,
        extra_raw={
            "anchor_input": heat2d.init_state,
            "anchor_observed": anchor_observed,
            "anchor_product_formula": anchor_pf,
        },
        extra_fields={
            "boundary": "dirichlet_2d",
            "anchor_input_index": anchor_index,
            "anchor_grid_row": 1,
            "anchor_grid_col": 1,
            "anchor_postselection_probability": anchor.postselection_probability,
            "wiring_crosscheck_relative_error": wiring_2d,
            "eps_t_2d": eps_t_2d,
            "anchor_relative_error": relative_2d,
            "agrees": relative_2d <= 1e-2,
            "eps_t_kind": "single_input_measured",
        },
    )
    row_2d["eps_t"] = eps_t_2d
    rows.append(row_2d)
    raw_paths["heat_2d_4x4"] = raw_path

    input_artifacts = {
        "selected_params": Path("results_revision/selected_params.json"),
        "gate_summary": Path("results_revision/gate_summary.json"),
        **{f"coefficients_{name}": path for name, path in coefficient_paths.items()},
        **{f"raw_{name}": path for name, path in raw_paths.items()},
    }
    sidecar = build_sidecar(
        args,
        {
            "gate_evaluation": {
                "d8_boundary": boundary,
                "heat_2d_4x4_agrees": row_2d["agrees"],
                "gate_c_status": "pending_formal_snap_telemetry_and_pi_decision",
            },
            "evaluation": sorted(
                {method for row in rows for method in json.loads(row["evaluation"])}
            ),
            "state_prep_method": "law_eberly",
            "hbar_convention": "code_hbar1",
            "input_artifacts": input_artifacts,
        },
    )
    write_artifact(args.output, rows, sidecar)
    print(f"[breadth] wrote {args.output}; 2D agrees={row_2d['agrees']}")


def _parse_cutoff_pairs(raw: str) -> list[tuple[int, int]]:
    return [tuple(map(int, item.split(":"))) for item in raw.split(",")]


def run_scaling(args: argparse.Namespace) -> None:
    rows = []
    coefficient_paths: dict[str, Path] = {}
    raw_paths: dict[str, Path] = {}
    prep = StatePrepSpec(method="law_eberly")
    for total_time in (float(item) for item in args.times.split(",")):
        for n_coeff, n_fock in _parse_cutoff_pairs(args.cutoff_pairs):
            kernel = _selected_kernel(
                "dirichlet", n_coeff=n_coeff, n_fock=n_fock
            )
            coeffs, coefficient_path = _coefficients_for_finite(
                "dirichlet", kernel, args
            )
            coefficient_paths[f"N{n_coeff}_NF{n_fock}"] = coefficient_path
            oracle = prepare_cv_oracle(kernel, prep, coeffs=coeffs)
            system = _heat_system("dirichlet", total_time)
            n_t_values = (25, 50, 100) if total_time == 1.0 else (1,)
            for n_t in n_t_values:
                evolution = EvolutionSpec(n_trotter_steps=n_t)
                circuit_map = None
                evaluation = ["exact_finite_map"]
                suffix = "exact"
                compile_resources = False
                extra_fields: dict[str, Any] = {"boundary": "dirichlet"}
                extra_raw: dict[str, Any] = {}
                if total_time == 1.0:
                    circuit_map = reconstruct_circuit_map(
                        system, kernel, prep, evolution, coeffs, oracle
                    )
                    pf_column = _product_formula_column(
                        replace(system, init_state=basis_state(4, 0)),
                        kernel,
                        oracle.prepared_state,
                        evolution,
                        basis_state(4, 0),
                    )
                    wiring = float(
                        np.linalg.norm(circuit_map[:, 0] - pf_column)
                        / max(np.linalg.norm(pf_column), 1e-15)
                    )
                    if wiring > 1e-8:
                        raise RuntimeError(
                            f"scaling wiring crosscheck failed: {wiring:.3e}"
                        )
                    evaluation = ["exact_finite_map", "circuit_statevector"]
                    suffix = f"nt{n_t}"
                    compile_resources = True
                    extra_fields["wiring_crosscheck_relative_error"] = wiring
                    extra_raw["product_formula_column_0"] = pf_column
                candidate_id = (
                    f"T{total_time:.1f}_N{n_coeff}_NF{n_fock}_{suffix}"
                )
                raw_path = Path(
                    f"results_revision/scaling_{candidate_id}_raw.npz"
                )
                row = _map_case_row(
                    case="scaling_dirichlet_d4",
                    system=system,
                    kernel=kernel,
                    coeffs=coeffs,
                    oracle=oracle,
                    evolution=evolution,
                    evaluation=evaluation,
                    coefficient_path=coefficient_path,
                    raw_path=raw_path,
                    circuit_map=circuit_map,
                    compile_resources=compile_resources,
                    extra_raw=extra_raw,
                    extra_fields={"candidate_id": candidate_id, **extra_fields},
                )
                row["achieved_total_error"] = row["eps_tot"]
                row["achieved_error_kind"] = (
                    "measured_circuit_total"
                    if circuit_map is not None
                    else "exact_map_total"
                )
                if circuit_map is None:
                    row["n_trotter_steps"] = ""
                    row["eps_t_bound"] = ""
                rows.append(row)
                raw_paths[candidate_id] = raw_path

    sidecar = build_sidecar(
        args,
        {
            "gate_evaluation": {
                "candidate_count": len(rows),
                "gate_c_status": "pending_formal_snap_telemetry_and_pi_decision",
            },
            "evaluation": ["exact_finite_map", "circuit_statevector"],
            "state_prep_method": "law_eberly",
            "hbar_convention": "code_hbar1",
            "input_artifacts": {
                "selected_params": Path("results_revision/selected_params.json"),
                **{
                    f"coefficients_{name}": path
                    for name, path in coefficient_paths.items()
                },
                **{f"raw_{name}": path for name, path in raw_paths.items()},
            },
        },
    )
    write_artifact(args.output, rows, sidecar)
    print(f"[scaling] wrote {args.output} ({len(rows)} rows)")


def _snap_totals(records: list[dict]) -> dict:
    return {
        "starts": len(records),
        "nit": sum(int(row["nit"]) for row in records),
        "nfev": sum(int(row["nfev"]) for row in records),
        "wall_seconds": float(sum(float(row["wall_seconds"]) for row in records)),
    }


def _load_snap_warm_start(
    path: Path,
    target: np.ndarray,
    n_fock: int,
) -> OraclePreparation:
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = document.get(
        "best_parameter_payload",
        document.get("snap_parameter_payload", document),
    )
    return oracle_from_snap_parameter_payload(
        target, n_fock=n_fock, payload=payload, verify_fidelity=False
    )


def run_snap(args: argparse.Namespace) -> None:
    kernel = _selected_kernel(
        "dirichlet", n_coeff=args.n_coeff, n_fock=args.n_fock
    )
    coeffs, coefficient_path = _coefficients_for_finite(
        "dirichlet", kernel, args
    )
    target = padded_seed_state(coeffs, kernel.n_fock)
    warm_oracle = (
        _load_snap_warm_start(args.warm_start_from, target, kernel.n_fock)
        if args.warm_start_from
        else None
    )
    depths = (
        [int(item) for item in args.depth_schedule.split(",")]
        if args.depth_schedule
        else [args.depth]
    )
    levels = []
    pipeline_records = []
    oracle = warm_oracle
    n_snap = min(kernel.n_coeff, kernel.n_fock)
    # The wall-time budget covers the whole run, split evenly over the levels
    # of the depth schedule.
    level_budget = (
        args.wall_time_seconds / len(depths)
        if args.wall_time_seconds is not None
        else None
    )
    for depth in depths:
        initial_guess = snap_d_initial_guess_from_oracle(
            oracle, depth=depth, n_snap=n_snap
        )
        oracle = optimize_snap_d(
            target,
            n_fock=kernel.n_fock,
            depth=depth,
            n_snap=n_snap,
            n_restarts=args.restarts,
            maxiter=args.maxiter,
            maxfun=args.maxfun,
            time_budget_seconds=level_budget,
            initial_guess=initial_guess,
            random_seed=args.seed,
        )
        oracle = align_oracle_global_phase(oracle)
        records = oracle.metadata["start_records"]
        pipeline_records.extend(records)
        levels.append(
            {
                "depth": depth,
                "oracle_fidelity": float(oracle.oracle_fidelity),
                "start_records": records,
                "restart_summary": oracle.metadata["restart_summary"],
                "restart_summary_by_kind": oracle.metadata[
                    "restart_summary_by_kind"
                ],
                "totals": _snap_totals(records),
                "best_parameter_payload": snap_parameter_payload(oracle),
            }
        )

    final_records = oracle.metadata["start_records"]
    final_totals = _snap_totals(final_records)
    pipeline_totals = _snap_totals(pipeline_records)
    document = {
        "n_coeff": kernel.n_coeff,
        "n_fock": kernel.n_fock,
        "seed": args.seed,
        "depth": depths[-1],
        "depth_schedule": depths,
        "restarts": args.restarts,
        "maxiter": args.maxiter,
        "maxfun": args.maxfun,
        "wall_time_seconds": args.wall_time_seconds,
        "oracle_fidelity": float(oracle.oracle_fidelity),
        "start_records": final_records,
        "restart_summary": oracle.metadata["restart_summary"],
        "restart_summary_by_kind": oracle.metadata[
            "restart_summary_by_kind"
        ],
        "total_nit": final_totals["nit"],
        "total_nfev": final_totals["nfev"],
        "total_wall_seconds": final_totals["wall_seconds"],
        "final_depth_totals": final_totals,
        "pipeline_totals": pipeline_totals,
        "best_parameter_payload": snap_parameter_payload(oracle),
        "levels": levels,
        "coefficient_artifact": str(coefficient_path),
    }
    sidecar = build_sidecar(
        args,
        {
            "state_prep_method": "snap_d",
            "n_coeff": kernel.n_coeff,
            "n_fock": kernel.n_fock,
            "hbar_convention": "code_hbar1",
            "input_artifacts": {
                "coefficients_dirichlet": coefficient_path,
                **(
                    {"warm_start": args.warm_start_from}
                    if args.warm_start_from
                    else {}
                ),
            },
            "gate_c_status": "telemetry_only_pi_decision_pending",
        },
    )
    write_artifact(args.output, document, sidecar)
    print(f"[snap] wrote {args.output}; fidelity={oracle.oracle_fidelity:.8f}")


PAPER_FIELD_CONVERSIONS = {
    "gamma_code": ("gamma_paper", to_paper_gamma),
    "lambda_code": ("lambda_paper", to_paper_lambda),
    "x_norm_bound_code": ("x_norm_bound_paper", lambda value: SQRT2 * value),
    "L_coupling_norm_code": ("L_norm_paper", to_paper_l_norm),
    "q_code": ("x_paper", to_paper_coordinate),
}
PAPER_NAMED_FIELDS = {
    "gamma_paper",
    "lambda_paper",
    "x_norm_bound_paper",
    "L_norm_paper",
    "x_paper",
}


def _convert_paper_value(value: Any, conversion: Any) -> Any:
    if value == "" or value is None:
        return value
    parsed = json.loads(value) if isinstance(value, str) else value
    converted = conversion(np.asarray(parsed))
    if np.ndim(converted) == 0:
        return float(converted)
    return json.dumps(np.asarray(converted).tolist(), separators=(",", ":"))


def convert_paper_rows(rows: list[dict]) -> list[dict]:
    """Convert only explicitly named code-convention fields."""

    converted_rows = []
    ambiguous = {"gamma", "lambda", "x_norm_bound", "l_norm", "q", "x"}
    for source in rows:
        converted = {}
        labels = {}
        for name, value in source.items():
            if name == "hbar_convention":
                continue
            if name in PAPER_FIELD_CONVERSIONS:
                output_name, conversion = PAPER_FIELD_CONVERSIONS[name]
                converted[output_name] = _convert_paper_value(value, conversion)
                labels[output_name] = "[converted code_hbar1->paper_hbar2]"
            elif name in PAPER_NAMED_FIELDS:
                converted[name] = value
                labels[name] = "[paper_hbar2]"
            elif name.lower() in ambiguous:
                raise ValueError(
                    f"field '{name}' has no explicit hbar convention"
                )
            else:
                output_name = "N_Fock" if name == "n_fock" else name
                converted[output_name] = value
                labels[output_name] = "[invariant]"
        converted["field_labels"] = json.dumps(labels, sort_keys=True)
        converted["hbar_convention"] = "paper_hbar2"
        converted_rows.append(converted)
    return converted_rows


def run_export_paper(args: argparse.Namespace) -> None:
    paper_root = (Path.cwd() / "results_revision/paper_hbar2").resolve()
    if not args.output.resolve().is_relative_to(paper_root):
        raise ValueError("export-paper output must be under results_revision/paper_hbar2")
    sources = {
        "finite-map": Path("results_revision/finite_map_m4.csv"),
        "breadth": Path("results_revision/breadth.csv"),
        "scaling": Path("results_revision/scaling.csv"),
        "retuning-grid": Path("results_revision/retuning_grid.csv"),
    }
    source = args.input or sources[args.table]
    with source.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    converted = convert_paper_rows(rows)
    sidecar = build_sidecar(
        args,
        {
            "hbar_convention": "paper_hbar2",
            "input_artifacts": {"source_table": source},
            "field_mapping": {
                name: target for name, (target, _) in PAPER_FIELD_CONVERSIONS.items()
            },
            "invariant_label": "[invariant]",
        },
    )
    write_artifact(args.output, converted, sidecar)
    print(f"[paper hbar=2] wrote {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    convention = subparsers.add_parser("convention")
    convention.add_argument(
        "--output",
        type=Path,
        default=Path("results_revision/convention.json"),
    )
    convention.set_defaults(_handler=run_convention)

    coeff_backend = subparsers.add_parser("coeff-backend")
    coeff_backend.add_argument(
        "--output",
        type=Path,
        default=Path("results_revision/coefficient_backend.csv"),
    )
    coeff_backend.set_defaults(_handler=run_coeff_backend)

    finite_map = subparsers.add_parser("finite-map")
    finite_map.add_argument("--suite", choices=("m4",), default="m4")
    finite_map.add_argument(
        "--boundaries",
        default="dirichlet,neumann,periodic",
    )
    finite_map.add_argument("--params-file", type=Path)
    finite_map.add_argument("--retuning-grid", type=Path)
    finite_map.add_argument("--output", type=Path)
    finite_map.set_defaults(_handler=run_finite_map)

    breadth = subparsers.add_parser("breadth")
    breadth.add_argument(
        "--d8-boundary",
        choices=("dirichlet", "neumann", "periodic"),
        default="dirichlet",
    )
    breadth.add_argument(
        "--output", type=Path, default=Path("results_revision/breadth.csv")
    )
    breadth.set_defaults(_handler=run_breadth)

    scaling = subparsers.add_parser("scaling")
    scaling.add_argument("--times", default="0.5,1,2")
    scaling.add_argument("--cutoff-pairs", default="12:16,24:32,48:64")
    scaling.add_argument(
        "--output", type=Path, default=Path("results_revision/scaling.csv")
    )
    scaling.set_defaults(_handler=run_scaling)

    snap = subparsers.add_parser("snap")
    snap.add_argument("--n-coeff", type=int, required=True)
    snap.add_argument("--n-fock", type=int, default=None)
    snap.add_argument("--seed", type=int, required=True)
    snap.add_argument("--depth", type=int, required=True)
    snap.add_argument("--depth-schedule")
    snap.add_argument("--restarts", type=int, required=True)
    snap.add_argument("--maxiter", type=int, required=True)
    snap.add_argument("--maxfun", type=int, default=None)
    snap.add_argument("--wall-time-seconds", type=float, default=None)
    snap.add_argument("--warm-start-from", type=Path)
    snap.add_argument("--output", type=Path, required=True)
    snap.set_defaults(_handler=run_snap)

    export_paper = subparsers.add_parser("export-paper")
    export_paper.add_argument(
        "--table",
        choices=("finite-map", "breadth", "scaling", "retuning-grid"),
        required=True,
    )
    export_paper.add_argument("--input", type=Path)
    export_paper.add_argument("--output", type=Path, required=True)
    export_paper.set_defaults(_handler=run_export_paper)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args._handler(args)


if __name__ == "__main__":
    main()
