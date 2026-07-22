"""Reconstruct the frozen 4x4 2D heat instance at full circuit level."""

import csv
import hashlib
import json
import signal
import time
import types
from pathlib import Path

import numpy as np

import revision_eval
from clean_core import EvolutionSpec, StatePrepSpec, pauli_sum_matrix
from revision_eval import (
    _coefficients_for_finite,
    _exact_case_maps,
    _map_case_row,
    _product_formula_column,
    _selected_kernel,
    build_pauli_system,
    build_sidecar,
    decompose_matrix_to_pauli_terms,
    prepare_cv_oracle,
)

OUT_DIR = Path("results_revision")
OUTPUT = OUT_DIR / "breadth_2d_circuit.csv"
RAW = OUT_DIR / "breadth_2d_4x4_circuit_raw.npz"
META = OUT_DIR / "breadth_2d_circuit.meta.json"
COEFFICIENT_SHA256 = (
    "3c5729bbcdc7c525fc5ea92847a18957e0526f85f16f897ff578a54bf3dc6644"
)
L_NORM = 7.23606797749979
LAMBDA_MIN = 0.7639320225002102
ANCHOR_INDEX = 5
ANCHOR_EPS_T = 3.552982606303818e-4
ANCHOR_RELATIVE_ERROR = 2.6103688480868586e-3


def laplacian_2d() -> np.ndarray:
    lap4 = 2.0 * np.eye(4)
    lap4 += np.diag(-np.ones(3), 1) + np.diag(-np.ones(3), -1)
    return np.kron(lap4, np.eye(4)) + np.kron(np.eye(4), lap4)


def _timed_circuit_map(system, kernel, prep, evolution, coeffs, oracle):
    """Use the shared reconstruction helper while timing/alarming each column."""

    wall_times = []
    original_run = revision_eval.run_clean_lchs

    def abort(_signum, _frame):
        raise TimeoutError("2D circuit column exceeded its 30-minute wall-time budget")

    def timed_run(*args, **kwargs):
        index = len(wall_times)
        start = time.monotonic()
        signal.alarm(30 * 60)
        try:
            return original_run(*args, **kwargs)
        finally:
            signal.alarm(0)
            elapsed = time.monotonic() - start
            wall_times.append(elapsed)
            print(f"[column {index:02d}] {elapsed:.3f}s", flush=True)

    previous_alarm = signal.signal(signal.SIGALRM, abort)
    revision_eval.run_clean_lchs = timed_run
    try:
        circuit_map = revision_eval.reconstruct_circuit_map(
            system, kernel, prep, evolution, coeffs, oracle
        )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        revision_eval.run_clean_lchs = original_run
    if len(wall_times) != system.dv_dim:
        raise RuntimeError(f"expected {system.dv_dim} column timings, got {len(wall_times)}")
    return circuit_map, np.asarray(wall_times)


def main() -> None:
    prep = StatePrepSpec(method="law_eberly")
    evolution = EvolutionSpec(n_trotter_steps=100)
    kernel = _selected_kernel("dirichlet")
    expected_kernel = (1.6, 0.25, 0.5, 32, 64, 240)
    actual_kernel = (
        kernel.r_target,
        kernel.r_prime,
        kernel.beta,
        kernel.n_coeff,
        kernel.n_fock,
        kernel.n_quad,
    )
    if actual_kernel != expected_kernel:
        raise RuntimeError(f"unexpected shared operating point: {actual_kernel}")

    args = types.SimpleNamespace(coeff_backend="explicit_overlap", n_quad=240)
    coeffs, coefficient_path = _coefficients_for_finite("dirichlet", kernel, args)
    coefficient_sha256 = hashlib.sha256(coefficient_path.read_bytes()).hexdigest()
    if coefficient_sha256 != COEFFICIENT_SHA256:
        raise RuntimeError(f"coefficient sha256 mismatch: {coefficient_sha256}")

    lap2d = laplacian_2d()
    eigenvalues = np.linalg.eigvalsh(lap2d)
    np.testing.assert_allclose(np.linalg.norm(lap2d, 2), L_NORM, rtol=1e-10)
    np.testing.assert_allclose(eigenvalues[0], LAMBDA_MIN, rtol=1e-10)
    terms = decompose_matrix_to_pauli_terms(lap2d)
    np.testing.assert_allclose(pauli_sum_matrix(terms), lap2d, atol=1e-10, rtol=0.0)
    system = build_pauli_system(
        l_terms=terms,
        h_terms=[],
        total_time=1.0,
        init_state=np.eye(16, dtype=complex)[:, ANCHOR_INDEX],
        label="qst_heat_2d_4x4",
    )

    oracle = prepare_cv_oracle(kernel, prep, coeffs=coeffs)
    target_map, prepared_map = _exact_case_maps(system, kernel, coeffs, oracle)
    circuit_map, wall_times = _timed_circuit_map(
        system, kernel, prep, evolution, coeffs, oracle
    )

    wiring_errors = []
    for index in range(system.dv_dim):
        input_state = np.eye(system.dv_dim, dtype=complex)[:, index]
        reference = _product_formula_column(
            system, kernel, oracle.prepared_state, evolution, input_state
        )
        error = float(
            np.linalg.norm(circuit_map[:, index] - reference)
            / max(np.linalg.norm(reference), 1e-15)
        )
        wiring_errors.append(error)
        print(f"[wiring {index:02d}] {error:.16e}", flush=True)
    wiring_errors = np.asarray(wiring_errors)
    if float(wiring_errors.max()) > 1e-8:
        raise RuntimeError(f"2D wiring crosscheck failed: {wiring_errors.max():.3e}")

    eps_t_2d = float(np.linalg.norm(circuit_map[:, ANCHOR_INDEX] - prepared_map[:, ANCHOR_INDEX]))
    anchor_relative_error = float(
        eps_t_2d / max(np.linalg.norm(prepared_map[:, ANCHOR_INDEX]), 1e-15)
    )
    np.testing.assert_allclose(eps_t_2d, ANCHOR_EPS_T, rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(
        anchor_relative_error, ANCHOR_RELATIVE_ERROR, rtol=1e-6, atol=0.0
    )

    probabilities = np.sum(np.abs(circuit_map) ** 2, axis=0)
    row = _map_case_row(
        case="heat_2d_4x4_circuit",
        system=system,
        kernel=kernel,
        coeffs=coeffs,
        oracle=oracle,
        evolution=evolution,
        evaluation=["exact_finite_map", "circuit_statevector"],
        coefficient_path=coefficient_path,
        raw_path=RAW,
        circuit_map=circuit_map,
        target_map=target_map,
        prepared_map=prepared_map,
        extra_raw={
            "per_input_wiring_errors": wiring_errors,
            "per_input_p": probabilities,
            "per_column_wall_times": wall_times,
            "circuit_total_wall_time": np.asarray(wall_times.sum()),
        },
        extra_fields={
            "boundary": "dirichlet_2d",
            "wiring_crosscheck_relative_error": float(wiring_errors.max()),
            "hbar_convention": "code_hbar1",
            "eps_t_kind": "full_circuit_map_measured",
            "eps_t_anchor_d4": "",
            "eps_t_anchor_d8": "",
            "eps_t_measured": "",
            "anchor_input_index": ANCHOR_INDEX,
            "anchor_grid_row": 1,
            "anchor_grid_col": 1,
            "anchor_postselection_probability": float(probabilities[ANCHOR_INDEX]),
            "eps_t_2d": eps_t_2d,
            "anchor_relative_error": anchor_relative_error,
            "agrees": anchor_relative_error <= 1e-2,
        },
    )

    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)

    sidecar = build_sidecar(
        args,
        {
            "output": str(OUTPUT),
            "raw_artifact": str(RAW),
            "evaluation": ["exact_finite_map", "circuit_statevector"],
            "state_prep_method": "law_eberly",
            "coefficient_artifact": str(coefficient_path),
            "coefficient_sha256": coefficient_sha256,
            "input_artifacts": {
                "selected_params_v2": OUT_DIR / "selected_params_v2.json",
                "selected_params": OUT_DIR / "selected_params.json",
                "gate_summary": OUT_DIR / "gate_summary.json",
                "coefficients_dirichlet": coefficient_path,
            },
        },
    )
    sidecar["source_sha256"] = {
        "make_2d_circuit.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **sidecar["source_sha256"],
    }
    META.write_text(json.dumps({"sidecar": sidecar}, indent=2) + "\n")
    print(
        f"wrote {OUTPUT}, {RAW}, and {META}; circuit wall time {wall_times.sum():.3f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
