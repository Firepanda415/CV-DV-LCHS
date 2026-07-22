"""Build the M=32 heat and advection-diffusion circuit-level breadth rows."""

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
    _heat_system,
    _map_case_row,
    _product_formula_column,
    _selected_kernel,
    build_pauli_system,
    build_sidecar,
    decompose_matrix_to_pauli_terms,
    prepare_cv_oracle,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = Path("results_revision")
OUTPUT = OUT_DIR / "breadth_m32.csv"
META = OUT_DIR / "breadth_m32.meta.json"
COEFFICIENT_SHA256 = "3c5729bbcdc7c525fc5ea92847a18957e0526f85f16f897ff578a54bf3dc6644"
SELECTED_SHA256 = "46dbc23aba2237a363a52dfbe44d1ba6db10838ab72ec1349bfa286a1a1caf13"
SOURCE_FILES = ("make_m32_breadth.py", "make_m32_dv_extension.py", "make_m32_dv_circuit.py")


def m32_matrices() -> tuple[np.ndarray, np.ndarray]:
    lap = 2.0 * np.eye(32)
    lap -= np.diag(np.ones(31), 1) + np.diag(np.ones(31), -1)
    central_difference = (
        np.diag(np.ones(31), 1) - np.diag(np.ones(31), -1)
    ) / 2.0
    return lap, -1.0j * central_difference


def nonnormality(l_mat: np.ndarray, h_mat: np.ndarray) -> tuple[float, float]:
    generator = l_mat + 1.0j * h_mat
    eigenvalues, vectors = np.linalg.eig(generator)
    fro2 = np.linalg.norm(generator, "fro") ** 2
    departure = np.sqrt(max(fro2 - np.sum(np.abs(eigenvalues) ** 2), 0.0))
    return float(np.linalg.cond(vectors)), float(departure / np.sqrt(fro2))


def _assert_system_pins(lap: np.ndarray, h_mat: np.ndarray) -> None:
    expected_norm = 2.0 + 2.0 * np.cos(np.pi / 33.0)
    expected_min = 2.0 - 2.0 * np.cos(np.pi / 33.0)
    np.testing.assert_allclose(np.linalg.norm(lap, 2), expected_norm, rtol=1e-10)
    np.testing.assert_allclose(np.linalg.eigvalsh(lap)[0], expected_min, rtol=1e-10)
    if np.linalg.norm(lap @ h_mat - h_mat @ lap) <= 1e-12:
        raise RuntimeError("M=32 advection-diffusion commutator unexpectedly vanished")


def _assert_round_trip(terms, matrix: np.ndarray, label: str) -> None:
    error = float(np.linalg.norm(pauli_sum_matrix(tuple(terms)) - matrix))
    if error > 1e-10:
        raise RuntimeError(f"{label} Pauli round-trip failed: {error:.3e}")


def _timed_circuit_map(case, system, kernel, prep, evolution, coeffs, oracle):
    wall_times = []
    original_run = revision_eval.run_clean_lchs

    def abort(_signum, _frame):
        raise TimeoutError(f"{case} circuit column exceeded 30 minutes")

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
            print(f"[{case} column {index:02d}] {elapsed:.3f}s", flush=True)

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
        raise RuntimeError(f"{case}: expected {system.dv_dim} timings, got {len(wall_times)}")
    return circuit_map, np.asarray(wall_times)


def _wiring_errors(case, system, kernel, prepared_state, evolution, circuit_map):
    errors = []

    def abort(_signum, _frame):
        raise TimeoutError(f"{case} dense wiring column exceeded 30 minutes")

    previous_alarm = signal.signal(signal.SIGALRM, abort)
    try:
        for index in range(system.dv_dim):
            input_state = np.eye(system.dv_dim, dtype=complex)[:, index]
            signal.alarm(30 * 60)
            try:
                reference = _product_formula_column(
                    system, kernel, prepared_state, evolution, input_state
                )
            finally:
                signal.alarm(0)
            error = float(
                np.linalg.norm(circuit_map[:, index] - reference)
                / max(np.linalg.norm(reference), 1e-15)
            )
            errors.append(error)
            print(f"[{case} wiring {index:02d}] {error:.16e}", flush=True)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
    errors = np.asarray(errors)
    if float(errors.max()) > 1e-8:
        raise RuntimeError(f"{case} wiring crosscheck failed: {errors.max():.3e}")
    return errors


def _assert_heat_ledger(row: dict) -> None:
    ledgers = json.loads(row["resource_ledger"])
    expected_step = {"1q": 386, "cx": 196, "D": 1, "cD": 31}
    expected_total = {"1q": 38600, "cx": 19600, "D": 100, "cD": 3100}
    for ledger in ledgers:
        step = ledger["C_step"]
        total = ledger["n_t_C_step"]
        actual_step = {
            "1q": step.get("h", 0) + step.get("rz", 0),
            "cx": step.get("cx", 0),
            "D": step.get("D", 0),
            "cD": step.get("cD", 0),
        }
        actual_total = {
            "1q": total.get("h", 0) + total.get("rz", 0),
            "cx": total.get("cx", 0),
            "D": total.get("D", 0),
            "cD": total.get("cD", 0),
        }
        if actual_step != expected_step or actual_total != expected_total:
            raise RuntimeError(
                f"heat ledger mismatch: step={actual_step}, total={actual_total}"
            )


def _run_case(case, raw_path, system, kernel, coeffs, oracle, prep, evolution, extra):
    target_map, prepared_map = _exact_case_maps(system, kernel, coeffs, oracle)
    circuit_map, wall_times = _timed_circuit_map(
        case, system, kernel, prep, evolution, coeffs, oracle
    )
    wiring = _wiring_errors(
        case, system, kernel, oracle.prepared_state, evolution, circuit_map
    )
    probabilities = np.sum(np.abs(circuit_map) ** 2, axis=0)
    row = _map_case_row(
        case=case,
        system=system,
        kernel=kernel,
        coeffs=coeffs,
        oracle=oracle,
        evolution=evolution,
        evaluation=["exact_finite_map", "circuit_statevector"],
        coefficient_path=Path(
            "results_revision/coefficients_dirichlet_r1p6_rp0p25_b0p5_N32_nq240.json"
        ),
        raw_path=raw_path,
        circuit_map=circuit_map,
        target_map=target_map,
        prepared_map=prepared_map,
        extra_raw={
            "per_input_wiring_errors": wiring,
            "per_input_p": probabilities,
            "per_column_wall_times": wall_times,
            "circuit_total_wall_time": np.asarray(wall_times.sum()),
        },
        extra_fields={
            "boundary": extra.pop("boundary"),
            "wiring_crosscheck_relative_error": float(wiring.max()),
            "hbar_convention": "code_hbar1",
            "eps_t_kind": "full_circuit_map_measured",
            **extra,
        },
    )
    if float(row["eps_synth"]) > 1e-12:
        raise RuntimeError(f"{case} eps_synth gate failed: {row['eps_synth']}")
    eps_f_percent = 100.0 * float(row["rel_frobenius_error"])
    if case == "heat_d32":
        if not 0.8 <= eps_f_percent <= 5.0:
            raise RuntimeError(f"heat_d32 eps_F outside [0.8, 5.0]%: {eps_f_percent}")
        if float(row["p_min"]) < 0.02:
            raise RuntimeError(f"heat_d32 p_min below 2%: {row['p_min']}")
        _assert_heat_ledger(row)
    elif not 0.5 <= eps_f_percent <= 15.0:
        raise RuntimeError(
            f"advdiff_d32 advisory eps_F outside [0.5, 15]%: {eps_f_percent}"
        )
    print(
        f"[{case}] epsF={eps_f_percent:.6f}% p_min={100*float(row['p_min']):.6f}% "
        f"wiring_max={wiring.max():.3e}",
        flush=True,
    )
    return row


def main() -> None:
    coefficient_path = OUT_DIR / "coefficients_dirichlet_r1p6_rp0p25_b0p5_N32_nq240.json"
    selected_path = OUT_DIR / "selected_params_v2.json"
    if hashlib.sha256(coefficient_path.read_bytes()).hexdigest() != COEFFICIENT_SHA256:
        raise RuntimeError("coefficient seed sha256 mismatch")
    if hashlib.sha256(selected_path.read_bytes()).hexdigest() != SELECTED_SHA256:
        raise RuntimeError("selection seed sha256 mismatch")
    if (OUT_DIR / "selected_params.json").read_bytes() != selected_path.read_bytes():
        raise RuntimeError("selected_params.json is not byte-identical to v2")

    lap, h_mat = m32_matrices()
    _assert_system_pins(lap, h_mat)
    l_terms = decompose_matrix_to_pauli_terms(lap)
    h_terms = decompose_matrix_to_pauli_terms(h_mat)
    _assert_round_trip(l_terms, lap, "M=32 L")
    _assert_round_trip(h_terms, h_mat, "M=32 H")

    prep = StatePrepSpec(method="law_eberly")
    evolution = EvolutionSpec(n_trotter_steps=100)
    kernel = _selected_kernel("dirichlet")
    actual_kernel = (
        kernel.r_target,
        kernel.r_prime,
        kernel.beta,
        kernel.n_coeff,
        kernel.n_fock,
        kernel.n_quad,
    )
    if actual_kernel != (1.6, 0.25, 0.5, 32, 64, 240):
        raise RuntimeError(f"unexpected shared operating point: {actual_kernel}")
    args = types.SimpleNamespace(coeff_backend="explicit_overlap", n_quad=240)
    coeffs, loaded_path = _coefficients_for_finite("dirichlet", kernel, args)
    if loaded_path != coefficient_path:
        raise RuntimeError(f"unexpected coefficient path: {loaded_path}")
    oracle = prepare_cv_oracle(kernel, prep, coeffs=coeffs)

    heat = _heat_system("dirichlet", 1.0, num_qubits=5)
    _assert_round_trip(heat.l_terms, lap, "M=32 heat L")
    kappa, henrici = nonnormality(lap, h_mat)
    advdiff = build_pauli_system(
        l_terms=l_terms,
        h_terms=h_terms,
        total_time=1.0,
        init_state=np.eye(32, dtype=complex)[:, 1],
        label="qst_advdiff_d32",
    )
    rows = [
        _run_case(
            "heat_d32",
            OUT_DIR / "breadth_d32_raw.npz",
            heat,
            kernel,
            coeffs,
            oracle,
            prep,
            evolution,
            {"boundary": "dirichlet"},
        ),
        _run_case(
            "advdiff_d32",
            OUT_DIR / "breadth_advdiff_d32_raw.npz",
            advdiff,
            kernel,
            coeffs,
            oracle,
            prep,
            evolution,
            {
                "boundary": "advdiff_dirichlet",
                "advection_nu": 1.0,
                "advection_c": 1.0,
                "mesh_peclet": 0.5,
                "kappa_eigenvector": kappa,
                "henrici_departure": henrici,
            },
        ),
    ]

    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    sidecar = build_sidecar(
        args,
        {
            "output": str(OUTPUT),
            "raw_artifacts": [
                str(OUT_DIR / "breadth_d32_raw.npz"),
                str(OUT_DIR / "breadth_advdiff_d32_raw.npz"),
            ],
            "evaluation": ["exact_finite_map", "circuit_statevector"],
            "state_prep_method": "law_eberly",
            "input_artifacts": {
                "selected_params_v2": selected_path,
                "selected_params": OUT_DIR / "selected_params.json",
                "gate_summary": OUT_DIR / "gate_summary.json",
                "coefficients_dirichlet": coefficient_path,
            },
        },
    )
    sidecar["source_sha256"] = {
        **{name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES},
        **sidecar["source_sha256"],
    }
    META.write_text(json.dumps({"sidecar": sidecar}, indent=2) + "\n")
    print(f"wrote {OUTPUT} and {META}", flush=True)


if __name__ == "__main__":
    main()
