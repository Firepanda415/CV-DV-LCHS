import argparse
import json
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.special import eval_hermite

from clean_core import (
    PAPER_TO_CODE_L_SCALE,
    EvolutionSpec,
    KernelSpec,
    PauliSystemSpec,
    PauliTerm,
    StatePrepSpec,
    basis_state,
    build_dirichlet_heat_system,
    build_pauli_system,
    code_l_terms,
    compute_lchs_coefficients,
    compute_lchs_coefficients_explicit,
    exact_reference_map,
    exact_truncated_cv_map,
    exact_truncated_cv_map_h0,
    gamma_hbar1,
    kernel_g_beta,
    kernel_g_beta_code,
    pauli_sum_matrix,
    position_operator,
    scaled_map_metrics,
    squeeze_operator,
    state_fidelity,
    system_blocks,
    truncated_oscillator_states,
    zeroth_moment_scale,
)
from clean_hybrid import (
    _bosonic_sq_for_postselect,
    _bosonic_sq_for_prepare,
    _conditional_displacement_alpha,
    build_hybrid_circuit,
    circuit_resource_report,
    run_clean_lchs,
)
from clean_oracles import (
    OraclePreparation,
    align_oracle_global_phase,
    displacement_matrix,
    optimize_snap_d,
    oracle_from_snap_parameter_payload,
    prepare_cv_oracle,
    snap_parameter_payload,
)
import revision_eval
from revision_eval import (
    convert_paper_rows,
    evaluate_exact_map_row,
    first_order_trotter_bound,
    map_error_chain,
    reconstruct_circuit_map,
    resource_ledger,
    to_code_coordinate,
    to_code_gamma,
    to_code_l_coupling_norm,
    to_code_lambda,
    to_paper_coordinate,
    to_paper_gamma,
    to_paper_l_norm,
    to_paper_lambda,
    x_norm_bound_code,
    x_norm_bound_paper,
)


def _simulate_single_mode(build):
    pytest.importorskip("bosonic_qiskit")
    from bosonic_qiskit import CVCircuit, QumodeRegister
    from bosonic_qiskit import util as cv_util

    qmr = QumodeRegister(num_qumodes=1, num_qubits_per_qumode=3)
    qc = CVCircuit(qmr)
    build(qc, qmr[0])
    state, _, _ = cv_util.simulate(
        qc,
        shots=1,
        return_fockcounts=False,
        add_save_statevector=True,
    )
    return np.asarray(state.data, dtype=complex)


def _complex_quad(fn):
    real = quad(
        lambda value: float(np.real(fn(value))),
        -np.inf,
        np.inf,
        epsabs=1e-11,
        epsrel=1e-10,
        limit=200,
    )[0]
    imag = quad(
        lambda value: float(np.imag(fn(value))),
        -np.inf,
        np.inf,
        epsabs=1e-11,
        epsrel=1e-10,
        limit=200,
    )[0]
    return real + 1.0j * imag


def test_t1_dense_broad_x_squeezed_vacuum_matches_analytic_projection():
    r = 0.5
    n_fock = 48
    state = squeeze_operator(n_fock, -r) @ basis_state(n_fock, 0)
    x_hat = position_operator(n_fock)
    variance = float(np.real(np.vdot(state, x_hat @ x_hat @ state)))
    expected_variance = np.exp(2.0 * r) / 2.0

    analytic = np.zeros(n_fock, dtype=complex)
    for m in range(n_fock // 2):
        analytic[2 * m] = (
            math.sqrt(math.factorial(2 * m))
            / (2**m * math.factorial(m))
            * np.tanh(r) ** m
        )
    analytic /= np.linalg.norm(analytic)

    assert abs(variance - expected_variance) / expected_variance <= 1e-6
    assert state_fidelity(state, analytic) >= 1.0 - 1e-8


def test_t2_coefficient_and_fock_cutoffs_are_independent():
    kernel = KernelSpec(
        r_target=1.0,
        r_prime=0.5,
        beta=0.5,
        n_coeff=6,
        n_fock=32,
        n_quad=40,
    )
    seed = basis_state(kernel.n_coeff, kernel.n_coeff - 1)
    squeezed, _ = truncated_oscillator_states(kernel, seed)

    assert np.sum(np.abs(squeezed[kernel.n_coeff :]) ** 2) > 1e-6
    assert kernel.n_coeff == 6
    assert kernel.n_fock == 32


def test_t3_bosonic_prepare_segment_matches_dense_broad_x_state():
    r_prime = 0.3
    seed = np.zeros(8, dtype=complex)
    seed[:3] = [0.8, 0.3j, -0.2]
    seed /= np.linalg.norm(seed)

    prepared = _simulate_single_mode(
        lambda qc, mode: (
            qc.cv_initialize(seed, mode),
            qc.cv_sq(_bosonic_sq_for_prepare(r_prime), mode),
        )
    )
    dense = squeeze_operator(8, -r_prime) @ seed

    assert state_fidelity(prepared, dense) >= 1.0 - 1e-8


def test_t4_bosonic_postselection_bra_matches_dense_state():
    r_target = 0.4
    indices = (0, 2, 4, 6)
    circuit_bra = []
    for index in indices:
        state = _simulate_single_mode(
            lambda qc, mode, index=index: (
                qc.cv_initialize(index, mode),
                qc.cv_sq(_bosonic_sq_for_postselect(r_target), mode),
            )
        )
        circuit_bra.append(state[0])

    phi_post = squeeze_operator(8, -r_target) @ basis_state(8, 0)
    dense_bra = np.conj(phi_post[list(indices)])

    assert state_fidelity(circuit_bra, dense_bra) >= 1.0 - 1e-8


def test_t5_paper_and_code_displacement_pairing_is_invariant():
    n_fock = 8
    lam = 0.13
    x_paper = np.sqrt(2.0) * position_operator(n_fock)
    paper_displacement = expm(-1.0j * lam * x_paper)
    lambda_code = np.sqrt(2.0) * lam
    code_displacement = displacement_matrix(
        _conditional_displacement_alpha(lambda_code),
        n_fock,
    )

    rel_error = np.linalg.norm(paper_displacement - code_displacement, "fro")
    rel_error /= np.linalg.norm(paper_displacement, "fro")
    assert rel_error <= 1e-12


def test_t6_paper_and_code_coefficient_integrals_match():
    r_target = 0.9
    r_prime = 0.3
    beta = 0.5
    sigma_prime = np.exp(r_prime)
    gamma_paper = 0.25 * (
        np.exp(-2.0 * r_prime) - np.exp(-2.0 * r_target)
    )
    gamma_code = gamma_hbar1(r_target, r_prime)

    for n in range(5):
        paper = _complex_quad(
            lambda x: eval_hermite(n, x / (np.sqrt(2.0) * sigma_prime))
            * kernel_g_beta(np.array([x]), beta)[0]
            * np.exp(-gamma_paper * x * x)
        )
        code = _complex_quad(
            lambda q: eval_hermite(n, q / sigma_prime)
            * kernel_g_beta_code(np.array([q]), beta)[0]
            * np.exp(-gamma_code * q * q)
        )
        assert abs(paper - code) / abs(paper) <= 1e-8


def test_t7_l_coupling_conversion_preserves_joint_generator():
    system = build_pauli_system(
        l_terms=(("I", 1.0), ("X", 0.2)),
        h_terms=(("Z", 0.3),),
        total_time=0.7,
        init_state=basis_state(2, 0),
    )
    l_paper, h_block = system_blocks(system)
    l_code = pauli_sum_matrix(code_l_terms(system.l_terms))
    x_code = position_operator(6)
    identity = np.eye(6)
    paper_generator = np.kron(np.sqrt(2.0) * x_code, l_paper) + np.kron(
        identity, h_block
    )
    code_generator = np.kron(x_code, l_code) + np.kron(identity, h_block)

    np.testing.assert_allclose(code_generator, paper_generator, atol=1e-15, rtol=0.0)
    assert np.linalg.norm(code_generator, 2) == pytest.approx(
        np.linalg.norm(paper_generator, 2), abs=1e-12
    )
    assert PAPER_TO_CODE_L_SCALE == pytest.approx(np.sqrt(2.0), abs=0.0)
    assert system.total_time == 0.7


def test_t8_zeroth_moment_scale_normalizes_zero_time_map():
    kernel = KernelSpec(
        r_target=1.2,
        r_prime=0.2,
        beta=0.5,
        n_coeff=4,
        n_fock=8,
        n_quad=40,
    )
    system = build_dirichlet_heat_system(
        num_qubits=1,
        alpha=1.0,
        grid_spacing=1.0,
        total_time=0.0,
    )

    defect = (
        zeroth_moment_scale(kernel) * exact_truncated_cv_map(system, kernel)
        - np.eye(system.dv_dim)
    )
    assert np.linalg.norm(defect, 2) <= 1e-12


def test_t9_fixed_scale_metric_detects_hidden_amplitude_error():
    system = build_dirichlet_heat_system(
        num_qubits=1,
        alpha=1.0,
        grid_spacing=1.0,
        total_time=0.3,
    )
    reference = exact_reference_map(system)
    alpha = 2.0
    synthetic_map = 0.5 * reference / alpha
    metrics = scaled_map_metrics(synthetic_map, reference, scale=alpha)

    assert metrics["rel_frobenius_error"] == pytest.approx(0.5, abs=1e-12)
    assert metrics["rel_frobenius_error"] > 0.1
    assert all(
        row["conditional_fidelity"] >= 1.0 - 1e-12
        for row in metrics["per_input"]
    )


def test_t10_map_error_chain_uses_target_scale_and_closes():
    system = build_dirichlet_heat_system(
        num_qubits=1,
        alpha=0.1,
        grid_spacing=1.0,
        total_time=0.2,
        init_basis_index=0,
    )
    kernel = KernelSpec(
        r_target=1.2,
        r_prime=0.5,
        beta=0.5,
        n_coeff=4,
        n_fock=8,
        n_quad=40,
    )
    coeffs = compute_lchs_coefficients(kernel)
    prep = StatePrepSpec(method="law_eberly")
    evolution = EvolutionSpec(n_trotter_steps=5)
    oracle = prepare_cv_oracle(kernel, prep, coeffs=coeffs)
    circuit_map = reconstruct_circuit_map(
        system, kernel, prep, evolution, coeffs, oracle
    )
    chain = map_error_chain(
        system, kernel, coeffs, oracle.prepared_state, circuit_map
    )

    for name in ("eps_model", "eps_synth", "eps_t", "eps_tot"):
        assert chain[name] >= 0.0
    assert chain["eps_tot"] <= chain["triangle_bound"] + 1e-10
    assert chain["alpha_target"] == pytest.approx(
        zeroth_moment_scale(kernel, coeffs), abs=1e-12
    )
    assert chain["prepared_scale"]["decomposition_residual"] <= 1e-12


def test_t11_h0_fast_map_matches_dense_map_and_guards_assumptions():
    kernel = KernelSpec(
        r_target=1.2,
        r_prime=0.2,
        beta=0.5,
        n_coeff=4,
        n_fock=8,
        n_quad=40,
    )
    seed = compute_lchs_coefficients(kernel)
    h0_system = build_dirichlet_heat_system(
        num_qubits=1,
        alpha=1.0,
        grid_spacing=1.0,
        total_time=0.3,
    )
    dense = exact_truncated_cv_map(h0_system, kernel, seed)
    fast = exact_truncated_cv_map_h0(h0_system, kernel, seed)
    np.testing.assert_allclose(fast, dense, atol=1e-10, rtol=0.0)

    nonzero_h = build_pauli_system(
        l_terms=(("I", 1.0),),
        h_terms=(("Z", 0.2),),
        total_time=0.3,
        init_state=basis_state(2, 0),
    )
    with pytest.raises(ValueError, match="H = 0"):
        exact_truncated_cv_map_h0(nonzero_h, kernel, seed)

    nonhermitian_l = PauliSystemSpec(
        l_terms=(PauliTerm("I", 1.0), PauliTerm("X", 1.0j)),
        h_terms=(PauliTerm("I", 0.0),),
        total_time=0.3,
        init_state=basis_state(2, 0),
    )
    with pytest.raises(ValueError, match="Hermitian L"):
        exact_truncated_cv_map_h0(nonhermitian_l, kernel, seed)


def test_t12_explicit_coefficients_are_stable_under_interval_refinement():
    kernel = KernelSpec(
        r_target=1.2,
        r_prime=0.2,
        beta=0.5,
        n_coeff=8,
        n_fock=8,
        n_quad=80,
    )
    coeffs = {
        scale: compute_lchs_coefficients_explicit(
            kernel,
            k_max_scale=scale,
            epsabs=1e-12,
            epsrel=1e-11,
        )
        for scale in (0.75, 1.0, 1.5)
    }
    reference = coeffs[1.0]

    for candidate in coeffs.values():
        overlap = np.vdot(candidate, reference)
        aligned = candidate * overlap / abs(overlap)
        assert np.linalg.norm(aligned - reference) <= 1e-3


def test_t16_hbar_conversion_round_trips():
    rng = np.random.default_rng(20260720)
    values = rng.normal(size=32)
    positive_values = rng.uniform(0.01, 10.0, size=32)

    for to_paper, to_code, samples in (
        (to_paper_coordinate, to_code_coordinate, values),
        (to_paper_lambda, to_code_lambda, values),
        (to_paper_gamma, to_code_gamma, values),
        (to_paper_l_norm, to_code_l_coupling_norm, positive_values),
    ):
        np.testing.assert_allclose(
            to_code(to_paper(samples)),
            samples,
            atol=1e-14,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            to_paper(to_code(samples)),
            samples,
            atol=1e-14,
            rtol=0.0,
        )

    n_fock = 64
    assert x_norm_bound_paper(n_fock) / x_norm_bound_code(n_fock) == pytest.approx(
        np.sqrt(2.0), abs=1e-14
    )
    r_target = 1.2
    r_prime = 0.5
    assert to_paper_gamma(gamma_hbar1(r_target, r_prime)) == pytest.approx(
        0.25 * (np.exp(-2.0 * r_prime) - np.exp(-2.0 * r_target)),
        abs=1e-14,
    )


def test_t17a_resource_ledger_reconciles_and_rejects_fake_operator(monkeypatch):
    system = build_dirichlet_heat_system(
        num_qubits=1,
        alpha=0.1,
        grid_spacing=1.0,
        total_time=0.2,
        init_basis_index=1,
    )
    kernel = KernelSpec(
        r_target=1.2,
        r_prime=0.5,
        beta=0.5,
        n_coeff=4,
        n_fock=8,
        n_quad=40,
    )
    coeffs = compute_lchs_coefficients(kernel)
    prep = StatePrepSpec(method="law_eberly")
    evolution = EvolutionSpec(n_trotter_steps=5)
    oracle = prepare_cv_oracle(kernel, prep, coeffs=coeffs)
    qc, _, _ = build_hybrid_circuit(
        system, kernel, prep, evolution, coeffs=coeffs, oracle=oracle
    )
    ledger = resource_ledger(
        qc,
        oracle=oracle,
        evolution=evolution,
        system=system,
        kernel=kernel,
    )
    predicted = revision_eval._add_counts(
        ledger["C_osc_prep"],
        ledger["C_input"],
        ledger["n_t_C_step"],
        ledger["C_post"],
    )
    assert predicted == circuit_resource_report(qc)["count_ops"]

    original = revision_eval._resource_segments

    def with_fake_operator(*args, **kwargs):
        segments = original(*args, **kwargs)
        segments["C_input"] = {**segments["C_input"], "fake": 1}
        return segments

    monkeypatch.setattr(revision_eval, "_resource_segments", with_fake_operator)
    with pytest.raises(ValueError, match="resource ledger mismatch"):
        resource_ledger(
            qc,
            oracle=oracle,
            evolution=evolution,
            system=system,
            kernel=kernel,
        )


def test_global_phase_alignment_restores_target_gauge():
    kernel = KernelSpec(
        r_target=1.2,
        r_prime=0.5,
        beta=0.5,
        n_coeff=4,
        n_fock=8,
        n_quad=40,
    )
    coeffs = compute_lchs_coefficients(kernel)
    oracle = prepare_cv_oracle(kernel, StatePrepSpec(method="law_eberly"), coeffs=coeffs)
    overlap = np.vdot(oracle.target_state, oracle.prepared_state)
    assert abs(np.imag(overlap)) <= 1e-12
    assert np.real(overlap) > 0.0
    assert "global_phase_alignment" in oracle.metadata

    initial_angle = oracle.metadata["global_phase_alignment"]
    oracle.prepared_state = -oracle.prepared_state
    realigned = align_oracle_global_phase(oracle)
    flipped_overlap = np.vdot(realigned.target_state, realigned.prepared_state)
    assert np.real(flipped_overlap) > 0.0
    assert abs(np.imag(flipped_overlap)) <= 1e-12
    assert (
        abs(
            abs(realigned.metadata["global_phase_alignment"] - initial_angle)
            - np.pi
        )
        <= 1e-12
    )

    prepared = realigned.prepared_state.copy()
    total_angle = realigned.metadata["global_phase_alignment"]
    realigned_twice = align_oracle_global_phase(realigned)
    np.testing.assert_array_equal(realigned_twice.prepared_state, prepared)
    assert realigned_twice.metadata["global_phase_alignment"] == total_angle
    assert realigned_twice.metadata["statevector_gauge"] == "raw_circuit_output"


def test_t13_nonnormal_benchmark_is_positive_and_nonnormal():
    system = build_pauli_system(
        l_terms=[("II", 0.75), ("ZI", 0.25), ("IZ", 0.15)],
        h_terms=[("XI", 0.30), ("IX", 0.20)],
        total_time=1.0,
        init_state=np.eye(4, dtype=complex)[:, 1],
        label="qst_nonnormal_4d",
    )
    l_block, h_block = system_blocks(system)
    commutator = l_block @ h_block - h_block @ l_block
    generator = l_block + 1.0j * h_block
    np.testing.assert_allclose(
        generator @ generator.conj().T - generator.conj().T @ generator,
        -2.0j * commutator,
        atol=1e-15,
        rtol=0.0,
    )
    assert np.min(np.linalg.eigvalsh(l_block)) == pytest.approx(0.35)
    assert np.linalg.norm(commutator) > 0.0

    bound = first_order_trotter_bound(system, 8, 1.0, 10)
    assert bound["Gamma_1"] == pytest.approx(
        bound["Gamma_1_LL"] + bound["Gamma_1_LH"] + bound["Gamma_1_HH"]
    )
    assert bound["eps_t_bound"] == pytest.approx(bound["Gamma_1"] / 20.0)


def test_t14_snap_seed_reproduces_deterministic_telemetry():
    target = np.array([1.0, 0.2j, -0.1, 0.05], dtype=complex)
    deterministic = (
        "start_index",
        "kind",
        "objective",
        "nit",
        "nfev",
        "success",
        "status",
        "message",
    )
    runs = [
        optimize_snap_d(
            target,
            n_fock=4,
            depth=1,
            n_snap=4,
            n_restarts=1,
            maxiter=50,
            random_seed=11,
        )
        for _ in range(2)
    ]
    records = [run.metadata["start_records"] for run in runs]
    assert len(records[0]) == 1
    for left, right in zip(records[0], records[1]):
        assert {key: left[key] for key in deterministic} == {
            key: right[key] for key in deterministic
        }
        assert left["wall_seconds"] > 0.0
        assert right["wall_seconds"] > 0.0

    values = np.asarray([row["objective"] for row in records[0]])
    expected = {
        "objective_min": float(np.min(values)),
        "objective_median": float(np.median(values)),
        "objective_max": float(np.max(values)),
        "optimizer_success_fraction": float(
            np.mean([row["success"] for row in records[0]])
        ),
    }
    assert runs[0].metadata["restart_summary"] == expected
    by_kind = runs[0].metadata["restart_summary_by_kind"]
    assert by_kind["random_restart"] == {"count": 1, **expected}


def test_t15_breadth_scaling_row_schema_fixture():
    row = evaluate_exact_map_row(
        {
            "r_target": 1.2,
            "r_prime": 0.5,
            "beta": 0.5,
            "n_coeff": 4,
            "n_fock": 8,
            "n_quad": 40,
        },
        "dirichlet",
        0.1,
    )
    assert {
        "n_coeff",
        "n_fock",
        "oracle_fidelity",
        "evaluation",
        "state_prep_method",
    } <= row.keys()


def test_t17b_export_paper_converts_only_named_fields(tmp_path):
    import csv

    source = {
        "gamma_code": 2.0,
        "lambda_code": 3.0,
        "x_norm_bound_code": 4.0,
        "L_coupling_norm_code": 5.0,
        "q_code": "[1.0,2.0]",
        "n_fock": 64,
        "rel_frobenius_error": 0.125,
        "hbar_convention": "code_hbar1",
    }
    source_path = tmp_path / "source.csv"
    output_path = tmp_path / "paper.csv"
    with source_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(source))
        writer.writeheader()
        writer.writerow(source)
    with source_path.open(newline="", encoding="utf-8") as handle:
        converted = convert_paper_rows(list(csv.DictReader(handle)))
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(converted[0]))
        writer.writeheader()
        writer.writerows(converted)
    with output_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))

    assert float(row["gamma_paper"]) == to_paper_gamma(2.0)
    assert float(row["lambda_paper"]) == to_paper_lambda(3.0)
    assert float(row["x_norm_bound_paper"]) == pytest.approx(
        4.0 * np.sqrt(2.0)
    )
    assert float(row["L_norm_paper"]) == to_paper_l_norm(5.0)
    assert json.loads(row["x_paper"]) == pytest.approx(
        [np.sqrt(2.0), 2.0 * np.sqrt(2.0)]
    )
    assert row["N_Fock"] == "64"
    assert float(row["rel_frobenius_error"]) == 0.125
    assert json.loads(row["field_labels"])["rel_frobenius_error"] == "[invariant]"
    assert row["hbar_convention"] == "paper_hbar2"
    with pytest.raises(ValueError, match="no explicit hbar convention"):
        convert_paper_rows([{"gamma": 2.0}])


def test_job5_snap_payload_fidelity_guard():
    target = np.array([1.0, 0.0], dtype=complex)
    oracle = OraclePreparation(
        method="snap_d",
        target_state=target,
        prepared_state=target.copy(),
        apply_mode="snap_d_layers",
        oracle_fidelity=1.0,
        metadata={"snap_n_snap": 0},
    )
    payload = snap_parameter_payload(oracle)
    changed_target = np.array([0.0, 1.0], dtype=complex)

    assert payload["coefficient_convention"] == "kernel_g_beta_code"
    with pytest.raises(ValueError, match="verify_fidelity=False"):
        oracle_from_snap_parameter_payload(
            changed_target, n_fock=2, payload=payload
        )
    replayed = oracle_from_snap_parameter_payload(
        changed_target, n_fock=2, payload=payload, verify_fidelity=False
    )
    assert replayed.oracle_fidelity == 0.0

    payload.pop("oracle_fidelity")
    with pytest.raises(ValueError, match="verify_fidelity=False"):
        oracle_from_snap_parameter_payload(target, n_fock=2, payload=payload)


def test_job5_direct_injection_does_not_apply_alignment_twice():
    system = build_pauli_system(
        l_terms=(("I", 0.2),),
        h_terms=(),
        total_time=0.2,
        init_state=basis_state(2, 0),
    )
    kernel = KernelSpec(
        r_target=1.2,
        r_prime=0.5,
        beta=0.5,
        n_coeff=4,
        n_fock=8,
        n_quad=40,
    )
    coeffs = compute_lchs_coefficients(kernel)
    prep = StatePrepSpec(method="injection")
    oracle = prepare_cv_oracle(kernel, prep, coeffs=coeffs)
    oracle.prepared_state = -oracle.prepared_state
    oracle = align_oracle_global_phase(oracle)
    assert abs(oracle.metadata["global_phase_alignment"]) > 1.0

    result = run_clean_lchs(
        system,
        kernel,
        prep,
        EvolutionSpec(n_trotter_steps=1),
        coeffs=coeffs,
        oracle=oracle,
    )
    expected = exact_truncated_cv_map(
        system, kernel, seed_state=oracle.prepared_state
    ) @ basis_state(2, 0)
    relative_error = np.linalg.norm(result.observed_vector - expected) / np.linalg.norm(
        expected
    )
    assert relative_error <= 1e-10


def test_job5_coefficient_files_include_quadrature_and_are_write_once(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(command="test")
    kernels = [
        KernelSpec(
            r_target=1.2,
            r_prime=0.5,
            beta=0.5,
            n_coeff=4,
            n_fock=8,
            n_quad=n_quad,
        )
        for n_quad in (40, 80)
    ]
    frozen = [
        revision_eval._coefficients_for_finite("dirichlet", kernel, args)
        for kernel in kernels
    ]
    assert frozen[0][1] != frozen[1][1]
    assert "nq40" in frozen[0][1].name
    assert "nq80" in frozen[1][1].name

    unchanged = frozen[0][1].read_bytes()
    revision_eval._freeze_coefficients(
        "dirichlet",
        kernels[0],
        frozen[0][0],
        args,
        source="test",
        path=frozen[0][1],
    )
    assert frozen[0][1].read_bytes() == unchanged

    with pytest.raises(ValueError, match="write-once"):
        revision_eval._freeze_coefficients(
            "dirichlet",
            kernels[0],
            frozen[0][0] + 1.0,
            args,
            source="test",
            path=frozen[0][1],
        )


def test_job5_zero_probability_ledger_keeps_empty_inverse_fields():
    guarded = revision_eval._ledger_with_probability({"C_run": {"x": 2}}, 0.0)
    assert guarded["p_succ"] == 0.0
    assert guarded["inv_p_succ"] is None
    assert guarded["E_N_run"] is None
    assert guarded["E_C_accept"] is None


def test_job5_export_preserves_empty_converted_cells():
    converted = convert_paper_rows(
        [{"gamma_code": "", "lambda_code": None, "hbar_convention": "code_hbar1"}]
    )[0]
    assert converted["gamma_paper"] == ""
    assert converted["lambda_paper"] is None


def test_job5_alignment_is_idempotent():
    kernel = KernelSpec(
        r_target=1.2,
        r_prime=0.5,
        beta=0.5,
        n_coeff=4,
        n_fock=8,
        n_quad=40,
    )
    coeffs = compute_lchs_coefficients(kernel)
    oracle = prepare_cv_oracle(kernel, StatePrepSpec(method="law_eberly"), coeffs=coeffs)
    state_before = oracle.prepared_state.copy()
    angle_before = float(oracle.metadata["global_phase_alignment"])

    realigned = align_oracle_global_phase(oracle)

    np.testing.assert_allclose(realigned.prepared_state, state_before, atol=1e-14, rtol=0.0)
    assert float(realigned.metadata["global_phase_alignment"]) == pytest.approx(
        angle_before, abs=1e-14
    )


def test_dv_pauli_rotation_matches_dense_for_odd_and_even_y():
    from qiskit import QuantumCircuit, QuantumRegister
    from qiskit.quantum_info import Operator
    from scipy.linalg import expm as dense_expm

    from clean_core import PauliTerm, pauli_sum_matrix
    from clean_hybrid import _apply_dv_pauli_rotation

    dt = 0.37
    for label, coeff in (("Y", 0.4), ("YI", 0.3), ("XY", 0.25), ("YY", 0.2), ("ZY", 0.15)):
        qreg = QuantumRegister(len(label), "q")
        qc = QuantumCircuit(qreg)
        term = PauliTerm(label=label, coeff=complex(coeff))
        _apply_dv_pauli_rotation(qc, qreg, term, dt)
        compiled = Operator(qc.reverse_bits()).data
        expected = dense_expm(-1.0j * dt * pauli_sum_matrix((term,)))
        phase = compiled[0, 0] / expected[0, 0] if abs(expected[0, 0]) > 1e-12 else 1.0
        np.testing.assert_allclose(compiled, phase * expected, atol=1e-12)
        assert abs(abs(phase) - 1.0) < 1e-12
