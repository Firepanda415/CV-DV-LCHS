import numpy as np
import pytest

from clean_core import (
    EvolutionSpec,
    KernelSpec,
    StatePrepSpec,
    build_dirichlet_heat_system,
    compute_lchs_coefficients,
)
from clean_hybrid import _apply_oracle_to_circuit, run_clean_lchs
from clean_oracles import detect_statevector_layout, law_eberly_synthesis, prepare_cv_oracle


def _normalized(values):
    arr = np.asarray(values, dtype=complex)
    return arr / np.linalg.norm(arr)


@pytest.mark.parametrize(
    "target",
    [
        np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
        _normalized([1.0, 1.0j, -0.5, 0.25j]),
    ],
)
def test_law_eberly_compiler_prepares_deterministic_targets(target):
    oracle = law_eberly_synthesis(target, n_fock=4)

    assert oracle.method == "law_eberly"
    assert oracle.apply_mode == "law_eberly_pulses"
    assert oracle.oracle_fidelity == pytest.approx(1.0, abs=1e-12)
    assert oracle.metadata["le_aux_ground_probability"] == pytest.approx(1.0, abs=1e-12)
    assert oracle.metadata["n_jc_pulses"] == oracle.metadata["n_sqr_pulses"]


@pytest.mark.parametrize("n_fock", [4, 8])
def test_law_eberly_compiler_prepares_random_targets(n_fock):
    rng = np.random.default_rng(n_fock)
    target = _normalized(rng.normal(size=n_fock) + 1.0j * rng.normal(size=n_fock))

    oracle = law_eberly_synthesis(target, n_fock=n_fock)

    assert oracle.oracle_fidelity == pytest.approx(1.0, abs=1e-12)
    assert oracle.metadata["le_aux_ground_probability"] == pytest.approx(1.0, abs=1e-12)


def test_law_eberly_bosonic_qiskit_circuit_matches_target():
    bosonic_qiskit = pytest.importorskip("bosonic_qiskit")
    pytest.importorskip("qiskit")
    from bosonic_qiskit import CVCircuit, QumodeRegister
    from bosonic_qiskit import util as cv_util
    from qiskit import QuantumRegister

    n_fock = 4
    target = _normalized([0.4, -0.2j, 0.7, 0.3j])
    oracle = law_eberly_synthesis(target, n_fock=n_fock)

    qmr = QumodeRegister(num_qumodes=1, num_qubits_per_qumode=2)
    le = QuantumRegister(1, "le")
    qc = CVCircuit(le, qmr)
    _apply_oracle_to_circuit(qc, qmr[0], oracle, law_eberly_qubit=le[0])

    state, _, _ = cv_util.simulate(
        qc,
        shots=1,
        return_fockcounts=False,
        add_save_statevector=True,
    )
    statevector = np.asarray(state.data, dtype=complex)
    layout = detect_statevector_layout(n_fock, 1)
    if layout == "fock_major":
        prepared = np.array([statevector[n * 2 + 1] for n in range(n_fock)])
    elif layout == "qubit_major":
        prepared = statevector[n_fock : 2 * n_fock]
    else:
        raise AssertionError(f"Unexpected layout {layout}")

    assert np.linalg.norm(prepared) ** 2 == pytest.approx(1.0, abs=1e-12)
    assert abs(np.vdot(target, prepared)) ** 2 == pytest.approx(1.0, abs=1e-12)


def test_law_eberly_clean_lchs_preserves_dv_output_shape():
    kernel = KernelSpec(
        r_target=1.2,
        r_prime=0.2,
        beta=0.5,
        n_coeff=4,
        n_fock=4,
        n_quad=40,
    )
    system = build_dirichlet_heat_system(
        num_qubits=1,
        alpha=1.0,
        grid_spacing=1.0,
        total_time=1.0,
    )
    prep = StatePrepSpec(method="law_eberly")
    coeffs = compute_lchs_coefficients(kernel)
    oracle = prepare_cv_oracle(kernel, prep, coeffs=coeffs)

    result = run_clean_lchs(
        system,
        kernel,
        prep,
        EvolutionSpec(n_trotter_steps=2),
        coeffs=coeffs,
        oracle=oracle,
    )

    assert result.observed_vector.shape == (2,)
    assert result.oracle_fidelity == pytest.approx(1.0, abs=1e-12)
    assert result.fidelity_vs_truncated == pytest.approx(1.0, abs=1e-10)


def test_law_eberly_density_readout_projects_auxiliary_qubit():
    kernel = KernelSpec(
        r_target=1.2,
        r_prime=0.2,
        beta=0.5,
        n_coeff=4,
        n_fock=4,
        n_quad=40,
    )
    system = build_dirichlet_heat_system(
        num_qubits=1,
        alpha=1.0,
        grid_spacing=1.0,
        total_time=1.0,
    )
    prep = StatePrepSpec(method="law_eberly")
    coeffs = compute_lchs_coefficients(kernel)
    oracle = prepare_cv_oracle(kernel, prep, coeffs=coeffs)

    result = run_clean_lchs(
        system,
        kernel,
        prep,
        EvolutionSpec(n_trotter_steps=2, readout_mode="postselect_density_matrix"),
        coeffs=coeffs,
        oracle=oracle,
    )

    assert result.observed_vector.shape == (2,)
    assert result.fidelity_vs_truncated == pytest.approx(1.0, abs=1e-10)
