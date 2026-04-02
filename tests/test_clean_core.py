import numpy as np
from scipy.linalg import expm

from clean_core import build_dirichlet_heat_system, build_pauli_system, generator_matrix


def test_dirichlet_heat_reconstructs_tridiagonal_generator():
    system = build_dirichlet_heat_system(
        num_qubits=2,
        alpha=1.0,
        grid_spacing=1.0,
        total_time=1.0,
        init_basis_index=0,
    )

    expected = np.array(
        [
            [2.0, -1.0, 0.0, 0.0],
            [-1.0, 2.0, -1.0, 0.0],
            [0.0, -1.0, 2.0, -1.0],
            [0.0, 0.0, -1.0, 2.0],
        ],
        dtype=complex,
    )
    assert np.allclose(generator_matrix(system), expected, atol=1e-10)


def test_generic_pauli_system_matches_manual_matrix_and_exact_map():
    system = build_pauli_system(
        l_terms=[("I", 0.2 + 0.0j)],
        h_terms=[("Y", -0.7 + 0.0j)],
        total_time=0.5,
        init_state=np.array([1.0, 1.0j]),
        label="generic_test",
    )

    y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    expected_generator = 0.2 * np.eye(2, dtype=complex) + 1.0j * (-0.7 * y)
    assert np.allclose(generator_matrix(system), expected_generator, atol=1e-10)

    exact_map = expm(-expected_generator * 0.5)
    assert np.allclose(expm(-generator_matrix(system) * system.total_time), exact_map, atol=1e-10)
