import numpy as np

from clean_core import pauli_sum_matrix
from make_m32_breadth import decompose_matrix_to_pauli_terms, m32_matrices


def test_m32_dirichlet_spectral_pins():
    lap, _ = m32_matrices()
    assert np.isclose(np.linalg.norm(lap, 2), 2 + 2 * np.cos(np.pi / 33), rtol=1e-8)
    assert np.isclose(np.linalg.eigvalsh(lap)[0], 2 - 2 * np.cos(np.pi / 33), rtol=1e-8)


def test_m32_pauli_round_trips():
    for matrix in m32_matrices():
        terms = decompose_matrix_to_pauli_terms(matrix)
        assert np.linalg.norm(pauli_sum_matrix(tuple(terms)) - matrix) <= 1e-10
