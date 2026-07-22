import importlib
import sys

import numpy as np

from clean_core import pauli_sum_matrix


def test_make_2d_circuit_import_has_no_side_effects(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sys.modules.pop("make_2d_circuit", None)
    module = importlib.import_module("make_2d_circuit")
    assert callable(module.main)
    assert list(tmp_path.iterdir()) == []


def test_lap2d_pauli_decomposition_round_trips():
    module = importlib.import_module("make_2d_circuit")
    lap2d = module.laplacian_2d()
    terms = module.decompose_matrix_to_pauli_terms(lap2d)
    assert np.linalg.norm(pauli_sum_matrix(terms) - lap2d) <= 1e-10


def test_lap2d_spectral_pins():
    module = importlib.import_module("make_2d_circuit")
    lap2d = module.laplacian_2d()
    assert np.isclose(np.linalg.norm(lap2d, 2), 7.23606797749979, rtol=1e-8)
    assert np.isclose(np.linalg.eigvalsh(lap2d)[0], 0.7639320225002102, rtol=1e-8)
