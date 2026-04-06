import numpy as np

from clean_core import generator_matrix
from clean_sweep import BOUNDARY_CONDITIONS, RANKING_OBJECTIVES, _build_heat_system, _score, _score_columns


def _row() -> dict:
    return {
        "fidelity": 0.81,
        "postselection_probability": 0.25,
        "oracle_fidelity": 0.64,
        "fidelity_vs_truncated": 0.95,
    }


def test_supported_ranking_objectives_have_score_columns():
    columns = _score_columns(_row())
    assert set(columns) == {f"score_{objective}" for objective in RANKING_OBJECTIVES}


def test_prep_pde_score_penalizes_low_oracle_fidelity():
    row = _row()
    assert _score(row, "pde") == row["fidelity"]
    assert _score(row, "oracle") == row["oracle_fidelity"]
    assert _score(row, "truncated") == row["fidelity_vs_truncated"]
    assert _score(row, "prep_pde") < _score(row, "pde")
    assert _score(row, "balanced") < _score(row, "pde")


def test_boundary_builder_selects_distinct_heat_generators():
    assert set(BOUNDARY_CONDITIONS) == {"dirichlet", "neumann", "periodic"}

    systems = {
        bc: _build_heat_system(
            bc,
            num_qubits=2,
            alpha=1.0,
            grid_spacing=1.0,
            total_time=1.0,
            init_basis_index=0,
        )
        for bc in BOUNDARY_CONDITIONS
    }
    mats = {bc: generator_matrix(system) for bc, system in systems.items()}

    assert np.allclose(np.diag(mats["dirichlet"]), [2.0, 2.0, 2.0, 2.0])
    assert np.allclose(np.diag(mats["neumann"]), [1.0, 2.0, 2.0, 1.0])
    assert np.allclose(np.diag(mats["periodic"]), [2.0, 2.0, 2.0, 2.0])
    assert mats["periodic"][0, -1] == -1.0
    assert mats["periodic"][-1, 0] == -1.0
    assert mats["dirichlet"][0, -1] == 0.0
    assert mats["dirichlet"][-1, 0] == 0.0
