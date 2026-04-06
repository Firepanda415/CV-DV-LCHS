from clean_sweep import RANKING_OBJECTIVES, _score, _score_columns


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
