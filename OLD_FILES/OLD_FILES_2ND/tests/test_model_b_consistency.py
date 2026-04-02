import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CANDIDATE_CORE_PATHS = [
    _ROOT / "heat1d_lchs_cv_dv.py",
    _ROOT / "RESTART" / "heat1d_lchs_cv_dv.py",
]
_CORE_PATH = next((p for p in _CANDIDATE_CORE_PATHS if p.exists()), None)
if _CORE_PATH is None:
    raise RuntimeError(
        "Unable to locate heat1d_lchs_cv_dv.py. Tried: "
        + ", ".join(str(p) for p in _CANDIDATE_CORE_PATHS)
    )
_SPEC = importlib.util.spec_from_file_location("heat1d_lchs_cv_dv", _CORE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load module spec from {_CORE_PATH}")
sys.path.insert(0, str(_ROOT))
core = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = core
_SPEC.loader.exec_module(core)


def test_model_b_matches_model_a_for_single_fock_component():
    """
    For n_dim=1 there is only n=0, so coherent (Model B) and incoherent
    (Model A) aggregations must be numerically identical.
    """
    cfg = core.HeatEquationConfig(
        total_time=1.0,
        n_steps=3,
        n_dim=1,
        max_fock_level=32,
        hbar=2.0,
    )
    params = core.CVLCHSParams(
        r_target=1.2,
        r_prime=0.3,
        kernel_beta=0.8,
        disp_phase=0.0,
    )

    try:
        solver_a = core.create_solver(cfg, model="A")
        solver_b = core.create_solver(cfg, model="B")
    except ImportError as exc:
        pytest.skip(f"Bosonic backend not available: {exc}")

    a = solver_a.evaluate(params)
    b = solver_b.evaluate(params)

    keys = [
        "post_prob",
        "pde_error",
        "fidelity",
        "trace_distance",
        "pauli_rmse",
        "rho_purity",
    ]
    for key in keys:
        assert np.isclose(a[key], b[key], rtol=1e-9, atol=1e-9), (
            f"Mismatch on {key}: ModelA={a[key]} ModelB={b[key]}"
        )
