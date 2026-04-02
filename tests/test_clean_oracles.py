import numpy as np

from clean_core import KernelSpec, StatePrepSpec, padded_seed_state
from clean_oracles import prepare_cv_oracle


def _kernel() -> KernelSpec:
    return KernelSpec(
        r_target=1.0,
        r_prime=0.1,
        beta=0.7,
        n_coeff=3,
        n_fock=8,
        n_quad=24,
    )


def test_injection_oracle_is_exact():
    coeffs = np.array([1.0, 0.25, 0.1j], dtype=complex)
    oracle = prepare_cv_oracle(_kernel(), StatePrepSpec(method="injection"), coeffs=coeffs)
    target = padded_seed_state(coeffs, 8)
    assert np.allclose(oracle.prepared_state, target, atol=1e-12)
    assert oracle.oracle_fidelity == 1.0


def test_givens_oracle_reconstructs_target_state():
    coeffs = np.array([1.0, 0.3, -0.2j], dtype=complex)
    oracle = prepare_cv_oracle(_kernel(), StatePrepSpec(method="givens"), coeffs=coeffs)
    target = padded_seed_state(coeffs, 8)
    assert np.allclose(oracle.prepared_state, target, atol=1e-10)
    assert oracle.oracle_fidelity > 1.0 - 1e-10


def test_snap_d_oracle_returns_valid_preparation():
    np.random.seed(0)
    coeffs = np.array([1.0, 0.2, 0.1j], dtype=complex)
    oracle = prepare_cv_oracle(
        _kernel(),
        StatePrepSpec(method="snap_d", snap_depth=2, snap_restarts=1, snap_maxiter=40),
        coeffs=coeffs,
    )
    assert oracle.prepared_state.shape == (8,)
    assert 0.0 <= oracle.oracle_fidelity <= 1.0
