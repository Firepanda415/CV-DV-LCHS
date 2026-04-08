import numpy as np

from clean_core import KernelSpec, StatePrepSpec, padded_seed_state
from clean_oracles import (
    OraclePreparation,
    oracle_from_snap_parameter_payload,
    prepare_cv_oracle,
    snap_d_initial_guess_from_oracle,
    snap_parameter_payload,
)


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


def test_snap_d_warm_start_pads_new_layers_with_identity():
    oracle = OraclePreparation(
        method="snap_d",
        target_state=np.array([1.0, 0.0, 0.0], dtype=complex),
        prepared_state=np.array([1.0, 0.0, 0.0], dtype=complex),
        apply_mode="snap_d_layers",
        oracle_fidelity=1.0,
        metadata={"snap_n_snap": 3},
        snap_thetas_per_layer=(
            np.array([0.1, 0.2, 0.3], dtype=float),
            np.array([0.4, 0.5, 0.6], dtype=float),
        ),
        snap_alphas_per_layer=(0.25 + 0.5j, -0.75 + 0.125j),
    )

    guess = snap_d_initial_guess_from_oracle(oracle, depth=3, n_snap=4)

    assert guess is not None
    assert guess.shape == (18,)
    assert np.allclose(guess[:4], [0.1, 0.2, 0.3, 0.0])
    assert np.allclose(guess[4:6], [0.25, 0.5])
    assert np.allclose(guess[6:10], [0.4, 0.5, 0.6, 0.0])
    assert np.allclose(guess[10:12], [-0.75, 0.125])
    assert np.allclose(guess[12:], 0.0)


def test_snap_d_payload_round_trips_without_reoptimization():
    np.random.seed(0)
    coeffs = np.array([1.0, 0.2, 0.1j], dtype=complex)
    kernel = _kernel()
    oracle = prepare_cv_oracle(
        kernel,
        StatePrepSpec(method="snap_d", snap_depth=2, snap_restarts=1, snap_maxiter=40),
        coeffs=coeffs,
    )

    payload = snap_parameter_payload(oracle)
    assert payload is not None
    replay = oracle_from_snap_parameter_payload(
        padded_seed_state(coeffs, kernel.n_fock),
        n_fock=kernel.n_fock,
        payload=payload,
    )

    assert np.allclose(replay.prepared_state, oracle.prepared_state, atol=1e-12)
    assert replay.oracle_fidelity == oracle.oracle_fidelity
    assert replay.metadata["snap_replayed_from_payload"] is True
