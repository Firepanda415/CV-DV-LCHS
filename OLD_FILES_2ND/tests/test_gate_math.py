import importlib.util
from pathlib import Path
import sys

import numpy as np
import hybridlane as hqml

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
core = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = core
_SPEC.loader.exec_module(core)


def _annihilation(cutoff: int) -> np.ndarray:
    a = np.zeros((cutoff, cutoff), dtype=complex)
    for n in range(1, cutoff):
        a[n - 1, n] = np.sqrt(n)
    return a


def _phase_diff(a: float, b: float) -> float:
    return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def test_displacement_generator_xp_identity():
    r"""Check alpha*a^\dagger - alpha^*a against the explicit x/p decomposition."""
    amp = 0.37
    phi = -0.41
    cutoff = 16

    a = _annihilation(cutoff)
    adag = a.conj().T
    x_op = adag + a
    p_op = 1j * (adag - a)

    alpha = amp * np.exp(1j * phi)
    lhs = alpha * adag - np.conj(alpha) * a
    rhs = -1j * amp * (np.cos(phi) * p_op - np.sin(phi) * x_op)
    assert np.allclose(lhs, rhs, atol=1e-12)


def test_displacement_phase_to_quadrature_mapping():
    """
    Hybridlane uses the standard phase-space convention:
      dx = 2 a cos(phi), dp = 2 a sin(phi)  (hbar=2 scaling).
    """
    amp = 0.23
    for phi in (0.0, -np.pi / 2.0, np.pi / 2.0, 0.71):
        heis = hqml.Displacement._heisenberg_rep([amp, phi])
        dx = float(heis[1, 0])
        dp = float(heis[2, 0])
        assert np.isclose(dx, 2.0 * amp * np.cos(phi), atol=1e-12)
        assert np.isclose(dp, 2.0 * amp * np.sin(phi), atol=1e-12)

    # Directly pin the historical confusion point.
    heis_0 = hqml.Displacement._heisenberg_rep([amp, 0.0])
    heis_m90 = hqml.Displacement._heisenberg_rep([amp, -np.pi / 2.0])
    assert np.isclose(heis_0[2, 0], 0.0, atol=1e-12)  # phi=0: no p shift
    assert np.isclose(heis_m90[1, 0], 0.0, atol=1e-12)  # phi=-pi/2: no x shift


def test_conditional_displacement_matches_cp_d_ialpha_cpdag():
    r"""
    Verify decomposition contract:
      CD(alpha) = CP^\dagger D(i alpha) CP.
    """
    amp = 0.19
    phi = -0.27
    op = hqml.ConditionalDisplacement(amp, phi, wires=[0, "m0"])
    decomp = op.decomposition()

    assert len(decomp) == 3
    assert decomp[0].name == "Adjoint(ConditionalParity)"
    assert decomp[1].name == "Displacement"
    assert decomp[2].name == "ConditionalParity"

    d_amp = float(decomp[1].parameters[0])
    d_phi = float(decomp[1].parameters[1])
    assert np.isclose(d_amp, amp, atol=1e-12)
    assert abs(_phase_diff(d_phi, phi + np.pi / 2.0)) < 1e-12


def test_single_trotter_step_has_expected_displacement_terms():
    """
    Construct (do not execute) one fock-component tape and validate the four
    displacement couplings used in one first-order Trotter step.
    """
    cfg = core.HeatEquationConfig(
        total_time=0.01,
        n_steps=1,
        n_dim=4,
        max_fock_level=4,
        init_qubits=(0, 1),
    )
    solver = core.Heat1DLCHSSolver(cfg)

    test_phase = -np.pi / 2.0
    args = (
        0,  # fock_n
        cfg.total_time,
        cfg.n_steps,
        cfg.init_qubits[0],
        cfg.init_qubits[1],
        1.2,  # r_target
        0.3,  # r_prime
        test_phase,
    )
    tape = solver.qnode.construct(args, {})

    disp_ops = [op for op in tape.operations if op.name in ("Displacement", "ConditionalDisplacement")]
    assert len(disp_ops) == 4

    amps = [float(op.parameters[0]) for op in disp_ops]
    phis = [float(op.parameters[1]) for op in disp_ops]

    dt = cfg.total_time / cfg.n_steps
    coeffs = core.pauli_coefficients_for_heat(cfg.h)
    expected_amps = [coeffs["I"] * dt, coeffs["IX"] * dt, coeffs["XX"] * dt, coeffs["YY"] * dt]

    assert np.allclose(amps, expected_amps, atol=1e-12)
    assert np.allclose(phis, [test_phase] * 4, atol=1e-12)
