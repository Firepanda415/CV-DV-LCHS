#!/usr/bin/env python3
"""Independent CV oracle preparation and readout helpers.

This module owns the oscillator-side logic of the clean stack:

- construction of the target LCHS coefficient state,
- simulator-side preparation models for injection, SNAP+D, and Law-Eberly,
- backend-agnostic extraction of DV vectors from joint CV-DV simulator output.

For readability, the main state-preparation models are:

1. SNAP+D:

       |psi(theta, alpha)> = prod_ell D(alpha_ell) SNAP(theta_ell) |0>

2. Law-Eberly qubit-assisted state synthesis using auxiliary-qubit rotations
   and Jaynes-Cummings exchange pulses, with a separate selective-rotation
   variant for the Bosonic Qiskit ``cv_sqr`` realization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

from clean_core import (
    KernelSpec,
    PauliSystemSpec,
    StatePrepSpec,
    compute_lchs_coefficients as core_compute_lchs_coefficients,
    normalize_vector,
    padded_seed_state,
    physics_to_qiskit_permutation,
    state_fidelity,
)


_LE_EXCITED = 0
_LE_GROUND = 1


@dataclass(frozen=True)
class LawEberlyPulse:
    """One circuit-level Law-Eberly preparation pulse.

    Attributes:
        kind: Either ``"jc"`` for a Jaynes-Cummings exchange pulse, ``"r"``
            for a global auxiliary-qubit rotation, or ``"sqr"`` for a
            Fock-conditioned qubit rotation.
        level: Photon number addressed by the pulse. For ``"jc"``, this is
            the upper Fock level ``n`` in ``|e,n-1> <-> |g,n>``. For
            ``"sqr"``, this is the conditioned oscillator level. For ``"r"``,
            this records the recursion level but does not condition the gate.
        theta: Bosonic Qiskit gate angle.
        phi: Bosonic Qiskit gate phase.
    """

    kind: str
    level: int
    theta: float
    phi: float


@dataclass
class OraclePreparation:
    """Prepared CV oracle plus metadata needed by the runtime.

    Attributes:
        method: State-preparation method name.
        target_state: Desired oscillator target state in the truncated Fock basis.
        prepared_state: Realized oscillator state in the same basis.
        apply_mode: Runtime instruction for how to place the state in circuit.
        oracle_fidelity: Fidelity between ``prepared_state`` and ``target_state``.
        metadata: Method-specific diagnostics and resource counts.
        snap_thetas_per_layer: Per-layer SNAP phases for the SNAP+D ansatz.
        snap_alphas_per_layer: Per-layer displacement amplitudes for SNAP+D.
        law_eberly_pulses: Circuit pulses for the Law-Eberly preparation.
    """

    method: str
    target_state: np.ndarray
    prepared_state: np.ndarray
    apply_mode: str
    oracle_fidelity: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    snap_thetas_per_layer: Tuple[np.ndarray, ...] = ()
    snap_alphas_per_layer: Tuple[complex, ...] = ()
    law_eberly_pulses: Tuple[LawEberlyPulse, ...] = ()


def snap_parameter_payload(oracle: OraclePreparation | None) -> Optional[Dict[str, Any]]:
    """Serialize a SNAP+D oracle into a replayable parameter payload.

    The payload is intentionally limited to the exact layer parameters needed
    to reconstruct the prepared oscillator seed later. It excludes large
    diagnostic arrays such as the target and prepared states.

    Args:
        oracle: Oracle result to serialize.

    Returns:
        JSON-serializable payload or ``None`` when ``oracle`` is not a SNAP+D
        preparation.
    """

    if oracle is None or oracle.method != "snap_d":
        return None

    thetas = [
        [float(theta) for theta in np.asarray(layer, dtype=float).tolist()]
        for layer in oracle.snap_thetas_per_layer
    ]
    alphas = [
        {"real": float(np.real(alpha)), "imag": float(np.imag(alpha))}
        for alpha in oracle.snap_alphas_per_layer
    ]
    n_snap = int(oracle.metadata.get("snap_n_snap", len(thetas[0]) if thetas else 0))

    return {
        "format": "snap_d_layers_v1",
        "method": "snap_d",
        "apply_mode": "snap_d_layers",
        "coefficient_convention": "kernel_g_beta_code",
        "oracle_fidelity": float(oracle.oracle_fidelity),
        "snap_depth": int(len(thetas)),
        "snap_n_snap": n_snap,
        "snap_thetas_per_layer": thetas,
        "snap_alphas_per_layer": alphas,
    }


def oracle_from_snap_parameter_payload(
    target_state: np.ndarray,
    *,
    n_fock: int,
    payload: Mapping[str, Any],
    verify_fidelity: bool = True,
) -> OraclePreparation:
    """Rebuild a SNAP+D oracle from a saved parameter payload.

    Args:
        target_state: Desired oscillator target state in the truncated basis.
        n_fock: Oscillator truncation dimension.
        payload: Serialized SNAP+D layer data produced by
            :func:`snap_parameter_payload`.
        verify_fidelity: Require the reconstructed fidelity to match the value
            stored in the payload.

    Returns:
        ``OraclePreparation`` matching the saved SNAP+D ansatz exactly.

    Raises:
        ValueError: If the payload is malformed or inconsistent.
    """

    if str(payload.get("format", "")) != "snap_d_layers_v1":
        raise ValueError(
            "Unsupported SNAP+D payload format. Expected 'snap_d_layers_v1'."
        )

    thetas_raw = payload.get("snap_thetas_per_layer", [])
    alphas_raw = payload.get("snap_alphas_per_layer", [])
    if not isinstance(thetas_raw, Sequence) or not isinstance(alphas_raw, Sequence):
        raise ValueError("SNAP+D payload must contain per-layer theta and alpha lists.")
    if len(thetas_raw) != len(alphas_raw):
        raise ValueError("SNAP+D payload has mismatched theta/alpha layer counts.")

    depth = int(payload.get("snap_depth", len(thetas_raw)))
    if depth != len(thetas_raw):
        raise ValueError("SNAP+D payload depth does not match serialized layer count.")

    n_snap = int(payload.get("snap_n_snap", len(thetas_raw[0]) if thetas_raw else 0))

    thetas: List[np.ndarray] = []
    alphas: List[complex] = []
    flat_params: List[float] = []

    for layer_thetas_raw, alpha_raw in zip(thetas_raw, alphas_raw):
        layer_thetas = np.asarray(layer_thetas_raw, dtype=float).reshape(-1)
        if layer_thetas.size != n_snap:
            raise ValueError(
                "SNAP+D payload layer theta count does not match snap_n_snap."
            )
        if not isinstance(alpha_raw, Mapping) or "real" not in alpha_raw or "imag" not in alpha_raw:
            raise ValueError(
                "SNAP+D payload alpha entries must map 'real' and 'imag'."
            )
        alpha = complex(float(alpha_raw["real"]), float(alpha_raw["imag"]))
        thetas.append(layer_thetas.copy())
        alphas.append(alpha)
        flat_params.extend(float(theta) for theta in layer_thetas)
        flat_params.extend([float(np.real(alpha)), float(np.imag(alpha))])

    target = padded_seed_state(target_state, n_fock)
    prepared = simulate_snap_d_state(
        np.asarray(flat_params, dtype=float),
        n_fock=n_fock,
        depth=depth,
        n_snap=n_snap,
    )
    fidelity = abs(np.vdot(target, prepared)) ** 2
    if verify_fidelity:
        if "oracle_fidelity" not in payload:
            raise ValueError(
                "SNAP+D payload has no stored oracle_fidelity and cannot be replayed "
                "as a result; use verify_fidelity=False only for a warm-start."
            )
        if abs(fidelity - float(payload["oracle_fidelity"])) > 1e-6:
            raise ValueError(
                "SNAP+D payload was optimized for a different target or convention "
                "and cannot be replayed as a result; use verify_fidelity=False only "
                "for a warm-start."
            )
    return OraclePreparation(
        method="snap_d",
        target_state=target,
        prepared_state=prepared,
        apply_mode=str(payload.get("apply_mode", "snap_d_layers")),
        oracle_fidelity=float(fidelity),
        metadata={
            "snap_depth": int(depth),
            "snap_n_snap": int(n_snap),
            "snap_restarts": 0,
            "snap_maxiter": 0,
            "snap_total_iterations": 0,
            "snap_total_starts": 0,
            "snap_used_warm_start": False,
            "snap_replayed_from_payload": True,
        },
        snap_thetas_per_layer=tuple(thetas),
        snap_alphas_per_layer=tuple(alphas),
    )


def compute_lchs_coefficients(kernel: KernelSpec) -> np.ndarray:
    """Forward to the independent coefficient generator in ``clean_core``."""

    return core_compute_lchs_coefficients(kernel)


def annihilation_operator(n_fock: int) -> np.ndarray:
    """Return the truncated annihilation operator."""

    op = np.zeros((n_fock, n_fock), dtype=complex)
    for n in range(1, n_fock):
        op[n - 1, n] = np.sqrt(n)
    return op


def displacement_matrix(alpha: complex, n_fock: int) -> np.ndarray:
    """Return the truncated displacement operator ``D(alpha)``."""

    a = annihilation_operator(n_fock)
    adag = a.conj().T
    return expm(alpha * adag - np.conj(alpha) * a)


def simulate_snap_d_state(
    params: np.ndarray,
    *,
    n_fock: int,
    depth: int,
    n_snap: int,
) -> np.ndarray:
    """Simulate the dense SNAP+D ansatz in the truncated Fock basis.

    Each layer applies a diagonal SNAP phase update on the first ``n_snap``
    amplitudes followed by a displacement.

    Args:
        params: Flattened real parameter vector. Each layer stores ``n_snap``
            phases followed by the real and imaginary parts of one
            displacement amplitude.
        n_fock: Oscillator truncation dimension.
        depth: Number of SNAP+D layers.
        n_snap: Number of Fock levels assigned explicit SNAP phases per layer.

    Returns:
        Normalized oscillator state prepared by the ansatz.
    """

    state = np.zeros(n_fock, dtype=complex)
    state[0] = 1.0
    params_per_layer = n_snap + 2

    for layer in range(depth):
        offset = layer * params_per_layer
        thetas = params[offset : offset + n_snap]
        alpha = complex(params[offset + n_snap], params[offset + n_snap + 1])

        # SNAP is diagonal in the Fock basis, so it updates amplitudes by phase
        # multiplication only.
        state[:n_snap] *= np.exp(1.0j * thetas)
        state = displacement_matrix(alpha, n_fock) @ state

    return normalize_vector(state)


def optimize_snap_d(
    target_state: np.ndarray,
    *,
    n_fock: int,
    depth: int,
    n_snap: int,
    n_restarts: int,
    maxiter: int,
    initial_guess: Optional[np.ndarray] = None,
    random_seed: Optional[int] = None,
) -> OraclePreparation:
    """Optimize a dense SNAP+D ansatz against a target state.

    The objective is the infidelity

        1 - |<target | psi(params)>|^2.

    Args:
        target_state: Desired oscillator target state.
        n_fock: Oscillator truncation dimension.
        depth: Number of SNAP+D layers.
        n_snap: Number of Fock levels with explicit SNAP phases per layer.
        n_restarts: Number of random restarts for the local optimizer.
        maxiter: Maximum L-BFGS-B iterations per restart.
        initial_guess: Optional flattened SNAP+D parameter vector used as a
            warm start before the random restarts. This is the hook used for
            depth-continuation across increasing ansatz depths.
        random_seed: Optional seed for the local random-restart generator.

    Returns:
        ``OraclePreparation`` describing the best SNAP+D fit found.

    Raises:
        ValueError: If ``depth`` is not positive.
        RuntimeError: If the optimizer never returns a candidate.
    """

    if depth <= 0:
        raise ValueError("snap_d requires snap_depth > 0.")

    target = padded_seed_state(target_state, n_fock)
    params_per_layer = n_snap + 2
    n_params = depth * params_per_layer

    if initial_guess is not None:
        initial_guess = np.asarray(initial_guess, dtype=float).reshape(-1)
        if initial_guess.shape != (n_params,):
            raise ValueError(
                "initial_guess must have shape "
                f"({n_params},), got {initial_guess.shape}."
            )

    def _cost(x: np.ndarray) -> float:
        prepared = simulate_snap_d_state(x, n_fock=n_fock, depth=depth, n_snap=n_snap)
        return 1.0 - abs(np.vdot(target, prepared)) ** 2

    best_x: Optional[np.ndarray] = None
    best_cost = float("inf")
    total_iterations = 0
    rng = np.random.default_rng(random_seed)

    guesses: List[np.ndarray] = []
    if initial_guess is not None:
        # Identity-padded continuation lets a deeper ansatz start from the best
        # shallower fit instead of sampling an unrelated basin.
        guesses.append(initial_guess.copy())

    for _ in range(n_restarts):
        guess = np.zeros(n_params, dtype=float)
        for layer in range(depth):
            offset = layer * params_per_layer
            # Random restarts help sample different phase/displacement basins.
            guess[offset : offset + n_snap] = rng.uniform(-np.pi, np.pi, size=n_snap)
            guess[offset + n_snap] = rng.uniform(-0.5, 0.5)
            guess[offset + n_snap + 1] = rng.uniform(-0.5, 0.5)
        guesses.append(guess)

    start_records: List[Dict[str, Any]] = []
    for start_index, guess in enumerate(guesses):
        started = perf_counter()
        result = minimize(
            _cost,
            guess,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-10},
        )
        start_records.append(
            {
                "start_index": int(start_index),
                "kind": (
                    "warm_start"
                    if initial_guess is not None and start_index == 0
                    else "random_restart"
                ),
                "objective": float(result.fun),
                "nit": int(getattr(result, "nit", 0)),
                "nfev": int(getattr(result, "nfev", 0)),
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "wall_seconds": float(perf_counter() - started),
            }
        )
        total_iterations += int(getattr(result, "nit", 0))
        if float(result.fun) < best_cost:
            best_cost = float(result.fun)
            best_x = np.asarray(result.x, dtype=float)

    if best_x is None:
        raise RuntimeError("SNAP+D optimization failed to produce a result.")

    prepared = simulate_snap_d_state(best_x, n_fock=n_fock, depth=depth, n_snap=n_snap)
    thetas: List[np.ndarray] = []
    alphas: List[complex] = []
    for layer in range(depth):
        offset = layer * params_per_layer
        thetas.append(best_x[offset : offset + n_snap].copy())
        alphas.append(complex(best_x[offset + n_snap], best_x[offset + n_snap + 1]))

    fidelity = abs(np.vdot(target, prepared)) ** 2
    objectives = np.asarray([row["objective"] for row in start_records])
    restart_summary = {
        "objective_min": float(np.min(objectives)),
        "objective_median": float(np.median(objectives)),
        "objective_max": float(np.max(objectives)),
        "optimizer_success_fraction": float(
            np.mean([row["success"] for row in start_records])
        ),
    }
    restart_summary_by_kind = {}
    for kind in sorted({row["kind"] for row in start_records}):
        subset = [row for row in start_records if row["kind"] == kind]
        values = np.asarray([row["objective"] for row in subset])
        restart_summary_by_kind[kind] = {
            "count": len(subset),
            "objective_min": float(np.min(values)),
            "objective_median": float(np.median(values)),
            "objective_max": float(np.max(values)),
            "optimizer_success_fraction": float(
                np.mean([row["success"] for row in subset])
            ),
        }
    return OraclePreparation(
        method="snap_d",
        target_state=target,
        prepared_state=prepared,
        apply_mode="snap_d_layers",
        oracle_fidelity=float(fidelity),
        metadata={
            "snap_depth": int(depth),
            "snap_restarts": int(n_restarts),
            "snap_maxiter": int(maxiter),
            "snap_n_snap": int(n_snap),
            "snap_total_iterations": int(total_iterations),
            "snap_total_starts": int(len(guesses)),
            "snap_used_warm_start": bool(initial_guess is not None),
            "random_seed": random_seed,
            "start_records": start_records,
            "restart_summary": restart_summary,
            "restart_summary_by_kind": restart_summary_by_kind,
        },
        snap_thetas_per_layer=tuple(thetas),
        snap_alphas_per_layer=tuple(alphas),
    )


def snap_d_initial_guess_from_oracle(
    oracle: OraclePreparation | None,
    *,
    depth: int,
    n_snap: int,
) -> Optional[np.ndarray]:
    """Create a flattened SNAP+D warm start from a prior SNAP+D oracle.

    The continuation rule is intentionally simple:

    1. copy the previously optimized layers verbatim,
    2. truncate them if the new depth is smaller,
    3. pad any additional layers with the identity action
       (zero SNAP phases and zero displacement).

    This preserves the prepared state of the shallower circuit exactly while
    giving the deeper optimizer extra degrees of freedom to improve from that
    point.

    Args:
        oracle: Previous oracle result. Non-SNAP or ``None`` inputs return
            ``None``.
        depth: Target SNAP+D depth for the new optimization.
        n_snap: Number of explicit SNAP phases per layer in the new run.

    Returns:
        Flattened parameter vector or ``None`` when no warm start applies.
    """

    if oracle is None or oracle.method != "snap_d":
        return None

    params_per_layer = n_snap + 2
    guess = np.zeros(depth * params_per_layer, dtype=float)

    old_thetas = tuple(oracle.snap_thetas_per_layer)
    old_alphas = tuple(oracle.snap_alphas_per_layer)
    if not old_thetas or not old_alphas:
        return guess

    old_n_snap = int(oracle.metadata.get("snap_n_snap", len(old_thetas[0])))
    shared_depth = min(depth, len(old_thetas), len(old_alphas))
    shared_n_snap = min(n_snap, old_n_snap)

    for layer in range(shared_depth):
        offset = layer * params_per_layer
        guess[offset : offset + shared_n_snap] = old_thetas[layer][:shared_n_snap]
        alpha = old_alphas[layer]
        guess[offset + n_snap] = float(np.real(alpha))
        guess[offset + n_snap + 1] = float(np.imag(alpha))

    return guess


def _apply_law_eberly_jc(
    state: np.ndarray,
    *,
    theta: float,
    phi: float,
) -> np.ndarray:
    """Apply a dense Jaynes-Cummings pulse in Law-Eberly conventions.

    Args:
        state: Joint qubit-oscillator state with shape ``(2, n_fock)``. The
            qubit basis follows Bosonic Qiskit's convention ``|e> = |0>`` and
            ``|g> = |1>``.
        theta: Bosonic Qiskit ``cv_jc`` angle.
        phi: Bosonic Qiskit ``cv_jc`` phase.

    Returns:
        Updated joint state.
    """

    out = np.asarray(state, dtype=complex).copy()
    n_fock = out.shape[1]
    for n in range(1, n_fock):
        a = out[_LE_EXCITED, n - 1]
        b = out[_LE_GROUND, n]
        angle = float(theta) * np.sqrt(n)
        c = np.cos(angle)
        s = np.sin(angle)
        out[_LE_EXCITED, n - 1] = c * a - 1.0j * np.exp(-1.0j * phi) * s * b
        out[_LE_GROUND, n] = c * b - 1.0j * np.exp(1.0j * phi) * s * a
    return out


def _apply_law_eberly_sqr(
    state: np.ndarray,
    *,
    theta: float,
    phi: float,
    level: int,
) -> np.ndarray:
    """Apply a dense Fock-conditioned qubit rotation.

    Args:
        state: Joint qubit-oscillator state with shape ``(2, n_fock)``.
        theta: Bosonic Qiskit ``cv_sqr`` angle.
        phi: Bosonic Qiskit ``cv_sqr`` phase.
        level: Oscillator Fock level on which the qubit rotation is applied.

    Returns:
        Updated joint state.
    """

    out = np.asarray(state, dtype=complex).copy()
    a = out[_LE_EXCITED, level]
    b = out[_LE_GROUND, level]
    c = np.cos(0.5 * theta)
    s = np.sin(0.5 * theta)
    out[_LE_EXCITED, level] = c * a - 1.0j * np.exp(-1.0j * phi) * s * b
    out[_LE_GROUND, level] = -1.0j * np.exp(1.0j * phi) * s * a + c * b
    return out


def _apply_law_eberly_r(
    state: np.ndarray,
    *,
    theta: float,
    phi: float,
) -> np.ndarray:
    """Apply a dense global auxiliary-qubit rotation.

    This is the circuit-level counterpart of the original Law-Eberly ``C_j``
    pulse. It acts on the auxiliary qubit and as identity on the oscillator.
    """

    out = np.asarray(state, dtype=complex).copy()
    excited = out[_LE_EXCITED, :].copy()
    ground = out[_LE_GROUND, :].copy()
    c = np.cos(0.5 * theta)
    s = np.sin(0.5 * theta)
    out[_LE_EXCITED, :] = c * excited - 1.0j * np.exp(-1.0j * phi) * s * ground
    out[_LE_GROUND, :] = -1.0j * np.exp(1.0j * phi) * s * excited + c * ground
    return out


def _apply_law_eberly_pulse(state: np.ndarray, pulse: LawEberlyPulse) -> np.ndarray:
    """Apply one Law-Eberly pulse to a dense joint state.

    Args:
        state: Joint qubit-oscillator state with shape ``(2, n_fock)``.
        pulse: Pulse descriptor to apply.

    Returns:
        Updated joint state.

    Raises:
        ValueError: If the pulse kind is unsupported.
    """

    if pulse.kind == "jc":
        return _apply_law_eberly_jc(state, theta=pulse.theta, phi=pulse.phi)
    if pulse.kind == "r":
        return _apply_law_eberly_r(state, theta=pulse.theta, phi=pulse.phi)
    if pulse.kind == "sqr":
        return _apply_law_eberly_sqr(
            state,
            theta=pulse.theta,
            phi=pulse.phi,
            level=pulse.level,
        )
    raise ValueError(f"Unknown Law-Eberly pulse kind '{pulse.kind}'.")


def _adjoint_law_eberly_pulse(pulse: LawEberlyPulse) -> LawEberlyPulse:
    """Return the adjoint pulse in Bosonic Qiskit's parameter convention.

    Args:
        pulse: Pulse descriptor to invert.

    Returns:
        Pulse descriptor for the adjoint operation.
    """

    return LawEberlyPulse(
        kind=pulse.kind,
        level=pulse.level,
        theta=-float(pulse.theta),
        phi=float(pulse.phi),
    )


def _law_eberly_synthesis(
    target_state: np.ndarray,
    *,
    n_fock: int,
    rotation_kind: str,
    method: str,
) -> OraclePreparation:
    """Compile a target oscillator state into a Law-Eberly-style pulse sequence.

    ``rotation_kind`` selects the second primitive in the inverse recursion:
    ``"r"`` for the original Law-Eberly global auxiliary-qubit rotation, or
    ``"sqr"`` for the number-selective Bosonic Qiskit variant.

    Args:
        target_state: Desired oscillator state in the truncated Fock basis.
        n_fock: Oscillator truncation dimension.
        rotation_kind: Either ``"r"`` or ``"sqr"``.
        method: Public state-preparation method name to store in the result.

    Returns:
        ``OraclePreparation`` with circuit-level Law-Eberly pulses.

    Raises:
        ValueError: If ``target_state`` has zero norm.
    """

    if rotation_kind not in {"r", "sqr"}:
        raise ValueError(f"Unknown Law-Eberly rotation kind '{rotation_kind}'.")

    target = padded_seed_state(target_state, n_fock)

    n_active = 0
    for idx in range(n_fock - 1, -1, -1):
        if abs(target[idx]) > 1e-15:
            n_active = idx + 1
            break
    if n_active == 0:
        raise ValueError("Target state has zero norm.")

    state = np.zeros((2, n_fock), dtype=complex)
    state[_LE_GROUND, :] = target
    inverse_pulses: List[LawEberlyPulse] = []

    for n in range(n_active - 1, 0, -1):
        excited_lower = state[_LE_EXCITED, n - 1]
        ground_upper = state[_LE_GROUND, n]
        if abs(ground_upper) > 1e-14:
            if abs(excited_lower) < 1e-14:
                angle = 0.5 * np.pi
                phi = 0.0
            else:
                angle = float(np.arctan2(abs(ground_upper), abs(excited_lower)))
                phi = float(np.angle(ground_upper) - np.angle(excited_lower) - 0.5 * np.pi)
            pulse = LawEberlyPulse(
                kind="jc",
                level=n,
                theta=float(angle / np.sqrt(n)),
                phi=phi,
            )
            state = _apply_law_eberly_pulse(state, pulse)
            inverse_pulses.append(pulse)

        excited = state[_LE_EXCITED, n - 1]
        ground = state[_LE_GROUND, n - 1]
        if abs(excited) > 1e-14:
            if abs(ground) < 1e-14:
                angle = 0.5 * np.pi
                phi = 0.0
            else:
                angle = float(np.arctan2(abs(excited), abs(ground)))
                phi = float(-np.angle(excited) + np.angle(ground) + 0.5 * np.pi)
            pulse = LawEberlyPulse(
                kind=rotation_kind,
                level=n - 1,
                theta=float(2.0 * angle),
                phi=phi,
            )
            state = _apply_law_eberly_pulse(state, pulse)
            inverse_pulses.append(pulse)

    prep_pulses = tuple(_adjoint_law_eberly_pulse(pulse) for pulse in reversed(inverse_pulses))
    prepared_joint = np.zeros((2, n_fock), dtype=complex)
    prepared_joint[_LE_GROUND, 0] = 1.0
    for pulse in prep_pulses:
        prepared_joint = _apply_law_eberly_pulse(prepared_joint, pulse)

    ground_component = np.asarray(prepared_joint[_LE_GROUND, :], dtype=complex)
    aux_ground_probability = float(np.linalg.norm(ground_component) ** 2)
    prepared = normalize_vector(ground_component)
    fidelity = abs(np.vdot(target, prepared)) ** 2
    return OraclePreparation(
        method=method,
        target_state=target,
        prepared_state=prepared,
        apply_mode="law_eberly_pulses",
        oracle_fidelity=float(fidelity),
        metadata={
            "n_active_fock_levels": int(n_active),
            "n_jc_pulses": int(sum(pulse.kind == "jc" for pulse in prep_pulses)),
            "n_sqr_pulses": int(sum(pulse.kind == "sqr" for pulse in prep_pulses)),
            "n_qubit_rotations": int(sum(pulse.kind == "r" for pulse in prep_pulses)),
            "n_law_eberly_pulses": int(len(prep_pulses)),
            "law_eberly_variant": "original" if rotation_kind == "r" else "selective",
            "le_aux_ground_probability": aux_ground_probability,
            "le_residual_excited_norm": float(np.linalg.norm(prepared_joint[_LE_EXCITED, :])),
            "le_residual_ground_tail_norm": float(np.linalg.norm(state[_LE_GROUND, 1:])),
        },
        law_eberly_pulses=prep_pulses,
    )


def law_eberly_synthesis(target_state: np.ndarray, *, n_fock: int) -> OraclePreparation:
    """Compile a target oscillator state with the original Law-Eberly pulses.

    The second primitive is the global auxiliary-qubit ``C_j`` rotation from the
    Law-Eberly construction, emitted as an ordinary Qiskit qubit rotation.
    """

    return _law_eberly_synthesis(
        target_state,
        n_fock=n_fock,
        rotation_kind="r",
        method="law_eberly",
    )


def law_eberly_selective_synthesis(target_state: np.ndarray, *, n_fock: int) -> OraclePreparation:
    """Compile the selective-rotation Law-Eberly variant used by ``cv_sqr``."""

    return _law_eberly_synthesis(
        target_state,
        n_fock=n_fock,
        rotation_kind="sqr",
        method="law_eberly_selective",
    )


def prepare_cv_oracle(
    kernel: KernelSpec,
    prep: StatePrepSpec,
    *,
    coeffs: Optional[np.ndarray] = None,
    warm_start: Optional[OraclePreparation] = None,
) -> OraclePreparation:
    """Construct the requested CV state-preparation oracle.

    Args:
        kernel: Kernel hyperparameters and truncation.
        prep: State-preparation specification.
        coeffs: Optional precomputed target coefficient vector.
        warm_start: Optional prior SNAP+D oracle used to initialize a new
            SNAP+D optimization when depth continuation is desired.

    Returns:
        ``OraclePreparation`` for the requested method.
    """

    if coeffs is None:
        coeffs = compute_lchs_coefficients(kernel)
    target = padded_seed_state(coeffs, kernel.n_fock)

    if prep.method == "injection":
        oracle = OraclePreparation(
            method="injection",
            target_state=target,
            prepared_state=target.copy(),
            apply_mode="direct_injection",
            oracle_fidelity=1.0,
            metadata={"stateprep_seed_fidelity": 1.0},
        )
    elif prep.method == "law_eberly":
        oracle = law_eberly_synthesis(target, n_fock=kernel.n_fock)
    elif prep.method == "law_eberly_selective":
        oracle = law_eberly_selective_synthesis(target, n_fock=kernel.n_fock)
    elif prep.snap_parameter_payload is not None:
        oracle = oracle_from_snap_parameter_payload(
            target,
            n_fock=kernel.n_fock,
            payload=prep.snap_parameter_payload,
        )
    else:
        n_snap = min(kernel.n_coeff, kernel.n_fock)
        oracle = optimize_snap_d(
            target,
            n_fock=kernel.n_fock,
            depth=prep.snap_depth,
            n_snap=n_snap,
            n_restarts=prep.snap_restarts,
            maxiter=prep.snap_maxiter,
            initial_guess=snap_d_initial_guess_from_oracle(
                warm_start,
                depth=prep.snap_depth,
                n_snap=n_snap,
            ),
            random_seed=prep.snap_seed,
        )
    return align_oracle_global_phase(oracle)


def align_oracle_global_phase(oracle: OraclePreparation) -> OraclePreparation:
    """Fix the free global phase so that <target|prepared> is real positive.

    A synthesized preparation unitary is only defined up to a global phase,
    but the target-scale map error uses alpha_{N,r} from the target state, so
    prepared states must share the target's phase gauge. The compensating
    angle is recorded so the circuit builder can apply the same convention.
    """

    overlap = np.vdot(oracle.target_state, oracle.prepared_state)
    oracle.metadata["statevector_gauge"] = "raw_circuit_output"
    if abs(overlap) < 1e-12:
        oracle.metadata.setdefault("global_phase_alignment", 0.0)
        return oracle
    phase = overlap / abs(overlap)
    angle = float(-np.angle(phase))
    oracle.prepared_state = oracle.prepared_state * np.conj(phase)
    oracle.metadata["global_phase_alignment"] = float(
        oracle.metadata.get("global_phase_alignment", 0.0)
    ) + angle
    return oracle


def _qiskit_to_physics_permutation(n_dv_qubits: int) -> np.ndarray:
    """Return the permutation needed to recover physics ordering from Qiskit."""

    return physics_to_qiskit_permutation(n_dv_qubits)


def _conditioned_qubit_indices(
    *,
    n_dv_qubits: int,
    n_prefix_qubits: int,
    prefix_qubit_values: Sequence[int],
) -> np.ndarray:
    """Return Qiskit-basis qubit indices for a fixed prefix register.

    Args:
        n_dv_qubits: Number of DV output qubits.
        n_prefix_qubits: Number of auxiliary qubits before the DV register.
        prefix_qubit_values: Fixed computational-basis values for the prefix
            qubits in circuit order.

    Returns:
        Qiskit-basis joint-qubit indices ordered by the DV register's native
        Qiskit little-endian index.

    Raises:
        ValueError: If the prefix specification is inconsistent.
    """

    values = tuple(int(value) for value in prefix_qubit_values)
    if len(values) != n_prefix_qubits:
        raise ValueError("prefix_qubit_values length must match n_prefix_qubits.")
    if any(value not in (0, 1) for value in values):
        raise ValueError("prefix_qubit_values entries must be 0 or 1.")

    prefix_index = sum(value << idx for idx, value in enumerate(values))
    dv_indices = np.arange(2**n_dv_qubits, dtype=int)
    return prefix_index + (dv_indices << n_prefix_qubits)


def detect_statevector_layout(max_fock_level: int, n_dv_qubits: int) -> str:
    """Detect bosonic-qiskit's flattened joint-state ordering.

    The backend may flatten the joint CV-DV state either with oscillator index
    running slowest or fastest. This helper injects a simple probe state and
    infers whether the layout is oscillator-major or qubit-major.

    Args:
        max_fock_level: Oscillator truncation dimension.
        n_dv_qubits: Number of DV qubits.

    Returns:
        Either ``"fock_major"`` or ``"qubit_major"``.
    """

    try:
        from bosonic_qiskit import CVCircuit, QumodeRegister
        from bosonic_qiskit import util as cv_util
        from qiskit import QuantumRegister
    except ImportError as exc:
        raise ImportError(
            "detect_statevector_layout requires bosonic_qiskit and qiskit."
        ) from exc

    qmr = QumodeRegister(
        num_qumodes=1,
        num_qubits_per_qumode=int(np.log2(max_fock_level)),
    )
    qbr = QuantumRegister(n_dv_qubits, "q")
    qc = CVCircuit(qbr, qmr)

    probe = np.zeros(max_fock_level, dtype=complex)
    probe[1] = 1.0
    qc.cv_initialize(probe, qmr[0])

    state, _, _ = cv_util.simulate(
        qc,
        shots=1,
        return_fockcounts=False,
        add_save_statevector=True,
    )
    vec = np.asarray(state.data, dtype=complex)
    nonzero = np.where(np.abs(vec) > 1e-12)[0]
    if nonzero.size != 1:
        raise RuntimeError(f"Unexpected layout probe state: found {nonzero.size} nonzero entries.")
    idx = int(nonzero[0])
    dv_dim = 2**n_dv_qubits

    if idx == dv_dim:
        return "fock_major"
    if idx == 1:
        return "qubit_major"
    raise RuntimeError(f"Unrecognized statevector layout probe index {idx}.")


def extract_direct_dv_slice(
    statevector: np.ndarray,
    *,
    layout: str,
    max_fock_level: int,
    n_dv_qubits: int,
    fock_index: int = 0,
    n_prefix_qubits: int = 0,
    prefix_qubit_values: Sequence[int] = (),
) -> np.ndarray:
    """Extract a fixed-Fock DV slice from a pure joint statevector.

    Args:
        statevector: Flattened joint CV-DV statevector.
        layout: Joint-state layout reported by ``detect_statevector_layout``.
        max_fock_level: Oscillator truncation dimension.
        n_dv_qubits: Number of DV qubits.
        fock_index: Oscillator basis index to slice out.
        n_prefix_qubits: Number of auxiliary qubits before the DV register.
        prefix_qubit_values: Fixed computational-basis values for the prefix
            qubits. The Law-Eberly auxiliary ground state is Qiskit ``|1>``.

    Returns:
        DV slice in physics qubit ordering.
    """

    dv_dim = 2**n_dv_qubits
    qubit_indices = _conditioned_qubit_indices(
        n_dv_qubits=n_dv_qubits,
        n_prefix_qubits=n_prefix_qubits,
        prefix_qubit_values=prefix_qubit_values,
    )
    vec = np.asarray(statevector, dtype=complex).reshape(-1)
    if layout == "fock_major":
        total_qubit_dim = 2 ** (n_dv_qubits + n_prefix_qubits)
        start = fock_index * total_qubit_dim
        dv_qiskit = vec[start + qubit_indices]
    elif layout == "qubit_major":
        dv_qiskit = vec[qubit_indices * max_fock_level + fock_index]
    else:
        raise ValueError(f"Unknown layout '{layout}'.")
    if dv_qiskit.size != dv_dim:
        raise RuntimeError("Failed to extract the requested DV slice.")
    return dv_qiskit[_qiskit_to_physics_permutation(n_dv_qubits)]


def extract_fock_zero_dv_statevector(
    statevector: np.ndarray,
    *,
    layout: str,
    max_fock_level: int,
    n_dv_qubits: int,
    n_prefix_qubits: int = 0,
    prefix_qubit_values: Sequence[int] = (),
) -> np.ndarray:
    """Extract the oscillator ``|0>`` DV slice from a pure statevector."""

    return extract_direct_dv_slice(
        statevector,
        layout=layout,
        max_fock_level=max_fock_level,
        n_dv_qubits=n_dv_qubits,
        fock_index=0,
        n_prefix_qubits=n_prefix_qubits,
        prefix_qubit_values=prefix_qubit_values,
    )


def extract_fock_zero_dv_density_matrix(
    density_matrix: np.ndarray,
    *,
    layout: str,
    max_fock_level: int,
    n_dv_qubits: int,
    n_prefix_qubits: int = 0,
    prefix_qubit_values: Sequence[int] = (),
) -> Tuple[np.ndarray, float]:
    """Extract the oscillator ``|0><0|`` DV block from a density matrix.

    Args:
        density_matrix: Flattened joint CV-DV density matrix.
        layout: Joint-state layout reported by ``detect_statevector_layout``.
        max_fock_level: Oscillator truncation dimension.
        n_dv_qubits: Number of DV qubits.
        n_prefix_qubits: Number of auxiliary qubits before the DV register.
        prefix_qubit_values: Fixed computational-basis values for the prefix
            qubits. The Law-Eberly auxiliary ground state is Qiskit ``|1>``.

    Returns:
        Tuple ``(rho_dv, post_prob)`` containing the DV block in physics
        ordering and its trace.
    """

    dv_dim = 2**n_dv_qubits
    qubit_indices = _conditioned_qubit_indices(
        n_dv_qubits=n_dv_qubits,
        n_prefix_qubits=n_prefix_qubits,
        prefix_qubit_values=prefix_qubit_values,
    )
    rho = np.asarray(density_matrix, dtype=complex)

    if layout == "fock_major":
        indices = qubit_indices
    elif layout == "qubit_major":
        indices = qubit_indices * max_fock_level
    else:
        raise ValueError(f"Unknown layout '{layout}'.")

    rho_qiskit = rho[np.ix_(indices, indices)]
    perm = _qiskit_to_physics_permutation(n_dv_qubits)
    rho_physics = rho_qiskit[np.ix_(perm, perm)]
    post_prob = float(np.real(np.trace(rho_physics)))
    return rho_physics, post_prob


def principal_statevector_from_density_matrix(rho: np.ndarray) -> np.ndarray:
    """Return the dominant pure-state component of a density matrix."""

    vals, vecs = np.linalg.eigh(np.asarray(rho, dtype=complex))
    idx = int(np.argmax(np.real(vals)))
    eigval = max(float(np.real(vals[idx])), 0.0)
    if eigval < 1e-15:
        return np.zeros(rho.shape[0], dtype=complex)
    principal = vecs[:, idx] * np.sqrt(eigval)
    return np.asarray(principal, dtype=complex)


def fidelity_density_matrix_vs_pure(rho: np.ndarray, pure_state: np.ndarray) -> float:
    """Return fidelity between a density matrix and a pure target state."""

    pure = normalize_vector(pure_state)
    rho_arr = np.asarray(rho, dtype=complex)
    trace = float(np.real(np.trace(rho_arr)))
    if trace < 1e-15:
        return 0.0
    return float(np.real(np.conj(pure) @ rho_arr @ pure) / trace)


def postselect_cv_output(
    raw_state: np.ndarray,
    *,
    readout_mode: str,
    layout: str,
    max_fock_level: int,
    n_dv_qubits: int,
    n_prefix_qubits: int = 0,
    prefix_qubit_values: Sequence[int] = (),
) -> Dict[str, Any]:
    """Apply the requested CV readout rule to simulator output.

    Supported modes are:

    - ``postselect_statevector`` for pure-state ``|0>`` postselection,
    - ``direct_statevector`` for raw direct slicing of a fixed Fock level,
    - ``postselect_density_matrix`` for density-matrix ``|0><0|`` extraction.

    Args:
        raw_state: Statevector or density matrix returned by the backend.
        readout_mode: Clean-stack readout mode.
        layout: Joint-state layout reported by ``detect_statevector_layout``.
        max_fock_level: Oscillator truncation dimension.
        n_dv_qubits: Number of DV qubits.
        n_prefix_qubits: Number of auxiliary qubits before the DV register.
        prefix_qubit_values: Fixed computational-basis values for prefix
            qubits that should be projected alongside the oscillator ``|0>``.

    Returns:
        Dictionary containing the observed DV vector, postselection
        probability, and the postselected density matrix when available.

    Raises:
        ValueError: If ``readout_mode`` is unsupported.
    """

    if readout_mode == "postselect_statevector":
        observed = extract_fock_zero_dv_statevector(
            raw_state,
            layout=layout,
            max_fock_level=max_fock_level,
            n_dv_qubits=n_dv_qubits,
            n_prefix_qubits=n_prefix_qubits,
            prefix_qubit_values=prefix_qubit_values,
        )
        return {
            "observed_vector": observed,
            "postselection_probability": float(np.linalg.norm(observed) ** 2),
        }

    if readout_mode == "direct_statevector":
        observed = extract_direct_dv_slice(
            raw_state,
            layout=layout,
            max_fock_level=max_fock_level,
            n_dv_qubits=n_dv_qubits,
            fock_index=0,
            n_prefix_qubits=n_prefix_qubits,
            prefix_qubit_values=prefix_qubit_values,
        )
        return {
            "observed_vector": observed,
            "postselection_probability": float(np.linalg.norm(observed) ** 2),
        }

    if readout_mode == "postselect_density_matrix":
        rho_post, post_prob = extract_fock_zero_dv_density_matrix(
            raw_state,
            layout=layout,
            max_fock_level=max_fock_level,
            n_dv_qubits=n_dv_qubits,
            n_prefix_qubits=n_prefix_qubits,
            prefix_qubit_values=prefix_qubit_values,
        )
        principal = principal_statevector_from_density_matrix(rho_post)
        return {
            "observed_vector": principal,
            "postselection_probability": float(post_prob),
            "postselected_density_matrix": rho_post,
        }

    raise ValueError(f"Unknown readout_mode '{readout_mode}'.")
