#!/usr/bin/env python3
r"""
Gate-level CV state preparation for the formed CV-DV LCHS pipeline.

Two methods are provided:

1. SNAP+Displacement (optimization-based):
   - Heeres et al., PRL 115, 137002 (2015): SNAP gate concept
   - Liu et al., arXiv:2407.10381 Sec. VI.E: numerical compilation via
     alternating SNAP+D layers with gradient-based optimization
   - Circuit: U = D(α_L) SNAP(θ_L) ... D(α_1) SNAP(θ_1)
   - Parameters optimized to maximize |⟨ψ_target|U|0⟩|²

2. Givens rotation decomposition (deterministic):
   - Prepares |ψ⟩ = Σ C_n |n⟩ from |0⟩ using N-1 adjacent Fock-level rotations
   - Each G(n, n+1; θ, φ) is a Jaynes-Cummings interaction pulse on the qumode
   - Rotation angles computed analytically (no optimization needed)
   - Exact fidelity = 1 up to numerical precision

Usage:
  # SNAP+D benchmark
  python formed_stateprep.py --method snap_d --r-target 0.85 \
      --r-prime 0.003 --beta 0.35 --n-dim 16 --max-fock 64 --depth 12

  # Givens rotation
  python formed_stateprep.py --method givens --r-target 0.85 \
      --r-prime 0.003 --beta 0.35 --n-dim 16 --max-fock 64

  # As a module
  from formed_stateprep import (
      optimize_snap_d_params, apply_snap_d_circuit,
      givens_decomposition, apply_givens_circuit,
  )
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Fock-basis matrix elements
# ---------------------------------------------------------------------------

def _annihilation_op(n_fock: int) -> np.ndarray:
    """Annihilation operator a in the Fock basis."""
    a = np.zeros((n_fock, n_fock), dtype=complex)
    for n in range(1, n_fock):
        a[n - 1, n] = np.sqrt(n)
    return a


def displacement_matrix(alpha: complex, n_fock: int) -> np.ndarray:
    """Compute D(α) = exp(α a† - α* a) in the Fock basis via matrix exponentiation."""
    a = _annihilation_op(n_fock)
    a_dag = a.T.conj()
    return expm(alpha * a_dag - np.conj(alpha) * a)


# ---------------------------------------------------------------------------
# Circuit simulation in Fock basis
# ---------------------------------------------------------------------------

def simulate_snap_d_circuit(
    params: np.ndarray,
    n_fock: int,
    depth: int,
    n_snap: int = 0,
) -> np.ndarray:
    """Simulate SNAP+D circuit on vacuum and return final state.

    Args:
        params: Optimization parameters.
        n_fock: Total Fock space dimension for displacement matrices.
        depth: Number of SNAP+D layers.
        n_snap: Number of SNAP phases per layer (default: n_fock).
            Using n_snap < n_fock reduces the parameter count while
            keeping displacement matrices at full dimension for accuracy.

    params layout: [θ_1(n_snap), Re(α_1), Im(α_1), θ_2(n_snap), ...]
    """
    if n_snap <= 0:
        n_snap = n_fock
    params_per_layer = n_snap + 2
    state = np.zeros(n_fock, dtype=complex)
    state[0] = 1.0  # vacuum

    for layer in range(depth):
        offset = layer * params_per_layer
        thetas = params[offset: offset + n_snap]
        re_alpha = params[offset + n_snap]
        im_alpha = params[offset + n_snap + 1]
        alpha = re_alpha + 1j * im_alpha

        # Apply SNAP (diagonal — element-wise multiply on first n_snap levels)
        state[:n_snap] *= np.exp(1j * thetas)

        # Apply displacement (full n_fock matrix for accuracy)
        D = displacement_matrix(alpha, n_fock)
        state = D @ state

    return state


def fidelity_cost(
    params: np.ndarray,
    target: np.ndarray,
    n_fock: int,
    depth: int,
    n_snap: int = 0,
) -> float:
    """Cost function: 1 - |⟨target|prepared⟩|²."""
    prepared = simulate_snap_d_circuit(params, n_fock, depth, n_snap)
    overlap = np.abs(np.vdot(target, prepared)) ** 2
    return 1.0 - overlap


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

@dataclass
class SNAPDResult:
    """Result of SNAP+D optimization."""
    fidelity: float
    depth: int
    n_fock: int
    params: np.ndarray
    thetas_per_layer: List[np.ndarray]
    alphas_per_layer: List[complex]
    target_coeffs: np.ndarray
    n_iterations: int
    optimization_time: float

    def to_dict(self) -> Dict:
        return {
            "fidelity": self.fidelity,
            "infidelity": 1.0 - self.fidelity,
            "depth": self.depth,
            "n_fock": self.n_fock,
            "n_iterations": self.n_iterations,
            "optimization_time_s": self.optimization_time,
            "alphas_per_layer": [
                {"re": float(a.real), "im": float(a.imag)}
                for a in self.alphas_per_layer
            ],
            "n_snap_phases_per_layer": len(self.thetas_per_layer[0])
            if self.thetas_per_layer
            else 0,
        }


def optimize_snap_d_params(
    target_coeffs: np.ndarray,
    n_fock: int,
    depth: int,
    *,
    n_snap: int = 0,
    n_restarts: int = 3,
    maxiter: int = 2000,
    verbose: bool = False,
) -> SNAPDResult:
    """Optimize SNAP+D circuit parameters to prepare target Fock superposition.

    Args:
        target_coeffs: Target state coefficients in Fock basis (will be normalized).
        n_fock: Fock space truncation dimension for displacement matrices.
        depth: Number of SNAP+D layers.
        n_snap: Number of SNAP phases per layer (default: len(target_coeffs)).
            Using n_snap < n_fock reduces parameter count while keeping
            displacement at full dimension. Should be >= number of active Fock levels.
        n_restarts: Number of random restarts for optimization.
        maxiter: Maximum iterations per optimization run.
        verbose: Print progress.

    Returns:
        SNAPDResult with optimized parameters and fidelity.
    """
    # Normalize target and pad to n_fock
    target = np.zeros(n_fock, dtype=complex)
    target[: len(target_coeffs)] = target_coeffs
    norm = np.linalg.norm(target)
    if norm < 1e-15:
        raise ValueError("Target state has zero norm.")
    target /= norm

    # Default n_snap: use the number of target coefficients (not full n_fock)
    if n_snap <= 0:
        n_snap = len(target_coeffs)
    n_snap = min(n_snap, n_fock)

    params_per_layer = n_snap + 2
    n_params = depth * params_per_layer

    best_result = None
    best_fidelity = -1.0
    total_iters = 0
    t_start = time.time()

    for restart in range(n_restarts):
        # Random initialization: small SNAP phases, small displacements
        x0 = np.zeros(n_params)
        for layer in range(depth):
            offset = layer * params_per_layer
            x0[offset: offset + n_snap] = np.random.uniform(
                -np.pi, np.pi, n_snap
            )
            x0[offset + n_snap] = np.random.uniform(-0.5, 0.5)
            x0[offset + n_snap + 1] = np.random.uniform(-0.5, 0.5)

        result = minimize(
            fidelity_cost,
            x0,
            args=(target, n_fock, depth, n_snap),
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-10},
        )

        fid = 1.0 - result.fun
        total_iters += result.nit

        if verbose:
            print(
                f"  Restart {restart + 1}/{n_restarts}: "
                f"fidelity = {fid:.8f}, iters = {result.nit}"
            )

        if fid > best_fidelity:
            best_fidelity = fid
            best_result = result

    t_elapsed = time.time() - t_start
    params = best_result.x

    # Extract per-layer parameters
    thetas_list = []
    alphas_list = []
    for layer in range(depth):
        offset = layer * params_per_layer
        thetas = params[offset: offset + n_snap].copy()
        re_a = params[offset + n_snap]
        im_a = params[offset + n_snap + 1]
        thetas_list.append(thetas)
        alphas_list.append(complex(re_a, im_a))

    return SNAPDResult(
        fidelity=best_fidelity,
        depth=depth,
        n_fock=n_fock,
        params=params,
        thetas_per_layer=thetas_list,
        alphas_per_layer=alphas_list,
        target_coeffs=target_coeffs.copy(),
        n_iterations=total_iters,
        optimization_time=t_elapsed,
    )


# ---------------------------------------------------------------------------
# bosonic_qiskit circuit construction
# ---------------------------------------------------------------------------

def apply_snap_d_circuit(
    qc,
    mode: Sequence,
    snap_d_result: SNAPDResult,
) -> None:
    """Apply optimized SNAP+D layers to a bosonic_qiskit CVCircuit.

    Args:
        qc: CVCircuit instance.
        mode: Qumode register (qmr[0]).
        snap_d_result: Result from optimize_snap_d_params.
    """
    for layer in range(snap_d_result.depth):
        thetas = snap_d_result.thetas_per_layer[layer]
        alpha = snap_d_result.alphas_per_layer[layer]

        # Apply SNAP: one phase per Fock level
        # (cv_snap with scalar n works; list version has a qargs bug)
        for n_idx in range(len(thetas)):
            theta_val = float(thetas[n_idx])
            if abs(theta_val) > 1e-12:  # skip near-zero phases
                qc.cv_snap(theta_val, n_idx, mode)

        # Apply displacement
        qc.cv_d(alpha, mode)


def stateprep_resource_stats(snap_d_result: SNAPDResult) -> Dict:
    """Gate counts for the SNAP+D state preparation circuit."""
    d = snap_d_result.depth
    return {
        "state_prep_method": "gate_based_snap_d",
        "snap_d_depth": d,
        "n_snap_gates": d,
        "n_displacement_gates": d,
        "total_stateprep_gates": 2 * d,
        "optimization_fidelity": snap_d_result.fidelity,
        "optimization_infidelity": 1.0 - snap_d_result.fidelity,
    }


# ---------------------------------------------------------------------------
# Givens rotation decomposition (deterministic)
# ---------------------------------------------------------------------------

@dataclass
class GivensResult:
    """Result of Givens rotation decomposition for Fock state preparation."""
    fidelity: float
    n_active: int
    n_fock: int
    rotations: List[Dict]  # [{n: int, theta: float, phi: float}, ...]
    target_coeffs: np.ndarray
    prepared_state: np.ndarray

    def to_dict(self) -> Dict:
        return {
            "fidelity": self.fidelity,
            "infidelity": 1.0 - self.fidelity,
            "n_active_levels": self.n_active,
            "n_fock": self.n_fock,
            "n_jc_pulses": len(self.rotations),
            "rotations": self.rotations,
        }


def _givens_rotation_matrix(n: int, theta: float, phi: float, dim: int) -> np.ndarray:
    """Construct Givens rotation G(n, n+1; θ, φ) acting on Fock space.

    G mixes |n⟩ and |n+1⟩:
      |n⟩   →  cos(θ)|n⟩   + e^{iφ} sin(θ)|n+1⟩
      |n+1⟩ → -e^{-iφ} sin(θ)|n⟩ + cos(θ)|n+1⟩

    All other levels are left unchanged (identity).
    """
    G = np.eye(dim, dtype=complex)
    c, s = np.cos(theta), np.sin(theta)
    G[n, n] = c
    G[n, n + 1] = np.exp(1j * phi) * s
    G[n + 1, n] = -np.exp(-1j * phi) * s
    G[n + 1, n + 1] = c
    return G


def givens_decomposition(
    target_coeffs: np.ndarray,
    n_fock: int,
    *,
    verbose: bool = False,
) -> GivensResult:
    """Decompose target Fock state into a sequence of adjacent Givens rotations.

    Computes rotation angles analytically so that
      G(0,1) G(1,2) ... G(N-2,N-1) |0⟩ = |ψ_target⟩

    The algorithm works backwards: starting from the target state, apply
    inverse Givens rotations G†(N-2,N-1), G†(N-3,N-2), ..., G†(0,1) to
    reduce the state to |0⟩. The preparation circuit is the reverse sequence.

    Each Givens rotation G(n, n+1) corresponds to a Jaynes-Cummings (JC)
    interaction pulse on the qumode, making this physically realizable.

    Args:
        target_coeffs: Target state coefficients C_n in Fock basis.
        n_fock: Fock space truncation dimension.
        verbose: Print decomposition details.

    Returns:
        GivensResult with rotation parameters and verification fidelity.
    """
    # Normalize and pad target to n_fock
    target = np.zeros(n_fock, dtype=complex)
    n_coeffs = min(len(target_coeffs), n_fock)
    target[:n_coeffs] = target_coeffs[:n_coeffs]
    norm = np.linalg.norm(target)
    if norm < 1e-15:
        raise ValueError("Target state has zero norm.")
    target /= norm

    # Find the highest active Fock level
    n_active = n_fock
    for k in range(n_fock - 1, -1, -1):
        if abs(target[k]) > 1e-15:
            n_active = k + 1
            break

    if verbose:
        print(f"  Givens decomposition: {n_active} active Fock levels")

    # Work backwards: compute rotations that reduce target → |0⟩
    # Then the preparation circuit applies them in reverse order.
    state = target.copy()
    inverse_rotations = []  # stored in reduction order

    for k in range(n_active - 1, 0, -1):
        # Zero out state[k] by rotating (k-1, k)
        a = state[k - 1]
        b = state[k]
        r = np.sqrt(abs(a) ** 2 + abs(b) ** 2)

        if r < 1e-15:
            # Both components are zero, skip
            inverse_rotations.append({"n": k - 1, "theta": 0.0, "phi": 0.0})
            continue

        # We want G†(k-1,k) to map (a, b) → (r, 0)
        # G†: |k-1⟩ →  cos(θ)|k-1⟩ - e^{iφ} sin(θ)|k⟩
        #      |k⟩   →  e^{-iφ} sin(θ)|k-1⟩ + cos(θ)|k⟩
        #
        # For the (k-1, k) subspace:
        #   [cos(θ), -e^{iφ}sin(θ)]   [a]   [r]
        #   [e^{-iφ}sin(θ), cos(θ) ] · [b] = [0]
        #
        # From second row: e^{-iφ}sin(θ)·a + cos(θ)·b = 0
        # → tan(θ) = -b·e^{iφ}/a  ... more directly:
        #
        # cos(θ) = |a|/r, sin(θ) = |b|/r
        # Phase: φ = arg(a) - arg(b) + π  (to get the cancellation right)

        theta = np.arctan2(abs(b), abs(a))
        phi = np.angle(a) - np.angle(b) + np.pi

        # Apply G†(k-1, k) to state
        c, s = np.cos(theta), np.sin(theta)
        new_km1 = c * state[k - 1] - np.exp(1j * phi) * s * state[k]
        new_k = np.exp(-1j * phi) * s * state[k - 1] + c * state[k]
        state[k - 1] = new_km1
        state[k] = new_k

        inverse_rotations.append({
            "n": k - 1,
            "theta": float(theta),
            "phi": float(phi),
        })

        if verbose and k >= n_active - 3:
            print(f"    G†({k-1},{k}): θ={theta:.6f}, φ={phi:.6f}, "
                  f"|state[{k}]|={abs(state[k]):.2e}")

    # After all reductions, state should be ~ e^{iδ}|0⟩
    global_phase = np.angle(state[0])
    if verbose:
        print(f"  Residual: |state[0]|={abs(state[0]):.8f}, "
              f"global phase={global_phase:.6f}")
        tail_norm = np.linalg.norm(state[1:])
        print(f"  Tail norm (should be ~0): {tail_norm:.2e}")

    # Preparation circuit = reverse order of inverse rotations,
    # with each rotation using the FORWARD G (not G†).
    # Forward G(n, n+1; θ, φ) has the same θ but we need to
    # conjugate the rotation direction.
    #
    # G† used θ_inv, φ_inv to reduce.
    # G_prep uses the same θ, φ but applied in reverse order.
    # Since G†G = I, applying G with same params undoes G†.
    prep_rotations = list(reversed(inverse_rotations))

    # Also need to account for the global phase on |0⟩
    # by absorbing it into the first rotation's phi.
    if abs(global_phase) > 1e-12 and prep_rotations:
        prep_rotations[0] = dict(prep_rotations[0])
        prep_rotations[0]["phi"] = prep_rotations[0]["phi"] + global_phase

    # Verify by applying preparation circuit to |0⟩
    prepared = np.zeros(n_fock, dtype=complex)
    prepared[0] = 1.0
    for rot in prep_rotations:
        n_level = rot["n"]
        theta = rot["theta"]
        phi = rot["phi"]
        c, s = np.cos(theta), np.sin(theta)
        a = prepared[n_level]
        b = prepared[n_level + 1]
        prepared[n_level] = c * a + np.exp(1j * phi) * s * b
        prepared[n_level + 1] = -np.exp(-1j * phi) * s * a + c * b

    fidelity = abs(np.vdot(target, prepared)) ** 2

    if verbose:
        print(f"  Verification fidelity: {fidelity:.10f}")
        print(f"  Number of JC pulses: {len(prep_rotations)}")

    return GivensResult(
        fidelity=fidelity,
        n_active=n_active,
        n_fock=n_fock,
        rotations=prep_rotations,
        target_coeffs=target_coeffs.copy(),
        prepared_state=prepared,
    )


def apply_givens_circuit(
    qc,
    mode: Sequence,
    givens_result: GivensResult,
) -> None:
    """Apply Givens rotation state preparation to a bosonic_qiskit CVCircuit.

    Each rotation G(n, n+1; θ, φ) is implemented as:
      - cv_snap(φ, n+1, mode)     — phase gate on level n+1
      - cv_bs(θ, mode_a, mode_b)  — beamsplitter-like JC coupling

    Since bosonic_qiskit may not have a direct JC gate, we use
    cv_snap for the phase and a sequence that achieves the Fock-level
    rotation. The exact gate mapping depends on the available gate set.

    For now, we use cv_initialize with the pre-computed prepared state
    as a fallback when native JC gates aren't available, and store the
    rotation parameters for resource counting.

    Args:
        qc: CVCircuit instance.
        mode: Qumode register (qmr[0]).
        givens_result: Result from givens_decomposition.
    """
    # Use cv_initialize with the analytically prepared state.
    # The Givens decomposition gives us exact angles, but bosonic_qiskit
    # doesn't have a native JC pulse gate. We inject the prepared state
    # and report the Givens gate count for resource estimation.
    # `mode` is typically a list of qubits encoding one qumode, so len(mode)
    # is log2(n_fock), not the Fock cutoff itself.
    n_fock = int(givens_result.n_fock)
    inject = np.zeros(n_fock, dtype=complex)
    prepared = np.asarray(givens_result.prepared_state, dtype=complex)
    n_copy = min(len(prepared), n_fock)
    inject[:n_copy] = prepared[:n_copy]
    qc.cv_initialize(inject, mode)


def givens_resource_stats(givens_result: GivensResult) -> Dict:
    """Gate counts for the Givens rotation state preparation circuit."""
    n_rot = len(givens_result.rotations)
    n_nontrivial = sum(1 for r in givens_result.rotations if abs(r["theta"]) > 1e-12)
    return {
        "state_prep_method": "givens_rotation",
        "n_active_fock_levels": givens_result.n_active,
        "n_givens_rotations_total": n_rot,
        "n_givens_rotations_nontrivial": n_nontrivial,
        "n_jc_pulses": n_nontrivial,
        "total_stateprep_gates": n_nontrivial,
        "preparation_fidelity": givens_result.fidelity,
        "preparation_infidelity": 1.0 - givens_result.fidelity,
        "note": (
            "Each Givens rotation G(n,n+1) is a Jaynes-Cummings "
            "interaction pulse between adjacent Fock levels. "
            f"For {givens_result.n_active} active levels, "
            f"{n_nontrivial} non-trivial JC pulses are needed."
        ),
    }


# ---------------------------------------------------------------------------
# Depth sweep benchmark
# ---------------------------------------------------------------------------

def benchmark_stateprep(
    target_coeffs: np.ndarray,
    n_fock: int,
    depths: Sequence[int],
    *,
    n_restarts: int = 3,
    maxiter: int = 2000,
    verbose: bool = True,
) -> List[SNAPDResult]:
    """Sweep circuit depth and report fidelity vs depth."""
    results = []
    for depth in depths:
        if verbose:
            print(f"\n--- Depth {depth} ---")
        res = optimize_snap_d_params(
            target_coeffs,
            n_fock,
            depth,
            n_restarts=n_restarts,
            maxiter=maxiter,
            verbose=verbose,
        )
        if verbose:
            print(
                f"  Best fidelity: {res.fidelity:.8f} "
                f"(infidelity: {1-res.fidelity:.2e}), "
                f"time: {res.optimization_time:.1f}s"
            )
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CV state preparation for LCHS (SNAP+D or Givens)"
    )
    p.add_argument(
        "--method",
        choices=["snap_d", "givens", "both"],
        default="both",
        help="State prep method: snap_d, givens, or both for comparison",
    )
    p.add_argument("--r-target", type=float, default=0.85)
    p.add_argument("--r-prime", type=float, default=0.003)
    p.add_argument("--beta", type=float, default=0.35)
    p.add_argument("--n-dim", type=int, default=16, help="Fock coefficients to compute")
    p.add_argument("--n-quad", type=int, default=300)
    from formed_qutip import COEFF_METHODS

    p.add_argument(
        "--coeff-method",
        choices=list(COEFF_METHODS),
        default="explicit_overlap",
        help="Coefficient backend shared with formed_qutip.py.",
    )
    p.add_argument(
        "--max-fock",
        type=int,
        default=32,
        help="Fock truncation for simulation",
    )
    p.add_argument("--depth", type=int, default=0, help="SNAP+D depth (0 = sweep)")
    p.add_argument(
        "--depths",
        type=str,
        default="2,4,6,8,10,12",
        help="Comma-separated depths for sweep mode",
    )
    p.add_argument("--n-restarts", type=int, default=3)
    p.add_argument("--maxiter", type=int, default=2000)
    p.add_argument("--output-json", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from formed_qutip import lchs_coefficients

    print("=== CV State Preparation for LCHS ===")
    print(f"LCHS params: r={args.r_target}, r'={args.r_prime}, beta={args.beta}")
    print(
        f"Fock coefficients: n_dim={args.n_dim}, n_quad={args.n_quad}, "
        f"coeff_method={args.coeff_method}"
    )
    print(f"Simulation Fock dim: {args.max_fock}")
    print(f"Method: {args.method}")
    print()

    # Compute target coefficients
    coeffs = lchs_coefficients(
        r_target=args.r_target,
        r_prime=args.r_prime,
        beta=args.beta,
        n_coeff=args.n_dim,
        n_quad=args.n_quad,
        method=args.coeff_method,
    )
    coeffs = np.asarray(coeffs, dtype=complex)

    # Show coefficient profile
    print(f"Target state: Σ C_n |n⟩  (n_dim={len(coeffs)})")
    print(f"  Norm: {np.linalg.norm(coeffs):.6f}")
    n_active = int(np.sum(np.abs(coeffs) > 1e-6))
    print(f"  Active levels (|C_n| > 1e-6): {n_active}")
    peak_n = int(np.argmax(np.abs(coeffs)))
    print(f"  Peak at n={peak_n}, |C_{peak_n}|={np.abs(coeffs[peak_n]):.6f}")
    print(f"  First 8 |C_n|: {np.abs(coeffs[:8])}")
    print()

    all_results = {}

    # --- Givens rotation ---
    if args.method in ("givens", "both"):
        print("=" * 50)
        print("=== Givens Rotation Decomposition ===")
        print()
        givens_res = givens_decomposition(
            coeffs, args.max_fock, verbose=True,
        )
        stats = givens_resource_stats(givens_res)
        print(f"\n  Fidelity: {givens_res.fidelity:.10f}")
        print(f"  Infidelity: {1 - givens_res.fidelity:.2e}")
        print(f"\n  Resource stats:")
        for k, v in stats.items():
            if k != "note":
                print(f"    {k}: {v}")
        print(f"    note: {stats['note']}")
        all_results["givens"] = givens_res.to_dict()

    # --- SNAP+D ---
    if args.method in ("snap_d", "both"):
        print()
        print("=" * 50)
        print("=== SNAP+D Optimization ===")
        print()

        if args.depth > 0:
            print(f"Optimizing at depth {args.depth} ...")
            result = optimize_snap_d_params(
                coeffs,
                args.max_fock,
                args.depth,
                n_restarts=args.n_restarts,
                maxiter=args.maxiter,
                verbose=True,
            )
            print(f"\nFinal fidelity: {result.fidelity:.8f}")
            print(f"Infidelity: {1 - result.fidelity:.2e}")
            print(f"Total optimization time: {result.optimization_time:.1f}s")
            stats = stateprep_resource_stats(result)
            print(f"\nGate counts: {stats}")
            snap_results = [result]
        else:
            depths = [int(d) for d in args.depths.split(",")]
            print(f"Sweeping depths: {depths}")
            snap_results = benchmark_stateprep(
                coeffs,
                args.max_fock,
                depths,
                n_restarts=args.n_restarts,
                maxiter=args.maxiter,
                verbose=True,
            )

            print("\n\n=== SNAP+D Depth Sweep Summary ===")
            print(f"{'Depth':>6s} {'Fidelity':>12s} {'Infidelity':>12s} {'Time (s)':>10s}")
            print("-" * 44)
            for r in snap_results:
                print(
                    f"{r.depth:>6d} {r.fidelity:>12.8f} {1-r.fidelity:>12.2e} "
                    f"{r.optimization_time:>10.1f}"
                )

        all_results["snap_d"] = [r.to_dict() for r in snap_results]

    # --- Comparison summary ---
    if args.method == "both":
        print()
        print("=" * 50)
        print("=== Comparison: Givens vs SNAP+D ===")
        g = givens_res
        print(f"  Givens:  fidelity={g.fidelity:.10f}, "
              f"JC pulses={sum(1 for r in g.rotations if abs(r['theta']) > 1e-12)}, "
              f"deterministic")
        for r in snap_results:
            print(f"  SNAP+D (depth={r.depth}): fidelity={r.fidelity:.8f}, "
                  f"gates={2*r.depth}, time={r.optimization_time:.1f}s")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "lchs_params": {
                "r_target": args.r_target,
                "r_prime": args.r_prime,
                "beta": args.beta,
                "n_dim": args.n_dim,
            },
            "max_fock": args.max_fock,
            "results": all_results,
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
