#!/usr/bin/env python3
r"""
Gate-level CV state preparation for the LCHS squeezed-Fock superposition.

Implements the SNAP+Displacement protocol for preparing arbitrary Fock
superpositions |ψ⟩ = Σ C_n |n⟩ from vacuum, based on:

  - Heeres et al., PRL 115, 137002 (2015): SNAP gate concept
  - Liu et al., arXiv:2407.10381 Sec. VI.E: numerical compilation via
    alternating SNAP+D layers with gradient-based optimization

The circuit ansatz is:
  U(params) = D(α_L) SNAP(θ_L) ... D(α_2) SNAP(θ_2) D(α_1) SNAP(θ_1)

Each layer has:
  - 1 SNAP gate: SNAP(θ₀,...,θ_{N-1}) = Σ_n e^{iθ_n} |n⟩⟨n|
  - 1 displacement: D(α) with complex α

Parameters are optimized to maximize |⟨ψ_target|U|0⟩|².

Usage:
  # Standalone benchmark
  python clacod_heat1d_stateprep.py --r-target 0.85 --r-prime 0.003 \
      --beta 0.35 --n-dim 16 --max-fock 64 --depth 12

  # As a module
  from clacod_heat1d_stateprep import optimize_snap_d_params, apply_snap_d_circuit
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
        description="SNAP+D gate-level CV state preparation for LCHS"
    )
    p.add_argument("--r-target", type=float, default=0.85)
    p.add_argument("--r-prime", type=float, default=0.003)
    p.add_argument("--beta", type=float, default=0.35)
    p.add_argument("--n-dim", type=int, default=16, help="Fock coefficients to compute")
    p.add_argument("--n-quad", type=int, default=300)
    from clacod_heat1d_qutip import COEFF_METHODS

    p.add_argument(
        "--coeff-method",
        choices=list(COEFF_METHODS),
        default="explicit_overlap",
        help="Coefficient backend shared with clacod_heat1d_qutip.py.",
    )
    p.add_argument(
        "--max-fock",
        type=int,
        default=32,
        help="Fock truncation for SNAP+D simulation",
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

    from clacod_heat1d_qutip import lchs_coefficients

    print("=== SNAP+D Gate-Level CV State Preparation ===")
    print(f"LCHS params: r={args.r_target}, r'={args.r_prime}, beta={args.beta}")
    print(
        f"Fock coefficients: n_dim={args.n_dim}, n_quad={args.n_quad}, "
        f"coeff_method={args.coeff_method}"
    )
    print(f"SNAP+D simulation Fock dim: {args.max_fock}")
    print()

    # Compute target coefficients
    coeffs = lchs_coefficients(
        r_target=args.r_target,
        r_prime=args.r_prime,
        beta=args.beta,
        n_fock=args.n_dim,
        n_quad=args.n_quad,
        method=args.coeff_method,
    )
    coeffs = np.asarray(coeffs, dtype=complex)

    # Show coefficient profile
    print(f"Target state: Σ C_n |n⟩  (n_dim={len(coeffs)})")
    print(f"  Norm: {np.linalg.norm(coeffs):.6f}")
    n_active = int(np.sum(np.abs(coeffs) > 1e-6))
    print(f"  Active levels (|C_n| > 1e-6): {n_active}")
    print(f"  First 8 |C_n|: {np.abs(coeffs[:8])}")
    print()

    if args.depth > 0:
        # Single depth run
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
        results = [result]
    else:
        # Depth sweep
        depths = [int(d) for d in args.depths.split(",")]
        print(f"Sweeping depths: {depths}")
        results = benchmark_stateprep(
            coeffs,
            args.max_fock,
            depths,
            n_restarts=args.n_restarts,
            maxiter=args.maxiter,
            verbose=True,
        )

        print("\n\n=== Depth Sweep Summary ===")
        print(f"{'Depth':>6s} {'Fidelity':>12s} {'Infidelity':>12s} {'Time (s)':>10s}")
        print("-" * 44)
        for r in results:
            print(
                f"{r.depth:>6d} {r.fidelity:>12.8f} {1-r.fidelity:>12.2e} "
                f"{r.optimization_time:>10.1f}"
            )

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
            "results": [r.to_dict() for r in results],
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote: {out}")


if __name__ == "__main__":
    main()
