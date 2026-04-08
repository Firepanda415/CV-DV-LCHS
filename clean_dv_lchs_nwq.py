#!/usr/bin/env python3
r"""Compute NWQlib-style DV LCHS quadrature size and optionally build the LCU circuit.

This script uses the prototype implementation in
``/Users/zhen002/GitHub/nwqlib/prototypes/lchs/src`` to evaluate the
quadrature-node count

.. math::

    M = 2 \left\lfloor \frac{K}{h_1} \right\rfloor Q,

for the homogeneous LCHS discretization of

.. math::

    \frac{d}{dt}u(t) = -A u(t),

with ``A`` taken to be the 1D heat operator on ``2**n`` grid points:

.. math::

    A = \frac{\alpha}{h^2}\,\mathcal{L}_{\mathrm{bc}},

where ``bc`` is the chosen one-dimensional heat boundary condition.

Unlike the simpler ancilla-grid implementation in the sibling DV repository,
the NWQlib prototype does *not* define the number of LCU branches from a fixed
ancilla width.  Instead:

1. it derives ``h1`` from :math:`\|L\|_2`,
2. truncates the infinite integral to ``[-K, K]``,
3. chooses a Gauss-Legendre order ``Q`` for each interval of width ``h1``,
4. and finally sets ``M = 2 (K/h1) Q``.

Only after this mathematical ``M`` is known does the code pad to
``ceil(log2(M))`` control qubits for the oracle circuits.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, qpy


DEFAULT_NWQ_LCHS_SRC = "/Users/zhen002/GitHub/nwqlib/prototypes/lchs/src"


@dataclass
class NwqDvLchsSummary:
    """Summary of the NWQlib DV LCHS discretization and optional circuit.

    Attributes:
        nwq_src: Filesystem path used to import the NWQlib prototype.
        num_system_qubits: Number of system qubits.
        grid_points: Number of grid points, equal to ``2**num_system_qubits``.
        boundary_condition: Boundary condition used in the matrix builder.
        total_time: End time ``T`` in the NWQlib formulas.
        alpha: Heat-equation diffusivity prefactor in ``A``.
        grid_spacing: Spatial spacing used in ``A``.
        beta: Kernel parameter from the NWQlib LCHS formulas.
        epsilon: Error tolerance used in the truncation and quadrature bounds.
        trunc_multiplier: Multiplicative safety factor used for ``K``.
        spectral_norm_L: Spectral norm of the Hermitian part ``L``.
        h1: Step size from ``step_size_h1``.
        K: Truncation range half-width.
        Q: Number of Gauss-Legendre nodes per interval.
        kh1: Integer interval count ``int(K / h1)``.
        M: Mathematical total number of LCU quadrature nodes.
        control_qubits: Padded ancilla/control width ``ceil(log2(M))``.
        padded_terms: Oracle basis size ``2**control_qubits``.
        coeff_l1_norm: The ``||c||_1`` value after quadrature coefficients are built.
        circuit_num_qubits: Total qubit count if a circuit was built, otherwise 0.
        circuit_depth: Qiskit depth if a circuit was built, otherwise 0.
        circuit_size: Qiskit size if a circuit was built, otherwise 0.
        operation_counts: Qiskit operation histogram if a circuit was built.
    """

    nwq_src: str
    num_system_qubits: int
    grid_points: int
    boundary_condition: str
    total_time: float
    alpha: float
    grid_spacing: float
    beta: float
    epsilon: float
    trunc_multiplier: float
    spectral_norm_L: float
    h1: float
    K: float
    Q: int
    kh1: int
    M: int
    control_qubits: int
    padded_terms: int
    coeff_l1_norm: float
    circuit_num_qubits: int
    circuit_depth: int
    circuit_size: int
    operation_counts: dict[str, int]


def _ensure_nwq_src_on_path(nwq_src: str) -> Path:
    """Add the NWQlib prototype ``src`` directory to ``sys.path``.

    Args:
        nwq_src: Filesystem path to the NWQlib prototype ``src`` directory.

    Returns:
        Resolved path object.

    Raises:
        FileNotFoundError: If the source path does not exist.
    """

    src_path = Path(nwq_src).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"NWQlib prototype src path not found: {src_path}")
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return src_path


def _import_nwq_modules(nwq_src: str) -> dict[str, Any]:
    """Import the NWQlib prototype helpers used by this wrapper.

    Args:
        nwq_src: Path to ``nwqlib/prototypes/lchs/src``.

    Returns:
        Dictionary of imported functions.
    """

    _ensure_nwq_src_on_path(nwq_src)
    lchs_mod = importlib.import_module("lchs")
    lcu_mod = importlib.import_module("lcu")
    utils_mod = importlib.import_module("utils_synth")
    return {
        "cart_decomp": lchs_mod.cart_decomp,
        "step_size_h1": lchs_mod.step_size_h1,
        "trunc_K": lchs_mod.trunc_K,
        "n_node_Q": lchs_mod.n_node_Q,
        "gauss_quadrature": lchs_mod.gauss_quadrature,
        "gk": lchs_mod.gk,
        "utk": lchs_mod.utk,
        "utk_L": lchs_mod.utk_L,
        "utk_H": lchs_mod.utk_H,
        "lcu_generator": lcu_mod.lcu_generator,
        "nearest_num_qubit": utils_mod.nearest_num_qubit,
    }


def build_heat_matrix(
    num_system_qubits: int,
    *,
    boundary_condition: str = "dirichlet",
    alpha: float = 1.0,
    grid_spacing: float = 1.0,
) -> np.ndarray:
    """Build the 1D heat matrix ``A`` used by NWQlib's ``du/dt=-Au``.

    Args:
        num_system_qubits: Number of spatial/system qubits.
        boundary_condition: One of ``dirichlet``, ``neumann``, or
            ``periodic``.
        alpha: Diffusivity prefactor.
        grid_spacing: Spatial finite-difference spacing.

    Returns:
        Dense complex-valued matrix of shape ``(2**n, 2**n)``.
    """

    dim = 2 ** num_system_qubits
    scale = alpha / (grid_spacing ** 2)
    A = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        if boundary_condition == "dirichlet":
            A[i, i] = 2.0 * scale
        elif boundary_condition == "neumann":
            A[i, i] = (1.0 if i in {0, dim - 1} else 2.0) * scale
        elif boundary_condition == "periodic":
            A[i, i] = 2.0 * scale
        else:
            raise ValueError(
                f"Unknown boundary_condition '{boundary_condition}'. "
                "Expected 'dirichlet', 'neumann', or 'periodic'."
            )
        if i > 0:
            A[i, i - 1] = -1.0 * scale
        if i + 1 < dim:
            A[i, i + 1] = -1.0 * scale
    if boundary_condition == "periodic" and dim > 1:
        A[0, dim - 1] = -1.0 * scale
        A[dim - 1, 0] = -1.0 * scale
    return A


def _build_basis_state_circuit(num_qubits: int, basis_index: int) -> QuantumCircuit:
    """Create a basis-state preparation circuit in Qiskit qubit order.

    Args:
        num_qubits: Number of system qubits.
        basis_index: Computational basis index to prepare.

    Returns:
        Qiskit circuit containing the appropriate ``X`` gates.
    """

    if basis_index < 0 or basis_index >= 2 ** num_qubits:
        raise ValueError(
            f"basis index {basis_index} is outside [0, {2**num_qubits - 1}]"
        )
    circuit = QuantumCircuit(num_qubits, name=f"|{basis_index:0{num_qubits}b}>")
    bitstring = format(basis_index, f"0{num_qubits}b")
    for qiskit_qubit, bit in enumerate(reversed(bitstring)):
        if bit == "1":
            circuit.x(qiskit_qubit)
    return circuit


def _collect_nwq_lchs_terms(
    A: np.ndarray,
    *,
    total_time: float,
    beta: float,
    epsilon: float,
    trunc_multiplier: float,
    trotter_lh: bool,
    nwq: dict[str, Any],
) -> tuple[list[complex], list[np.ndarray], dict[str, float | int]]:
    """Reproduce the NWQlib quadrature construction without simulation.

    Args:
        A: Coefficient matrix for ``du/dt = -A u``.
        total_time: Final time ``T``.
        beta: Kernel parameter.
        epsilon: Error tolerance.
        trunc_multiplier: Safety factor applied to ``K``.
        trotter_lh: Whether to use the prototype's ``utk_L`` shortcut.
        nwq: Imported NWQlib helpers.

    Returns:
        Tuple ``(coeffs, unitaries, stats)`` where ``stats`` contains the
        mathematical quadrature metadata, including ``M``.
    """

    try:
        L, H = nwq["cart_decomp"](A)
    except Warning:
        # The NWQ prototype raises ``Warning`` when a numerically Hermitian
        # positive-semidefinite matrix picks up a tiny negative eigenvalue from
        # floating-point roundoff. For the heat operators used here, falling
        # back to the explicit Cartesian split is mathematically equivalent.
        L = 0.5 * (A + A.conj().T)
        H = 0.5j * (A.conj().T - A)
        min_eval = float(np.min(np.linalg.eigvalsh(L)))
        if np.linalg.norm(H, ord=2) > 1e-10 or min_eval < -1e-10:
            raise
    h1 = float(nwq["step_size_h1"](total_time, L))
    K = float(trunc_multiplier * nwq["trunc_K"](beta, epsilon, h1))
    Q = int(nwq["n_node_Q"](beta, epsilon, K))
    kh1 = int(K / h1)
    coeffs_unrot: list[complex] = []
    unitaries_unrot: list[np.ndarray] = []
    coeff_l1_norm = 0.0

    for shifted_m in range(2 * kh1):
        m = -kh1 + shifted_m
        k_nodes, weights = nwq["gauss_quadrature"](m * h1, h1, Q)
        for q_idx in range(Q):
            coeff = weights[q_idx] * nwq["gk"](beta, k_nodes[q_idx])
            coeff_l1_norm += float(np.abs(coeff))
            if trotter_lh:
                umat = nwq["utk_L"](total_time, k_nodes[q_idx], L)
            else:
                umat = nwq["utk"](total_time, k_nodes[q_idx], L, H)
            coeffs_unrot.append(coeff)
            unitaries_unrot.append(umat)

    M = len(coeffs_unrot)
    stats = {
        "spectral_norm_L": float(np.linalg.norm(L, ord=2)),
        "h1": h1,
        "K": K,
        "Q": Q,
        "kh1": kh1,
        "M": M,
        "coeff_l1_norm": coeff_l1_norm,
    }
    return coeffs_unrot, unitaries_unrot, stats


def build_nwq_dv_lchs_dirichlet(
    *,
    nwq_src: str,
    num_system_qubits: int,
    boundary_condition: str,
    total_time: float,
    alpha: float,
    grid_spacing: float,
    beta: float,
    epsilon: float,
    trunc_multiplier: float,
    qiskit_api: bool,
    trotter_lh: bool,
    build_circuit: bool,
    init_basis_index: int | None,
) -> tuple[QuantumCircuit | None, NwqDvLchsSummary]:
    """Compute NWQlib-style ``M`` and optionally build the padded LCU circuit.

    Args:
        nwq_src: Filesystem path to the NWQlib prototype ``src`` directory.
        num_system_qubits: Number of system qubits.
        boundary_condition: One-dimensional heat boundary condition used to
            define ``A``.
        total_time: Final time ``T``.
        alpha: Heat-equation diffusivity prefactor in ``A``.
        grid_spacing: Spatial spacing used in ``A``.
        beta: Kernel parameter.
        epsilon: Error tolerance for the NWQlib formulas.
        trunc_multiplier: Multiplicative safety factor for ``K``.
        qiskit_api: Forwarded to NWQlib's ``lcu_generator``.
        trotter_lh: Whether to use the prototype's ``utk_L`` shortcut.
        build_circuit: Whether to assemble the actual LCU circuit.
        init_basis_index: Optional system basis state to prepend.

    Returns:
        Pair ``(circuit_or_none, summary)``.
    """

    nwq = _import_nwq_modules(nwq_src)
    A = build_heat_matrix(
        num_system_qubits,
        boundary_condition=boundary_condition,
        alpha=alpha,
        grid_spacing=grid_spacing,
    )
    coeffs_unrot, unitaries_unrot, stats = _collect_nwq_lchs_terms(
        A,
        total_time=total_time,
        beta=beta,
        epsilon=epsilon,
        trunc_multiplier=trunc_multiplier,
        trotter_lh=trotter_lh,
        nwq=nwq,
    )

    M = int(stats["M"])
    control_qubits = int(nwq["nearest_num_qubit"](M))
    padded_terms = 2 ** control_qubits

    circuit = None
    operation_counts: dict[str, int] = {}
    circuit_num_qubits = 0
    circuit_depth = 0
    circuit_size = 0

    if build_circuit:
        initial_state_circ = None
        if init_basis_index is not None:
            initial_state_circ = _build_basis_state_circuit(
                num_system_qubits,
                init_basis_index,
            )
        circuit, _coef_abs, _absorbed_unitaries, _coef_l1 = nwq["lcu_generator"](
            coeffs_unrot,
            unitaries_unrot,
            initial_state_circ=initial_state_circ,
            verbose=1,
            qiskit_api=qiskit_api,
            debug=False,
        )
        operation_counts = {
            str(name): int(count) for name, count in dict(circuit.count_ops()).items()
        }
        circuit_num_qubits = int(circuit.num_qubits)
        circuit_depth = int(circuit.depth() or 0)
        circuit_size = int(circuit.size())

    summary = NwqDvLchsSummary(
        nwq_src=str(Path(nwq_src).expanduser().resolve()),
        num_system_qubits=num_system_qubits,
        grid_points=2 ** num_system_qubits,
        boundary_condition=boundary_condition,
        total_time=total_time,
        alpha=alpha,
        grid_spacing=grid_spacing,
        beta=beta,
        epsilon=epsilon,
        trunc_multiplier=trunc_multiplier,
        spectral_norm_L=float(stats["spectral_norm_L"]),
        h1=float(stats["h1"]),
        K=float(stats["K"]),
        Q=int(stats["Q"]),
        kh1=int(stats["kh1"]),
        M=M,
        control_qubits=control_qubits,
        padded_terms=padded_terms,
        coeff_l1_norm=float(stats["coeff_l1_norm"]),
        circuit_num_qubits=circuit_num_qubits,
        circuit_depth=circuit_depth,
        circuit_size=circuit_size,
        operation_counts=operation_counts,
    )
    return circuit, summary


def _write_outputs(
    circuit: QuantumCircuit | None,
    summary: NwqDvLchsSummary,
    *,
    output_json: str | None,
    output_qpy: str | None,
    output_text: str | None,
) -> None:
    """Write optional summary and circuit artifacts.

    Args:
        circuit: Optional built circuit.
        summary: Summary dataclass.
        output_json: Optional path for JSON metadata.
        output_qpy: Optional path for QPY circuit serialization.
        output_text: Optional path for ASCII circuit output.
    """

    if output_json:
        path = Path(output_json).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(summary), indent=2) + "\n")

    if circuit is None:
        return

    if output_qpy:
        path = Path(output_qpy).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            qpy.dump(circuit, fh)

    if output_text:
        path = Path(output_text).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(circuit.draw(output="text")) + "\n")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Compute the NWQlib DV LCHS quadrature size M for the 1D "
            "heat equation and optionally build the padded LCU circuit."
        )
    )
    parser.add_argument(
        "--nwq-src",
        default=DEFAULT_NWQ_LCHS_SRC,
        help="Path to /Users/.../nwqlib/prototypes/lchs/src.",
    )
    parser.add_argument(
        "--num-system-qubits",
        type=int,
        default=2,
        help="Number of system qubits. Default 2 gives a 4x4 heat matrix.",
    )
    parser.add_argument(
        "--boundary-condition",
        choices=("dirichlet", "neumann", "periodic"),
        default="dirichlet",
        help="One-dimensional heat boundary condition used to build A.",
    )
    parser.add_argument(
        "--total-time",
        type=float,
        default=1.0,
        help="Final time T in the NWQlib formulas.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Diffusivity prefactor in the heat matrix A.",
    )
    parser.add_argument(
        "--grid-spacing",
        type=float,
        default=1.0,
        help="Finite-difference grid spacing h used in A.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.9,
        help="NWQlib kernel parameter beta.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="NWQlib error tolerance epsilon.",
    )
    parser.add_argument(
        "--trunc-multiplier",
        type=float,
        default=2.0,
        help="Safety factor applied to the truncation range K.",
    )
    parser.add_argument(
        "--trotter-lh",
        action="store_true",
        help="Use NWQlib's trotterLH option for U(T,k).",
    )
    parser.add_argument(
        "--build-circuit",
        action="store_true",
        help="Actually build the NWQlib LCU circuit. This can be expensive when M is large.",
    )
    parser.add_argument(
        "--init-basis-index",
        type=int,
        help="Optional system basis state to prepend when --build-circuit is used.",
    )
    parser.add_argument(
        "--qiskit-api",
        action="store_true",
        help="Use NWQlib's Qiskit API path for PREP/SELECT synthesis.",
    )
    parser.add_argument(
        "--print-circuit",
        action="store_true",
        help="Print the ASCII circuit diagram to stdout after building it.",
    )
    parser.add_argument("--output-json", help="Optional path for summary JSON.")
    parser.add_argument("--output-qpy", help="Optional path for circuit QPY.")
    parser.add_argument("--output-text", help="Optional path for ASCII circuit text.")
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    circuit, summary = build_nwq_dv_lchs_dirichlet(
        nwq_src=args.nwq_src,
        num_system_qubits=args.num_system_qubits,
        boundary_condition=args.boundary_condition,
        total_time=args.total_time,
        alpha=args.alpha,
        grid_spacing=args.grid_spacing,
        beta=args.beta,
        epsilon=args.epsilon,
        trunc_multiplier=args.trunc_multiplier,
        qiskit_api=args.qiskit_api,
        trotter_lh=args.trotter_lh,
        build_circuit=args.build_circuit,
        init_basis_index=args.init_basis_index,
    )

    _write_outputs(
        circuit,
        summary,
        output_json=args.output_json,
        output_qpy=args.output_qpy,
        output_text=args.output_text,
    )

    print(json.dumps(asdict(summary), indent=2))
    if args.print_circuit and circuit is not None:
        print()
        print(circuit.draw(output="text"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
