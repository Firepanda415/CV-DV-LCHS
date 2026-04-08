#!/usr/bin/env python3
r"""Build a DV LCHS circuit for the 1D Dirichlet heat equation.

This wrapper lives in the CV-DV repo but uses the sibling
``lchs-quantum-pde`` implementation to build an actual discrete-variable (DV)
LCHS circuit for the same benchmark family studied in the clean CV-DV work.

The intended default keeps the same Dirichlet heat benchmark already used for
the ``SNAP+D`` experiments,

.. math::

    \frac{d}{dt}u(t) = -A u(t), \qquad
    A =
    \begin{bmatrix}
    2&-1&0&0\\
    -1&2&-1&0\\
    0&-1&2&-1\\
    0&0&-1&2
    \end{bmatrix},
    \qquad
    u(0)=|01\rangle,

with :math:`M=4` grid points, :math:`\alpha = 1`, :math:`h = 1`, and
:math:`T = 1`, but it does *not* reuse the CV-optimal kernel exponent.  The
default DV kernel exponent is chosen independently as :math:`\beta = 0.8`,
which is consistent with the LCHS literature and the local DV tutorial
material for near-optimal DV quadrature behavior.

This script uses the actual ancilla-preparation path of
``lchs-quantum-pde``:

1. compute the DV LCHS kernel amplitudes,
2. compress them to an MPS,
3. convert that MPS to a quantum circuit,
4. assemble the full DV LCHS evolution circuit.

To align the DV ancilla width with the resource study already carried out in
this repo, the default ancilla register size is derived from the NWQ-style DV
quadrature count and padded to ``ceil(log2(M_DV))`` control qubits. The
``lchs-quantum-pde`` implementation uses a dyadic two's-complement ``k`` grid,
so its step size is set by the closest dyadic approximation to the NWQ step
size :math:`h_1`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from qiskit import QuantumCircuit, qpy

from clean_dv_lchs_nwq import build_nwq_dv_lchs_dirichlet


DEFAULT_DV_REPO_SRC = "/Users/zhen002/GitHub/lchs-quantum-pde/src"
DEFAULT_NWQ_SRC = "/Users/zhen002/GitHub/nwqlib/prototypes/lchs/src"


@dataclass
class DvLchsSummary:
    """Structured summary of the built DV LCHS circuit.

    Attributes:
        dv_repo_src: Filesystem path used to import the upstream DV repo.
        nwq_src: Filesystem path used to import the NWQ reference code.
        num_system_qubits: Number of system qubits for the heat equation.
        grid_points: Number of spatial grid points, equal to ``2**n``.
        boundary_condition: Boundary condition used by the operator builder.
        initial_basis_index: Optional basis state prepended on the system
            register before the DV LCHS evolution block.
        num_qubits_lcu: Number of ancilla qubits in the DV LCHS discretization.
        lcu_branch_count: Number of logical DV LCHS branches,
            ``2**num_qubits_lcu``.
        lsb_pos: Binary scaling exponent for the two's-complement ancilla grid.
        dyadic_step: Actual ``k``-grid spacing, equal to ``2**lsb_pos``.
        kappa: Heat equation diffusivity coefficient.
        h: Grid spacing used in the finite-difference Laplacian.
        total_time: Total simulated evolution time.
        num_time_steps: Number of repeated controlled-evolution slices used to
            emulate a product-formula time discretization in the DV circuit.
        step_dt: Time increment used in each repeated controlled-evolution
            slice, equal to ``total_time / num_time_steps``.
        kernel: Ancilla kernel family.
        kernel_projection: Projection used for the parametrized kernel.
        beta: Parametrized-kernel exponent.
        use_kernel_phases: Whether a diagonal ancilla phase correction is
            inserted for a complex parametrized kernel.
        num_layers: Number of MPS disentangling layers used in the upstream
            MPS-to-circuit conversion.
        nwq_h1: NWQ-derived step size ``h_1``.
        nwq_K: NWQ-derived truncation half-width.
        nwq_Q: NWQ-derived Gauss-Legendre order.
        nwq_M: NWQ-derived total quadrature term count.
        nwq_control_qubits: NWQ-derived padded control width.
        k_values: Ordered list of discretized ``k`` values represented by the
            ancilla computational basis.
        heat_operator_term_count: Number of terms in the upstream heat operator.
        antihermitian_term_count: Number of anti-Hermitian strings used by the
            LCHS evolution block.
        antihermitian_terms: The actual anti-Hermitian operator strings.
        kernel_mps_ranks: Bond-dimension profile of the ancilla kernel MPS.
        kernel_prep_depth: Qiskit depth of the ancilla PREP subcircuit.
        kernel_prep_size: Qiskit instruction count of the ancilla PREP
            subcircuit.
        kernel_prep_operation_counts: Gate histogram of the ancilla PREP block.
        controlled_evolution_depth: Qiskit depth of one ancilla-controlled
            evolution slice.
        controlled_evolution_size: Qiskit instruction count of one
            ancilla-controlled evolution slice.
        controlled_evolution_operation_counts: Gate histogram of one repeated
            ancilla-controlled evolution slice.
        circuit_num_qubits: Total qubit count of the generated circuit.
        circuit_num_clbits: Total classical-bit count of the generated circuit.
        circuit_depth: Qiskit depth of the assembled circuit.
        circuit_size: Qiskit instruction count of the assembled circuit.
        operation_counts: Qiskit operation histogram.
    """

    dv_repo_src: str
    nwq_src: str
    num_system_qubits: int
    grid_points: int
    boundary_condition: str
    initial_basis_index: int | None
    num_qubits_lcu: int
    lcu_branch_count: int
    lsb_pos: int
    dyadic_step: float
    kappa: float
    h: float
    total_time: float
    num_time_steps: int
    step_dt: float
    kernel: str
    kernel_projection: str
    beta: float
    use_kernel_phases: bool
    num_layers: int
    nwq_h1: float
    nwq_K: float
    nwq_Q: int
    nwq_M: int
    nwq_control_qubits: int
    k_values: list[float]
    heat_operator_term_count: int
    antihermitian_term_count: int
    antihermitian_terms: list[str]
    kernel_mps_ranks: list[int]
    kernel_prep_depth: int
    kernel_prep_size: int
    kernel_prep_operation_counts: dict[str, int]
    controlled_evolution_depth: int
    controlled_evolution_size: int
    controlled_evolution_operation_counts: dict[str, int]
    circuit_num_qubits: int
    circuit_num_clbits: int
    circuit_depth: int
    circuit_size: int
    operation_counts: dict[str, int]


def _ensure_dv_repo_on_path(dv_repo_src: str) -> Path:
    """Insert the upstream DV repo ``src`` directory into ``sys.path``.

    Args:
        dv_repo_src: Absolute path to the upstream ``src`` directory.

    Returns:
        Resolved path object.

    Raises:
        FileNotFoundError: If the requested path does not exist.
    """

    src_path = Path(dv_repo_src).expanduser().resolve()
    if not src_path.exists():
        raise FileNotFoundError(f"DV repo src path does not exist: {src_path}")
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    return src_path


def _import_upstream_modules(dv_repo_src: str) -> dict[str, Any]:
    """Import the upstream DV LCHS modules after amending ``sys.path``.

    Args:
        dv_repo_src: Path to ``lchs-quantum-pde/src``.

    Returns:
        Dictionary of imported classes and helper functions.
    """

    _ensure_dv_repo_on_path(dv_repo_src)

    pde_ops = importlib.import_module(
        "lchs_quantum_pde.operators.pde_hamiltonians"
    )
    hsim_mod = importlib.import_module(
        "lchs_quantum_pde.quantum_solvers.hamiltonian_simulation"
    )
    kernels_mod = importlib.import_module(
        "lchs_quantum_pde.quantum_solvers.kernels"
    )

    return {
        "HeatEquationEvolution": pde_ops.HeatEquationEvolution,
        "HamiltonianSimulation": hsim_mod.HamiltonianSimulation,
        "LinearCombinationHamiltonianSimulation": (
            hsim_mod.LinearCombinationHamiltonianSimulation
        ),
        "cauchy_kernel_amplitudes": kernels_mod.cauchy_kernel_amplitudes,
        "kernel_amplitudes_to_mps": kernels_mod.kernel_amplitudes_to_mps,
        "parametrized_kernel_amplitudes": (
            kernels_mod.parametrized_kernel_amplitudes
        ),
        "parametrized_kernel_phase_angles": (
            kernels_mod.parametrized_kernel_phase_angles
        ),
        "k_values": kernels_mod.k_values,
    }


def _build_basis_state_circuit(num_qubits: int, basis_index: int) -> QuantumCircuit:
    """Create a basis-state preparation circuit in Qiskit qubit order.

    Args:
        num_qubits: Number of system qubits.
        basis_index: Computational-basis index in physics/MSB ordering.

    Returns:
        Circuit containing the appropriate ``X`` gates.
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


def _nearest_dyadic_lsb_pos(step_size: float) -> int:
    """Return the closest integer ``lsb_pos`` for dyadic step ``2**lsb_pos``."""

    if step_size <= 0:
        raise ValueError(f"step_size must be positive, got {step_size}")
    return int(round(math.log2(step_size)))


def _build_mps_kernel(
    *,
    num_qubits_lcu: int,
    lsb_pos: int,
    kernel: str,
    kernel_projection: str,
    beta: float,
    use_kernel_phases: bool,
    upstream: dict[str, Any],
) -> tuple[Any, list[float] | None]:
    """Build the upstream kernel MPS used for ancilla preparation.

    Args:
        num_qubits_lcu: Number of ancilla qubits.
        lsb_pos: Binary scaling exponent for the ``k`` grid.
        kernel: Either ``"cauchy"`` or ``"parametrized"``.
        kernel_projection: Projection mode for the parametrized kernel.
        beta: Parametrized-kernel exponent.
        use_kernel_phases: Whether to apply the complex kernel phase correction.
        upstream: Imported upstream helpers.

    Returns:
        Pair ``(kernel_mps, phase_angles_or_none)``.

    Raises:
        ValueError: If the kernel name is not supported.
    """

    if kernel == "cauchy":
        amplitudes = upstream["cauchy_kernel_amplitudes"](num_qubits_lcu, lsb_pos)
        phase_angles = None
    elif kernel == "parametrized":
        amplitudes = upstream["parametrized_kernel_amplitudes"](
            num_qubits_lcu,
            beta=beta,
            lsb_pos=lsb_pos,
            projection=kernel_projection,
        )
        phase_angles = (
            upstream["parametrized_kernel_phase_angles"](
                num_qubits_lcu,
                beta=beta,
                lsb_pos=lsb_pos,
            )
            if use_kernel_phases
            else None
        )
    else:
        raise ValueError(f"Unsupported kernel: {kernel!r}")

    kernel_mps = upstream["kernel_amplitudes_to_mps"](amplitudes)
    return kernel_mps, phase_angles


def build_dv_lchs_dirichlet_heat_circuit(
    *,
    dv_repo_src: str,
    nwq_src: str,
    num_system_qubits: int,
    num_qubits_lcu: int | None,
    lsb_pos: int | None,
    h: float,
    kappa: float,
    total_time: float,
    num_time_steps: int,
    barrier: bool,
    measure: bool,
    kernel: str,
    kernel_projection: str,
    beta: float,
    use_kernel_phases: bool,
    num_layers: int,
    initial_basis_index: int | None,
    nwq_epsilon: float,
    nwq_trunc_multiplier: float,
) -> tuple[QuantumCircuit, DvLchsSummary]:
    """Build the 1D Dirichlet DV LCHS circuit and return a structured summary.

    Args:
        dv_repo_src: Path to the upstream DV repo ``src`` directory.
        nwq_src: Path to the NWQlib prototype ``src`` directory.
        num_system_qubits: Number of spatial/system qubits.
        num_qubits_lcu: Number of ancilla qubits for the DV LCHS. When ``None``,
            use the NWQ-derived padded control width.
        lsb_pos: Binary scaling exponent for the ``k`` discretization. When
            ``None``, use the closest dyadic step to the NWQ ``h_1``.
        h: Spatial grid spacing in the Laplacian.
        kappa: Heat equation diffusivity coefficient.
        total_time: Total simulated evolution time.
        num_time_steps: Number of repeated ancilla-controlled evolution slices
            used to emulate a product-formula discretization while keeping the
            ancilla PREP and unPREP outside the repeated middle block.
        barrier: Whether to request Qiskit barriers in the generated circuit.
        measure: Whether to include ancilla post-selection measurements.
        kernel: Ancilla kernel family.
        kernel_projection: Projection used for the parametrized kernel.
        beta: Parametrized-kernel exponent.
        use_kernel_phases: Whether to add the upstream complex-kernel phase
            correction.
        num_layers: Number of MPS disentangling layers for ancilla PREP.
        initial_basis_index: Optional system basis state prepended to the
            evolution circuit.
        nwq_epsilon: NWQ-style epsilon used only to size the DV ancilla.
        nwq_trunc_multiplier: NWQ-style truncation multiplier used only to size
            the DV ancilla.

    Returns:
        Pair ``(circuit, summary)``.
    """

    if num_time_steps <= 0:
        raise ValueError("num_time_steps must be positive.")
    step_dt = total_time / num_time_steps

    upstream = _import_upstream_modules(dv_repo_src)
    _nwq_circuit, nwq_summary = build_nwq_dv_lchs_dirichlet(
        nwq_src=nwq_src,
        num_system_qubits=num_system_qubits,
        boundary_condition="dirichlet",
        total_time=total_time,
        alpha=kappa,
        grid_spacing=h,
        beta=beta,
        epsilon=nwq_epsilon,
        trunc_multiplier=nwq_trunc_multiplier,
        qiskit_api=True,
        trotter_lh=False,
        build_circuit=False,
        init_basis_index=initial_basis_index,
    )

    if num_qubits_lcu is None:
        num_qubits_lcu = nwq_summary.control_qubits
    if lsb_pos is None:
        lsb_pos = _nearest_dyadic_lsb_pos(nwq_summary.h1)

    heat = upstream["HeatEquationEvolution"](
        num_qubits_x=num_system_qubits,
        dim=1,
        h=h,
        periodic=False,
        kappa=kappa,
    )
    split = upstream["HamiltonianSimulation"](heat)

    kernel_mps, phase_angles = _build_mps_kernel(
        num_qubits_lcu=num_qubits_lcu,
        lsb_pos=lsb_pos,
        kernel=kernel,
        kernel_projection=kernel_projection,
        beta=beta,
        use_kernel_phases=use_kernel_phases,
        upstream=upstream,
    )

    lchs = upstream["LinearCombinationHamiltonianSimulation"](
        heat,
        num_qubits_lcu=num_qubits_lcu,
        lsb_pos=lsb_pos,
        kernel=kernel_mps,
        kernel_phase_angles=phase_angles,
        num_layers=num_layers,
    )

    n_int = lchs._num_qubits_lcu + lchs._lsb_pos - 1
    k_bit_weights = [-2**n_int] + [2**k for k in reversed(range(lchs._lsb_pos, n_int))]
    dt_list = [step_dt * k for k in k_bit_weights]

    if num_time_steps == 1:
        evolve_circuit = lchs.get_evolve_circ(
            dt=step_dt,
            barrier=barrier,
            measure=measure,
        )
        controlled_evolution = lchs._get_evolve_circ_part(
            dt_list,
            lchs.op_a_list,
            lchs.coeffs_a,
            control=True,
            barrier=barrier,
        )
    else:
        from qiskit import ClassicalRegister, QuantumRegister

        q_sys = QuantumRegister(lchs._num_qubits_system, r"q_{sys}")
        q_anc = QuantumRegister(lchs._num_qubits_lcu, r"q_{anc}")
        evolve_circuit = QuantumCircuit(q_sys, q_anc)

        if lchs.with_hermitian:
            circ_h = lchs._get_evolve_circ_part(
                total_time,
                lchs.op_h_list,
                lchs.coeffs_h,
                barrier=barrier,
            )
            evolve_circuit = evolve_circuit.compose(
                circ_h,
                qubits=evolve_circuit.qubits[: lchs._num_qubits_system],
            )

        controlled_evolution = lchs._get_evolve_circ_part(
            dt_list,
            lchs.op_a_list,
            lchs.coeffs_a,
            control=True,
            barrier=barrier,
        )
        evolve_circuit = evolve_circuit.compose(
            lchs._lcu_state_preparation_circ,
            qubits=evolve_circuit.qubits[-lchs._num_qubits_lcu :],
        )
        for _ in range(num_time_steps):
            evolve_circuit = evolve_circuit.compose(
                controlled_evolution,
                qubits=evolve_circuit.qubits,
            )
        if lchs._kernel_phase_circ is not None:
            evolve_circuit = evolve_circuit.compose(
                lchs._kernel_phase_circ,
                qubits=evolve_circuit.qubits[-lchs._num_qubits_lcu :],
            )
        evolve_circuit = evolve_circuit.compose(
            lchs._lcu_state_preparation_circ.inverse(),
            qubits=evolve_circuit.qubits[-lchs._num_qubits_lcu :],
        )
        if measure:
            c_reg = ClassicalRegister(lchs._num_qubits_lcu)
            evolve_circuit.add_register(c_reg)
            evolve_circuit.measure(q_anc, c_reg)

    if initial_basis_index is not None:
        init_circuit = _build_basis_state_circuit(
            num_system_qubits,
            initial_basis_index,
        )
        circuit = QuantumCircuit(
            evolve_circuit.num_qubits,
            evolve_circuit.num_clbits,
        )
        circuit.compose(
            init_circuit,
            qubits=range(num_system_qubits),
            inplace=True,
        )
        circuit.compose(evolve_circuit, inplace=True)
    else:
        circuit = evolve_circuit

    kernel_prep = lchs._lcu_state_preparation_circ
    k_vals = upstream["k_values"](num_qubits_lcu, lsb_pos)
    op_counts = {
        str(name): int(count) for name, count in dict(circuit.count_ops()).items()
    }

    summary = DvLchsSummary(
        dv_repo_src=str(Path(dv_repo_src).expanduser().resolve()),
        nwq_src=str(Path(nwq_src).expanduser().resolve()),
        num_system_qubits=num_system_qubits,
        grid_points=2 ** num_system_qubits,
        boundary_condition="dirichlet",
        initial_basis_index=initial_basis_index,
        num_qubits_lcu=num_qubits_lcu,
        lcu_branch_count=2 ** num_qubits_lcu,
        lsb_pos=lsb_pos,
        dyadic_step=float(2.0 ** lsb_pos),
        kappa=kappa,
        h=h,
        total_time=total_time,
        num_time_steps=int(num_time_steps),
        step_dt=float(step_dt),
        kernel=kernel,
        kernel_projection=kernel_projection,
        beta=beta,
        use_kernel_phases=use_kernel_phases,
        num_layers=num_layers,
        nwq_h1=float(nwq_summary.h1),
        nwq_K=float(nwq_summary.K),
        nwq_Q=int(nwq_summary.Q),
        nwq_M=int(nwq_summary.M),
        nwq_control_qubits=int(nwq_summary.control_qubits),
        k_values=[float(v) for v in k_vals],
        heat_operator_term_count=len(heat.op_list),
        antihermitian_term_count=len(split.op_a_list),
        antihermitian_terms=[str(term) for term in split.op_a_list],
        kernel_mps_ranks=[int(v) for v in getattr(lchs.kernel_mps, "ranks", [])],
        kernel_prep_depth=int(kernel_prep.depth() or 0),
        kernel_prep_size=int(kernel_prep.size()),
        kernel_prep_operation_counts={
            str(name): int(count) for name, count in dict(kernel_prep.count_ops()).items()
        },
        controlled_evolution_depth=int(controlled_evolution.depth() or 0),
        controlled_evolution_size=int(controlled_evolution.size()),
        controlled_evolution_operation_counts={
            str(name): int(count)
            for name, count in dict(controlled_evolution.count_ops()).items()
        },
        circuit_num_qubits=int(circuit.num_qubits),
        circuit_num_clbits=int(circuit.num_clbits),
        circuit_depth=int(circuit.depth() or 0),
        circuit_size=int(circuit.size()),
        operation_counts=op_counts,
    )
    return circuit, summary


def _write_optional_outputs(
    circuit: QuantumCircuit,
    summary: DvLchsSummary,
    *,
    output_json: str | None,
    output_text: str | None,
    output_qpy: str | None,
) -> None:
    """Write optional circuit artifacts to disk.

    Args:
        circuit: Built DV LCHS circuit.
        summary: Structured summary object.
        output_json: Optional path for JSON metadata.
        output_text: Optional path for text circuit drawing.
        output_qpy: Optional path for QPY circuit serialization.
    """

    if output_json:
        json_path = Path(output_json).expanduser().resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(asdict(summary), indent=2) + "\n")

    if output_text:
        text_path = Path(output_text).expanduser().resolve()
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(str(circuit.draw(output="text")) + "\n")

    if output_qpy:
        qpy_path = Path(output_qpy).expanduser().resolve()
        qpy_path.parent.mkdir(parents=True, exist_ok=True)
        with qpy_path.open("wb") as fh:
            qpy.dump(circuit, fh)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Build a DV LCHS circuit for the 1D Dirichlet heat equation using "
            "lchs-quantum-pde, with Dirichlet heat-equation defaults matched "
            "to the clean benchmark and a DV-specific kernel beta."
        )
    )
    parser.add_argument(
        "--dv-repo-src",
        default=os.environ.get("DV_LCHS_REPO_SRC", DEFAULT_DV_REPO_SRC),
        help="Path to the sibling lchs-quantum-pde/src directory.",
    )
    parser.add_argument(
        "--nwq-src",
        default=os.environ.get("NWQ_LCHS_REPO_SRC", DEFAULT_NWQ_SRC),
        help="Path to the NWQlib prototype src directory used for DV sizing.",
    )
    parser.add_argument(
        "--num-system-qubits",
        type=int,
        default=2,
        help="Number of spatial/system qubits. Default matches the M=4 benchmark.",
    )
    parser.add_argument(
        "--num-lcu-qubits",
        type=int,
        help=(
            "Number of DV LCHS ancilla qubits. If omitted, use the NWQ-derived "
            "padded control width."
        ),
    )
    parser.add_argument(
        "--lsb-pos",
        type=int,
        help=(
            "Least-significant-bit position for the DV ancilla k-grid. If "
            "omitted, use the nearest dyadic approximation to the NWQ h1."
        ),
    )
    parser.add_argument(
        "--h",
        type=float,
        default=1.0,
        help="Grid spacing used in the finite-difference Laplacian.",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=1.0,
        help="Heat equation diffusivity coefficient.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Total simulated evolution time. Default matches the clean T=1 benchmark.",
    )
    parser.add_argument(
        "--num-time-steps",
        type=int,
        default=1,
        help=(
            "Repeat the ancilla-controlled evolution block this many times using "
            "step size dt/num_time_steps, while keeping ancilla PREP and unPREP "
            "outside the repeated middle block."
        ),
    )
    parser.add_argument(
        "--kernel",
        choices=("cauchy", "parametrized"),
        default="parametrized",
        help="Ancilla kernel family. Default uses the DV parametrized kernel.",
    )
    parser.add_argument(
        "--kernel-projection",
        choices=("positive_real", "magnitude"),
        default="positive_real",
        help="Projection used when --kernel parametrized is selected.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.8,
        help=(
            "Parametrized-kernel exponent for the DV LCHS kernel. Default "
            "0.8 is DV-specific and is not tied to the CV-optimal beta."
        ),
    )
    parser.add_argument(
        "--use-kernel-phases",
        action="store_true",
        help="Apply the upstream complex-kernel ancilla phase correction.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="MPS disentangling layers for the ancilla PREP circuit.",
    )
    parser.add_argument(
        "--initial-basis-index",
        type=int,
        default=1,
        help="Optional system basis state. Default |01> matches the clean benchmark.",
    )
    parser.add_argument(
        "--nwq-epsilon",
        type=float,
        default=0.1,
        help="NWQ-style epsilon used only to size the DV ancilla register.",
    )
    parser.add_argument(
        "--nwq-trunc-multiplier",
        type=float,
        default=2.0,
        help="NWQ-style truncation multiplier used only to size the DV ancilla register.",
    )
    parser.add_argument(
        "--barrier",
        action="store_true",
        help="Insert Qiskit barriers in the generated circuit.",
    )
    parser.add_argument(
        "--no-measure",
        action="store_true",
        help="Skip the ancilla post-selection measurements.",
    )
    parser.add_argument(
        "--print-circuit",
        action="store_true",
        help="Print an ASCII circuit diagram to stdout.",
    )
    parser.add_argument(
        "--output-json",
        help="Optional path to save the summary JSON.",
    )
    parser.add_argument(
        "--output-text",
        help="Optional path to save the ASCII circuit drawing.",
    )
    parser.add_argument(
        "--output-qpy",
        help="Optional path to save a QPY circuit file.",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""

    args = parse_args()
    circuit, summary = build_dv_lchs_dirichlet_heat_circuit(
        dv_repo_src=args.dv_repo_src,
        nwq_src=args.nwq_src,
        num_system_qubits=args.num_system_qubits,
        num_qubits_lcu=args.num_lcu_qubits,
        lsb_pos=args.lsb_pos,
        h=args.h,
        kappa=args.kappa,
        total_time=args.dt,
        num_time_steps=args.num_time_steps,
        barrier=args.barrier,
        measure=not args.no_measure,
        kernel=args.kernel,
        kernel_projection=args.kernel_projection,
        beta=args.beta,
        use_kernel_phases=args.use_kernel_phases,
        num_layers=args.num_layers,
        initial_basis_index=args.initial_basis_index,
        nwq_epsilon=args.nwq_epsilon,
        nwq_trunc_multiplier=args.nwq_trunc_multiplier,
    )

    _write_optional_outputs(
        circuit,
        summary,
        output_json=args.output_json,
        output_text=args.output_text,
        output_qpy=args.output_qpy,
    )

    print(json.dumps(asdict(summary), indent=2))
    if args.print_circuit:
        print()
        print(circuit.draw(output="text"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
