"""State-preparation fidelity of the two-layer MPS ancilla PREP circuit.

The compiled DV LCHS circuits load the nine-qubit quadrature ancilla with
magnitude amplitudes |g_beta(k_j)|^(1/2), beta = 0.6, on the dyadic grid with
lsb_pos = -3 (512 nodes), through kernel_amplitudes_to_mps followed by the
iterative-disentangling MPS-to-circuit conversion with num_layers = 2. The
exact amplitude vector has full MPS bond dimension (16 at the central bond),
so the finite-layer circuit is an approximate loading routine. This script
rebuilds that exact construction, hard-gates its gate histogram and depth
against the archived circuit summaries (M = 8, 16, 32 revision summaries and
the M = 4 trotter100 summary), and reports the statevector fidelity of the
prepared state against the exact normalized amplitude profile for
num_layers = 1..4. The MPS is orthonormalized with TT.ortho() before the
conversion, replicating the solver path in the upstream package
(hamiltonian_simulation.py builds _lcu_state_preparation_circ as
kernel.ortho() followed by mps_to_circuit(kernel, D=num_layers)); no dense
amplitude-loading routine (qiskit StatePreparation or initialize) appears
anywhere in that path.

The upstream conversion completes each extracted isometry to a unitary with
unseeded random columns (numpy.random.rand in mps_to_circuit). Those columns
do not touch the num_layers = 1 state, which is therefore deterministic, but
they feed the inverse-update step and perturb the prepared state for
num_layers >= 2, and the specific draw inside the archived compiled circuits
is not recoverable. The gate histogram and depth are draw independent. The
script therefore samples NUM_DRAWS conversions per layer count and records
the fidelity distribution; the num_layers = 2 statistics back the
state-preparation infidelity quoted in the DV comparison subsection.

Method provenance: the MPS-to-circuit conversion is the iterative
disentangling scheme of Ran, Phys. Rev. A 101, 032310 (2020). As a
numerically exact reference alternative, the script also compiles the same
profile with qiskit's StatePreparation, which synthesizes through the
isometry decomposition of Iten, Colbeck, Kukuljan, Home, and Christandl,
Phys. Rev. A 93, 032318 (2016) (StatePreparation._define_synthesis_isom ->
Isometry(params, 0, 0) in qiskit 2.0.0), decomposed to the same {1Q, CX}
reporting basis. Those exact-loading counts back the exact ancilla PREP rows
of the DV circuit table.

Run with the legacy environment python (scikit_tt available), e.g.
    /Users/zhen002/miniconda3/envs/cvdv/bin/python -W ignore \
        make_dv_prep_fidelity.py
Writes results_revision_v2/dv_prep_fidelity.json (+ meta).
"""

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/zhen002/GitHub/lchs-quantum-pde/src")

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

from lchs_quantum_pde.quantum_solvers.kernels import (
    kernel_amplitudes_to_mps,
    parametrized_kernel_amplitudes,
)
from lchs_quantum_pde.tensor_network.mps_circuit import mps_to_circuit

N_LCU = 9
BETA = 0.6
LSB_POS = -3
COMPILED_LAYERS = 2
NUM_DRAWS = 50
SUMMARY_SOURCES = {
    4: Path("results_clean_dv_lchs_dirichlet/trotter100_summary.json"),
    8: Path("results_revision_v2/dv_circuit_m8_summary.json"),
    16: Path("results_revision_v2/dv_circuit_m16_summary.json"),
    32: Path("results_revision_v2/dv_circuit_m32_summary.json"),
}
OUTPUT = Path("results_revision_v2/dv_prep_fidelity.json")
META = Path("results_revision_v2/dv_prep_fidelity.meta.json")


def main():
    amps = parametrized_kernel_amplitudes(
        N_LCU, beta=BETA, lsb_pos=LSB_POS, projection="magnitude"
    )
    target = np.asarray(amps, dtype=complex)
    target = target / np.linalg.norm(target)

    mps = kernel_amplitudes_to_mps(amps)
    mps.ortho()
    bond_dims = [int(r) for r in mps.ranks]
    if max(bond_dims) != 2 ** (N_LCU // 2):
        raise RuntimeError(f"unexpected MPS bond profile: {bond_dims}")

    cores = [np.asarray(c)[:, :, 0, :] for c in mps.cores]
    vec = cores[0]
    for core in cores[1:]:
        vec = np.tensordot(vec, core, axes=([-1], [0]))
    mps_vec = vec.reshape(-1)
    mps_vec = mps_vec / np.linalg.norm(mps_vec)
    mps_fid = float(abs(np.vdot(target, mps_vec)) ** 2)
    if abs(mps_fid - 1.0) > 1e-10:
        raise RuntimeError(f"untruncated MPS is not exact: fidelity {mps_fid}")

    rng_seed = 20260722
    np.random.seed(rng_seed)
    layer_rows = {}
    for layers in (1, 2, 3, 4):
        fids = []
        counts = None
        depth = None
        for _ in range(NUM_DRAWS):
            circ = mps_to_circuit(mps, D=layers)
            flat = circ.decompose()
            draw_counts = {name: int(v) for name, v in flat.count_ops().items()}
            if counts is None:
                counts, depth = draw_counts, int(flat.depth())
            elif draw_counts != counts or int(flat.depth()) != depth:
                raise RuntimeError(f"layers={layers}: counts vary across draws")
            sv = np.asarray(Statevector.from_instruction(circ), dtype=complex)
            fids.append(float(abs(np.vdot(target, sv)) ** 2))
        fids_arr = np.asarray(fids)
        layer_rows[layers] = {
            "operation_counts": counts,
            "depth": depth,
            "state_infidelity_mean": float(np.mean(1.0 - fids_arr)),
            "state_infidelity_min": float(np.min(1.0 - fids_arr)),
            "state_infidelity_max": float(np.max(1.0 - fids_arr)),
            "state_infidelity_std": float(np.std(1.0 - fids_arr)),
            "num_draws": NUM_DRAWS,
        }
        print(
            f"layers={layers}: infidelity mean={np.mean(1.0 - fids_arr):.3e} "
            f"min={np.min(1.0 - fids_arr):.3e} max={np.max(1.0 - fids_arr):.3e} "
            f"counts={counts} depth={depth}",
            flush=True,
        )

    dense = QuantumCircuit(N_LCU)
    dense.append(StatePreparation(target), range(N_LCU))
    work = dense
    for _ in range(30):
        present = {inst.operation.name for inst in work.data}
        todo = present - {"u", "rz", "ry", "rx", "p", "x", "h", "gphase", "cx", "barrier"}
        if not todo:
            break
        work = work.decompose(gates_to_decompose=sorted(todo))
    dense_counts = {name: int(v) for name, v in work.count_ops().items() if name != "barrier"}
    dense_cx = dense_counts.get("cx", 0)
    dense_1q = sum(v for k, v in dense_counts.items() if k != "cx")
    dense_sv = np.asarray(Statevector.from_instruction(dense), dtype=complex)
    dense_fid = float(abs(np.vdot(target, dense_sv)) ** 2)
    if abs(dense_fid - 1.0) > 1e-10:
        raise RuntimeError(f"exact isometry loading is not exact: fidelity {dense_fid}")
    t2 = transpile(dense, basis_gates=["u", "cx"], optimization_level=2, seed_transpiler=1234)
    t2_counts = dict(t2.count_ops())
    if int(t2_counts.get("cx", 0)) != dense_cx:
        raise RuntimeError(
            f"exact-loading CX count changes under opt-2 transpile: {t2_counts} vs {dense_cx}"
        )
    print(
        f"exact isometry loading: 1q={dense_1q} cx={dense_cx} depth={work.depth()} "
        f"(opt-2 transpile CX unchanged), fidelity={dense_fid:.12f}",
        flush=True,
    )

    compiled = layer_rows[COMPILED_LAYERS]
    for m_grid, path in SUMMARY_SOURCES.items():
        summary = json.loads(path.read_text())
        if summary["num_layers"] != COMPILED_LAYERS or summary["beta"] != BETA:
            raise RuntimeError(f"M={m_grid} summary settings differ from this build")
        if summary["lsb_pos"] != LSB_POS or summary["num_qubits_lcu"] != N_LCU:
            raise RuntimeError(f"M={m_grid} summary grid differs from this build")
        expected_size = sum(compiled["operation_counts"].values())
        if summary["kernel_prep_size"] != expected_size:
            raise RuntimeError(
                f"M={m_grid} kernel_prep_size {summary['kernel_prep_size']} "
                f"!= rebuilt {expected_size}"
            )
        if summary["kernel_prep_depth"] != compiled["depth"]:
            raise RuntimeError(
                f"M={m_grid} kernel_prep_depth {summary['kernel_prep_depth']} "
                f"!= rebuilt {compiled['depth']}"
            )
        archived_counts = summary.get("kernel_prep_operation_counts")
        if archived_counts is not None and dict(archived_counts) != compiled[
            "operation_counts"
        ]:
            raise RuntimeError(
                f"M={m_grid} archived PREP histogram {archived_counts} "
                f"!= rebuilt {compiled['operation_counts']}"
            )
        print(f"[M={m_grid}] archived PREP block matches the rebuilt circuit", flush=True)

    OUTPUT.write_text(
        json.dumps(
            {
                "target_state": (
                    "normalized |g_beta(k_j)|^(1/2) profile, beta = 0.6, "
                    "9-qubit dyadic grid with lsb_pos = -3 (512 nodes)"
                ),
                "mps_bond_dims": bond_dims,
                "untruncated_mps_fidelity": mps_fid,
                "compiled_num_layers": COMPILED_LAYERS,
                "sampling_seed": rng_seed,
                "exact_isometry_loading": {
                    "method": (
                        "qiskit StatePreparation (isometry decomposition, Iten et al., "
                        "PRA 93, 032318 (2016)), structural decomposition to {1Q, CX}"
                    ),
                    "one_qubit_gates": dense_1q,
                    "cx_gates": dense_cx,
                    "depth": int(work.depth()),
                    "state_fidelity": dense_fid,
                    "opt2_transpile_cx_unchanged": True,
                },
                "layers": {str(k): v for k, v in layer_rows.items()},
            },
            indent=2,
        )
        + "\n"
    )
    META.write_text(
        json.dumps(
            {
                "output": str(OUTPUT),
                "source_sha256": hashlib.sha256(
                    Path(__file__).resolve().read_bytes()
                ).hexdigest(),
                "summary_inputs": {str(k): str(v) for k, v in SUMMARY_SOURCES.items()},
                "validation_gates": (
                    "full-rank bond profile, untruncated-MPS exactness, and "
                    "per-size equality of the rebuilt two-layer PREP histogram "
                    "and depth with the archived circuit summaries"
                ),
                "compile_settings": {
                    "qiskit": importlib.metadata.version("qiskit"),
                    "kernel": "parametrized",
                    "kernel_projection": "magnitude",
                    "beta": BETA,
                    "lsb_pos": LSB_POS,
                    "num_qubits_lcu": N_LCU,
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUTPUT} and {META}", flush=True)


if __name__ == "__main__":
    main()
