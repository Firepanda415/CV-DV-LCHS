#!/usr/bin/env python3
"""Measure full 100-slice DV circuit depths in the published {1Q, CX} basis.

Builds the complex-kernel Dirichlet DV LCHS circuit at every grid size with
all 100 controlled-evolution slices instantiated, decomposes every gate,
including each CRZ through its standard two-CX identity, down to the
{1Q, CX} reporting basis, and records exact gate counts and DAG depth. The
full-circuit gate counts must reproduce the published inventory at every
size, which is enforced as a hard validation gate before any result is
written.

Run from the repository root with the legacy environment python
(scikit_tt available), e.g.
    /Users/zhen002/miniconda3/envs/cvdv/bin/python -W ignore make_dv_full_depths.py
Writes results_revision_v2/dv_circuit_full_depths.json.
"""

import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_dv_circuit_complex_kernel import ONE_QUBIT, build

OUTPUT = Path("results_revision_v2/dv_circuit_full_depths.json")
PUBLISHED_FULL_COUNTS = {
    4: {"1q": 7872, "cx": 6206},
    8: {"1q": 28072, "cx": 19206},
    16: {"1q": 64472, "cx": 41406},
    32: {"1q": 115272, "cx": 78206},
}


def decompose_to_1q_cx(circuit):
    work = circuit
    for _ in range(40):
        present = {inst.operation.name for inst in work.data}
        todo = present - ONE_QUBIT - {"cx", "barrier", "measure"}
        if not todo:
            break
        work = work.decompose(gates_to_decompose=sorted(todo))
    counts = {}
    for inst in work.data:
        name = inst.operation.name
        if name in ("barrier", "measure"):
            continue
        counts[name] = counts.get(name, 0) + 1
    leftovers = set(counts) - ONE_QUBIT - {"cx"}
    if leftovers:
        raise RuntimeError(f"gates left outside {{1Q, CX}}: {leftovers}")
    return {
        "1q": sum(v for k, v in counts.items() if k != "cx"),
        "cx": counts.get("cx", 0),
        "depth": work.depth(),
    }


def main() -> None:
    results = {}
    for num_system_qubits, m_grid in ((2, 4), (3, 8), (4, 16), (5, 32)):
        t0 = time.time()
        circuit, _ = build(num_system_qubits, 100)
        measured = decompose_to_1q_cx(circuit)
        ref = PUBLISHED_FULL_COUNTS[m_grid]
        if measured["1q"] != ref["1q"] or measured["cx"] != ref["cx"]:
            raise RuntimeError(
                f"M={m_grid} full-circuit count gate failed: {measured} vs {ref}"
            )
        results[str(m_grid)] = {
            **measured,
            "wall_seconds": round(time.time() - t0, 1),
        }
        print(f"[M={m_grid}] {results[str(m_grid)]}", flush=True)
    document = {
        "description": (
            "Directly measured full-circuit depths of the complex-kernel "
            "Dirichlet DV LCHS circuits with all 100 controlled-evolution "
            "slices instantiated, in the {1Q, CX} reporting basis with each "
            "CRZ decomposed into two CX and two single-qubit rotations. "
            "Full-circuit 1Q and CX counts reproduce the published "
            "inventory at every size (hard gate)."
        ),
        "results": results,
        "environment": {
            "python": platform.python_version(),
            "qiskit": importlib.metadata.version("qiskit"),
        },
        "source": "make_dv_full_depths.py",
    }
    OUTPUT.write_text(json.dumps(document, indent=1) + "\n")
    print(f"wrote {OUTPUT}")


main()
