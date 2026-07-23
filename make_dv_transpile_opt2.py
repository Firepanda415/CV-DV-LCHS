#!/usr/bin/env python3
"""Transpile the full DV circuits at optimization level 2 and record the counts.

Builds the complex-kernel Dirichlet DV LCHS circuit at every grid size with
all 100 controlled-evolution slices instantiated and transpiles it with
Qiskit at optimization level 2 in the {U, CX} basis with a fixed transpiler
seed and no coupling map, so no routing pass runs and the result is
deterministic. Records the transpiled one-qubit count, CX count, and depth
next to the published identity-decomposition values. The published CX counts
must be unchanged by the transpilation, which is enforced as a hard gate.

Run from the repository root with the legacy environment python
(scikit_tt available), e.g.
    /Users/zhen002/miniconda3/envs/cvdv/bin/python -W ignore make_dv_transpile_opt2.py
Writes results_revision_v2/dv_circuit_opt2_transpile.json.
"""

import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qiskit import transpile

from make_dv_circuit_complex_kernel import build

OUTPUT = Path("results_revision_v2/dv_circuit_opt2_transpile.json")
PUBLISHED = {
    4: {"1q": 7872, "cx": 6206, "depth": 12874},
    8: {"1q": 28072, "cx": 19206, "depth": 36975},
    16: {"1q": 64472, "cx": 41406, "depth": 77466},
    32: {"1q": 115272, "cx": 78206, "depth": 132964},
}


def main() -> None:
    results = {}
    for num_system_qubits, m_grid in ((2, 4), (3, 8), (4, 16), (5, 32)):
        circuit, _ = build(num_system_qubits, 100)
        t0 = time.time()
        transpiled = transpile(
            circuit,
            basis_gates=["u", "cx"],
            optimization_level=2,
            seed_transpiler=11,
        )
        counts = {}
        for inst in transpiled.data:
            counts[inst.operation.name] = counts.get(inst.operation.name, 0) + 1
        leftovers = set(counts) - {"u", "cx"}
        if leftovers:
            raise RuntimeError(f"gates outside {{u, cx}}: {leftovers}")
        ref = PUBLISHED[m_grid]
        row = {
            "opt2_1q": counts.get("u", 0),
            "opt2_cx": counts.get("cx", 0),
            "opt2_depth": transpiled.depth(),
            "published_1q": ref["1q"],
            "published_cx": ref["cx"],
            "published_depth": ref["depth"],
            "wall_seconds": round(time.time() - t0, 1),
        }
        if row["opt2_cx"] != ref["cx"]:
            raise RuntimeError(
                f"M={m_grid} CX count changed under transpilation: "
                f"{row['opt2_cx']} vs {ref['cx']}"
            )
        results[str(m_grid)] = row
        print(f"[M={m_grid}] {row}", flush=True)
    document = {
        "description": (
            "Full 100-slice complex-kernel Dirichlet DV LCHS circuits "
            "transpiled with Qiskit at optimization level 2 in the {U, CX} "
            "basis, fixed transpiler seed 11, no coupling map. The CX count "
            "is unchanged at every size (hard gate), so the published "
            "identity-decomposition CX counts already coincide with the "
            "optimized ones. One-qubit counts and depths shrink through "
            "adjacent one-qubit merging."
        ),
        "transpile_settings": {
            "basis_gates": ["u", "cx"],
            "optimization_level": 2,
            "seed_transpiler": 11,
            "coupling_map": None,
        },
        "results": results,
        "environment": {
            "python": platform.python_version(),
            "qiskit": importlib.metadata.version("qiskit"),
        },
        "source": "make_dv_transpile_opt2.py",
    }
    OUTPUT.write_text(json.dumps(document, indent=1) + "\n")
    print(f"wrote {OUTPUT}")


main()
