"""Complex-kernel DV circuit inventory in the two-type {1Q, CX} reporting basis.

Same circuits as make_dv_circuit_complex_kernel.py (Dirichlet DV LCHS,
magnitude amplitudes plus the ancilla phase diagonal, beta = 0.6, n_t = 1 and
n_t = 2 builds differenced and extrapolated to 100 slices), with one change in
the accounting: every CRZ is decomposed through its standard identity into two
CX and two single-qubit rotations, so two-qubit cost is reported as a single
CX column and no separate CRZ column remains. Counts therefore convert exactly
from the three-type inventory (per CRZ: CX + 2, 1Q + 2), and that conversion
is enforced as a hard gate against dv_circuit_extension_complex_kernel.csv.
Depths do not convert arithmetically and are measured in this basis, with a
direct M = 4 build at n_t = 100 required to reproduce the extrapolation.

Run with the legacy environment python (scikit_tt available), e.g.
    /Users/zhen002/miniconda3/envs/cvdv/bin/python -W ignore \
        make_dv_circuit_complex_kernel_cx.py
Writes results_revision_v2/dv_circuit_extension_complex_kernel_cx.csv (+ meta).
"""

import csv
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_dv_circuit_complex_kernel as base

BASELINE = Path("results_revision_v2/dv_circuit_extension_complex_kernel.csv")
OUTPUT = Path("results_revision_v2/dv_circuit_extension_complex_kernel_cx.csv")
META = Path("results_revision_v2/dv_circuit_extension_complex_kernel_cx.meta.json")

ONE_QUBIT = base.ONE_QUBIT


def to_cx_basis(circuit):
    """Decompose to {1Q, CX}: CRZ, CZ, and CP are no longer terminal."""
    work = circuit
    for _ in range(24):
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
        raise RuntimeError(f"unexpected gates after decomposition: {leftovers}")
    return {
        "1q": sum(v for k, v in counts.items() if k in ONE_QUBIT),
        "cx": counts.get("cx", 0),
        "depth": work.depth(),
    }


def main():
    with BASELINE.open() as handle:
        three_type = {row["case"]: row for row in csv.DictReader(handle)}

    rows = []
    for nq in (2, 3, 4, 5):
        m_grid = 2**nq
        c1, summary = base.build(nq, 1)
        c2, _ = base.build(nq, 2)
        b1 = to_cx_basis(c1)
        b2 = to_cx_basis(c2)
        slice_counts = {k: b2[k] - b1[k] for k in ("1q", "cx")}
        prep_counts = {k: b1[k] - slice_counts[k] for k in ("1q", "cx")}
        total_100 = {k: prep_counts[k] + 100 * slice_counts[k] for k in ("1q", "cx")}
        depth_100 = b1["depth"] + 99 * (b2["depth"] - b1["depth"])

        ref = three_type[f"heat_m{m_grid}_dirichlet"]
        expected_slice = {
            "1q": int(ref["slice_1q"]) + 2 * int(ref["slice_crz"]),
            "cx": int(ref["slice_cx"]) + 2 * int(ref["slice_crz"]),
        }
        expected_prep = {
            "1q": int(ref["prep_unprep_1q"]) + 2 * int(ref["prep_unprep_crz"]),
            "cx": int(ref["prep_unprep_cx"]) + 2 * int(ref["prep_unprep_crz"]),
        }
        if slice_counts != expected_slice or prep_counts != expected_prep:
            raise RuntimeError(
                f"M={m_grid} CRZ-conversion gate failed: slice {slice_counts} vs "
                f"{expected_slice}, prep {prep_counts} vs {expected_prep}"
            )

        rows.append(
            {
                "case": f"heat_m{m_grid}_dirichlet",
                "grid_points": m_grid,
                "lcu_qubits": summary.num_qubits_lcu,
                "circuit_qubits": summary.circuit_num_qubits,
                "slice_1q": slice_counts["1q"],
                "slice_cx": slice_counts["cx"],
                "prep_unprep_1q": prep_counts["1q"],
                "prep_unprep_cx": prep_counts["cx"],
                "full100_1q": total_100["1q"],
                "full100_cx": total_100["cx"],
                "full100_depth_est": depth_100,
            }
        )
        print(
            f"[M={m_grid}] CRZ-conversion gate passed; slice={slice_counts} "
            f"prep+unprep+phase={prep_counts} full100={total_100} depth~{depth_100}",
            flush=True,
        )

    c_full, _ = base.build(2, 100)
    b_full = to_cx_basis(c_full)
    m4 = rows[0]
    if (
        b_full["1q"] != m4["full100_1q"]
        or b_full["cx"] != m4["full100_cx"]
        or b_full["depth"] != m4["full100_depth_est"]
    ):
        raise RuntimeError(
            f"M=4 direct n_t=100 build gate failed: measured {b_full}, "
            f"extrapolated {m4}"
        )
    print(
        f"M=4 direct n_t=100 build reproduces the extrapolation exactly: {b_full}",
        flush=True,
    )

    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    META.write_text(
        json.dumps(
            {
                "output": str(OUTPUT),
                "source_sha256": hashlib.sha256(
                    Path(__file__).resolve().read_bytes()
                ).hexdigest(),
                "baseline_input": str(BASELINE),
                "reporting_basis": ["1Q", "CX"],
                "crz_convention": (
                    "each CRZ decomposed via its standard identity into two CX "
                    "and two single-qubit rotations; structural decomposition "
                    "only, no transpiler optimization or gate merging"
                ),
                "conversion_gate": (
                    "per size, slice and PREP+unPREP counts must equal the "
                    "three-type inventory with CX + 2 crz and 1Q + 2 crz"
                ),
                "m4_direct_build": b_full,
                "compile_settings": {
                    "qiskit": importlib.metadata.version("qiskit"),
                    "kernel": "parametrized",
                    "kernel_projection": "magnitude",
                    "use_kernel_phases": True,
                    "beta": 0.6,
                    "num_layers": 2,
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUTPUT} and {META}", flush=True)


if __name__ == "__main__":
    main()
