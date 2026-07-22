"""Extend the DV circuit inventory (tab:dv_circuit_compare) to heat M=8, 16.

Mirrors the published methodology: build the Dirichlet DV LCHS circuit
(positive-real parametrized kernel, beta = 0.6, phases disabled, MPS-compressed
ancilla PREP) with n_t = 1 and n_t = 2 slices, decompose multi-controlled gates
to the published reporting basis (1Q, CX, CRZ), difference the two builds to
obtain the per-slice and PREP+unPREP costs, and extrapolate to the 100-slice
circuit. Validation gate: M=4 must reproduce the published block counts
(PREP 130 1Q + 48 CX; slice 53 1Q + 38 CX + 9 CRZ; full 5561/3896/900).

Run with the legacy environment python (scikit_tt available), e.g.
    /Users/zhen002/miniconda3/envs/cvdv/bin/python make_dv_circuit_ext.py
Writes results_revision_v2/dv_circuit_extension.csv.
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clean_dv_lchs_dirichlet import build_dv_lchs_dirichlet_heat_circuit

NWQ_SRC = "/Users/zhen002/GitHub/nwtoolchain/prototypes_nwqlib/prototypes/lchs/src"
DV_SRC = "/Users/zhen002/GitHub/lchs-quantum-pde/src"
OUTPUT = Path("results_revision_v2/dv_circuit_extension.csv")

ONE_QUBIT = {"u", "u1", "u2", "u3", "rz", "p", "h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "sx"}
MULTI = {"ccx", "mcx", "mcx_vchain", "mcx_vchain_dg", "mcx_gray", "rccx", "c3x", "c4x", "cswap"}


def to_reporting_basis(circuit):
    work = circuit
    for _ in range(12):
        present = {inst.operation.name for inst in work.data}
        todo = present & MULTI
        if not todo:
            break
        work = work.decompose(gates_to_decompose=sorted(todo))
    counts = {}
    for inst in work.data:
        name = inst.operation.name
        if name in ("barrier", "measure"):
            continue
        counts[name] = counts.get(name, 0) + 1
    leftovers = set(counts) - ONE_QUBIT - {"cx", "crz", "cz", "cp"}
    if leftovers:
        raise RuntimeError(f"unexpected gates after decomposition: {leftovers}")
    one_q = sum(v for k, v in counts.items() if k in ONE_QUBIT)
    two_q_extra = {k: v for k, v in counts.items() if k in ("cz", "cp")}
    return {
        "1q": one_q,
        "cx": counts.get("cx", 0),
        "crz": counts.get("crz", 0),
        **{f"extra_{k}": v for k, v in two_q_extra.items()},
        "depth": work.depth(),
    }


def build(num_system_qubits, num_time_steps):
    circuit, summary = build_dv_lchs_dirichlet_heat_circuit(
        dv_repo_src=DV_SRC,
        nwq_src=NWQ_SRC,
        num_system_qubits=num_system_qubits,
        num_qubits_lcu=None,
        lsb_pos=None,
        h=1.0,
        kappa=1.0,
        total_time=1.0,
        num_time_steps=num_time_steps,
        barrier=False,
        measure=False,
        kernel="parametrized",
        kernel_projection="positive_real",
        beta=0.6,
        use_kernel_phases=False,
        num_layers=2,
        initial_basis_index=1,
        nwq_epsilon=0.1,
        nwq_trunc_multiplier=1.0,
    )
    return circuit, summary


def main():
    rows = []
    for nq in (2, 3, 4):
        m_grid = 2**nq
        c1, summary = build(nq, 1)
        c2, _ = build(nq, 2)
        b1 = to_reporting_basis(c1)
        b2 = to_reporting_basis(c2)
        slice_counts = {k: b2[k] - b1[k] for k in ("1q", "cx", "crz")}
        prep_counts = {k: b1[k] - slice_counts[k] for k in ("1q", "cx", "crz")}
        total_100 = {k: prep_counts[k] + 100 * slice_counts[k] for k in ("1q", "cx", "crz")}
        depth_slice = b2["depth"] - b1["depth"]
        depth_100 = b1["depth"] + 99 * depth_slice
        row = {
            "case": f"heat_m{m_grid}_dirichlet",
            "grid_points": m_grid,
            "lcu_qubits": summary.num_qubits_lcu,
            "circuit_qubits": summary.circuit_num_qubits,
            "slice_1q": slice_counts["1q"],
            "slice_cx": slice_counts["cx"],
            "slice_crz": slice_counts["crz"],
            "prep_unprep_1q": prep_counts["1q"],
            "prep_unprep_cx": prep_counts["cx"],
            "prep_unprep_crz": prep_counts["crz"],
            "full100_1q": total_100["1q"],
            "full100_cx": total_100["cx"],
            "full100_crz": total_100["crz"],
            "full100_depth_est": depth_100,
        }
        rows.append(row)
        print(
            f"[M={m_grid}] lcu={row['lcu_qubits']} slice(1q/cx/crz)="
            f"{slice_counts['1q']}/{slice_counts['cx']}/{slice_counts['crz']} "
            f"prep+unprep={prep_counts['1q']}/{prep_counts['cx']}/{prep_counts['crz']} "
            f"full100={total_100['1q']}/{total_100['cx']}/{total_100['crz']} "
            f"depth~{depth_100}",
            flush=True,
        )
        if m_grid == 4:
            if (
                slice_counts != {"1q": 53, "cx": 38, "crz": 9}
                or total_100 != {"1q": 5561, "cx": 3896, "crz": 900}
            ):
                raise RuntimeError(f"M=4 validation failed: {row}")
            print("M=4 validation gate passed", flush=True)

    fieldnames = list(rows[0])
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
