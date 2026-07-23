"""Rebuild the DV circuit inventory with the complex LCHS kernel realized exactly.

Same construction as make_dv_circuit_ext.py (Dirichlet DV LCHS, beta = 0.6,
MPS-compressed ancilla PREP, n_t = 1 and n_t = 2 builds differenced into
per-slice and PREP+unPREP costs, extrapolation to the 100-slice circuit),
except the ancilla kernel now carries the full complex weights g(k): magnitude
amplitudes |g(k)|^(1/2) in PREP together with the diagonal phase layer
exp(i arg g(k)) applied once between the last controlled-evolution slice and
unPREP. The post-selected map is then proportional to sum_k g(k) U_k, the
complex-kernel LCHS quadrature on the circuit's dyadic k grid.

The phase layer sits outside the repeated slice block, so the per-slice counts
must be identical to the phase-free inventory. That equality is enforced as a
hard validation gate against results_revision_v2/dv_circuit_extension.csv and
dv_circuit_extension_m32.csv before any row is written. The diagonal enters
the one-time PREP+unPREP overhead after decomposition to the published
reporting basis (1Q, CX, CRZ).

Run with the legacy environment python (scikit_tt available), e.g.
    /Users/zhen002/miniconda3/envs/cvdv/bin/python -W ignore \
        make_dv_circuit_complex_kernel.py
Writes results_revision_v2/dv_circuit_extension_complex_kernel.csv (+ meta).
"""

import csv
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clean_dv_lchs_dirichlet import build_dv_lchs_dirichlet_heat_circuit

NWQ_SRC = "/Users/zhen002/GitHub/nwtoolchain/prototypes_nwqlib/prototypes/lchs/src"
DV_SRC = "/Users/zhen002/GitHub/lchs-quantum-pde/src"
BASELINE_MAIN = Path("results_revision_v2/dv_circuit_extension.csv")
BASELINE_M32 = Path("results_revision_v2/dv_circuit_extension_m32.csv")
OUTPUT = Path("results_revision_v2/dv_circuit_extension_complex_kernel.csv")
META = Path("results_revision_v2/dv_circuit_extension_complex_kernel.meta.json")

ONE_QUBIT = {"u", "u1", "u2", "u3", "rz", "p", "h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "sx"}
TERMINAL_TWO_QUBIT = {"cx", "crz", "cz", "cp"}


def to_reporting_basis(circuit):
    """Decompose to the published reporting basis {1Q, CX, CRZ}.

    Unlike the phase-free script, the circuit now contains a DiagonalGate whose
    Qiskit synthesis passes through uniformly controlled rotations, so the
    decomposition iterates over every non-terminal gate name rather than a
    fixed multi-controlled list. CRZ, CZ, and CP stay terminal exactly as in
    the published counting.
    """
    work = circuit
    for _ in range(24):
        present = {inst.operation.name for inst in work.data}
        todo = present - ONE_QUBIT - TERMINAL_TWO_QUBIT - {"barrier", "measure"}
        if not todo:
            break
        work = work.decompose(gates_to_decompose=sorted(todo))
    counts = {}
    for inst in work.data:
        name = inst.operation.name
        if name in ("barrier", "measure"):
            continue
        counts[name] = counts.get(name, 0) + 1
    leftovers = set(counts) - ONE_QUBIT - TERMINAL_TWO_QUBIT
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
        kernel_projection="magnitude",
        beta=0.6,
        use_kernel_phases=True,
        num_layers=2,
        initial_basis_index=1,
        nwq_epsilon=0.1,
        nwq_trunc_multiplier=1.0,
    )
    return circuit, summary


def load_baseline_rows():
    rows = {}
    for path in (BASELINE_MAIN, BASELINE_M32):
        with path.open() as handle:
            for row in csv.DictReader(handle):
                rows[int(row["grid_points"])] = row
    return rows


def main():
    baseline = load_baseline_rows()
    rows = []
    deltas = {}
    for nq in (2, 3, 4, 5):
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

        ref = baseline[m_grid]
        ref_slice = {k: int(ref[f"slice_{'1q' if k == '1q' else k}"]) for k in ("1q", "cx", "crz")}
        if slice_counts != ref_slice:
            raise RuntimeError(
                f"M={m_grid} per-slice parity gate failed: {slice_counts} vs {ref_slice}"
            )
        if summary.num_qubits_lcu != 9:
            raise RuntimeError(f"M={m_grid} LCU width != 9: {summary.num_qubits_lcu}")
        deltas[f"m{m_grid}"] = {
            k: prep_counts[k] - int(ref[f"prep_unprep_{'1q' if k == '1q' else k}"])
            for k in ("1q", "cx", "crz")
        }

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
            f"[M={m_grid}] slice parity gate passed; prep+unprep+phase="
            f"{prep_counts['1q']}/{prep_counts['cx']}/{prep_counts['crz']} "
            f"(delta vs phase-free {deltas[f'm{m_grid}']}) "
            f"full100={total_100['1q']}/{total_100['cx']}/{total_100['crz']} "
            f"depth~{depth_100}",
            flush=True,
        )

    fieldnames = list(rows[0])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
                "baseline_inputs": [str(BASELINE_MAIN), str(BASELINE_M32)],
                "compile_settings": {
                    "qiskit": importlib.metadata.version("qiskit"),
                    "reporting_basis": ["1Q", "CX", "CRZ"],
                    "nwq_src": NWQ_SRC,
                    "dv_src": DV_SRC,
                    "kernel": "parametrized",
                    "kernel_projection": "magnitude",
                    "use_kernel_phases": True,
                    "beta": 0.6,
                    "num_layers": 2,
                    "nwq_trunc_multiplier": 1.0,
                },
                "per_slice_parity": "per-slice counts equal the phase-free inventory at every size (hard gate)",
                "prep_unprep_delta_vs_phase_free": deltas,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUTPUT} and {META}", flush=True)


if __name__ == "__main__":
    main()
