"""Build the M=32 heat DV circuit inventory after the frozen M=4 gate.

Run from a fresh working directory with the legacy cvdv environment python
(scikit_tt available), mirroring make_m32_breadth.py, e.g.
    cd workdir_m32 && /Users/zhen002/miniconda3/envs/cvdv/bin/python \
        -W ignore ../make_m32_dv_circuit.py
Writes dv_circuit_extension_m32.csv (+ meta and the 100-step summary JSON)
into results_revision/ relative to the working directory.
"""

import csv
import hashlib
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path

import make_dv_circuit_ext as base

ROOT = Path(__file__).resolve().parent
OUT_DIR = Path("results_revision")
SUMMARY = OUT_DIR / "dv_circuit_m32_summary.json"
OUTPUT = OUT_DIR / "dv_circuit_extension_m32.csv"
META = OUT_DIR / "dv_circuit_extension_m32.meta.json"
PINNED_NWQ_SRC = "/Users/zhen002/GitHub/nwtoolchain/prototypes_nwqlib/prototypes/lchs/src"
PROBE_SLICE_CX = 758


def _inventory(num_system_qubits: int):
    circuit1, summary = base.build(num_system_qubits, 1)
    circuit2, _ = base.build(num_system_qubits, 2)
    basis1 = base.to_reporting_basis(circuit1)
    basis2 = base.to_reporting_basis(circuit2)
    slice_counts = {key: basis2[key] - basis1[key] for key in ("1q", "cx", "crz")}
    prep_counts = {key: basis1[key] - slice_counts[key] for key in ("1q", "cx", "crz")}
    full_counts = {
        key: prep_counts[key] + 100 * slice_counts[key] for key in ("1q", "cx", "crz")
    }
    depth_slice = basis2["depth"] - basis1["depth"]
    depth_100 = basis1["depth"] + 99 * depth_slice
    return summary, slice_counts, prep_counts, full_counts, depth_100


def main() -> None:
    if base.NWQ_SRC != PINNED_NWQ_SRC:
        raise RuntimeError(f"unexpected NWQ source: {base.NWQ_SRC}")

    _, m4_slice, _, m4_full, m4_depth = _inventory(2)
    if (
        m4_slice != {"1q": 53, "cx": 38, "crz": 9}
        or m4_full != {"1q": 5561, "cx": 3896, "crz": 900}
        or m4_depth != 9227
    ):
        raise RuntimeError(
            f"M=4 validation failed: slice={m4_slice}, full={m4_full}, depth={m4_depth}"
        )
    print("M=4 DV circuit validation gate passed", flush=True)

    summary1, slice_counts, prep_counts, full_counts, depth_100 = _inventory(5)
    if summary1.num_qubits_lcu != 9:
        raise RuntimeError(f"M=32 LCU width != 9: {summary1.num_qubits_lcu}")
    if slice_counts["crz"] != 9:
        raise RuntimeError(f"M=32 CRZ per slice != 9: {slice_counts['crz']}")
    if prep_counts != {"1q": 261, "cx": 96, "crz": 0}:
        raise RuntimeError(f"M=32 PREP+unPREP gate failed: {prep_counts}")

    _, summary100 = base.build(5, 100)
    row = {
        "case": "heat_m32_dirichlet",
        "grid_points": 32,
        "lcu_qubits": summary1.num_qubits_lcu,
        "circuit_qubits": summary1.circuit_num_qubits,
        "slice_1q": slice_counts["1q"],
        "slice_cx": slice_counts["cx"],
        "slice_crz": slice_counts["crz"],
        "prep_unprep_1q": prep_counts["1q"],
        "prep_unprep_cx": prep_counts["cx"],
        "prep_unprep_crz": prep_counts["crz"],
        "full100_1q": full_counts["1q"],
        "full100_cx": full_counts["cx"],
        "full100_crz": full_counts["crz"],
        "full100_depth_est": depth_100,
        "prior_probe_slice_cx": PROBE_SLICE_CX,
        "slice_cx_delta_from_probe": slice_counts["cx"] - PROBE_SLICE_CX,
    }
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    SUMMARY.write_text(json.dumps(asdict(summary100), indent=2) + "\n")
    META.write_text(
        json.dumps(
            {
                "output": str(OUTPUT),
                "summary": str(SUMMARY),
                "source_sha256": hashlib.sha256(
                    (ROOT / "make_m32_dv_circuit.py").read_bytes()
                ).hexdigest(),
                "m4_validation": {
                    "slice": m4_slice,
                    "full100": m4_full,
                    "depth100": m4_depth,
                },
                "compile_settings": {
                    "qiskit": importlib.metadata.version("qiskit"),
                    "reporting_basis": ["1Q", "CX", "CRZ"],
                    "multi_gate_decomposition_passes": 12,
                    "nwq_src": base.NWQ_SRC,
                    "dv_src": base.DV_SRC,
                    "kernel": "parametrized",
                    "kernel_projection": "positive_real",
                    "beta": 0.6,
                    "num_layers": 2,
                    "nwq_trunc_multiplier": 1.0,
                },
                "probe_comparison": {
                    "prior_slice_cx": PROBE_SLICE_CX,
                    "measured_slice_cx": slice_counts["cx"],
                    "delta": slice_counts["cx"] - PROBE_SLICE_CX,
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"[M=32] slice={slice_counts} prep+unprep={prep_counts} "
        f"full100={full_counts} depth~{depth_100}; probe delta="
        f"{slice_counts['cx'] - PROBE_SLICE_CX:+d}",
        flush=True,
    )
    print(f"wrote {OUTPUT}, {SUMMARY}, and {META}", flush=True)


if __name__ == "__main__":
    main()
