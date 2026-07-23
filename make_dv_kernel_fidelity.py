"""Conditional fidelity of the DV circuit's dyadic-grid LCHS map, complex kernel vs projection.

The DV heat circuit realizes, after post-selection, the quadrature
    u_hat = sum_j c_j exp(-i T (k_j L + H)) u0
on its own dyadic k grid (9 ancilla qubits, lsb_pos = -3, so 512 nodes with
k in [-32, 31.875] and step 1/8). With magnitude amplitudes and the diagonal
phase layer the weights are the complex kernel values c_j = g(k_j), with
g(k) = 1 / ((1 - ik) C_beta exp((1 + ik)^beta)) and beta = 0.6. The published
phase-free circuit instead carries the positive-real projection
c_j = |Re g(k_j)|. This script evaluates both maps by matrix computation with
exact branch propagators and reports the conditional fidelity
|<u_hat/||u_hat||, u_exact>|^2 against u_exact = exp(-T A) u0, A = L + iH,
for the heat cases M = 4, 8, 16, 32 (Dirichlet, u0 = basis index 1, T = 1).

Branch propagators are exact matrix exponentials, so the reported numbers
isolate the kernel-quadrature error and exclude Trotter slicing and the
two-layer MPS compression of PREP. They therefore bound the corresponding
circuit-level fidelities from above.

The k grid is read from the archived circuit summaries (M = 4, 8, 16, 32) and
required to be identical across sizes and equal to its closed form.

Run with the miniconda base python (same environment as make_dv_extension.py):
    python3 make_dv_kernel_fidelity.py
Writes results_revision_v2/dv_kernel_fidelity.csv (+ meta).
"""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clean_core import system_blocks
from make_dv_extension import kernel_weight
from revision_eval import _heat_system

BETA = 0.6
TOTAL_TIME = 1.0
GRID_SOURCES = {
    4: Path("results_clean_dv_lchs_dirichlet/trotter100_summary.json"),
    8: Path("results_revision_v2/dv_circuit_m8_summary.json"),
    16: Path("results_revision_v2/dv_circuit_m16_summary.json"),
    32: Path("results_revision_v2/dv_circuit_m32_summary.json"),
}
OUTPUT = Path("results_revision_v2/dv_kernel_fidelity.csv")
META = Path("results_revision_v2/dv_kernel_fidelity.meta.json")


def load_k_grid():
    grids = {}
    lsb = {}
    for m_grid, path in GRID_SOURCES.items():
        summary = json.loads(path.read_text())
        if summary["beta"] != BETA or summary["num_qubits_lcu"] != 9:
            raise RuntimeError(f"unexpected kernel settings in {path}")
        grids[m_grid] = summary["k_values"]
        lsb[m_grid] = summary["lsb_pos"]
    reference = grids[4]
    if any(grids[m] != reference or lsb[m] != -3 for m in grids):
        raise RuntimeError("k grids differ across circuit summaries")
    ks = np.asarray(reference, dtype=float)
    n_int = 9 - 3 - 1
    closed_form = np.sort(-(2.0**n_int) + np.arange(512) * 0.125)
    if not np.array_equal(np.sort(ks), closed_form):
        raise RuntimeError("k grid does not match its closed form")
    return ks


def dyadic_map_infidelity(l_mat, h_mat, u0, ks, weights):
    a_mat = l_mat + 1.0j * h_mat
    u_exact = expm(-TOTAL_TIME * a_mat) @ u0
    u_exact = u_exact / np.linalg.norm(u_exact)
    u_hat = np.zeros_like(u0, dtype=complex)
    for k, c in zip(ks, weights):
        u_hat = u_hat + c * (expm(-1.0j * TOTAL_TIME * (k * l_mat + h_mat)) @ u0)
    fid = abs(np.vdot(u_hat / np.linalg.norm(u_hat), u_exact)) ** 2
    return 1.0 - fid


def main():
    ks = load_k_grid()
    dk = 0.125
    g_values = np.asarray([kernel_weight(k, BETA) for k in ks])
    complex_weights = g_values * dk
    positive_real_weights = np.abs(g_values.real) * dk
    print(
        f"grid: 512 nodes, k in [{ks.min()}, {ks.max()}], dk={dk}; "
        f"sum g dk = {np.sum(complex_weights):.6f}, "
        f"|c|_1 complex = {np.sum(np.abs(complex_weights)):.4f}, "
        f"|c|_1 positive-real = {np.sum(positive_real_weights):.4f}",
        flush=True,
    )

    rows = []
    for num_qubits in (2, 3, 4, 5):
        m_grid = 2**num_qubits
        system = _heat_system("dirichlet", TOTAL_TIME, num_qubits=num_qubits)
        l_mat, h_mat = system_blocks(system)
        u0 = np.eye(l_mat.shape[0], dtype=complex)[:, 1]
        inf_complex = dyadic_map_infidelity(l_mat, h_mat, u0, ks, complex_weights)
        inf_positive_real = dyadic_map_infidelity(
            l_mat, h_mat, u0, ks, positive_real_weights
        )
        row = {
            "case": f"heat_m{m_grid}_dirichlet",
            "grid_points": m_grid,
            "branches": len(ks),
            "lsb_pos": -3,
            "beta": BETA,
            "c_l1_complex": float(np.sum(np.abs(complex_weights))),
            "c_l1_positive_real": float(np.sum(positive_real_weights)),
            "one_minus_F_complex_kernel": inf_complex,
            "one_minus_F_positive_real": inf_positive_real,
        }
        rows.append(row)
        print(
            f"[M={m_grid}] 1-F complex kernel = {inf_complex:.3e}, "
            f"1-F positive-real projection = {inf_positive_real:.3e}",
            flush=True,
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
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
                "grid_sources": {m: str(p) for m, p in GRID_SOURCES.items()},
                "methodology": (
                    "conditional fidelity |<u_hat/||u_hat||, u_exact>|^2 with "
                    "u_exact = exp(-T(L+iH))u0 normalized, u_hat = sum_j c_j "
                    "exp(-iT(k_j L + H))u0 on the circuit's dyadic grid "
                    "(512 nodes, lsb_pos=-3), c_j = g(k_j) dk for the complex "
                    "kernel and c_j = |Re g(k_j)| dk for the positive-real "
                    "projection, beta = 0.6, u0 = basis index 1, T = 1; branch "
                    "propagators are exact matrix exponentials, so the values "
                    "exclude Trotter and PREP-compression error and bound the "
                    "circuit fidelity from above"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUTPUT} and {META}", flush=True)


if __name__ == "__main__":
    main()
