"""Fixed-scale map error and sampling overhead of the DV LCHS baseline.

Completes the DV side of the Section 7.5 comparison with the two quantities
that `tab:dv_resource_compare` does not report: the fixed-scale map error of
the prescribed quadrature and the LCU postselection probability of the
benchmark input.  For each of the ten instances of the table, at the
published beta_best and the published sizing (eta = 1, eps = 0.1, T = 1),
the script evaluates

    K_DV      = sum_j c_j exp(-i T (k_j L + H)),        c_j = glw_j w(k_j),
    eps_F^DV  = ||K_DV - exp(-A T)||_F / ||exp(-A T)||_F,
    p_DV      = ||K_DV u0||^2 / ||c||_1^2,              u0 = basis index 1,

with exact branch propagators, so no Trotter or PREP-compression error
enters.  The convention mirrors the CV route's fixed-scale error: the
physical LCU block is K_DV / ||c||_1 with ||c||_1 known a priori from the
classical coefficients, and multiplying back by that known scale gives K_DV
with no fitted rescaling.  p_DV is the acceptance probability of the
all-zero ancilla branch of a PREP-SELECT-unPREP LCU circuit.

Validation gate (runs before anything is written): the full beta scan of
`make_dv_extension.dv_case` is re-run for all ten instances and must
reproduce the published M = 4 rows digit-for-digit (beta and M_DV exact,
1 - F_DV at four significant digits, ||c||_1 at four decimals) and match
every row of `results_revision_v2/dv_extension.csv` and
`dv_extension_m32.csv` to relative tolerance 1e-10.

Outputs: results_revision_v2/dv_fixed_scale.csv (+ .meta.json sidecar).
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from make_dv_extension import TOTAL_TIME, dv_case, quadrature, tridiag  # noqa: E402
from revision_eval import _heat_system  # noqa: E402
from clean_core import system_blocks  # noqa: E402

OUTPUT = REPO / "results_revision_v2" / "dv_fixed_scale.csv"
META = REPO / "results_revision_v2" / "dv_fixed_scale.meta.json"


def build_instances():
    """Return (label, beta_best, L, H, u0) in the row order of the table."""
    out = []
    for boundary, beta in (("dirichlet", 0.60), ("neumann", 0.90), ("periodic", 0.80)):
        l_mat, h_mat = system_blocks(_heat_system(boundary, TOTAL_TIME))
        out.append((f"heat_m4_{boundary}", beta, l_mat, h_mat, np.eye(4, dtype=complex)[:, 1]))
    for num_qubits, label in ((3, "heat_m8_dirichlet"), (4, "heat_m16_dirichlet")):
        l_mat, h_mat = system_blocks(_heat_system("dirichlet", TOTAL_TIME, num_qubits=num_qubits))
        out.append((label, 0.60, l_mat, h_mat, np.eye(l_mat.shape[0], dtype=complex)[:, 1]))
    lap32 = tridiag(32)
    out.append(("heat_m32_dirichlet", 0.60, lap32, np.zeros_like(lap32),
                np.eye(32, dtype=complex)[:, 1]))
    for m_dim in (8, 16, 32):
        lap = tridiag(m_dim)
        dc = (np.diag(np.ones(m_dim - 1), 1) - np.diag(np.ones(m_dim - 1), -1)) / 2.0
        out.append((f"advdiff_m{m_dim}", 0.60, lap, -1.0j * dc,
                    np.eye(m_dim, dtype=complex)[:, 1]))
    lap2d = np.kron(tridiag(4), np.eye(4)) + np.kron(np.eye(4), tridiag(4))
    out.append(("heat_2d_4x4", 0.60, lap2d, np.zeros_like(lap2d),
                np.eye(16, dtype=complex)[:, 1]))
    return out


def validation_gate(instances):
    """Re-run the published beta scan and check every row digit-for-digit."""
    published_m4 = {
        "heat_m4_dirichlet": (0.60, 320, "2.334e-03", "0.9357"),
        "heat_m4_neumann": (0.90, 192, "2.260e-03", "1.2073"),
        "heat_m4_periodic": (0.80, 248, "2.777e-04", "1.0740"),
    }
    csv_rows = {}
    for rel in ("results_revision_v2/dv_extension.csv",
                "results_revision_v2/dv_extension_m32.csv"):
        with (REPO / rel).open() as handle:
            for row in csv.DictReader(handle):
                csv_rows[row["case"]] = row

    failures = []
    for label, _beta, l_mat, h_mat, u0 in instances:
        row = dv_case(label, l_mat, h_mat, u0)
        if label in published_m4:
            b_ref, m_ref, inf_ref, c_ref = published_m4[label]
            if row["beta_opt"] != b_ref:
                failures.append(f"{label}: beta {row['beta_opt']} != {b_ref}")
            if row["M_DV"] != m_ref:
                failures.append(f"{label}: M_DV {row['M_DV']} != {m_ref}")
            if f"{row['one_minus_F_DV']:.3e}" != inf_ref:
                failures.append(f"{label}: 1-F {row['one_minus_F_DV']:.3e} != {inf_ref}")
            if f"{row['c_l1']:.4f}" != c_ref:
                failures.append(f"{label}: |c|_1 {row['c_l1']:.4f} != {c_ref}")
        ref = csv_rows[label]
        if abs(row["beta_opt"] - float(ref["beta_opt"])) > 1e-12:
            failures.append(f"{label}: beta vs CSV {row['beta_opt']} != {ref['beta_opt']}")
        if row["M_DV"] != int(ref["M_DV"]) or row["Q"] != int(ref["Q"]):
            failures.append(f"{label}: M_DV/Q vs CSV mismatch")
        for key in ("h1", "K", "c_l1", "one_minus_F_DV", "L_norm"):
            if not math.isclose(row[key], float(ref[key]), rel_tol=1e-10, abs_tol=1e-14):
                failures.append(f"{label}: {key} {row[key]!r} != CSV {ref[key]}")
    if failures:
        raise RuntimeError("VALIDATION GATE FAILED:\n" + "\n".join(failures))
    print("validation gate PASSED: all 10 rows reproduce the published table "
          "(M=4 digit-for-digit; breadth and M=32 rows match the CSVs to <1e-10)",
          flush=True)


def analyze(instances):
    results = []
    for label, beta, l_mat, h_mat, u0 in instances:
        l_norm = float(np.linalg.norm(l_mat, 2))
        _h1, _big_k, q_order, m_dv, ks, cs = quadrature(beta, l_norm)
        dim = l_mat.shape[0]
        k_dv = np.zeros((dim, dim), dtype=complex)
        for k, c in zip(ks, cs):
            k_dv = k_dv + c * expm(-1.0j * TOTAL_TIME * (k * l_mat + h_mat))
        target = expm(-TOTAL_TIME * (l_mat + 1.0j * h_mat))
        t_norm = np.linalg.norm(target, "fro")
        eps_f = float(np.linalg.norm(k_dv - target, "fro") / t_norm)
        c_l1 = float(np.sum(np.abs(cs)))
        p_dv = float(np.linalg.norm(k_dv @ u0) ** 2 / c_l1**2)
        p_cols = np.sum(np.abs(k_dv) ** 2, axis=0) / c_l1**2
        results.append({
            "case": label, "beta": beta, "M_DV": m_dv, "Q": q_order,
            "c_l1": c_l1, "eps_F_DV": eps_f,
            "p_DV_basis1": p_dv, "inv_p_DV_basis1": 1.0 / p_dv,
            "p_DV_min": float(np.min(p_cols)), "p_DV_max": float(np.max(p_cols)),
            "norm_Kdv_u0_sq": float(np.linalg.norm(k_dv @ u0) ** 2),
        })
        print(f"{label:<20} beta={beta:.2f} M_DV={m_dv:4d} |c|_1={c_l1:.4f} "
              f"eps_F={eps_f:.4e} p_DV={p_dv:.4e} 1/p={1.0 / p_dv:.2f} "
              f"p_range=[{np.min(p_cols):.4f},{np.max(p_cols):.4f}]")
    return results


def main():
    instances = build_instances()
    validation_gate(instances)
    results = analyze(instances)
    fieldnames = list(results[0].keys())
    OUTPUT.parent.mkdir(exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    META.write_text(
        json.dumps(
            {
                "output": str(OUTPUT.relative_to(REPO)),
                "methodology": (
                    "same sizing as tab:dv_resource_compare (eta=1, eps=0.1, "
                    "T=1, published beta_best per row, composite Gauss-Legendre, "
                    "complex kernel weights, exact branch propagators). "
                    "eps_F_DV = ||K_DV - e^{-AT}||_F/||e^{-AT}||_F with "
                    "K_DV = sum_j c_j e^{-iT(k_j L + H)} unscaled (fixed-scale "
                    "convention, scale ||c||_1 known a priori, no fitted "
                    "rescaling). p_DV_basis1 = ||K_DV u0||^2/||c||_1^2 is the "
                    "all-zero ancilla acceptance probability of the LCU block "
                    "for u0 = basis index 1, and p_DV_min/p_DV_max give its "
                    "range over all computational-basis inputs (columns of "
                    "K_DV). Validation gate: full beta-scan reproduction of "
                    "the published table before writing."
                ),
                "versions": {
                    "numpy": np.__version__,
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUTPUT.relative_to(REPO)} ({len(results)} rows)")


if __name__ == "__main__":
    main()
