"""Extend the classical DV-LCHS quadrature table to the new breadth cases.

Reproduces the published methodology of Table `tab:dv_resource_compare`
exactly: for each generator, scan beta in [0.60, 0.95], build the composite
Gauss--Legendre quadrature with
    h1 = 1/(e T ||L||_2),
    K = eta * ceil((ln(1/eps))^(1/beta) / h1) * h1,
    Q = ceil(log(8K/(3 C_beta eps)) / log 4),  C_beta = 2 pi exp(-2^beta),
    M_DV = 2 floor(K/h1) Q,   m_c = ceil(log2 M_DV),
(eta = 1, eps = 0.1), evaluate the classical DV output
    u_DV = sum_j c_j exp(-i T (k_j L + H)) u0,   c_j = glw_j * w(k_j),
    w(k) = e^{2^beta} / (2 pi (1 - i k) e^{(1+i k)^beta}),
and report the beta maximizing the conditional fidelity F(u_DV, u_exact).

Validation gate: the M=4 heat rows must reproduce the published values
digit-for-digit (beta_opt 0.60/0.90/0.80 and M_DV 320/192/248 exactly;
1-F_DV equal to 2.334e-3/2.260e-3/2.777e-4 at four significant digits)
before the new rows are written. New cases: 1D heat M=8 and M=16 (Dirichlet),
advection-diffusion M=8 and M=16 (Dirichlet, nu=c=h=1), and the 4x4 2D heat
instance. u0 is basis index 1 throughout (the benchmark convention).

Writes results_revision_v2/dv_extension.csv.
"""

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from revision_eval import _heat_system
from clean_core import system_blocks

EPS = 0.1
ETA = 1.0
TOTAL_TIME = 1.0
BETAS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

OUTPUT = Path("results_revision_v2/dv_extension.csv")
META = Path("results_revision_v2/dv_extension.meta.json")


def kernel_weight(k, beta):
    return np.exp(2.0**beta) / (
        2.0 * np.pi * (1.0 - 1.0j * k) * np.exp((1.0 + 1.0j * k) ** beta)
    )


def quadrature(beta, l_norm):
    h1 = 1.0 / (math.e * TOTAL_TIME * l_norm)
    n_half = math.ceil((math.log(1.0 / EPS)) ** (1.0 / beta) / h1)
    big_k = ETA * n_half * h1
    c_beta = 2.0 * math.pi * math.exp(-(2.0**beta))
    q_order = math.ceil(math.log(8.0 * big_k / (3.0 * c_beta * EPS)) / math.log(4.0))
    nodes, weights = leggauss(q_order)
    ks, cs = [], []
    # K is n_half * h1 by construction (eta = 1), so use the integer directly
    # instead of re-flooring K/h1, which is ulp-fragile at the boundary.
    n_sub = 2 * n_half
    for idx in range(n_sub):
        left = -big_k + idx * h1
        centre = left + h1 / 2.0
        for node, weight in zip(nodes, weights):
            k = centre + node * h1 / 2.0
            ks.append(k)
            cs.append(weight * h1 / 2.0 * kernel_weight(k, beta))
    m_dv = n_sub * q_order
    return h1, big_k, q_order, m_dv, np.asarray(ks), np.asarray(cs)


def dv_case(label, l_mat, h_mat, u0):
    a_mat = l_mat + 1.0j * h_mat
    u_exact = expm(-TOTAL_TIME * a_mat) @ u0
    u_exact = u_exact / np.linalg.norm(u_exact)
    l_norm = float(np.linalg.norm(l_mat, 2))
    best = None
    for beta in BETAS:
        h1, big_k, q_order, m_dv, ks, cs = quadrature(beta, l_norm)
        u_dv = np.zeros_like(u0, dtype=complex)
        for k, c in zip(ks, cs):
            u_dv = u_dv + c * (expm(-1.0j * TOTAL_TIME * (k * l_mat + h_mat)) @ u0)
        fid = abs(np.vdot(u_dv / np.linalg.norm(u_dv), u_exact)) ** 2
        row = {
            "case": label,
            "beta_opt": beta,
            "h1": h1,
            "K": big_k,
            "Q": q_order,
            "M_DV": m_dv,
            "m_c": math.ceil(math.log2(m_dv)),
            "c_l1": float(np.sum(np.abs(cs))),
            "one_minus_F_DV": 1.0 - fid,
            "L_norm": l_norm,
        }
        if best is None or row["one_minus_F_DV"] < best["one_minus_F_DV"]:
            best = row
    print(
        f"[{label}] beta={best['beta_opt']:.2f} h1={best['h1']:.5f} K={best['K']:.5f} "
        f"Q={best['Q']} M_DV={best['M_DV']} m_c={best['m_c']} "
        f"|c|1={best['c_l1']:.4f} 1-F_DV={best['one_minus_F_DV']:.3e}",
        flush=True,
    )
    return best


def tridiag(m_dim):
    lap = 2.0 * np.eye(m_dim)
    lap -= np.diag(np.ones(m_dim - 1), 1) + np.diag(np.ones(m_dim - 1), -1)
    return lap


def main():
    rows = []

    # Validation gate: M=4 heat rows must reproduce the published table.
    published = {
        "heat_m4_dirichlet": (0.60, 320, "2.334e-03"),
        "heat_m4_neumann": (0.90, 192, "2.260e-03"),
        "heat_m4_periodic": (0.80, 248, "2.777e-04"),
    }
    for boundary in ("dirichlet", "neumann", "periodic"):
        system = _heat_system(boundary, TOTAL_TIME)
        l_mat, h_mat = system_blocks(system)
        u0 = np.eye(4, dtype=complex)[:, 1]
        row = dv_case(f"heat_m4_{boundary}", l_mat, h_mat, u0)
        beta_ref, m_ref, inf_ref = published[row["case"]]
        if row["beta_opt"] != beta_ref or row["M_DV"] != m_ref:
            raise RuntimeError(f"validation failed for {row['case']}: {row}")
        if f"{row['one_minus_F_DV']:.3e}" != inf_ref:
            raise RuntimeError(
                f"validation infidelity mismatch for {row['case']}: "
                f"{row['one_minus_F_DV']:.3e} vs {inf_ref}"
            )
        row["validated_against_published"] = True
        rows.append(row)
    print("validation gate passed", flush=True)

    for num_qubits, label in ((3, "heat_m8_dirichlet"), (4, "heat_m16_dirichlet")):
        system = _heat_system("dirichlet", TOTAL_TIME, num_qubits=num_qubits)
        l_mat, h_mat = system_blocks(system)
        u0 = np.eye(l_mat.shape[0], dtype=complex)[:, 1]
        rows.append(dv_case(label, l_mat, h_mat, u0))

    for m_dim, label in ((8, "advdiff_m8"), (16, "advdiff_m16")):
        l_mat = tridiag(m_dim)
        dc = (np.diag(np.ones(m_dim - 1), 1) - np.diag(np.ones(m_dim - 1), -1)) / 2.0
        h_mat = -1.0j * dc
        u0 = np.eye(m_dim, dtype=complex)[:, 1]
        rows.append(dv_case(label, l_mat, h_mat, u0))

    lap2d = np.kron(tridiag(4), np.eye(4)) + np.kron(np.eye(4), tridiag(4))
    rows.append(
        dv_case("heat_2d_4x4", lap2d, np.zeros_like(lap2d), np.eye(16, dtype=complex)[:, 1])
    )

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    META.write_text(
        json.dumps(
            {
                "output": str(OUTPUT),
                "methodology": (
                    "identical to tab:dv_resource_compare: eta=1, eps=0.1, "
                    "beta scan [0.60, 0.95] step 0.05, composite Gauss-Legendre, "
                    "complex kernel weights, u0 = basis index 1, conditional "
                    "fidelity vs exp(-AT)u0; M=4 heat rows validated "
                    "digit-for-digit against the published table (beta and "
                    "M_DV exact, 1-F_DV at four significant digits) before "
                    "new rows are accepted"
                ),
                "betas": BETAS,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUTPUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
