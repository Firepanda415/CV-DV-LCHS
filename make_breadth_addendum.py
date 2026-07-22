"""Breadth addendum at the shared operating point (LE preparation only).

Three additional circuit-level cases requested after the v2 rerun:
- d16_1d_heat_circuit: the M=16 1D Dirichlet heat instance at full circuit
  level (the v2 breadth table carries it at model level only), so the measured
  eps_t can be contrasted with the vacuous first-order bound.
- advdiff_d8 / advdiff_d16: 1D advection-diffusion u_t = nu*u_xx - c*u_x with
  Dirichlet boundaries, central differences, nu = c = h = 1 (mesh Peclet 0.5)
  at M = 8 and M = 16. The Hermitian split is L = nu*T (positive definite) and
  H = -i*c*Dc; [L, H] != 0 and the generator is genuinely non-normal, with
  eigenvector condition number kappa(V) ~ 48 (M=8) and ~ 4.0e3 (M=16). Both
  kappa(V) and the Henrici departure from normality are recorded per row.

Run from the v2 workdir. Writes results_revision/breadth_addendum.csv plus one
raw npz per case.
"""

import csv
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clean_core import EvolutionSpec, StatePrepSpec, basis_state
from revision_eval import (
    _coefficients_for_finite,
    _heat_system,
    _map_case_row,
    _product_formula_column,
    _selected_kernel,
    build_pauli_system,
    decompose_matrix_to_pauli_terms,
    prepare_cv_oracle,
    reconstruct_circuit_map,
)

OUT_DIR = Path("results_revision")
OUTPUT = OUT_DIR / "breadth_addendum.csv"
META = OUT_DIR / "breadth_addendum.meta.json"

NU = 1.0
C_ADV = 1.0


def advdiff_matrices(m_dim):
    lap = 2.0 * np.eye(m_dim)
    lap -= np.diag(np.ones(m_dim - 1), 1) + np.diag(np.ones(m_dim - 1), -1)
    dc = (np.diag(np.ones(m_dim - 1), 1) - np.diag(np.ones(m_dim - 1), -1)) / 2.0
    l_mat = NU * lap
    h_mat = -1.0j * C_ADV * dc
    return l_mat, h_mat


def nonnormality(l_mat, h_mat):
    a_mat = l_mat + 1.0j * h_mat
    eigenvalues, vectors = np.linalg.eig(a_mat)
    kappa = float(np.linalg.cond(vectors))
    fro2 = np.linalg.norm(a_mat, "fro") ** 2
    dep = float(
        np.sqrt(max(fro2 - np.sum(np.abs(eigenvalues) ** 2), 0.0)) / np.sqrt(fro2)
    )
    return kappa, dep


def check_decomposition(terms, matrix, label):
    from clean_core import pauli_sum_matrix

    rebuilt = pauli_sum_matrix(tuple(terms))
    err = np.linalg.norm(rebuilt - matrix)
    if err > 1e-10:
        raise RuntimeError(f"{label} Pauli decomposition mismatch: {err:.3e}")


def run_case(case, system, kernel, coeffs, oracle, evolution, coefficient_path, extra):
    start = time.time()
    circuit_map = reconstruct_circuit_map(
        system, kernel, StatePrepSpec(method="law_eberly"), evolution, coeffs, oracle
    )
    pf_column = _product_formula_column(
        replace(system, init_state=basis_state(system.dv_dim, 0)),
        kernel,
        oracle.prepared_state,
        evolution,
        basis_state(system.dv_dim, 0),
    )
    wiring = float(
        np.linalg.norm(circuit_map[:, 0] - pf_column)
        / max(np.linalg.norm(pf_column), 1e-15)
    )
    if wiring > 1e-8:
        raise RuntimeError(f"{case} wiring crosscheck failed: {wiring:.3e}")
    row = _map_case_row(
        case=case,
        system=system,
        kernel=kernel,
        coeffs=coeffs,
        oracle=oracle,
        evolution=evolution,
        evaluation=["exact_finite_map", "circuit_statevector"],
        coefficient_path=coefficient_path,
        raw_path=OUT_DIR / f"breadth_{case}_raw.npz",
        circuit_map=circuit_map,
        extra_raw={"product_formula_column_0": pf_column},
        extra_fields={"wiring_crosscheck_relative_error": wiring, **extra},
    )
    print(
        f"[{case}] epsF={float(row['rel_frobenius_error'])*100:.3f}% "
        f"worstF={float(row['worst_conditional_fidelity']):.6f} "
        f"eps_t={row['eps_t']} ({time.time() - start:.0f}s)",
        flush=True,
    )
    return row


def main():
    prep = StatePrepSpec(method="law_eberly")
    evolution = EvolutionSpec(n_trotter_steps=100)
    kernel = _selected_kernel("dirichlet")
    import types

    args = types.SimpleNamespace(coeff_backend="explicit_overlap", n_quad=240)
    coeffs, coefficient_path = _coefficients_for_finite("dirichlet", kernel, args)
    oracle = prepare_cv_oracle(kernel, prep, coeffs=coeffs)
    rows = []

    d16 = _heat_system("dirichlet", 1.0, num_qubits=4)
    rows.append(
        run_case(
            "d16_1d_heat_circuit",
            d16,
            kernel,
            coeffs,
            oracle,
            evolution,
            coefficient_path,
            {"boundary": "dirichlet"},
        )
    )

    for m_dim, label in ((8, "advdiff_d8"), (16, "advdiff_d16")):
        l_mat, h_mat = advdiff_matrices(m_dim)
        l_terms = decompose_matrix_to_pauli_terms(l_mat)
        h_terms = decompose_matrix_to_pauli_terms(h_mat)
        check_decomposition(l_terms, l_mat, f"{label} L")
        check_decomposition(h_terms, h_mat, f"{label} H")
        kappa, henrici = nonnormality(l_mat, h_mat)
        system = build_pauli_system(
            l_terms=l_terms,
            h_terms=h_terms,
            total_time=1.0,
            init_state=np.eye(m_dim, dtype=complex)[:, 1],
            label=f"qst_{label}",
        )
        rows.append(
            run_case(
                label,
                system,
                kernel,
                coeffs,
                oracle,
                evolution,
                coefficient_path,
                {
                    "boundary": "advdiff_dirichlet",
                    "advection_nu": NU,
                    "advection_c": C_ADV,
                    "mesh_peclet": C_ADV / (2.0 * NU),
                    "kappa_eigenvector": kappa,
                    "henrici_departure": henrici,
                },
            )
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
                "note": (
                    "breadth addendum at the shared operating point, LE only: "
                    "M=16 1D heat at circuit level; advection-diffusion "
                    "(Dirichlet, central differences, nu=c=h=1, mesh Peclet "
                    "0.5) at M=8 and M=16 with kappa(V) and Henrici departure "
                    "recorded."
                ),
                "coefficient_artifact": str(coefficient_path),
                "coefficient_sha256": hashlib.sha256(
                    Path(coefficient_path).read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUTPUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
