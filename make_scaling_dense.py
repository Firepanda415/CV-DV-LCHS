"""Dense scaling data for the scaling-study figure (visualization sweep).

Produces results_revision/viz_dense/scaling_dense.csv with two row families at
the Dirichlet-selected kernel point:
- model rows: exact-truncated-map evaluation over T in an 8-point grid for the
  three cutoff pairs (N, N_Fock) in {(12,16), (24,32), (48,64)}, each with its
  raw map/reference/alpha frozen to a per-row npz (raw_artifact column);
- circuit rows: circuit-level evaluation at T=1 over an 8-point n_t ladder for
  the same cutoff pairs, with the dense product-formula wiring crosscheck.

This is a descriptive densification of the frozen results_revision/scaling.csv
(whose T in {0.5,1,2} and n_t in {25,50,100} points are a subgrid and are
crosschecked for agreement); it changes no selection and no gate evidence.
"""

import csv
import hashlib
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clean_core import EvolutionSpec, StatePrepSpec, basis_state
from revision_eval import (
    _coefficients_for_finite,
    _heat_system,
    _kernel_fields,
    _map_case_row,
    _product_formula_column,
    _selected_kernel,
    evaluate_exact_map_row,
    prepare_cv_oracle,
    reconstruct_circuit_map,
)

OUT_DIR = Path("results_revision/viz_dense")
OUTPUT = OUT_DIR / "scaling_dense.csv"
META = OUT_DIR / "scaling_dense.meta.json"

T_VALUES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
NT_VALUES = [25, 35, 50, 71, 100, 141, 200, 283]
CUTOFF_PAIRS = [(12, 16), (24, 32), (48, 64)]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    args = types.SimpleNamespace(coeff_backend="explicit_overlap", n_quad=240)
    prep = StatePrepSpec(method="law_eberly")
    rows = []

    for n_coeff, n_fock in CUTOFF_PAIRS:
        kernel = _selected_kernel("dirichlet", n_coeff=n_coeff, n_fock=n_fock)
        kwargs = {
            "r_target": kernel.r_target,
            "r_prime": kernel.r_prime,
            "beta": kernel.beta,
            "n_coeff": kernel.n_coeff,
            "n_fock": kernel.n_fock,
            "n_quad": kernel.n_quad,
        }
        for total_time in T_VALUES:
            row, raw = evaluate_exact_map_row(
                kwargs, "dirichlet", total_time, return_raw=True
            )
            per_input = json.loads(row["per_input"])
            row["case"] = "scaling_dense_model"
            row["n_trotter_steps"] = ""
            # benchmark input u(0)=|01> is basis index 1
            row["p_succ"] = per_input[1]["success_probability"]
            raw_path = (
                OUT_DIR / f"scaling_dense_T{total_time}_N{n_coeff}_NF{n_fock}_exact_raw.npz"
            )
            np.savez(raw_path, **raw)
            row["raw_artifact"] = str(raw_path)
            rows.append(row)
            print(
                f"[model] N={n_coeff} NF={n_fock} T={total_time}: "
                f"epsF={row['rel_frobenius_error']:.6f} p={row['p_succ']:.5f}",
                flush=True,
            )

        coeffs, coefficient_path = _coefficients_for_finite("dirichlet", kernel, args)
        oracle = prepare_cv_oracle(kernel, prep, coeffs=coeffs)
        system = _heat_system("dirichlet", 1.0)
        for n_t in NT_VALUES:
            evolution = EvolutionSpec(n_trotter_steps=n_t)
            circuit_map = reconstruct_circuit_map(
                system, kernel, prep, evolution, coeffs, oracle
            )
            pf_column = _product_formula_column(
                replace(system, init_state=basis_state(4, 0)),
                kernel,
                oracle.prepared_state,
                evolution,
                basis_state(4, 0),
            )
            wiring = float(
                np.linalg.norm(circuit_map[:, 0] - pf_column)
                / max(np.linalg.norm(pf_column), 1e-15)
            )
            if wiring > 1e-8:
                raise RuntimeError(f"wiring crosscheck failed: {wiring:.3e}")
            candidate_id = f"T1.0_N{n_coeff}_NF{n_fock}_nt{n_t}"
            row = _map_case_row(
                case="scaling_dense_circuit",
                system=system,
                kernel=kernel,
                coeffs=coeffs,
                oracle=oracle,
                evolution=evolution,
                evaluation=["exact_finite_map", "circuit_statevector"],
                coefficient_path=coefficient_path,
                raw_path=OUT_DIR / f"scaling_dense_{candidate_id}_raw.npz",
                circuit_map=circuit_map,
                compile_resources=False,
                extra_fields={
                    "candidate_id": candidate_id,
                    "boundary": "dirichlet",
                    "wiring_crosscheck_relative_error": wiring,
                },
            )
            rows.append(row)
            print(
                f"[circuit] N={n_coeff} NF={n_fock} nt={n_t}: "
                f"epsF={row['rel_frobenius_error']:.6f} eps_t={row['eps_t']:.4e}",
                flush=True,
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
                    "post-hoc dense VISUALIZATION sweep for the scaling figure; "
                    "not gate evidence. Frozen scaling.csv subgrid points are "
                    "crosschecked separately."
                ),
                "t_values": T_VALUES,
                "nt_values": NT_VALUES,
                "cutoff_pairs": CUTOFF_PAIRS,
                "exact_map_raw_artifacts": (
                    "scaling_dense_T{T}_N{N}_NF{NF}_exact_raw.npz per model row "
                    "(observed_vectors, reference_map, input_vectors, alpha_target)"
                ),
                "selected_params_sha256": hashlib.sha256(
                    Path("results_revision/selected_params.json").read_bytes()
                ).hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUTPUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
