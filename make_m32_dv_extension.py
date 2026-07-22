"""Build the two M=32 classical DV quadrature rows after the M=4 gate."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from clean_core import system_blocks
from make_dv_extension import TOTAL_TIME, dv_case, tridiag
from revision_eval import _heat_system

ROOT = Path(__file__).resolve().parent
OUTPUT = Path("results_revision/dv_extension_m32.csv")
META = Path("results_revision/dv_extension_m32.meta.json")


def _validation_gate() -> list[dict]:
    expected = {
        "heat_m4_dirichlet": (0.60, 320, "2.334224e-03", "0.9357"),
        "heat_m4_neumann": (0.90, 192, "2.260403e-03", "1.2073"),
        "heat_m4_periodic": (0.80, 248, "2.777025e-04", "1.0740"),
    }
    rows = []
    for boundary in ("dirichlet", "neumann", "periodic"):
        l_mat, h_mat = system_blocks(_heat_system(boundary, TOTAL_TIME))
        row = dv_case(
            f"heat_m4_{boundary}",
            l_mat,
            h_mat,
            np.eye(4, dtype=complex)[:, 1],
        )
        beta, m_dv, infidelity, c_l1 = expected[row["case"]]
        actual = (
            row["beta_opt"],
            row["M_DV"],
            f"{row['one_minus_F_DV']:.6e}",
            f"{row['c_l1']:.4f}",
        )
        if actual != (beta, m_dv, infidelity, c_l1):
            raise RuntimeError(f"M=4 DV validation failed for {row['case']}: {actual}")
        rows.append(row)
    print("M=4 DV quadrature validation gate passed", flush=True)
    return rows


def _json_row(row: dict) -> dict:
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in row.items()
    }


def main() -> None:
    validation = _validation_gate()
    lap = tridiag(32)
    central_difference = (
        np.diag(np.ones(31), 1) - np.diag(np.ones(31), -1)
    ) / 2.0
    u0 = np.eye(32, dtype=complex)[:, 1]
    rows = [
        dv_case("heat_m32_dirichlet", lap, np.zeros_like(lap), u0),
        dv_case("advdiff_m32", lap, -1.0j * central_difference, u0),
    ]
    for row in rows:
        if not 350 <= row["M_DV"] <= 370:
            raise RuntimeError(f"{row['case']} M_DV outside [350, 370]: {row['M_DV']}")
        if row["m_c"] != 9:
            raise RuntimeError(f"{row['case']} m_c != 9: {row['m_c']}")

    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    META.write_text(
        json.dumps(
            {
                "output": str(OUTPUT),
                "source_sha256": hashlib.sha256(
                    (ROOT / "make_m32_dv_extension.py").read_bytes()
                ).hexdigest(),
                "m4_validation": [_json_row(row) for row in validation],
                "methodology": (
                    "epsilon=0.1, eta=1, beta=0.60..0.95 step 0.05, "
                    "complex kernel weights, u0=basis index 1"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUTPUT} and {META}", flush=True)


if __name__ == "__main__":
    main()
