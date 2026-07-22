#!/usr/bin/env python3
"""Recompute the relative-entropy non-Gaussianity values used in Table 3.

The script reads the retained Law-Eberly and replayable SNAP+D result folders.
It recomputes the target finite oscillator seed state from the saved kernel
parameters and reconstructs the SNAP+D prepared seed state from the serialized
payload, so no variational optimization is rerun.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from clean_core import (
    KernelSpec,
    StatePrepSpec,
    compute_lchs_coefficients,
    padded_seed_state,
)
from clean_oracles import oracle_from_snap_parameter_payload, prepare_cv_oracle


BOUNDARIES = ("dirichlet", "neumann", "periodic")


def relative_entropy_nongaussianity(state: Sequence[complex]) -> float:
    """Return the single-mode pure-state relative-entropy non-Gaussianity.

    The logarithm is natural, matching the manuscript formula. The input is a
    Fock-basis statevector for the unsqueezed seed state.
    """

    psi = np.asarray(state, dtype=complex).reshape(-1)
    norm = np.linalg.norm(psi)
    if norm <= 0.0:
        raise ValueError("state must have nonzero norm")
    psi = psi / norm

    n_fock = psi.size
    if n_fock == 1:
        return 0.0

    alpha = np.sum(
        np.conj(psi[:-1]) * psi[1:] * np.sqrt(np.arange(1, n_fock, dtype=float))
    )
    nbar = float(np.sum(np.arange(n_fock, dtype=float) * np.abs(psi) ** 2))

    if n_fock >= 3:
        m = np.sum(
            np.conj(psi[:-2])
            * psi[2:]
            * np.sqrt(
                np.arange(1, n_fock - 1, dtype=float)
                * np.arange(2, n_fock, dtype=float)
            )
        )
    else:
        m = 0.0j

    n_centered = nbar - abs(alpha) ** 2
    m_centered = m - alpha**2
    det_v = float((n_centered + 0.5) ** 2 - abs(m_centered) ** 2)

    if det_v < 0.25 and 0.25 - det_v < 1e-12:
        det_v = 0.25
    if det_v < 0.25:
        raise ValueError(f"unphysical covariance determinant {det_v}")

    nu = math.sqrt(det_v)
    nth = max(nu - 0.5, 0.0)
    if nth <= 1e-15:
        return 0.0
    return float((nth + 1.0) * math.log(nth + 1.0) - nth * math.log(nth))


def read_best_row(result_dir: Path) -> Mapping[str, Any]:
    summary_path = result_dir / "sweep_summary.json"
    if summary_path.exists():
        with summary_path.open() as handle:
            summary = json.load(handle)
        if "best_row" in summary:
            return summary["best_row"]

    csv_path = result_dir / "sweep_all.csv"
    with csv_path.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("valid", "")).lower() == "true"
        ]
    if not rows:
        raise ValueError(f"No valid rows found in {csv_path}")
    return max(
        rows,
        key=lambda row: float(row.get("score", row.get("fidelity", 0.0))),
    )


def build_target_state(
    row: Mapping[str, Any],
    *,
    n_fock: int,
    n_quad: int,
    coeff_backend: str,
) -> np.ndarray:
    kernel = KernelSpec(
        r_target=float(row["r_target"]),
        r_prime=float(row["r_prime"]),
        beta=float(row["beta"]),
        n_coeff=int(row["n_coeff"]),
        n_fock=n_fock,
        n_quad=n_quad,
        coeff_backend=coeff_backend,
    )
    coeffs = compute_lchs_coefficients(kernel)
    return padded_seed_state(coeffs, n_fock)


def compute_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    root = Path(args.root)

    for boundary in BOUNDARIES:
        le_dir = root / f"results_clean_law_eberly_{boundary}"
        snap_dir = root / f"results_clean_snap_recreate_{boundary}"
        le_row = read_best_row(le_dir)
        snap_row = read_best_row(snap_dir)

        target_state = build_target_state(
            le_row,
            n_fock=args.n_fock,
            n_quad=args.n_quad,
            coeff_backend=args.coeff_backend,
        )
        delta_target = relative_entropy_nongaussianity(target_state)

        snap_target_state = build_target_state(
            snap_row,
            n_fock=args.n_fock,
            n_quad=args.n_quad,
            coeff_backend=args.coeff_backend,
        )
        payload_raw = snap_row.get("snap_parameter_payload_json")
        if not payload_raw:
            raise ValueError(f"Missing SNAP+D payload in {snap_dir}")
        snap_payload = (
            json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
        )
        snap_oracle = oracle_from_snap_parameter_payload(
            snap_target_state,
            n_fock=args.n_fock,
            payload=snap_payload,
        )
        delta_snap = relative_entropy_nongaussianity(snap_oracle.prepared_state)

        rows.append(
            {
                "boundary_condition": boundary,
                "r_target": float(le_row["r_target"]),
                "r_prime": float(le_row["r_prime"]),
                "beta": float(le_row["beta"]),
                "n_coeff": int(le_row["n_coeff"]),
                "delta_ng_target_le": delta_target,
                "delta_ng_snap": delta_snap,
                "snap_oracle_fidelity": float(snap_oracle.oracle_fidelity),
            }
        )

    return rows


def compute_from_coefficients(path: Path) -> Dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    kernel = KernelSpec(**document["kernel"])
    coeffs = np.asarray(document["coefficients_re"], dtype=float) + 1.0j * np.asarray(
        document["coefficients_im"], dtype=float
    )
    oracle = prepare_cv_oracle(
        kernel, StatePrepSpec(method="law_eberly"), coeffs=coeffs
    )
    n_max = int(np.flatnonzero(coeffs)[-1])
    return {
        "boundary_condition": document.get("boundary"),
        "coefficient_artifact": str(path),
        "coefficient_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "coefficient_convention": "kernel_g_beta_code",
        "kernel": document["kernel"],
        "state_prep_method": "law_eberly",
        "delta_ng_target": relative_entropy_nongaussianity(oracle.target_state),
        "delta_ng_law_eberly": relative_entropy_nongaussianity(
            oracle.prepared_state
        ),
        "law_eberly_oracle_fidelity": float(oracle.oracle_fidelity),
        "n_max": n_max,
        "stellar_rank": n_max,
        "stellar_rank_equals_n_coeff_minus_one": n_max == kernel.n_coeff - 1,
        "highest_coefficient_abs": float(abs(coeffs[-1])),
    }


def write_csv(rows: Iterable[Mapping[str, Any]], path: Path | None) -> None:
    rows = list(rows)
    fieldnames = [
        "boundary_condition",
        "r_target",
        "r_prime",
        "beta",
        "n_coeff",
        "delta_ng_target_le",
        "delta_ng_snap",
        "snap_oracle_fidelity",
    ]
    if path is None:
        handle = None
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("w", newline="")
    try:
        writer = csv.DictWriter(handle or sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    finally:
        if handle is not None:
            handle.close()


def print_markdown(rows: Sequence[Mapping[str, Any]]) -> None:
    print("| Boundary | delta_nG target/LE | delta_nG SNAP+D | SNAP+D oracle fidelity |")
    print("| --- | ---: | ---: | ---: |")
    for row in rows:
        print(
            "| {boundary_condition} | {delta_ng_target_le:.6f} | {delta_ng_snap:.6f} | "
            "{snap_oracle_fidelity:.9f} |".format(**row)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute the six Table 3 non-Gaussianity diagnostics."
    )
    parser.add_argument("--root", default=Path(__file__).resolve().parent)
    parser.add_argument("--n-fock", type=int, default=64)
    parser.add_argument("--n-quad", type=int, default=240)
    parser.add_argument("--coeff-backend", default="explicit_overlap")
    parser.add_argument("--coefficients", type=Path)
    parser.add_argument(
        "--format",
        choices=("markdown", "csv", "json"),
        default="markdown",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.coefficients is not None:
        text = json.dumps(compute_from_coefficients(args.coefficients), indent=2)
        if args.output is None:
            print(text)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n")
        return
    rows = compute_rows(args)

    if args.format == "markdown":
        print_markdown(rows)
    elif args.format == "csv":
        write_csv(rows, args.output)
    else:
        text = json.dumps(rows, indent=2)
        if args.output is None:
            print(text)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
