#!/usr/bin/env python3
"""Compute the ordinary-Fock tail of the prepared kernel state.

The Parameter Selection paragraph quotes ordinary-Fock tail masses for the
two squeezed vacua only.  This script supplies the matching numbers for the
prepared kernel state itself at the shared operating point:

* the tail mass ||(I - Pi_{N_Fock}) S(r') |chi_N>|| of the ideal (untruncated)
  squeezed kernel state beyond the embedding cutoff, and
* the infidelity between that ideal state and the finite construction of
  Eq. (finite-target-states), i.e. the state obtained by truncating the
  squeezing generator to N_Fock dimensions before exponentiation.

The ideal state is represented in a large dense space (D=512) whose adequacy
is verified by recomputing at D=768.  Coefficients C_n are loaded from the
frozen shared-point artifact.

Output: results_joint_tradeoff/kernel_tail.json (values + checks + sidecar).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parent
COEFF_PATH = (
    ROOT
    / "results_revision_v2"
    / "coefficients_dirichlet_r1p6_rp0p25_b0p5_N32_nq240.json"
)
OUTPUT_PATH = ROOT / "results_joint_tradeoff" / "kernel_tail.json"

R_PRIME = 0.25
N_COEFF = 32
N_FOCK = 64
DIM_MAIN = 512
DIM_CHECK = 768


def squeeze_operator(dim: int, r: float) -> np.ndarray:
    """Return exp[(r/2)(a^dag^2 - a^2)] on a dim-dimensional Fock space."""

    ladder = np.diag(np.sqrt(np.arange(1, dim)), 1)
    generator = 0.5 * r * (ladder.T @ ladder.T - ladder @ ladder)
    return expm(generator)


def kernel_state(dim: int, coeffs: np.ndarray) -> np.ndarray:
    """Return S(r')|chi_N> in a dim-dimensional Fock space (normalized)."""

    chi = np.zeros(dim, dtype=complex)
    chi[: len(coeffs)] = coeffs
    state = squeeze_operator(dim, R_PRIME) @ chi
    return state / np.linalg.norm(state)


def tail_and_infidelity(dim: int, coeffs: np.ndarray) -> Dict[str, float]:
    """Compute the beyond-N_Fock tail and the finite-construction infidelity."""

    ideal = kernel_state(dim, coeffs)
    tail = float(np.linalg.norm(ideal[N_FOCK:]))

    finite = kernel_state(N_FOCK, coeffs)
    overlap = abs(np.vdot(ideal[:N_FOCK], finite))
    infidelity = float(1.0 - overlap**2)
    return {"tail_mass_beyond_n_fock": tail, "finite_construction_infidelity": infidelity}


def main() -> None:
    with open(COEFF_PATH) as fh:
        stored = json.load(fh)
    coeffs = np.array(stored["coefficients_re"]) + 1j * np.array(
        stored["coefficients_im"]
    )
    assert len(coeffs) == N_COEFF

    main_vals = tail_and_infidelity(DIM_MAIN, coeffs)
    check_vals = tail_and_infidelity(DIM_CHECK, coeffs)
    dim_gap = {
        key: abs(main_vals[key] - check_vals[key]) for key in main_vals
    }

    result = {
        "kernel": {"r_prime": R_PRIME, "n_coeff": N_COEFF, "n_fock": N_FOCK},
        "coefficient_artifact": {
            "path": str(COEFF_PATH.relative_to(ROOT)),
            "sha256": hashlib.sha256(COEFF_PATH.read_bytes()).hexdigest(),
        },
        "values": main_vals,
        "checks": {
            "dense_dimension": DIM_MAIN,
            "check_dimension": DIM_CHECK,
            "dimension_convergence_gap": dim_gap,
        },
        "sidecar": {
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
            ).stdout.strip(),
            "command": " ".join(sys.argv),
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "source_sha256": {
                "make_kernel_tail.py": hashlib.sha256(
                    (ROOT / "make_kernel_tail.py").read_bytes()
                ).hexdigest()
            },
            "versions": {"python": sys.version.split()[0], "numpy": np.__version__},
        },
    }

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUTPUT_PATH, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"tail mass beyond N_Fock={N_FOCK}     : {main_vals['tail_mass_beyond_n_fock']:.6e}")
    print(f"finite-construction infidelity   : {main_vals['finite_construction_infidelity']:.6e}")
    print(f"dimension-convergence gaps       : {dim_gap}")
    print(f"-> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
