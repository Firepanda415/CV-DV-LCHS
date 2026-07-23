"""Evaluate the map-level L1 kernel error of the implemented finite-r construction.

Computes ||g_{N,r} - g||_{L1(R)} at the shared operating point
(r, r', beta, N) = (1.6, 0.25, 0.5, 32), where

    g_{N,r}(k) = alpha_{N,r} * phi_r(k) * psi_{N,r}(k)

is the implemented finite kernel on the code (hbar = 1) axis,
psi_{N,r} = sum_n C_n chi_n is the normalized finite squeezed-Fock kernel
state reconstructed from the recorded coefficients, phi_r is the
postselection squeezed vacuum, and alpha_{N,r} = 1/<phi_r|psi_{N,r}>.
The L1 distance is invariant under the sqrt(2) axis conversion between the
code and paper conventions because both g_{N,r} and g transform as densities.

Cross-checks enforced before the result is written:
  * the three per-boundary coefficient files hold the identical state,
  * sum_n |C_n|^2 = 1,
  * alpha_{N,r} matches the recorded landscape value at the operating point,
  * <n> and the ordinary-Fock tail norm beyond N_Fock = 64 match the
    values reported in the manuscript (1.61 and 3.75e-4).

Output: results_revision_v2/finite_r_l1.json (+ .meta.json sidecar).
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import clean_core as cc  # noqa: E402

COEFF_FILES = [
    REPO / "results_revision_v2" / f"coefficients_{b}_r1p6_rp0p25_b0p5_N32_nq240.json"
    for b in ("dirichlet", "neumann", "periodic")
]
LANDSCAPE_EXT = REPO / "results_revision" / "viz_dense" / "landscape_dense_ext.csv"
OUT = REPO / "results_revision_v2" / "finite_r_l1.json"
META = REPO / "results_revision_v2" / "finite_r_l1.meta.json"

K_INNER_MAX = 50.0
K_INNER_STEP = 1.0e-3
K_TAIL_MAX = 3000.0
K_TAIL_STEP = 2.0e-2
N_FOCK = 64


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hermite_functions(y: np.ndarray, count: int) -> np.ndarray:
    """Normalized hbar = 1 Hermite functions h_0 .. h_{count-1} by stable recurrence."""
    out = np.empty((count, y.size))
    out[0] = np.pi ** -0.25 * np.exp(-0.5 * y * y)
    if count > 1:
        out[1] = np.sqrt(2.0) * y * out[0]
    for m in range(1, count - 1):
        out[m + 1] = y * np.sqrt(2.0 / (m + 1)) * out[m] - np.sqrt(m / (m + 1)) * out[m - 1]
    return out


def main() -> None:
    payloads = [json.loads(p.read_text()) for p in COEFF_FILES]
    coeff_sets = [
        np.array(d["coefficients_re"]) + 1j * np.array(d["coefficients_im"]) for d in payloads
    ]
    for other in coeff_sets[1:]:
        assert np.array_equal(coeff_sets[0], other), "per-boundary coefficient files differ"
    coeffs = coeff_sets[0]
    kernel = payloads[0]["kernel"]
    r_target, r_prime = float(kernel["r_target"]), float(kernel["r_prime"])
    beta, n_coeff = float(kernel["beta"]), int(kernel["n_coeff"])
    assert abs(np.sum(np.abs(coeffs) ** 2) - 1.0) < 1e-10, "coefficients not normalized"

    sigma, sigma_prime = np.exp(r_target), np.exp(r_prime)
    k = np.arange(-K_INNER_MAX, K_INNER_MAX, K_INNER_STEP)

    chi = hermite_functions(k / sigma_prime, n_coeff) / np.sqrt(sigma_prime)
    psi = (coeffs[:, None] * chi).sum(axis=0)
    psi /= np.sqrt(np.trapezoid(np.abs(psi) ** 2, k))
    phi = np.pi ** -0.25 * sigma ** -0.5 * np.exp(-(k ** 2) / (2.0 * sigma ** 2))
    overlap = np.trapezoid(phi * psi, k)
    alpha = 1.0 / overlap

    with LANDSCAPE_EXT.open() as handle:
        recorded_alpha = next(
            complex(float(row["alpha_target_re"]), float(row["alpha_target_im"]))
            for row in csv.DictReader(handle)
            if row["boundary"] == "dirichlet"
            and float(row["r_target"]) == r_target
            and float(row["r_prime"]) == r_prime
            and float(row["beta"]) == beta
            and int(row["n_coeff"]) == n_coeff
        )
    assert abs(alpha - recorded_alpha) < 1e-5, "alpha mismatch with recorded landscape value"

    dpsi = np.gradient(psi, k)
    n_mean = (
        0.5 * np.trapezoid(k ** 2 * np.abs(psi) ** 2, k)
        + 0.5 * np.trapezoid(np.abs(dpsi) ** 2, k)
        - 0.5
    )
    assert abs(n_mean - 1.61) < 0.01, "mean photon number mismatch with manuscript value"

    fock = hermite_functions(k, 256)
    amplitudes = np.trapezoid(fock * psi[None, :], k, axis=1)
    tail_norm = float(np.sqrt(max(1.0 - np.sum(np.abs(amplitudes[:N_FOCK]) ** 2), 0.0)))
    assert abs(tail_norm - 3.75e-4) < 5e-6, "ordinary-Fock tail norm mismatch"

    g_inner = cc.kernel_g_beta_code(k, beta)
    g_impl = alpha * phi * psi
    l1_inner = float(np.trapezoid(np.abs(g_impl - g_inner), k))
    k_tail = np.arange(K_INNER_MAX, K_TAIL_MAX, K_TAIL_STEP)
    g_tail = float(2.0 * np.trapezoid(np.abs(cc.kernel_g_beta_code(k_tail, beta)), k_tail))
    l1_total = l1_inner + g_tail
    g_l1 = float(np.trapezoid(np.abs(g_inner), k)) + g_tail

    result = {
        "kernel": kernel,
        "alpha_N_r": [alpha.real, alpha.imag],
        "l1_error_total": l1_total,
        "l1_error_inner": l1_inner,
        "g_abs_tail_beyond_inner_grid": g_tail,
        "g_l1_norm": g_l1,
        "relative_l1_deviation": l1_total / g_l1,
        "checks": {
            "coefficient_norm": float(np.sum(np.abs(coeffs) ** 2)),
            "recorded_alpha": [recorded_alpha.real, recorded_alpha.imag],
            "mean_photon_number": float(n_mean),
            "fock_tail_norm_beyond_64": tail_norm,
            "int_g_impl": [
                float(np.trapezoid(g_impl.real, k)),
                float(np.trapezoid(g_impl.imag, k)),
            ],
        },
        "grid": {
            "inner_half_width": K_INNER_MAX,
            "inner_step": K_INNER_STEP,
            "tail_max": K_TAIL_MAX,
            "tail_step": K_TAIL_STEP,
        },
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")

    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    meta = {
        "output": str(OUT.relative_to(REPO)),
        "source_sha256": sha256(Path(__file__)),
        "input_sha256": {str(p.relative_to(REPO)): sha256(p) for p in COEFF_FILES},
        "landscape_ext_sha256": sha256(LANDSCAPE_EXT),
        "git_head": git_head,
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "methodology": (
            "||g_{N,r}-g||_{L1} on the code (hbar=1) axis with "
            "g_{N,r}=alpha*phi_r*psi_{N,r} reconstructed from the recorded normalized "
            "squeezed-Fock coefficients, g from kernel_g_beta_code, trapezoid rule on "
            "|k|<=50 at dk=1e-3 plus the |g| tail integrated to |k|=3000; invariant "
            "under the sqrt(2) code-to-paper axis conversion since both integrands "
            "transform as densities"
        ),
    }
    META.write_text(json.dumps(meta, indent=1) + "\n")

    print(f"alpha_N_r          = {alpha.real:.9f}{alpha.imag:+.2e}j (recorded {recorded_alpha.real:.9f})")
    print(f"<n>                = {n_mean:.4f}")
    print(f"tail norm (>=64)   = {tail_norm:.3e}")
    print(f"L1 inner / tail    = {l1_inner:.6f} / {g_tail:.3e}")
    print(f"L1 TOTAL           = {l1_total:.6f}")
    print(f"||g||_L1           = {g_l1:.6f}   relative deviation = {l1_total / g_l1:.4f}")
    print(f"wrote {OUT.name}, {META.name}")


if __name__ == "__main__":
    main()
