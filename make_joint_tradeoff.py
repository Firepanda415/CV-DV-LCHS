#!/usr/bin/env python3
"""Build the joint cutoff--depth tradeoff figure for the Referee 1A response.

This script evaluates the coefficient-truncation tail of the shared kernel
state numerically and combines it with the cutoff-dependent product-formula
bound of Theorem 3 (normalized prefactors) to produce the joint resource
proxy

    R(N; eps) = N * n_t(N; eps),
    n_t(N; eps) = ceil( [ Lambda_p(kappa N) / (eps - eps_tr(N)) ]^{1/p} ),

with the embedding ratio kappa in {2, 4/3} linking the coefficient cutoff N
to the ordinary-Fock cutoff N_Fock = ceil(kappa N).  All theorem-dependent
prefactors (c_p, t^{p+1}, and the nested-commutator sums Gamma) are set to
one, so the figure quantifies the trend of the analytical bound rather than
an absolute hardware gate count.

The truncation tail eps_tr(N) is the object controlled by Theorem 2: the
orthogonal squeezed-Fock coefficient projection of the ideal normalized
kernel state psi_inf = N g.  Its coefficients are

    a_n = Int pref_n H_n(k/sigma') g_code(k) exp(-k^2/(2 sigma'^2)) dk,

i.e. the Gaussian in the integrand is the one carried by the basis
function itself.  This series is L2-convergent (sum |a_n|^2 = ||psi_inf||^2)
and its integrand has a non-growing envelope, so plain float64 Simpson
quadrature is accurate.  eps_tr(N) = sqrt(sum_{n>=N} |a_n|^2).

This must not be confused with the finite-r coefficient-generating pairing
used by the numerical pipeline (``compute_lchs_coefficients_explicit``),
whose integrand carries exp(-gamma k^2) with
gamma = (exp(-2r') - exp(-2r))/2 < 1/(2 sigma'^2).  That pairing
corresponds to the non-normalizable formal state g/phi_r^* (the kernel
divided by the postselection Gaussian), whose coefficient series is NOT
square-summable: the coefficients decay only up to n of order 70 and then
grow again.  The manuscript already separates these two objects (the ideal
projection theorem does not control the finite-r pairing); the truncated,
renormalized N=32 pipeline state is finite and well defined, but its
N -> infinity coefficient limit does not exist, so the Theorem-2 projection
is the correct object for a truncation-error budget.

Numerical care: the integrand of a_n oscillates over the full classically
allowed region |k| < sigma' sqrt(2n) (k up to ~27 for n ~ 224), so the
quadrature domain must cover it entirely, and the normalized Hermite
recurrence psi_n = H_n / sqrt(2^n n!) is used because SciPy's
``eval_hermite`` overflows beyond n of roughly 150.  Richardson
extrapolation over two Simpson grids removes the leading discretization
error.

Validation: (i) an engine gate recomputes the finite-r pairing
coefficients with the same machinery and matches the archived shared-point
coefficients digit-for-digit at n < 32; (ii) the h -> 2h Richardson gap;
(iii) independent adaptive-quadrature spot checks of a_n at selected n;
(iv) the integrand magnitude at the domain edge.

Outputs (results_joint_tradeoff/):
    epsilon_tail.csv              eps_tr(N) for N = 2..128
    joint_tradeoff.csv            per-(kappa, eps, N) rows with n_t and R
    coefficients_shared_N224.npz  raw coefficients (unnormalized and normalized)
    joint_tradeoff_meta.json      parameters, conventions, and checks
Figure:
    figures/joint_cutoff_depth.pdf
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from clean_core import gamma_hbar1, kernel_g_beta_code

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results_joint_tradeoff"
FIGURE_PATH = ROOT / "figures" / "joint_cutoff_depth.pdf"
ARCHIVED_COEFFS = (
    ROOT
    / "results_revision_v2"
    / "coefficients_dirichlet_r1p6_rp0p25_b0p5_N32_nq240.json"
)

R_TARGET = 1.6
R_PRIME = 0.25
BETA = 0.5
N_MAX = 224
N_GRID = np.arange(2, 201)
TROTTER_ORDER = 2
KAPPAS: Tuple[Tuple[str, float], ...] = (("2", 2.0), ("4/3", 4.0 / 3.0))
EPSILONS = (1e-1, 5e-2, 3e-2, 1e-2, 5e-3)

K_MAX = 33.0
K_STEP = 5e-4

SIGMA_PRIME = float(np.exp(R_PRIME))
GAMMA = gamma_hbar1(R_TARGET, R_PRIME)


def _uniform_grid(half_width: float, step: float) -> np.ndarray:
    """Return a symmetric uniform grid with an even interval count."""

    n_intervals = int(round(2.0 * half_width / step))
    if n_intervals % 2 == 1:
        n_intervals += 1
    return np.linspace(-half_width, half_width, n_intervals + 1)


def _simpson_weights(n_points: int, step: float) -> np.ndarray:
    """Return composite Simpson weights for an odd number of points."""

    weights = np.ones(n_points)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return weights * (step / 3.0)


def raw_coefficients(
    step: float,
    gaussian_exponent: float,
    n_max: int = N_MAX,
    k_max: float = K_MAX,
) -> np.ndarray:
    """Evaluate unnormalized squeezed-Fock overlap coefficients.

    The integrand is pref_n H_n(k/sigma') g_code(k) exp(-q k^2) with
    pref_n = (2^n n! sqrt(pi) sigma')^{-1/2} and q = ``gaussian_exponent``.
    q = 1/(2 sigma'^2) gives the Theorem-2 projection coefficients of the
    ideal normalized kernel state; q = gamma gives the finite-r pairing
    used by the pipeline (engine-gate comparison only).  The integrand is
    assembled from the normalized Hermite recurrence psi_n = H_n /
    sqrt(2^n n!), which keeps every intermediate within float64 range on
    |k| <= K_MAX for n < N_MAX.

    Args:
        step: Uniform grid spacing for the composite Simpson rule.
        gaussian_exponent: Coefficient q of the Gaussian exp(-q k^2).
        n_max: Number of coefficients to return.
        k_max: Half-width of the quadrature domain; must cover the full
            oscillation region sigma' sqrt(2 n_max) plus a decay margin.

    Returns:
        Complex coefficient vector of length ``n_max`` (not normalized).
    """

    k_grid = _uniform_grid(k_max, step)
    weights = _simpson_weights(len(k_grid), k_grid[1] - k_grid[0])
    y = k_grid / SIGMA_PRIME
    envelope = (
        kernel_g_beta_code(k_grid, BETA)
        * np.exp(-gaussian_exponent * k_grid**2)
        * weights
        / np.sqrt(np.sqrt(np.pi) * SIGMA_PRIME)
    )

    coeffs = np.zeros(n_max, dtype=complex)
    psi_prev = np.zeros_like(y)
    psi_curr = np.ones_like(y)
    for n in range(n_max):
        coeffs[n] = np.dot(psi_curr, envelope)
        psi_next = np.sqrt(2.0 / (n + 1.0)) * y * psi_curr - np.sqrt(
            n / (n + 1.0)
        ) * psi_prev
        psi_prev, psi_curr = psi_curr, psi_next
    return coeffs


PROJECTION_EXPONENT = 0.5 / SIGMA_PRIME**2


def quad_spot_checks(indices: Tuple[int, ...]) -> Dict[int, float]:
    """Recompute projection coefficients with adaptive quadrature.

    The projection integrand has a non-growing envelope, so SciPy's
    adaptive rule is reliable here; each index is integrated piecewise
    between Hermite oscillation scales.

    Returns:
        Map from index n to the recomputed complex coefficient.
    """

    from scipy.integrate import quad

    def _psi_scalar(n: int, y: float) -> float:
        psi_prev, psi_curr = 0.0, 1.0
        for m in range(n):
            psi_prev, psi_curr = (
                psi_curr,
                np.sqrt(2.0 / (m + 1.0)) * y * psi_curr
                - np.sqrt(m / (m + 1.0)) * psi_prev,
            )
        return psi_curr

    results: Dict[int, complex] = {}
    for n in indices:
        pref = 1.0 / np.sqrt(np.sqrt(np.pi) * SIGMA_PRIME)

        def _f(k: float, part: str) -> float:
            val = (
                pref
                * _psi_scalar(n, k / SIGMA_PRIME)
                * kernel_g_beta_code(np.array([k]), BETA)[0]
                * np.exp(-PROJECTION_EXPONENT * k * k)
            )
            return float(np.real(val) if part == "re" else np.imag(val))

        edges = np.linspace(-K_MAX, K_MAX, max(4 * n, 32))
        total = 0.0 + 0.0j
        for left, right in zip(edges[:-1], edges[1:]):
            re = quad(lambda k: _f(k, "re"), left, right, limit=200)[0]
            im = quad(lambda k: _f(k, "im"), left, right, limit=200)[0]
            total += re + 1j * im
        results[n] = total
    return results


def edge_integrand_magnitude(gaussian_exponent: float, k_max: float = K_MAX) -> float:
    """Return max_n |integrand(k_max)| to confirm the domain suffices."""

    envelope = abs(
        kernel_g_beta_code(np.array([k_max]), BETA)[0]
        * np.exp(-gaussian_exponent * k_max**2)
        / np.sqrt(np.sqrt(np.pi) * SIGMA_PRIME)
    )
    y_edge = k_max / SIGMA_PRIME
    psi_prev, psi_curr = 0.0, 1.0
    largest = envelope
    for n in range(N_MAX):
        largest = max(largest, abs(psi_curr) * envelope)
        psi_next = np.sqrt(2.0 / (n + 1.0)) * y_edge * psi_curr - np.sqrt(
            n / (n + 1.0)
        ) * psi_prev
        psi_prev, psi_curr = psi_curr, psi_next
    return float(largest)


def truncation_tail(
    raw: np.ndarray, n_values: np.ndarray, kernel_norm_sq: float
) -> np.ndarray:
    """Return eps_tr(N) = ||(I - Pi_N) psi_inf|| for each N.

    The mass beyond the computed n < N_MAX window is recovered exactly from
    Parseval's identity: sum_n |a_n|^2 = ||g||^2, so the remainder
    ||g||^2 - sum_{n<N_MAX} |a_n|^2 is added to every tail.
    """

    parseval_gap = max(kernel_norm_sq - float(np.sum(np.abs(raw) ** 2)), 0.0)
    tail_sq = (
        np.concatenate([np.cumsum(np.abs(raw[::-1]) ** 2)[::-1], [0.0]])
        + parseval_gap
    )
    return np.sqrt(tail_sq[n_values] / kernel_norm_sq)


def lambda_p_normalized(n_fock: int, p: int) -> float:
    """Evaluate Lambda_p with all commutator sums set to one."""

    base = float(n_fock - 1)
    mixed = sum(2.0**a * base ** (a / 2.0) for a in range(1, p + 1))
    pure_l = 2.0 ** (p + 1) * base ** ((p + 1) / 2.0)
    return mixed + pure_l + 1.0


def joint_resource_rows(
    eps_tr: np.ndarray,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, Dict[str, float]]]]:
    """Assemble per-(kappa, eps, N) proxy rows and per-curve optima."""

    rows: List[Dict[str, object]] = []
    optima: Dict[str, Dict[str, Dict[str, float]]] = {}
    for kappa_label, kappa in KAPPAS:
        optima[kappa_label] = {}
        for eps in EPSILONS:
            best: Dict[str, float] = {}
            for idx, n_coeff in enumerate(N_GRID):
                margin = eps - eps_tr[idx]
                if margin <= 0.0:
                    continue
                n_fock = int(np.ceil(kappa * n_coeff))
                lam = lambda_p_normalized(n_fock, TROTTER_ORDER)
                n_t = int(np.ceil((lam / margin) ** (1.0 / TROTTER_ORDER)))
                resource = int(n_coeff) * n_t
                rows.append(
                    {
                        "kappa": kappa_label,
                        "epsilon": eps,
                        "n_coeff": int(n_coeff),
                        "n_fock": n_fock,
                        "eps_tr": float(eps_tr[idx]),
                        "n_t": n_t,
                        "resource": resource,
                    }
                )
                if not best or resource < best["resource"]:
                    best = {
                        "n_coeff": int(n_coeff),
                        "n_fock": n_fock,
                        "n_t": n_t,
                        "resource": resource,
                    }
            optima[kappa_label][f"{eps:g}"] = best
    return rows, optima


def make_figure(
    eps_tr: np.ndarray,
    rows: List[Dict[str, object]],
    optima: Dict[str, Dict[str, Dict[str, float]]],
    fit_params: Tuple[float, float],
) -> None:
    """Render the three-panel tradeoff figure."""

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8))
    fig.subplots_adjust(wspace=0.34)

    ax = axes[0]
    ax.semilogy(N_GRID, eps_tr, color="tab:blue", lw=1.4, label="numerical tail")
    fit_a, fit_c = fit_params
    ax.semilogy(
        N_GRID,
        np.exp(fit_a + fit_c * N_GRID**0.25),
        "--",
        color="gray",
        lw=1.0,
        label=rf"$\propto e^{{{fit_c:.2f}\,N^{{1/4}}}}$ fit",
    )
    power_law = eps_tr[N_GRID == 16][0] * (N_GRID / 16.0) ** (-2.0)
    ax.semilogy(N_GRID, power_law, ":", color="gray", lw=1.0, label=r"$N^{-2}$")
    ax.set_xlabel(r"coefficient cutoff $N$", fontsize=9)
    ax.set_ylabel(r"$\epsilon_{\mathrm{tr}}(N)$", fontsize=9)
    ax.set_title("(a) coefficient-truncation tail", fontsize=10)
    ax.set_ylim(1e-3, 1.0)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)

    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(EPSILONS)))
    resources = [row["resource"] for row in rows]
    shared_ylim = (0.6 * min(resources), 1.8 * max(resources))
    for panel, (kappa_label, _kappa) in enumerate(KAPPAS, start=1):
        ax = axes[panel]
        for color, eps in zip(colors, EPSILONS):
            curve = [
                (row["n_coeff"], row["resource"])
                for row in rows
                if row["kappa"] == kappa_label and row["epsilon"] == eps
            ]
            n_vals, r_vals = zip(*curve)
            ax.semilogy(
                n_vals,
                r_vals,
                color=color,
                lw=1.3,
                label=rf"$\epsilon={eps:g}$",
            )
            best = optima[kappa_label][f"{eps:g}"]
            ax.semilogy(
                [best["n_coeff"]],
                [best["resource"]],
                marker="o",
                ms=5,
                mfc=color,
                mec="black",
                mew=0.7,
            )
        ax.set_xlabel(r"coefficient cutoff $N$", fontsize=9)
        ax.set_ylabel(r"$\mathcal{R}(N;\epsilon)=N\,n_t(N;\epsilon)$", fontsize=9)
        ax.set_ylim(*shared_ylim)
        kappa_tex = "2" if kappa_label == "2" else "4/3"
        ax.set_title(
            rf"({chr(ord('a') + panel)}) joint proxy, $N_{{\rm Fock}}=\lceil {kappa_tex}\,N\rceil$",
            fontsize=10,
        )
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, bbox_inches="tight")
    plt.close(fig)


def _sidecar() -> Dict[str, object]:
    """Collect provenance metadata following the repo sidecar convention."""

    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except OSError:
        head, dirty = "unknown", True
    return {
        "git_head": head,
        "git_dirty": dirty,
        "command": " ".join(sys.argv),
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {
            "make_joint_tradeoff.py": _sha(ROOT / "make_joint_tradeoff.py"),
            "clean_core.py": _sha(ROOT / "clean_core.py"),
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    from scipy.integrate import quad

    raw_h = raw_coefficients(K_STEP, PROJECTION_EXPONENT)
    raw_2h = raw_coefficients(2.0 * K_STEP, PROJECTION_EXPONENT)
    # Composite Simpson converges as h^4; Richardson removes the leading term.
    raw = (16.0 * raw_h - raw_2h) / 15.0
    richardson_gap = float(np.max(np.abs(raw - raw_h)))

    kernel_norm_sq = quad(
        lambda k: float(np.abs(kernel_g_beta_code(np.array([k]), BETA)[0]) ** 2),
        -60.0,
        60.0,
        limit=2000,
    )[0]
    parseval_gap = kernel_norm_sq - float(np.sum(np.abs(raw) ** 2))

    # Engine gate: the same machinery with the finite-r pairing Gaussian and
    # the archived quadrature domain must reproduce the archived shared-point
    # coefficients digit-for-digit.
    with open(ARCHIVED_COEFFS) as fh:
        stored = json.load(fh)
    stored_vec = np.array(stored["coefficients_re"]) + 1j * np.array(
        stored["coefficients_im"]
    )
    k_tail_archive = float(np.sqrt(-np.log(1e-14) / GAMMA))
    k_max_archive = max(8.0 * SIGMA_PRIME, 1.15 * k_tail_archive, 12.0)
    pairing_32 = raw_coefficients(K_STEP, GAMMA, n_max=32, k_max=k_max_archive)
    pairing_32 /= np.linalg.norm(pairing_32)
    engine_gate_diff = float(np.max(np.abs(pairing_32 - stored_vec)))

    spot = quad_spot_checks((8, 32, 100, 200))
    spot_diffs = {n: float(abs(spot[n] - raw[n])) for n in spot}

    edge_magnitude = edge_integrand_magnitude(PROJECTION_EXPONENT)

    eps_tr = truncation_tail(raw, N_GRID, kernel_norm_sq)

    fit_range = (N_GRID >= 40) & (N_GRID <= 200)
    design = np.vstack(
        [np.ones(int(np.sum(fit_range))), N_GRID[fit_range] ** 0.25]
    ).T
    fit_coef, *_ = np.linalg.lstsq(design, np.log(eps_tr[fit_range]), rcond=None)
    fit_params = (float(fit_coef[0]), float(fit_coef[1]))
    fit_max_rel_err = float(
        np.max(np.abs(np.exp(design @ fit_coef) / eps_tr[fit_range] - 1.0))
    )

    rows, optima = joint_resource_rows(eps_tr)
    make_figure(eps_tr, rows, optima, fit_params)

    with open(OUTPUT_DIR / "epsilon_tail.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["n_coeff", "eps_tr"])
        for n_coeff, value in zip(N_GRID, eps_tr):
            writer.writerow([int(n_coeff), f"{value:.12e}"])

    with open(OUTPUT_DIR / "joint_tradeoff.csv", "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "kappa",
                "epsilon",
                "n_coeff",
                "n_fock",
                "eps_tr",
                "n_t",
                "resource",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    normalized = raw / np.sqrt(kernel_norm_sq)
    np.savez(
        OUTPUT_DIR / "coefficients_shared_N224.npz",
        raw_re=np.real(raw),
        raw_im=np.imag(raw),
        normalized_re=np.real(normalized),
        normalized_im=np.imag(normalized),
        kernel_norm_sq=kernel_norm_sq,
        r_target=R_TARGET,
        r_prime=R_PRIME,
        beta=BETA,
    )

    meta = {
        "kernel": {
            "r_target": R_TARGET,
            "r_prime": R_PRIME,
            "beta": BETA,
            "coefficient_convention": "kernel_g_beta_code, code hbar=1",
        },
        "n_max": N_MAX,
        "trotter_order": TROTTER_ORDER,
        "kappas": [label for label, _ in KAPPAS],
        "epsilons": list(EPSILONS),
        "normalization": (
            "c_p = t^{p+1} = 1 and all nested-commutator sums Gamma = 1; "
            "Lambda_p(N_Fock) = sum_{a=1}^p 2^a (N_Fock-1)^{a/2} "
            "+ 2^{p+1} (N_Fock-1)^{(p+1)/2} + 1"
        ),
        "quadrature": {
            "method": (
                "uniform composite Simpson over the full oscillation domain "
                "with normalized-Hermite recurrence and h->h/2 Richardson "
                "extrapolation"
            ),
            "k_max": K_MAX,
            "k_step": K_STEP,
        },
        "coefficient_object": (
            "Theorem-2 orthogonal projection of the ideal normalized kernel "
            "state psi_inf = N g onto the r'-squeezed-Fock basis "
            "(basis Gaussian exp(-k^2/(2 sigma'^2))). NOT the finite-r "
            "pairing exp(-gamma k^2) of the pipeline, whose coefficient "
            "series is not square-summable (formal state g/phi_r^* is not "
            "normalizable); the pairing enters only the engine gate below."
        ),
        "kernel_norm_sq": kernel_norm_sq,
        "parseval_gap": parseval_gap,
        "tail_fit": {
            "model": "eps_tr(N) ~ exp(a + c N^{1/4}) over N in [40, 200]",
            "a": fit_params[0],
            "c": fit_params[1],
            "max_rel_err": fit_max_rel_err,
        },
        "checks": {
            "richardson_gap_max_abs": richardson_gap,
            "engine_gate_pairing_vs_archive_max_abs_diff": engine_gate_diff,
            "quad_spot_abs_diff": {str(n): v for n, v in spot_diffs.items()},
            "edge_integrand_magnitude": edge_magnitude,
        },
        "optima": optima,
        "sidecar": _sidecar(),
    }
    with open(OUTPUT_DIR / "joint_tradeoff_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"richardson gap max |dc|       : {richardson_gap:.3e}")
    print(f"engine gate (pairing) vs arch : {engine_gate_diff:.3e}")
    for n in sorted(spot_diffs):
        print(f"quad spot n={n:3d} |dc|         : {spot_diffs[n]:.3e}  (|a_n|={abs(raw[n]):.3e})")
    print(f"edge integrand magnitude      : {edge_magnitude:.3e}")
    print(f"parseval gap (beyond n=223)   : {parseval_gap:.3e}")
    print(
        "tail fit: eps_tr ~ "
        f"exp({fit_params[0]:.3f} + {fit_params[1]:.3f} N^0.25), "
        f"max rel err {fit_max_rel_err:.1%}"
    )
    print("eps_tr samples:")
    for n_probe in (8, 16, 24, 32, 48, 64, 96, 128):
        print(f"  N={n_probe:4d}  eps_tr={eps_tr[N_GRID == n_probe][0]:.3e}")
    print("optima (kappa, epsilon) -> N*, n_t, R*:")
    for kappa_label, per_eps in optima.items():
        for eps_label, best in per_eps.items():
            print(
                f"  kappa={kappa_label:>3s} eps={eps_label:>6s} -> "
                f"N*={best['n_coeff']:3d} N_Fock={best['n_fock']:3d} "
                f"n_t={best['n_t']:6d} R*={best['resource']:8d}"
            )
    print(f"figure -> {FIGURE_PATH}")


if __name__ == "__main__":
    main()
