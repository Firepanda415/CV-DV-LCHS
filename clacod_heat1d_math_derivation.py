#!/usr/bin/env python3
r"""
Mathematical Derivation + Numerical Audit for CV-DV LCHS Coefficients.

Context
-------
This script is a derivation-driven diagnostics tool for the 1D heat-equation
CV-DV LCHS implementation used in:
  - clacod_heat1d_qutip.py
  - clacod_heat1d_bosonic.py

The goal is to make the coefficient math explicit, compare competing formulas,
and provide concrete numerical evidence for which expression is being evaluated
and how stable each expression is under finite Fock truncation.

LCHS identity for the homogeneous system (A = L, H = 0 here):
  e^{-tA} = ∫ g(k) e^{-it k A} dk

CV-DV realization:
  K(t) := <phi_r| exp(-i t x_hat ⊗ A) |psi>   ≈ e^{-tA} (up to a global scale)

Convention issue
----------------
There are two common oscillator conventions:
  hbar=2 style: x_hat = a + a†
  hbar=1 style: x_hat = (a + a†)/sqrt(2)   (used by QuTiP in this project)

The paper-level symbolic coefficient equation is often written in a form with
only exp(-gamma x^2). In hbar=1 wavefunction overlap form, one must track the
Hermite-function Gaussian carefully.

This script evaluates and compares three coefficient expressions:
  1) legacy_hbar1_gh:
     Historical pre-fix expression (kept only as a bug-reproduction baseline).
  2) corrected_hbar1_explicit:
     Direct integral for
       C_n ∝ ∫ H_n(y/sigma') g(y) exp(-gamma_hbar1 y^2) dy
     with gamma_hbar1 = 0.5*(exp(-2r') - exp(-2r)).
  3) corrected_hbar1_gh_comp:
     Same corrected integral, rewritten in Gauss-Hermite coordinates with
     compensation for absorbed weight:
       y = sqrt(2)*sigma'*xi
       C_n ∝ Σ w_i H_n(sqrt(2)xi_i) g(sqrt(2)sigma'xi_i) exp((sigma'^2/sigma^2)xi_i^2)

Important:
  This script does not modify production formulas; it reports numerical
  behavior (accuracy and truncation sensitivity) to support a math-level fix.

How to read the output
----------------------
1) If corrected_hbar1_explicit and corrected_hbar1_gh_comp agree closely, the
   corrected algebra and GH compensation are internally consistent.
2) If legacy_hbar1_gh differs strongly from corrected methods, production
   should update the coefficient expression (not the CV-DV framework itself).
3) If corrected methods show high tail mass near n_coeff, increase n_coeff and
   n_fock before judging map accuracy.

Example CLI usage
-----------------
  # Default derivation report + single-point numeric comparison
  python clacod_heat1d_math_derivation.py

  # Match a target parameter point
  python clacod_heat1d_math_derivation.py \
      --r-target 0.015 --r-prime 0.01 --beta 0.95 \
      --n-fock 32 --n-coeff 32 --n-quad 200

  # Stability scan across coefficient cutoffs
  python clacod_heat1d_math_derivation.py \
      --n-scan 24,32,40,48,56,64 --n-fock 64 --n-quad 200

  # Better-conditioned check where corrected explicit and GH should align tightly
  python clacod_heat1d_math_derivation.py \
      --r-target 1.5 --r-prime 0.1 --n-fock 64 --n-coeff 32 --n-quad 160

  # Write full machine-readable report
  python clacod_heat1d_math_derivation.py \
      --n-scan 24,32,40,48,56,64 --output-json results/coeff_derivation_report.json

  # Auto-search robust corrected settings with constraints
  python clacod_heat1d_math_derivation.py \
      --r-target 1.5 --auto-search \
      --search-r-prime 0.02,0.05,0.1,0.2,0.3 \
      --search-beta 0.35,0.5,0.65,0.8,0.95 \
      --search-n-coeff 24,32,40 \
      --search-max-tail4 0.2 --search-max-gh-vs-exp 1e-3
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from numpy.polynomial.hermite import hermgauss
from qutip import Qobj, basis, qeye, squeeze, tensor
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.special import eval_hermite, gammaln

from clacod_heat1d_qutip import (
    fit_global_scale,
    heat_matrix,
    kernel_g_beta,
)


@dataclass(frozen=True)
class DerivationConfig:
    alpha: float = 1.0
    h_grid: float = 1.0
    total_time: float = 1.0
    n_fock: int = 64
    n_coeff: int = 32
    r_target: float = 0.003
    r_prime: float = 0.001
    beta: float = 0.95
    n_quad: int = 200
    integration_abs_tol: float = 1e-10
    integration_rel_tol: float = 1e-9
    max_integration_range: float = 220.0


def gamma_hbar2(r_target: float, r_prime: float) -> float:
    return float(np.exp(-2.0 * r_prime) - np.exp(-2.0 * r_target))


def gamma_hbar1(r_target: float, r_prime: float) -> float:
    return 0.5 * gamma_hbar2(r_target, r_prime)


def coeff_tail_mass(coeffs: np.ndarray, tail_levels: int = 4) -> float:
    m = min(max(int(tail_levels), 0), len(coeffs))
    if m == 0:
        return 0.0
    w = np.abs(coeffs) ** 2
    s = np.sum(w)
    if np.isclose(s, 0.0):
        return float("nan")
    return float(np.sum(w[-m:]) / s)


def coeff_prefactor(n: int, sigma_prime: float) -> float:
    # Keep n-dependent normalization; global constants cancel after vector normalization.
    return float(np.exp(-0.5 * (n * np.log(2.0) + gammaln(n + 1.0))))


def classical_map(cfg: DerivationConfig) -> np.ndarray:
    a = heat_matrix(cfg.alpha, cfg.h_grid)
    return expm(-a * cfg.total_time)


def map_from_coefficients(cfg: DerivationConfig, coeffs: np.ndarray) -> np.ndarray:
    # Build the same CV-DV map as clacod_heat1d_qutip.py, but with injected coefficients.
    full = np.zeros(cfg.n_fock, dtype=complex)
    full[: len(coeffs)] = coeffs

    psi_seed = Qobj(full.reshape((-1, 1))).unit()
    psi_osc = (squeeze(cfg.n_fock, cfg.r_prime) * psi_seed).unit()
    phi_post = (squeeze(cfg.n_fock, cfg.r_target) * basis(cfg.n_fock, 0)).unit()

    a = heat_matrix(cfg.alpha, cfg.h_grid)
    a_qobj = Qobj(a, dims=[[4], [4]])
    # Match production convention x_hat = (a+a†)/sqrt(2).
    from qutip import destroy

    x_hat = (destroy(cfg.n_fock) + destroy(cfg.n_fock).dag()) / np.sqrt(2.0)
    u_joint = (-1.0j * cfg.total_time * tensor(x_hat, a_qobj)).expm()
    k_map = tensor(phi_post.dag(), qeye(4)) * u_joint * tensor(psi_osc, qeye(4))
    return np.asarray(k_map.full(), dtype=complex)


def coefficients_legacy_hbar1_gh(cfg: DerivationConfig) -> np.ndarray:
    """Historical pre-fix expression (kept for audit baseline).

    Old production formula mixed gamma_hbar2 with an additional Gaussian from the
    hbar=1 Hermite-function overlap:
      C_n ∝ ∫ H_n(k/sigma') g(k) exp(-gamma_hbar2 k^2) exp(-k^2/(2 sigma'^2)) dk
    """
    gamma2 = gamma_hbar2(cfg.r_target, cfg.r_prime)
    sigma_prime = float(np.exp(cfg.r_prime))
    roots, weights = hermgauss(cfg.n_quad)
    k_points = np.sqrt(2.0) * sigma_prime * roots
    scale = np.sqrt(2.0) * sigma_prime

    envelope = kernel_g_beta(k_points, cfg.beta) * np.exp(-gamma2 * k_points**2)
    coeffs = np.zeros(cfg.n_coeff, dtype=complex)
    for n in range(cfg.n_coeff):
        pref = coeff_prefactor(n, sigma_prime)
        herm = eval_hermite(n, k_points / sigma_prime)
        coeffs[n] = pref * np.sum(weights * herm * envelope) * scale

    norm = np.linalg.norm(coeffs)
    if np.isclose(norm, 0.0):
        raise RuntimeError("legacy_hbar1_gh baseline produced near-zero coefficient norm.")
    return coeffs / norm


def _integration_window(cfg: DerivationConfig, gamma_eff: float) -> float:
    # Estimate finite range from Gaussian tail; cap for runtime safety.
    if gamma_eff <= 0:
        return 60.0
    est = np.sqrt(max(-np.log(1e-14) / gamma_eff, 1.0)) * 1.2
    return float(min(cfg.max_integration_range, max(12.0, est)))


def coefficients_corrected_hbar1_explicit(cfg: DerivationConfig) -> np.ndarray:
    r"""Direct corrected expression:
      C_n ∝ ∫ H_n(y/sigma') g(y) exp(-gamma_hbar1 y^2) dy
      gamma_hbar1 = 0.5*(exp(-2r') - exp(-2r))
    """
    sigma_prime = float(np.exp(cfg.r_prime))
    gamma1 = gamma_hbar1(cfg.r_target, cfg.r_prime)
    if gamma1 <= 0:
        raise ValueError("Need r_prime < r_target for gamma_hbar1 > 0.")
    y_max = _integration_window(cfg, gamma1)

    coeffs = np.zeros(cfg.n_coeff, dtype=complex)
    for n in range(cfg.n_coeff):
        pref = coeff_prefactor(n, sigma_prime)

        def fr(y: float) -> float:
            v = pref * eval_hermite(n, y / sigma_prime) * kernel_g_beta(np.array([y]), cfg.beta)[0] * np.exp(
                -gamma1 * y * y
            )
            return float(np.real(v))

        def fi(y: float) -> float:
            v = pref * eval_hermite(n, y / sigma_prime) * kernel_g_beta(np.array([y]), cfg.beta)[0] * np.exp(
                -gamma1 * y * y
            )
            return float(np.imag(v))

        re = quad(
            fr,
            -y_max,
            y_max,
            epsabs=cfg.integration_abs_tol,
            epsrel=cfg.integration_rel_tol,
            limit=max(100, cfg.n_quad),
        )[0]
        im = quad(
            fi,
            -y_max,
            y_max,
            epsabs=cfg.integration_abs_tol,
            epsrel=cfg.integration_rel_tol,
            limit=max(100, cfg.n_quad),
        )[0]
        coeffs[n] = re + 1.0j * im

    norm = np.linalg.norm(coeffs)
    if np.isclose(norm, 0.0):
        raise RuntimeError("corrected_hbar1_explicit produced near-zero coefficient norm.")
    return coeffs / norm


def coefficients_corrected_hbar1_gh_comp(cfg: DerivationConfig) -> np.ndarray:
    r"""Gauss-Hermite evaluation of the same corrected expression.

    Start from corrected integral in y and use y = sqrt(2)*sigma'*xi:
      C_n ∝ ∫ e^{-xi^2} [ H_n(sqrt(2)xi) g(sqrt(2)sigma'xi) exp((sigma'^2/sigma^2)xi^2) ] dxi
    """
    sigma = float(np.exp(cfg.r_target))
    sigma_prime = float(np.exp(cfg.r_prime))
    ratio = (sigma_prime * sigma_prime) / (sigma * sigma)
    roots, weights = hermgauss(cfg.n_quad)
    y_points = np.sqrt(2.0) * sigma_prime * roots
    scale = np.sqrt(2.0) * sigma_prime

    coeffs = np.zeros(cfg.n_coeff, dtype=complex)
    for n in range(cfg.n_coeff):
        pref = coeff_prefactor(n, sigma_prime)
        herm = eval_hermite(n, np.sqrt(2.0) * roots)

        # Stabilize positive exponent with weighted log-scaling.
        b = ratio * (roots**2)
        logw = np.log(np.abs(weights) + 1e-300)
        shift = np.max(logw + b)
        scaled = np.sign(weights) * herm * kernel_g_beta(y_points, cfg.beta) * np.exp((logw + b) - shift)
        coeffs[n] = pref * np.exp(shift) * np.sum(scaled) * scale

    norm = np.linalg.norm(coeffs)
    if np.isclose(norm, 0.0):
        raise RuntimeError("corrected_hbar1_gh_comp produced near-zero coefficient norm.")
    return coeffs / norm


def evaluate_method(name: str, cfg: DerivationConfig, coeffs: np.ndarray, ref_map: np.ndarray) -> Dict[str, float]:
    k_map = map_from_coefficients(cfg, coeffs)
    _, map_rel = fit_global_scale(k_map, ref_map)
    peak_idx = int(np.argmax(np.abs(coeffs) ** 2))
    return {
        "map_rel_error_best_scale": float(map_rel),
        "coeff_tail_mass_last4": coeff_tail_mass(coeffs, tail_levels=4),
        "coeff_peak_index": peak_idx,
        "coeff_peak_prob": float(np.max(np.abs(coeffs) ** 2)),
        "coeff_n_eff": float(1.0 / np.sum((np.abs(coeffs) ** 2) ** 2)),
    }


def parse_int_list(text: str) -> List[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("List must contain at least one integer.")
    for v in vals:
        if v <= 0:
            raise ValueError("All list entries must be positive integers.")
    return vals


def parse_float_list(text: str) -> List[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("List must contain at least one float.")
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Derivation + numerical audit for CV-DV LCHS coefficients")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--h-grid", type=float, default=1.0)
    p.add_argument("--total-time", type=float, default=1.0)
    p.add_argument("--n-fock", type=int, default=64)
    p.add_argument("--n-coeff", type=int, default=32)
    p.add_argument("--r-target", type=float, default=0.003)
    p.add_argument("--r-prime", type=float, default=0.001)
    p.add_argument("--beta", type=float, default=0.95)
    p.add_argument("--n-quad", type=int, default=200)
    p.add_argument("--integration-abs-tol", type=float, default=1e-10)
    p.add_argument("--integration-rel-tol", type=float, default=1e-9)
    p.add_argument("--max-integration-range", type=float, default=220.0)
    p.add_argument(
        "--n-scan",
        type=str,
        default="",
        help="Optional comma-separated n_coeff scan (e.g. 24,32,40,48,56,64).",
    )
    p.add_argument(
        "--auto-search",
        action="store_true",
        help="Grid-search (r_prime, beta, n_coeff) using corrected backends and rank robust settings.",
    )
    p.add_argument(
        "--search-r-prime",
        type=str,
        default="0.02,0.05,0.1,0.2,0.3",
        help="Comma-separated candidate r_prime values (must satisfy r_prime < r_target).",
    )
    p.add_argument(
        "--search-beta",
        type=str,
        default="0.35,0.5,0.65,0.8,0.95",
        help="Comma-separated candidate beta values in (0,1).",
    )
    p.add_argument(
        "--search-n-coeff",
        type=str,
        default="24,32,40",
        help="Comma-separated candidate n_coeff values (must be <= n_fock).",
    )
    p.add_argument(
        "--search-max-tail4",
        type=float,
        default=0.2,
        help="Validity threshold for tail mass over the last 4 coefficients.",
    )
    p.add_argument(
        "--search-max-gh-vs-exp",
        type=float,
        default=1e-3,
        help="Validity threshold for corrected GH-vs-explicit coefficient rel error.",
    )
    p.add_argument(
        "--search-peak-margin",
        type=int,
        default=2,
        help="Require peak_n <= n_coeff - margin for a candidate to be valid.",
    )
    p.add_argument(
        "--search-top-k",
        type=int,
        default=12,
        help="Number of top-ranked auto-search candidates to print.",
    )
    p.add_argument("--output-json", type=str, default="")
    return p.parse_args()


def print_derivation_summary(cfg: DerivationConfig) -> None:
    sigma = np.exp(cfg.r_target)
    sigma_prime = np.exp(cfg.r_prime)
    g2 = gamma_hbar2(cfg.r_target, cfg.r_prime)
    g1 = gamma_hbar1(cfg.r_target, cfg.r_prime)

    print("=== CV-DV LCHS Coefficient Derivation Audit ===")
    print(
        "Config: "
        f"r={cfg.r_target}, r'={cfg.r_prime}, beta={cfg.beta}, "
        f"n_fock={cfg.n_fock}, n_coeff={cfg.n_coeff}, n_quad={cfg.n_quad}"
    )
    print(f"sigma = e^r = {sigma:.6f}, sigma' = e^r' = {sigma_prime:.6f}")
    print(f"gamma_hbar2 = exp(-2r') - exp(-2r) = {g2:.6e}")
    print(f"gamma_hbar1 = gamma_hbar2 / 2       = {g1:.6e}")
    print()
    print("Target corrected hbar=1 expression:")
    print("  C_n ∝ ∫ H_n(y/sigma') g(y) exp(-gamma_hbar1 y^2) dy")
    print("Gauss-Hermite compensated form:")
    print("  y = sqrt(2) sigma' xi,")
    print("  C_n ∝ Σ w_i H_n(sqrt(2)xi_i) g(sqrt(2)sigma'xi_i) exp((sigma'^2/sigma^2)xi_i^2)")
    print("Interpretation target:")
    print("  corrected_explicit and corrected_gh_comp should match each other closely.")
    print("  legacy deviation indicates a coefficient-expression mismatch.")
    print()


def print_fix_recommendation(single: Dict[str, object], cfg: DerivationConfig) -> None:
    legacy_err = float(single["legacy_hbar1_gh"]["map_rel_error_best_scale"])
    corr_exp_err = float(single["corrected_hbar1_explicit"]["map_rel_error_best_scale"])
    corr_gh_err = float(single["corrected_hbar1_gh_comp"]["map_rel_error_best_scale"])
    rel_legacy_exp = float(single["coeff_vector_rel_errors"]["legacy_vs_corrected_explicit"])
    rel_exp_gh = float(single["coeff_vector_rel_errors"]["corrected_explicit_vs_corrected_gh_comp"])

    tail_exp = float(single["corrected_hbar1_explicit"]["coeff_tail_mass_last4"])
    tail_gh = float(single["corrected_hbar1_gh_comp"]["coeff_tail_mass_last4"])

    print()
    print("--- Recommended Math Update (hbar=1 / QuTiP convention) ---")
    print("Patch target expression:")
    print("  gamma_hbar1 = 0.5 * (exp(-2 r_prime) - exp(-2 r_target))")
    print("  C_n ∝ pref_n * ∫ H_n(y/sigma') * g(y) * exp(-gamma_hbar1 y^2) dy")
    print("  pref_n = 1 / sqrt(2^n n!)  (global normalization absorbed at the end)")
    print()
    print("Equivalent GH-compensated implementation form:")
    print("  y = sqrt(2) sigma' xi")
    print("  C_n ∝ Σ w_i * H_n(sqrt(2)xi_i) * g(sqrt(2)sigma'xi_i) * exp((sigma'^2/sigma^2)xi_i^2)")
    print()
    print("Numerical consistency indicators:")
    print(f"  legacy vs corrected-explicit coeff rel error:      {rel_legacy_exp:.6e}")
    print(f"  corrected-explicit vs corrected-GH coeff rel error:{rel_exp_gh:.6e}")
    print(f"  map error legacy/corr-exp/corr-gh:                 {legacy_err:.6e} / {corr_exp_err:.6e} / {corr_gh_err:.6e}")

    if tail_exp > 1e-2 or tail_gh > 1e-2:
        print()
        print("Truncation warning:")
        print("  Corrected coefficients are still concentrated near the truncation edge.")
        print("  Increase --n-coeff and --n-fock (and possibly --n-quad) before final judgment.")
    if cfg.r_target - cfg.r_prime < 0.02:
        print()
        print("Conditioning note:")
        print("  r_target and r_prime are very close; gamma is small and coefficients spread wide.")
        print("  Expect larger required n_coeff/n_fock for stable corrected-formula numerics.")


def run_single_point(cfg: DerivationConfig) -> Dict[str, object]:
    ref_map = classical_map(cfg)
    coeffs_legacy = coefficients_legacy_hbar1_gh(cfg)
    coeffs_corr_exp = coefficients_corrected_hbar1_explicit(cfg)
    coeffs_corr_gh = coefficients_corrected_hbar1_gh_comp(cfg)

    _, rel_legacy_vs_exp = fit_global_scale(coeffs_legacy, coeffs_corr_exp)
    _, rel_exp_vs_gh = fit_global_scale(coeffs_corr_exp, coeffs_corr_gh)

    out = {
        "legacy_hbar1_gh": evaluate_method("legacy_hbar1_gh", cfg, coeffs_legacy, ref_map),
        "corrected_hbar1_explicit": evaluate_method("corrected_hbar1_explicit", cfg, coeffs_corr_exp, ref_map),
        "corrected_hbar1_gh_comp": evaluate_method("corrected_hbar1_gh_comp", cfg, coeffs_corr_gh, ref_map),
        "coeff_vector_rel_errors": {
            "legacy_vs_corrected_explicit": float(rel_legacy_vs_exp),
            "corrected_explicit_vs_corrected_gh_comp": float(rel_exp_vs_gh),
        },
    }
    return out


def run_scan(cfg: DerivationConfig, scan_vals: Sequence[int]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for n in scan_vals:
        cfg_n = DerivationConfig(**{**asdict(cfg), "n_coeff": n, "n_fock": max(cfg.n_fock, n)})
        ref_map = classical_map(cfg_n)
        corr = coefficients_corrected_hbar1_explicit(cfg_n)
        metrics = evaluate_method("corrected_hbar1_explicit", cfg_n, corr, ref_map)
        row = {"n_coeff": n}
        row.update(metrics)
        rows.append(row)
    return rows


def run_auto_search(
    cfg: DerivationConfig,
    *,
    r_prime_vals: Sequence[float],
    beta_vals: Sequence[float],
    n_coeff_vals: Sequence[int],
    max_tail4: float,
    max_gh_vs_exp: float,
    peak_margin: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    valid_n_coeff = [n for n in n_coeff_vals if 1 <= n <= cfg.n_fock]
    if not valid_n_coeff:
        raise ValueError("No valid search-n-coeff entries satisfy 1 <= n_coeff <= n_fock.")

    for r_prime in r_prime_vals:
        if r_prime >= cfg.r_target:
            continue
        for beta in beta_vals:
            if not (0.0 < beta < 1.0):
                continue
            for n_coeff in valid_n_coeff:
                cfg_i = DerivationConfig(
                    **{
                        **asdict(cfg),
                        "r_prime": float(r_prime),
                        "beta": float(beta),
                        "n_coeff": int(n_coeff),
                    }
                )
                ref_map = classical_map(cfg_i)
                try:
                    coeff_exp = coefficients_corrected_hbar1_explicit(cfg_i)
                    coeff_gh = coefficients_corrected_hbar1_gh_comp(cfg_i)
                    _, gh_vs_exp = fit_global_scale(coeff_gh, coeff_exp)
                    metrics = evaluate_method("corrected_hbar1_explicit", cfg_i, coeff_exp, ref_map)

                    map_err = float(metrics["map_rel_error_best_scale"])
                    tail4 = float(metrics["coeff_tail_mass_last4"])
                    peak_n = int(metrics["coeff_peak_index"])
                    margin = max(int(peak_margin), 0)
                    peak_ok = peak_n <= max(n_coeff - margin, 0)
                    valid = (gh_vs_exp <= max_gh_vs_exp) and (tail4 <= max_tail4) and peak_ok

                    penalty = 0.0
                    if gh_vs_exp > max_gh_vs_exp:
                        penalty += (gh_vs_exp / max(max_gh_vs_exp, 1e-15)) - 1.0
                    if tail4 > max_tail4:
                        penalty += (tail4 / max(max_tail4, 1e-15)) - 1.0
                    if not peak_ok:
                        penalty += 1.0 + (peak_n - (n_coeff - margin)) / max(float(n_coeff), 1.0)

                    score = map_err + 10.0 * penalty
                    row: Dict[str, object] = {
                        "valid": bool(valid),
                        "score": float(score),
                        "map_rel_error_best_scale": map_err,
                        "gh_vs_exp_rel_error": float(gh_vs_exp),
                        "coeff_tail_mass_last4": tail4,
                        "coeff_peak_index": peak_n,
                        "r_prime": float(r_prime),
                        "beta": float(beta),
                        "n_coeff": int(n_coeff),
                        "error": "",
                    }
                except Exception as exc:
                    row = {
                        "valid": False,
                        "score": float("inf"),
                        "map_rel_error_best_scale": float("inf"),
                        "gh_vs_exp_rel_error": float("inf"),
                        "coeff_tail_mass_last4": float("inf"),
                        "coeff_peak_index": -1,
                        "r_prime": float(r_prime),
                        "beta": float(beta),
                        "n_coeff": int(n_coeff),
                        "error": str(exc),
                    }
                rows.append(row)

    rows.sort(
        key=lambda r: (
            0 if bool(r["valid"]) else 1,
            float(r["score"]),
            float(r["map_rel_error_best_scale"]),
            float(r["gh_vs_exp_rel_error"]),
        )
    )
    return rows


def print_auto_search(
    rows: Sequence[Dict[str, object]],
    *,
    top_k: int,
    max_tail4: float,
    max_gh_vs_exp: float,
    peak_margin: int,
) -> None:
    print()
    print("--- Auto Search (Corrected Formula) ---")
    print(
        "Validity constraints: "
        f"tail4 <= {max_tail4:.3e}, gh_vs_exp <= {max_gh_vs_exp:.3e}, "
        f"peak_n <= n_coeff - {max(int(peak_margin), 0)}"
    )
    print(
        f"{'rank':>4s} {'ok':>3s} {'score':>10s} {'map_err':>10s} "
        f"{'gh_vs_exp':>10s} {'tail4':>10s} {'peak_n':>7s} "
        f"{'r_prime':>8s} {'beta':>6s} {'n_coeff':>8s}"
    )

    n_show = min(max(int(top_k), 1), len(rows))
    for i in range(n_show):
        r = rows[i]
        ok = "Y" if bool(r["valid"]) else "N"
        print(
            f"{i+1:4d} {ok:>3s} "
            f"{float(r['score']):10.3e} "
            f"{float(r['map_rel_error_best_scale']):10.3e} "
            f"{float(r['gh_vs_exp_rel_error']):10.3e} "
            f"{float(r['coeff_tail_mass_last4']):10.3e} "
            f"{int(r['coeff_peak_index']):7d} "
            f"{float(r['r_prime']):8.3f} "
            f"{float(r['beta']):6.2f} "
            f"{int(r['n_coeff']):8d}"
        )

    if rows:
        best = rows[0]
        print()
        print(
            "Best candidate: "
            f"valid={best['valid']}, map_err={float(best['map_rel_error_best_scale']):.6e}, "
            f"r'={float(best['r_prime']):.6f}, beta={float(best['beta']):.4f}, "
            f"n_coeff={int(best['n_coeff'])}"
        )


def main() -> None:
    args = parse_args()
    cfg = DerivationConfig(
        alpha=args.alpha,
        h_grid=args.h_grid,
        total_time=args.total_time,
        n_fock=args.n_fock,
        n_coeff=args.n_coeff,
        r_target=args.r_target,
        r_prime=args.r_prime,
        beta=args.beta,
        n_quad=args.n_quad,
        integration_abs_tol=args.integration_abs_tol,
        integration_rel_tol=args.integration_rel_tol,
        max_integration_range=args.max_integration_range,
    )

    if cfg.n_coeff > cfg.n_fock:
        raise ValueError("n_coeff must be <= n_fock.")
    if cfg.r_prime >= cfg.r_target:
        raise ValueError("Need r_prime < r_target for positive gamma.")

    print_derivation_summary(cfg)
    single = run_single_point(cfg)

    print("--- Single-Point Comparison ---")
    for method in ["legacy_hbar1_gh", "corrected_hbar1_explicit", "corrected_hbar1_gh_comp"]:
        m = single[method]
        print(
            f"{method:28s}  "
            f"map_err={m['map_rel_error_best_scale']:.6e}  "
            f"tail4={m['coeff_tail_mass_last4']:.6e}  "
            f"peak_n={m['coeff_peak_index']:>3d}  "
            f"peak_w={m['coeff_peak_prob']:.6e}"
        )
    ce = single["coeff_vector_rel_errors"]
    print(
        "Coeff vector rel errors: "
        f"legacy-vs-corrected-explicit={ce['legacy_vs_corrected_explicit']:.6e}, "
        f"corrected-explicit-vs-corrected-gh={ce['corrected_explicit_vs_corrected_gh_comp']:.6e}"
    )
    print_fix_recommendation(single, cfg)

    scan_rows: List[Dict[str, object]] = []
    if args.n_scan:
        scan_vals = parse_int_list(args.n_scan)
        scan_rows = run_scan(cfg, scan_vals)
        print()
        print("--- Corrected-Formula Stability Scan ---")
        print(f"{'n_coeff':>8s} {'map_err':>12s} {'tail4':>12s} {'peak_n':>8s} {'peak_w':>12s}")
        for row in scan_rows:
            print(
                f"{row['n_coeff']:8d} "
                f"{row['map_rel_error_best_scale']:12.6e} "
                f"{row['coeff_tail_mass_last4']:12.6e} "
                f"{row['coeff_peak_index']:8d} "
                f"{row['coeff_peak_prob']:12.6e}"
            )

    auto_rows: List[Dict[str, object]] = []
    if args.auto_search:
        auto_rows = run_auto_search(
            cfg,
            r_prime_vals=parse_float_list(args.search_r_prime),
            beta_vals=parse_float_list(args.search_beta),
            n_coeff_vals=parse_int_list(args.search_n_coeff),
            max_tail4=float(args.search_max_tail4),
            max_gh_vs_exp=float(args.search_max_gh_vs_exp),
            peak_margin=int(args.search_peak_margin),
        )
        print_auto_search(
            auto_rows,
            top_k=int(args.search_top_k),
            max_tail4=float(args.search_max_tail4),
            max_gh_vs_exp=float(args.search_max_gh_vs_exp),
            peak_margin=int(args.search_peak_margin),
        )

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(cfg),
            "single_point": single,
            "scan": scan_rows,
            "auto_search": auto_rows,
            "derivation": {
                "gamma_hbar2": gamma_hbar2(cfg.r_target, cfg.r_prime),
                "gamma_hbar1": gamma_hbar1(cfg.r_target, cfg.r_prime),
                "formula_corrected_hbar1": "C_n ∝ ∫ H_n(y/sigma') g(y) exp(-gamma_hbar1 y^2) dy",
                "formula_corrected_gh_comp": "C_n ∝ Σ w_i H_n(sqrt(2)xi_i) g(sqrt(2)sigma'xi_i) exp((sigma'^2/sigma^2)xi_i^2)",
            },
        }
        out.write_text(json.dumps(payload, indent=2))
        print()
        print(f"Wrote JSON: {out}")


if __name__ == "__main__":
    main()
