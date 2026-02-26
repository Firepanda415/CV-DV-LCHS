#!/usr/bin/env python3
"""
1D heat equation via CV-DV LCHS using qutip + scipy.

Problem:  du/dt = -A u,   u(0) = u₀
where A = (α/h²) T  (4×4 tridiagonal Laplacian, Dirichlet BC).

A is real symmetric  →  Cartesian decomposition: L = A, H = 0.
LCHS circuit:  U(t) = exp(-it x̂ ⊗ A).  (Single term, no Trotter needed.)

Two state-preparation methods:
  1. "paper": Squeezed-Fock expansion (paper Eq. 12-13, needs large r and N_dim)
  2. "xbasis": Direct x-eigenbasis encoding with integration weights (practical)

Reference:
  - Hybrid_CV_DV_LCHS.pdf, Section 3.5.2 (Eq. 45-48)
  - LCHS CHO.ipynb (working reference implementation)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite, gammaln
from scipy.linalg import expm
from numpy.polynomial.hermite import hermgauss
from qutip import basis, squeeze, tensor, sigmax, sigmay, identity, destroy, Qobj


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

def lchs_kernel(k, beta):
    """g(k) = 1 / [C_β · e^{(1+ik)^β} · (1-ik)],  C_β = 2π e^{-2^β}."""
    C_val = 2 * np.pi * np.exp(-(2.0 ** beta))
    return 1.0 / (C_val * np.exp((1 + 1j * k) ** beta) * (1 - 1j * k))


# ---------------------------------------------------------------------------
# State preparation (method 1): Paper's squeezed-Fock expansion
# ---------------------------------------------------------------------------

def get_lchs_states_paper(beta, r_target, r_prime, N_dim, n_quad=50000):
    """
    Paper Eq. 12-13:  C_n = K_n ∫ H_n(x/σ') g(x) e^{-γx²} dx
    where K_n = √(σ/σ') / √(2^n n!),  σ = e^r, σ' = e^{r'}.

    Returns (psi_osc, phi_post, cn_array).
    """
    gamma = np.exp(-2 * r_prime) - np.exp(-2 * r_target)
    assert gamma > 0, f"Need r' < r for γ > 0, got γ={gamma:.6f}"

    sigma = np.exp(r_target)
    sigma_prime = np.exp(r_prime)

    x_max = min(50.0, 8.0 / np.sqrt(gamma))
    x = np.linspace(-x_max, x_max, n_quad)
    dx = x[1] - x[0]

    g_weighted = lchs_kernel(x, beta) * np.exp(-gamma * x ** 2)
    K_base = np.sqrt(sigma / sigma_prime)

    cn_list = []
    for n in range(N_dim):
        K_n = K_base * np.exp(-0.5 * (n * np.log(2) + gammaln(n + 1)))
        H_val = eval_hermite(n, x / sigma_prime)
        val = K_n * np.sum(H_val * g_weighted) * dx
        cn_list.append(val)

    cn_array = np.array(cn_list)
    psi_seed = sum(cn_array[n] * basis(N_dim, n) for n in range(N_dim)).unit()
    psi_osc = squeeze(N_dim, r_prime) * psi_seed
    phi_post = squeeze(N_dim, r_target) * basis(N_dim, 0)

    return psi_osc, phi_post, cn_array


# ---------------------------------------------------------------------------
# State preparation (method 2): Direct x-eigenbasis encoding
# ---------------------------------------------------------------------------

def get_lchs_states_xbasis(beta, N_dim):
    """
    Encode g(x) directly using x̂ eigenstates with integration weights.

    The x̂ eigenvalues x_j provide a natural quadrature grid.
    We set: φ*(x_j)·ψ(x_j) = w_j · g(x_j)  where w_j are weights.

    Returns (psi_osc, phi_post).
    """
    # Build x_hat in Fock basis and diagonalize
    a_mat = np.diag(np.sqrt(np.arange(1, N_dim, dtype=float)), 1)
    x_op = (a_mat + a_mat.T) / 2
    x_vals, x_vecs = np.linalg.eigh(x_op)

    # Integration weights (midpoint rule)
    w = np.zeros(N_dim)
    w[0] = x_vals[1] - x_vals[0]
    w[-1] = x_vals[-1] - x_vals[-2]
    for j in range(1, N_dim - 1):
        w[j] = (x_vals[j + 1] - x_vals[j - 1]) / 2

    # Kernel values at x-eigenvalues
    g = lchs_kernel(x_vals, beta)

    # Factorize: φ*(x_j) ψ(x_j) = w_j g(x_j)
    # Choose: ψ_x(j) = √w_j · g(x_j),  φ_x(j) = √w_j
    psi_x = np.sqrt(w) * g
    phi_x = np.sqrt(w).astype(complex)

    # Convert to Fock basis via eigenvector transformation
    psi_fock = x_vecs @ psi_x
    phi_fock = x_vecs @ phi_x

    psi_osc = Qobj(psi_fock.reshape(-1, 1))
    phi_post = Qobj(phi_fock.reshape(-1, 1))

    # Normalize psi (phi stays unnormalized — part of the LCHS protocol)
    psi_osc = psi_osc.unit()

    return psi_osc, phi_post


# ---------------------------------------------------------------------------
# Heat equation operator
# ---------------------------------------------------------------------------

def heat_matrix(alpha=1.0, h=1.0):
    """4×4 discretized Laplacian: A = (α/h²) T.  (Paper Eq. 45-46.)"""
    T = np.array(
        [[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 2]],
        dtype=float,
    )
    return (alpha / h ** 2) * T


def heat_operator_qutip(alpha=1.0, h=1.0):
    """A as qutip operator on 2 qubits. Paper Eq. 46."""
    s = alpha / h ** 2
    I2 = identity(2)
    return (
        2 * s * tensor(I2, I2)
        - s * tensor(I2, sigmax())
        - 0.5 * s * tensor(sigmax(), sigmax())
        - 0.5 * s * tensor(sigmay(), sigmay())
    )


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_simulation(alpha, h_grid, T_total, n_steps, N_dim,
                   psi_osc, phi_post, init_q0=0, init_q1=1):
    """
    Run LCHS simulation for the 1D heat equation.
    Joint Hamiltonian: x̂ ⊗ A  (single term → no Trotter error).
    """
    A_op = heat_operator_qutip(alpha, h_grid)
    a = destroy(N_dim)
    K_op = (a.dag() + a) / 2  # x̂

    Ham = tensor(K_op, A_op)
    dt = T_total / n_steps
    U_step = (-1j * dt * Ham).expm()

    psi_qubit = tensor(basis(2, init_q0), basis(2, init_q1))
    psi_current = tensor(psi_osc, psi_qubit)

    Id_q = tensor(identity(2), identity(2))
    projector = tensor(phi_post.dag(), Id_q)

    # Normalization from t=0
    vec_0 = (projector * psi_current).full().flatten()
    norm_factor = np.linalg.norm(vec_0)
    print(f"   Norm factor (t=0): {norm_factor:.6f}")

    times, u_raw = [], []
    for step in range(n_steps + 1):
        qubit_vec = (projector * psi_current).full().flatten()
        times.append(step * dt)
        u_raw.append(np.real(qubit_vec))
        if step < n_steps:
            psi_current = U_step * psi_current

    return np.array(times), np.array(u_raw), norm_factor


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="1D heat equation via LCHS (qutip)")
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--h-grid", type=float, default=1.0)
    p.add_argument("--T-total", type=float, default=1.0)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--init-q0", type=int, default=0, choices=[0, 1])
    p.add_argument("--init-q1", type=int, default=1, choices=[0, 1])
    p.add_argument("--beta", type=float, default=0.8)
    p.add_argument("--N-dim", type=int, default=100)
    p.add_argument("--method", choices=["paper", "xbasis"], default="xbasis")
    p.add_argument("--r-target", type=float, default=3.0)
    p.add_argument("--r-prime", type=float, default=1.0)
    p.add_argument("--output-dir", type=str, default="results/heat1d_qutip")
    return p


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha, h_grid = args.alpha, args.h_grid
    T_total, n_steps = args.T_total, args.n_steps
    init_q0, init_q1 = args.init_q0, args.init_q1
    N_dim = args.N_dim

    # 1. Prepare states
    print(f"Method: {args.method}")
    if args.method == "paper":
        psi_osc, phi_post, _ = get_lchs_states_paper(
            args.beta, args.r_target, args.r_prime, N_dim,
        )
    else:
        psi_osc, phi_post = get_lchs_states_xbasis(args.beta, N_dim)
    print(f"   N_dim={N_dim}, beta={args.beta}")

    # 2. Run simulation
    print("Running simulation...")
    times, u_raw, nf = run_simulation(
        alpha, h_grid, T_total, n_steps, N_dim, psi_osc, phi_post,
        init_q0, init_q1,
    )
    u_lchs = u_raw / nf

    # 3. Classical reference
    A = heat_matrix(alpha, h_grid)
    u0 = np.zeros(4)
    u0[init_q0 * 2 + init_q1] = 1.0
    u_exact = np.array([expm(-t * A) @ u0 for t in times])

    # 4. Error analysis
    print("\n=== Results ===")
    sample_idx = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps]
    for idx in sample_idx:
        t = times[idx]
        err = np.linalg.norm(u_lchs[idx] - u_exact[idx]) / max(
            np.linalg.norm(u_exact[idx]), 1e-15
        )
        print(
            f"  t={t:.3f}: LCHS={np.array2string(u_lchs[idx], precision=4, suppress_small=True)}"
            f"  exact={np.array2string(u_exact[idx], precision=4, suppress_small=True)}"
            f"  err={err:.4e}"
        )

    final_err = np.linalg.norm(u_lchs[-1] - u_exact[-1]) / max(
        np.linalg.norm(u_exact[-1]), 1e-15
    )
    print(f"\n  Final PDE error: {final_err:.6e}")

    # 5. Plots
    labels = [r"$u_1$", r"$u_2$", r"$u_3$", r"$u_4$"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        ax.plot(times, u_lchs[:, i], "b.-", label="LCHS", lw=1.5, ms=3)
        ax.plot(times, u_exact[:, i], "k--", label="Exact", alpha=0.7, lw=1.5)
        ax.set_title(labels[i])
        ax.set_xlabel("Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.suptitle(
        f"1D Heat Equation via LCHS ({args.method})\n"
        rf"$\alpha$={alpha}, N_dim={N_dim}, $\beta$={args.beta}"
    )
    plt.tight_layout()
    fig.savefig(out_dir / "heat1d_lchs_vs_exact.png", dpi=150)
    plt.close()

    errors = [
        np.linalg.norm(u_lchs[i] - u_exact[i]) / max(np.linalg.norm(u_exact[i]), 1e-15)
        for i in range(len(times))
    ]
    plt.figure(figsize=(8, 4))
    plt.semilogy(times, errors, "r.-", lw=1.5, ms=3)
    plt.xlabel("Time")
    plt.ylabel("Relative error")
    plt.title("LCHS relative error vs time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "heat1d_error_vs_time.png", dpi=150)
    plt.close()

    # 6. Save
    results = {
        "method": args.method,
        "params": {
            "alpha": alpha, "h_grid": h_grid, "T_total": T_total,
            "n_steps": n_steps, "init": [init_q0, init_q1],
            "beta": args.beta, "N_dim": N_dim,
        },
        "norm_factor": float(nf),
        "final_pde_error": float(final_err),
        "u_lchs_final": u_lchs[-1].tolist(),
        "u_exact_final": u_exact[-1].tolist(),
    }
    if args.method == "paper":
        results["params"]["r_target"] = args.r_target
        results["params"]["r_prime"] = args.r_prime
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\n  Outputs: {out_dir}")


if __name__ == "__main__":
    main()
