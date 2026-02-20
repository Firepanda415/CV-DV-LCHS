import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial, eval_hermite
from numpy.polynomial.hermite import hermgauss
from qutip import basis, displace, Qobj


def lchs_coefficients(r_target, r_prime, n_dim, n_quad_points=100):
    """Compute normalized finite squeezed Fock expansion coefficients Cn."""
    width_param = np.exp(r_prime)
    scale_factor = np.sqrt(2) * width_param
    gamma = np.exp(-2 * r_prime) - np.exp(-2 * r_target)
    if gamma <= 0:
        raise ValueError("r_prime must be < r_target so gamma = e^{-2r'} - e^{-2r} > 0.")

    x_roots, weights = hermgauss(n_quad_points)
    p_points = x_roots * scale_factor

    g_part = 1.0 / (1.0 - 1.0j * p_points)
    target_part = np.exp(-gamma * p_points**2) * g_part

    cn_list = []
    sqrt_pi = np.sqrt(np.pi)
    basis_prefactor = 1.0 / np.sqrt(sqrt_pi * width_param)

    for n in range(n_dim):
        fock_norm = 1.0 / np.sqrt((2**n) * factorial(n))
        h_val = eval_hermite(n, p_points / width_param)
        integrand = target_part * (basis_prefactor * fock_norm * h_val)
        val = np.sum(weights * integrand) * scale_factor
        cn_list.append(val)

    cn_array = np.array(cn_list)
    norm = np.linalg.norm(cn_array)
    if np.isclose(norm, 0.0):
        raise RuntimeError("Computed LCHS coefficients have zero norm.")
    return cn_array / norm


def target_state_from_coeffs(coeffs):
    n_dim = len(coeffs)
    state = sum(coeffs[n] * basis(n_dim, n) for n in range(n_dim))
    return state.unit()


def snap_operator(phases, n_dim):
    phases = np.asarray(phases, dtype=float)
    diag = np.ones(n_dim, dtype=complex)
    diag[: len(phases)] = np.exp(-1j * phases)
    return Qobj(np.diag(diag))


def prepare_state(params, n_dim, n_layers, snap_levels):
    state = basis(n_dim, 0)
    idx = 0
    for _ in range(n_layers):
        alpha = params[idx] + 1j * params[idx + 1]
        idx += 2
        phases = params[idx : idx + snap_levels]
        idx += snap_levels
        state = displace(n_dim, alpha) * state
        state = snap_operator(phases, n_dim) * state
    return state.unit()


def fidelity(state_a, state_b):
    overlap = state_a.dag() * state_b
    if hasattr(overlap, "full"):
        val = overlap.full()[0, 0]
    else:
        val = complex(overlap)
    return float(np.abs(val) ** 2)


def optimize_snap_displacement(target, n_dim, n_layers, snap_levels, max_iter=200):
    n_params = n_layers * (2 + snap_levels)
    rng = np.random.default_rng(1234)
    init = np.zeros(n_params, dtype=float)
    init += 0.05 * rng.standard_normal(n_params)

    def cost(x):
        psi = prepare_state(x, n_dim, n_layers, snap_levels)
        return 1.0 - fidelity(psi, target)

    result = minimize(cost, init, method="Nelder-Mead", options={"maxiter": max_iter})
    return result


def unpack_params(params, n_layers, snap_levels):
    alphas = []
    phases = []
    idx = 0
    for _ in range(n_layers):
        alpha = params[idx] + 1j * params[idx + 1]
        idx += 2
        ph = params[idx : idx + snap_levels]
        idx += snap_levels
        alphas.append(alpha)
        phases.append(np.array(ph, dtype=float))
    return alphas, phases


def main():
    n_dim = 16
    r_target = 1.2
    r_prime = 1.0

    n_layers = 2
    snap_levels = 8
    max_iter = 200

    print("Coherent state prep via SNAP + Displacement (optimization)")
    print("Settings:")
    print("  n_dim:", n_dim)
    print("  r_target:", r_target)
    print("  r_prime:", r_prime)
    print("  n_layers:", n_layers)
    print("  snap_levels:", snap_levels)
    print("  max_iter:", max_iter)

    coeffs = lchs_coefficients(r_target, r_prime, n_dim)
    target = target_state_from_coeffs(coeffs)

    result = optimize_snap_displacement(
        target,
        n_dim=n_dim,
        n_layers=n_layers,
        snap_levels=snap_levels,
        max_iter=max_iter,
    )

    best_state = prepare_state(result.x, n_dim, n_layers, snap_levels)
    best_fid = fidelity(best_state, target)

    alphas, phases = unpack_params(result.x, n_layers, snap_levels)

    print("Optimization status:", result.message)
    print("Final fidelity:", best_fid)

    print("\nSuggested parameters for HybridLane:")
    for layer_idx, (alpha, ph) in enumerate(zip(alphas, phases), start=1):
        print(f"  Layer {layer_idx}:")
        print("    alpha:", alpha)
        print("    phases:", ph.tolist())

    print("\nHybridLane circuit sketch:")
    print("  for layer in layers:")
    print("    hqml.Displacement(abs(alpha), angle(alpha), wires=mode)")
    print("    for n, phi in enumerate(phases):")
    print("      hqml.SelectiveNumberArbitraryPhase(phi, n, wires=[ancilla, mode])")
    print("  qml.Squeezing(r_prime, 0.0, wires=mode)")


if __name__ == "__main__":
    main()
