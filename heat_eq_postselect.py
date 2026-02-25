import numpy as np
import pennylane as qml
import hybridlane as hqml
from scipy.linalg import expm
from scipy.special import factorial, eval_hermite
from numpy.polynomial.hermite import hermgauss
from qutip import basis, squeeze

# ---- Device: one qumode "m0" + two qubits ----
MAX_FOCK_LEVEL = 32
if MAX_FOCK_LEVEL <= 0 or (MAX_FOCK_LEVEL & (MAX_FOCK_LEVEL - 1)) != 0:
    raise ValueError("MAX_FOCK_LEVEL must be a positive power of two.")
DEV = qml.device(
    "bosonicqiskit.hybrid",
    wires=[0, 1, "m0"],
    shots=None,
    max_fock_level=MAX_FOCK_LEVEL,
    hbar=2.0,
)

# ---- Problem constants ----
h = 1.0
alpha = 1.0
lam_I = 2.0 / h**2
lam_X1 = -1.0 / h**2
lam_XX = -0.5 / h**2
lam_YY = -0.5 / h**2


# ---- LCHS State Preparation (non-Gaussian) ----
def kernel_function(k_points, kernel_beta):
    """
    LCHS kernel from the base reference:
      g(k) = f(k) / (1 - i k),
      f(k) = exp(-(1 + i k)^beta) / C_beta,
      C_beta = 2*pi*exp(-2*beta).
    """
    c_beta = 2.0 * np.pi * np.exp(-2.0 * kernel_beta)
    return np.exp(-((1.0 + 1.0j * k_points) ** kernel_beta)) / (
        c_beta * (1.0 - 1.0j * k_points)
    )


def lchs_coefficients(r_target, r_prime, n_dim, kernel_beta=0.0, n_quad_points=220):
    """Compute normalized finite squeezed Fock expansion coefficients Cn."""
    width_param = np.exp(r_prime)
    scale_factor = np.sqrt(2) * width_param
    gamma = np.exp(-2 * r_prime) - np.exp(-2 * r_target)
    if gamma <= 0:
        raise ValueError("r_prime must be < r_target so gamma = e^{-2r'} - e^{-2r} > 0.")

    x_roots, weights = hermgauss(n_quad_points)
    p_points = x_roots * scale_factor

    g_part = kernel_function(p_points, kernel_beta)
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


def get_lchs_states(r_target, r_prime, n_dim, kernel_beta=0.0, n_quad_points=200):
    """
    Calculates the LCHS coefficients C_n using Gauss-Hermite quadrature.
    Returns the CV initial state and the post-selection state in Fock basis.
    """
    print(f"Preparing LCHS states (N={n_dim}, r_target={r_target})...")

    cn_array = lchs_coefficients(
        r_target, r_prime, n_dim, kernel_beta=kernel_beta, n_quad_points=n_quad_points
    )
    psi_seed = sum([cn_array[n] * basis(n_dim, n) for n in range(n_dim)]).unit()

    # Apply squeezing (basis transform)
    s_op_prime = squeeze(n_dim, r_prime)
    psi_osc_init = s_op_prime * psi_seed

    # Post-selection state phi_post = S(r_target) |0>
    s_op_target = squeeze(n_dim, r_target)
    phi_post = s_op_target * basis(n_dim, 0)

    print("States ready.")
    return psi_osc_init.full().flatten(), phi_post.full().flatten()


def displacement_from_beta(beta, r_prime):
    return 1j * np.sqrt(2) * beta * np.exp(r_prime)


def _apply_displacement(alpha, mode):
    if alpha is None or np.isclose(alpha, 0.0):
        return
    amp = float(np.abs(alpha))
    phi = float(np.angle(alpha))
    hqml.Displacement(amp, phi, wires=mode)


def apply_gaussian_prep(mode, r, alpha=None):
    hqml.Squeezing(r, 0.0, wires=mode)
    _apply_displacement(alpha, mode)


def apply_inverse_gaussian_prep(mode, r, alpha=None):
    _apply_displacement(-alpha if alpha is not None else None, mode)
    hqml.Squeezing(-r, 0.0, wires=mode)


def _get_qnode_tape(qnode):
    for attr in ("qtape", "tape", "_tape"):
        tape = getattr(qnode, attr, None)
        if tape is not None:
            return tape
    return None


def _print_circuit_stats(qnode):
    tape = _get_qnode_tape(qnode)
    if tape is None:
        print("Circuit stats: (tape unavailable)")
        return
    op_counts = {}
    for op in tape.operations:
        op_counts[op.name] = op_counts.get(op.name, 0) + 1
    meas_counts = {}
    for m in tape.measurements:
        meas_name = type(m).__name__
        meas_counts[meas_name] = meas_counts.get(meas_name, 0) + 1
    wires = list(tape.wires)
    n_qumodes = sum(1 for w in wires if isinstance(w, str))
    n_qubits = len(wires) - n_qumodes
    print("Circuit stats:")
    print("  wires:", wires)
    print("  qubits:", n_qubits, "qumodes:", n_qumodes)
    print("  operations:", len(tape.operations))
    print("  measurements:", len(tape.measurements))
    print("  op counts:", op_counts)
    print("  measurement counts:", meas_counts)


# ---- Pauli utilities ----
PAULI_I = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_LABELS = ("I", "X", "Y", "Z")
PAULI_MATRICES = {
    "I": PAULI_I,
    "X": PAULI_X,
    "Y": PAULI_Y,
    "Z": PAULI_Z,
}
PAULI_COMBOS = [(a, b) for a in PAULI_LABELS for b in PAULI_LABELS]


def pauli_observable(label, wire):
    if label == "I":
        return qml.Identity(wires=wire)
    if label == "X":
        return qml.PauliX(wires=wire)
    if label == "Y":
        return qml.PauliY(wires=wire)
    if label == "Z":
        return qml.PauliZ(wires=wire)
    raise ValueError(f"Unknown Pauli label: {label}")


def rebuild_density_from_paulis(expectations):
    rho = np.zeros((4, 4), dtype=complex)
    for value, (label_a, label_b) in zip(expectations, PAULI_COMBOS):
        rho += value * np.kron(PAULI_MATRICES[label_a], PAULI_MATRICES[label_b])
    return rho / 4.0


def principal_statevector(rho):
    eigvals, eigvecs = np.linalg.eigh(rho)
    idx = np.argmax(eigvals)
    vec = eigvecs[:, idx]
    return vec / np.linalg.norm(vec)


def sanitize_density_matrix(rho, eps=1e-12):
    """Hermitize, normalize trace, and clip tiny negative eigenvalues from tomography noise."""
    rho_h = 0.5 * (rho + rho.conj().T)
    trace_before = np.trace(rho_h)
    trace_before_real = float(np.real(trace_before))
    if np.isclose(trace_before_real, 0.0):
        raise RuntimeError("Reconstructed density matrix has near-zero trace.")

    rho_h = rho_h / trace_before_real
    eigvals, eigvecs = np.linalg.eigh(rho_h)
    eigvals_real = np.real(eigvals)
    min_eig_before = float(np.min(eigvals_real))
    clipped_weight = float(np.sum(np.clip(-eigvals_real, 0.0, None)))

    eigvals_clipped = np.clip(eigvals_real, 0.0, None)
    clipped_sum = float(np.sum(eigvals_clipped))
    if np.isclose(clipped_sum, 0.0):
        raise RuntimeError("Density matrix became zero after eigenvalue clipping.")

    rho_psd = eigvecs @ np.diag(eigvals_clipped / clipped_sum) @ eigvecs.conj().T
    rho_psd = 0.5 * (rho_psd + rho_psd.conj().T)

    trace_after = float(np.real(np.trace(rho_psd)))
    purity = float(np.real(np.trace(rho_psd @ rho_psd)))

    info = {
        "trace_before": trace_before_real,
        "trace_after": trace_after,
        "min_eig_before": min_eig_before,
        "clipped_weight": clipped_weight,
        "purity": purity,
    }
    if abs(np.imag(trace_before)) > eps:
        info["trace_imag_before"] = float(np.imag(trace_before))
    return rho_psd, info


def dv_generator_matrix():
    term_I = lam_I * np.kron(PAULI_I, PAULI_I)
    term_X1 = lam_X1 * np.kron(PAULI_I, PAULI_X)
    term_xx = lam_XX * np.kron(PAULI_X, PAULI_X)
    term_yy = lam_YY * np.kron(PAULI_Y, PAULI_Y)
    return term_I + term_X1 + term_xx + term_yy


def _dv_wire_count(device):
    return sum(1 for w in device.wires if not isinstance(w, str))


def initial_dv_state(init_qubits):
    vec = np.zeros(4, dtype=complex)
    index = init_qubits[0] * 2 + init_qubits[1]
    vec[index] = 1.0
    return vec


def _fmt_vec(vec, precision=6):
    arr = np.real_if_close(np.asarray(vec))
    return np.array2string(arr, precision=precision, suppress_small=True)


def print_original_problem_statement():
    print("\n=== 1) Original problem statement and equation ===")
    print("Source: res_base/Hybrid_CV_DV_LCHS (2).pdf, Sec. 3.5.2, Eq. (41)-(46)")
    print("Continuous PDE:")
    print("  ∂u(x,t)/∂t = α ∂²u(x,t)/∂x²,  u(x,0)=f(x),  u(0,t)=u(L,t)=0")
    print("Finite-difference ODE (4 interior points, 6-point grid including boundaries):")
    print("  d/dt u(t) = -(α/h²) T u(t)")
    t_matrix = np.real_if_close(dv_generator_matrix())
    print("  T =")
    print(_fmt_vec(t_matrix))
    print("Pauli decomposition used in the circuit:")
    print("  T = 2(I⊗I) - (I⊗X) - 1/2[(X⊗X) + (Y⊗Y)]")


def print_classical_solution(total_time, init_qubits):
    print("\n=== 2) Classical solution ===")
    print("Continuous closed-form (Dirichlet BC):")
    print("  u(x,t) = Σ_{n=1}^∞ b_n sin(nπx/L) exp[-α(nπ/L)^2 t],")
    print("  b_n = (2/L)∫_0^L f(x) sin(nπx/L) dx")
    print("Discrete reference used for this mapped 4-point system:")
    print("  u_classical(t) = exp[-α t T] u(0), with h=1")

    u0 = initial_dv_state(init_qubits)
    t_matrix = dv_generator_matrix()
    u_classical = expm(-alpha * total_time * t_matrix) @ u0
    print("  u(0) =", _fmt_vec(u0))
    print(f"  u_classical(t={total_time}) =", _fmt_vec(u_classical))

    return u_classical


def print_lchs_theory_target(total_time, init_qubits, u_classical=None):
    print("\n=== 2b) CV-DV LCHS theoretical target ===")
    print("Ideal CV-DV LCHS target for original PDE operator:")
    print("  u_lchs_theory(t) = exp[-α t A_eff] u(0)")
    print("  A_eff = T (original PDE operator)")

    u0 = initial_dv_state(init_qubits)
    a_eff = dv_generator_matrix()
    u_lchs_theory = expm(-alpha * total_time * a_eff) @ u0
    print(f"  u_lchs_theory(t={total_time}) =", _fmt_vec(u_lchs_theory))
    if u_classical is not None:
        norm_classical = np.linalg.norm(u_classical)
        if not np.isclose(norm_classical, 0.0):
            gap = np.linalg.norm(u_lchs_theory - u_classical) / norm_classical
            print("  Relative gap to classical target ||u_lchs_theory-u_classical||/||u_classical|| =", gap)
    return u_lchs_theory


# ---- CV-DV Hamiltonian Simulation ----

def cond_disp(control_qubit, amp, mode="m0"):
    hqml.ConditionalDisplacement(amp, 0.0, wires=[control_qubit, mode])


def term_I(amp, mode="m0"):
    hqml.Displacement(amp, 0.0, wires=mode)


def term_X1(amp, mode="m0"):
    qml.H(1)
    cond_disp(1, amp, mode)
    qml.H(1)


def term_XX(amp, mode="m0"):
    qml.H(0)
    qml.H(1)
    qml.CNOT(wires=[0, 1])
    cond_disp(1, amp, mode)
    qml.CNOT(wires=[0, 1])
    qml.H(0)
    qml.H(1)


def term_YY(amp, mode="m0"):
    qml.RZ(np.pi / 2, wires=0)
    qml.RZ(np.pi / 2, wires=1)
    qml.H(0)
    qml.H(1)
    qml.CNOT(wires=[0, 1])
    cond_disp(1, amp, mode)
    qml.CNOT(wires=[0, 1])
    qml.H(0)
    qml.H(1)
    qml.RZ(-np.pi / 2, wires=0)
    qml.RZ(-np.pi / 2, wires=1)


def trotter_step(dt, mode="m0"):
    term_I(lam_I * dt, mode)
    term_X1(lam_X1 * dt, mode)
    term_XX(lam_XX * dt, mode)
    term_YY(lam_YY * dt, mode)


@qml.qnode(DEV)
def cvdv_heat_postselect(
    total_time=0.1,
    n_steps=10,
    init_qubits=(0, 0),
    cv_init_state=None,
    r_target=0.0,
    r_prime=0.0,
    beta=0.0,
    use_gaussian_prep=True,
    use_displacement=False,
    mode="m0",
):
    if use_gaussian_prep:
        alpha_init = displacement_from_beta(beta, r_prime) if use_displacement else None
        apply_gaussian_prep(mode, r_prime, alpha=alpha_init)
    else:
        if cv_init_state is None:
            raise ValueError("cv_init_state is required when use_gaussian_prep=False.")
        qml.FockStateVector(cv_init_state, wires=mode)

    if init_qubits is not None:
        if len(init_qubits) != 2:
            raise ValueError("init_qubits must specify two qubits for wires [0, 1].")
        for wire, bit in enumerate(init_qubits):
            if bit not in (0, 1):
                raise ValueError("init_qubits entries must be 0 or 1.")
            if bit:
                qml.PauliX(wire)

    dt = total_time / n_steps
    for _ in range(n_steps):
        trotter_step(dt, mode=mode)

    # Post-select on phi_post = S(r_target) |0> (and optional displacement) via inverse ops
    if use_gaussian_prep:
        alpha_post = alpha_init if use_displacement else None
        apply_inverse_gaussian_prep(mode, r_target, alpha=alpha_post)
    else:
        if not np.isclose(r_target, 0.0):
            qml.Squeezing(-r_target, 0.0, wires=mode)

    proj = hqml.FockStateProjector(np.array(0), wires=mode)
    measurements = [qml.expval(proj)]

    for label_a, label_b in PAULI_COMBOS:
        obs = proj @ (pauli_observable(label_a, 0) @ pauli_observable(label_b, 1))
        measurements.append(qml.expval(obs))

    return tuple(measurements)


@qml.qnode(DEV)
def cvdv_heat_postselect_fock_component(
    fock_n,
    total_time=0.1,
    n_steps=10,
    init_qubits=(0, 0),
    r_target=0.0,
    r_prime=0.0,
    mode="m0",
    fock_ancilla_wire=0,
):
    hqml.FockState(int(fock_n), wires=[fock_ancilla_wire, mode])

    if not np.isclose(r_prime, 0.0):
        qml.Squeezing(r_prime, 0.0, wires=mode)

    if init_qubits is not None:
        if len(init_qubits) != 2:
            raise ValueError("init_qubits must specify two qubits for wires [0, 1].")
        for wire, bit in enumerate(init_qubits):
            if bit not in (0, 1):
                raise ValueError("init_qubits entries must be 0 or 1.")
            if bit:
                qml.PauliX(wire)

    dt = total_time / n_steps
    for _ in range(n_steps):
        trotter_step(dt, mode=mode)

    if not np.isclose(r_target, 0.0):
        qml.Squeezing(-r_target, 0.0, wires=mode)

    proj = hqml.FockStateProjector(np.array(0), wires=mode)
    measurements = [qml.expval(proj)]

    for label_a, label_b in PAULI_COMBOS:
        obs = proj @ (pauli_observable(label_a, 0) @ pauli_observable(label_b, 1))
        measurements.append(qml.expval(obs))

    return tuple(measurements)


if __name__ == "__main__":
    total_time = 1
    n_steps = 100
    dv_qubits = _dv_wire_count(DEV)
    if dv_qubits != 2:
        raise ValueError("This script assumes 2 DV qubits; update dv_generator_matrix/PAULI_COMBOS if you change wires.")
    excited_indices = (dv_qubits - 1,)
    initial_qubits = tuple(1 if i in excited_indices else 0 for i in range(dv_qubits))

    n_dim = MAX_FOCK_LEVEL
    r_target = 1.2
    r_prime = 0.3
    beta = 0.7
    kernel_beta = 0.8
    use_gaussian_prep = False
    use_displacement = True
    use_fock_expansion = True
    fock_expansion_cutoff = 1e-8

    print("Settings:")
    print("  total_time:", total_time)
    print("  n_steps:", n_steps)
    print("  dt:", total_time / n_steps)
    print("  init_qubits:", initial_qubits)
    print("  dv_qubits:", dv_qubits)
    print("  n_dim:", n_dim)
    print("  r_target:", r_target)
    print("  r_prime:", r_prime)
    if use_gaussian_prep and use_displacement:
        print("  beta:", beta)
    print("  kernel_beta:", kernel_beta)
    print("  use_gaussian_prep:", use_gaussian_prep)
    print("  use_displacement:", use_displacement)
    print("  use_fock_expansion:", use_fock_expansion)
    print("  fock_expansion_cutoff:", fock_expansion_cutoff)
    print("  max_fock_level:", MAX_FOCK_LEVEL)
    print("  hbar:", 2.0)
    print_original_problem_statement()
    u_classical = print_classical_solution(
        total_time=total_time,
        init_qubits=initial_qubits,
    )
    u_lchs_theory = print_lchs_theory_target(
        total_time=total_time,
        init_qubits=initial_qubits,
        u_classical=u_classical,
    )

    psi_init = None
    psi_lchs_init = None
    if n_dim != MAX_FOCK_LEVEL:
        raise ValueError("n_dim must match MAX_FOCK_LEVEL in this backend/fock-expansion setup.")
    if not use_gaussian_prep:
        psi_lchs_init, _ = get_lchs_states(r_target, r_prime, n_dim, kernel_beta=kernel_beta)
    if not use_gaussian_prep:
        psi_init = psi_lchs_init

    if use_fock_expansion:
        if use_gaussian_prep:
            raise ValueError("use_fock_expansion=True is incompatible with use_gaussian_prep=True.")
        coeffs = lchs_coefficients(r_target, r_prime, n_dim, kernel_beta=kernel_beta)
        weights = np.abs(coeffs) ** 2
        post_prob = 0.0
        pauli_accum = np.zeros(len(PAULI_COMBOS), dtype=float)
        for n, w in enumerate(weights):
            if w < fock_expansion_cutoff:
                continue
            res = cvdv_heat_postselect_fock_component(
                fock_n=n,
                total_time=total_time,
                n_steps=n_steps,
                init_qubits=initial_qubits,
                r_target=r_target,
                r_prime=r_prime,
            )
            post_prob += float(w) * float(res[0])
            pauli_accum += float(w) * np.array(res[1:], dtype=float)
        results = (post_prob,) + tuple(pauli_accum.tolist())
        _print_circuit_stats(cvdv_heat_postselect_fock_component)
        print("Note: fock expansion uses incoherent weights |Cn|^2 (mixture approximation).")
    else:
        results = cvdv_heat_postselect(
            total_time=total_time,
            n_steps=n_steps,
            init_qubits=initial_qubits,
            cv_init_state=psi_init,
            r_target=r_target,
            r_prime=r_prime,
            beta=beta,
            use_gaussian_prep=use_gaussian_prep,
            use_displacement=use_displacement,
        )
        _print_circuit_stats(cvdv_heat_postselect)

    post_prob = float(results[0])
    pauli_expectations = results[1:]

    print("Post-selection probability:", post_prob)
    if np.isclose(post_prob, 0.0):
        raise RuntimeError("Post-selection probability is zero; cannot normalize.")

    rho_raw = rebuild_density_from_paulis(pauli_expectations) / post_prob
    rho_post, rho_info = sanitize_density_matrix(rho_raw)

    u_target = u_classical
    u_lchs = u_lchs_theory
    target_label = "original classical heat-equation solution"

    norm_target = np.linalg.norm(u_target)
    if np.isclose(norm_target, 0.0):
        raise RuntimeError("Classical target norm is zero; cannot normalize.")

    u_target_norm = u_target / norm_target
    fidelity_mixed = float(np.real(np.vdot(u_target_norm, rho_post @ u_target_norm)))
    fidelity_mixed = float(np.clip(fidelity_mixed, 0.0, 1.0))
    infidelity = 1.0 - fidelity_mixed

    rho_target = np.outer(u_target_norm, u_target_norm.conj())
    rho_delta = 0.5 * ((rho_post - rho_target) + (rho_post - rho_target).conj().T)
    eig_delta = np.linalg.eigvalsh(rho_delta)
    trace_distance = float(0.5 * np.sum(np.abs(eig_delta)))
    hs_distance = float(np.linalg.norm(rho_post - rho_target, ord="fro"))
    psi_principal = principal_statevector(rho_post)
    overlap = np.vdot(u_target_norm, psi_principal)
    if np.abs(overlap) > 0:
        psi_principal = psi_principal * np.exp(-1j * np.angle(overlap))
    u_cvdv = np.sqrt(post_prob) * psi_principal
    pde_vector_error = float(np.linalg.norm(u_target - u_cvdv) / norm_target)

    pauli_conditional = np.array(pauli_expectations, dtype=float) / post_prob
    pauli_target = []
    for label_a, label_b in PAULI_COMBOS:
        pmat = np.kron(PAULI_MATRICES[label_a], PAULI_MATRICES[label_b])
        pauli_target.append(float(np.real(np.trace(rho_target @ pmat))))
    pauli_target = np.array(pauli_target, dtype=float)
    pauli_err = pauli_conditional - pauli_target
    pauli_rmse = float(np.sqrt(np.mean(pauli_err**2)))
    pauli_max_abs = float(np.max(np.abs(pauli_err)))

    print("Density diagnostics:")
    print("  Tr(rho_post) before sanitize:", rho_info["trace_before"])
    if "trace_imag_before" in rho_info:
        print("  Im[Tr(rho_post)] before sanitize:", rho_info["trace_imag_before"])
    print("  min eig before sanitize:", rho_info["min_eig_before"])
    print("  clipped negative eig weight:", rho_info["clipped_weight"])
    print("  Tr(rho_post) after sanitize:", rho_info["trace_after"])
    print("  purity Tr(rho_post^2):", rho_info["purity"])
    if np.isclose(norm_target, 0.0):
        theory_vs_classical_rel = np.nan
    else:
        theory_vs_classical_rel = np.linalg.norm(u_lchs - u_classical) / norm_target

    print("\n=== 3) Correctness report (recommended) ===")
    print("Primary target for implementation correctness:", target_label)
    print("Algorithmic mismatch to original PDE target:")
    print(
        "  ε_alg = ||u_lchs_theory - u_classical|| / ||u_classical||:",
        theory_vs_classical_rel,
    )
    print("Implementation mismatch to target (density-level, no vector proxy):")
    print("  Relative PDE-vector error (principal-eigenvector proxy):", pde_vector_error)
    print("  Fidelity F=<u_target_hat|rho_post|u_target_hat>:", fidelity_mixed)
    print("  Infidelity (1-F):", infidelity)
    print("  Trace distance D_tr(rho_post, |u_target><u_target|):", trace_distance)
    print("  Hilbert-Schmidt distance ||rho_post-rho_target||_F:", hs_distance)
    print("Tomography consistency (conditional Pauli expectations):")
    print("  Pauli RMSE:", pauli_rmse)
    print("  Pauli max abs error:", pauli_max_abs)
