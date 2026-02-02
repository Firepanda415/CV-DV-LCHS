import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import hybridlane as hqml
from scipy.linalg import expm

# ---- Device: one qumode "m0" + two qubits ----
# Use a power-of-two Fock truncation (8, 16, ...)
DEV = qml.device("bosonicqiskit.hybrid", shots=None, max_fock_level=8, hbar=2.0)

# ---- Problem constants (you can change these) ----
h = 1.0                   # grid spacing
alpha = 1.0               # thermal diffusivity scaling; absorbed into A if desired
# Coefficients lambda_j in L = A = (1/h^2) * [ 2 I - (I ⊗ X) - 1/2 (XX + YY) ]
lam_I = 2.0 / h**2
lam_X1 = -1.0 / h**2
lam_XX = -0.5 / h**2
lam_YY = -0.5 / h**2


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


def dv_generator_matrix():
    """Two-qubit generator matching the discrete Laplacian coefficients."""
    term_I = lam_I * np.kron(PAULI_I, PAULI_I)
    term_X1 = lam_X1 * np.kron(PAULI_I, PAULI_X)
    term_xx = lam_XX * np.kron(PAULI_X, PAULI_X)
    term_yy = lam_YY * np.kron(PAULI_Y, PAULI_Y)
    return term_I + term_X1 + term_xx + term_yy


def initial_dv_state(init_qubits):
    vec = np.zeros(4, dtype=complex)
    index = init_qubits[0] * 2 + init_qubits[1]
    vec[index] = 1.0
    return vec


# ---- CV-DV Hamiltonian Simulation ----

def cond_disp(control_qubit, amp, mode="m0"):
    """Apply a conditional displacement D(amp) on `mode` controlled by Z on `control_qubit`."""
    hqml.ConditionalDisplacement(amp, 0.0, wires=[control_qubit, mode])


def term_I(amp, mode="m0"):
    """Unconditional displacement by amp (implements e^{-i amp p})"""
    hqml.Displacement(amp, 0.0, wires=mode)


def term_X1(amp, mode="m0"):
    """e^{-i amp (X1 ⊗ p)} via H to Z-basis, conditional displacement on qubit 1."""
    qml.H(1)
    cond_disp(1, amp, mode)
    qml.H(1)


def term_XX(amp, mode="m0"):
    """e^{-i amp (X0 X1 ⊗ p)} via H on both -> ZZ, parity trick with a single control."""
    qml.H(0)
    qml.H(1)
    qml.CNOT(wires=[0, 1])
    cond_disp(1, amp, mode)
    qml.CNOT(wires=[0, 1])
    qml.H(0)
    qml.H(1)


def term_YY(amp, mode="m0"):
    """e^{-i amp (Y0 Y1 ⊗ p)} via S;H on both -> ZZ, parity trick, then undo."""
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
    """One first-order Trotter step for e^{-i dt (L ⊗ p)} with L from Sec. 3.4.2."""
    term_I(lam_I * dt, mode)
    term_X1(lam_X1 * dt, mode)
    term_XX(lam_XX * dt, mode)
    term_YY(lam_YY * dt, mode)


# ---- Gaussian-only LCHS approximation ----

def displacement_from_beta(beta, r_prime):
    """Match the optional displacement used in LCHS_CHO_2 (Gaussian-only approximation)."""
    return 1j * np.sqrt(2) * beta * np.exp(r_prime)


def _apply_displacement(alpha, mode):
    if alpha is None:
        return
    if np.isclose(alpha, 0.0):
        return
    amp = float(np.abs(alpha))
    phi = float(np.angle(alpha))
    hqml.Displacement(amp, phi, wires=mode)


def apply_gaussian_prep(mode, r, alpha=None):
    """Prepare a Gaussian state by squeezing and optional displacement."""
    hqml.Squeezing(r, 0.0, wires=mode)
    _apply_displacement(alpha, mode)


def apply_inverse_gaussian_prep(mode, r, alpha=None):
    """Apply the inverse of apply_gaussian_prep (for post-selection)."""
    _apply_displacement(-alpha if alpha is not None else None, mode)
    hqml.Squeezing(-r, 0.0, wires=mode)


def theoretical_heat_solution(times, init_qubits):
    """Classical heat equation solution u(t) = exp(alpha * t * L) u(0)."""
    generator = dv_generator_matrix()
    u0 = initial_dv_state(init_qubits)
    sol = []
    for t in times:
        # Heat equation is u_t = -L u, so propagator is exp(-alpha * t * L)
        u_t = expm(-alpha * t * generator) @ u0
        sol.append(u_t)
    return np.array(sol)


def align_to_reference(reference, vectors):
    """Align global phase of vectors to the reference via maximal overlap."""
    aligned = []
    for vec in vectors:
        phase = np.vdot(reference, vec)
        if np.abs(phase) > 0:
            vec = vec * np.exp(-1j * np.angle(phase))
        aligned.append(vec)
    return np.array(aligned)


def sweep_cvdv(times, step_dt, **kwargs):
    """Run the CV-DV circuit at each time and return post-selected DV statevectors."""
    post_probs = []
    psi_posts = []
    for t in times:
        n_steps = max(1, int(np.ceil(t / step_dt))) if step_dt > 0 else 1
        results = cvdv_heat_gaussian(total_time=float(t), n_steps=n_steps, **kwargs)
        post_prob = float(results[0])
        if post_prob <= 0:
            post_probs.append(post_prob)
            psi_posts.append(np.zeros(4, dtype=complex))
            continue

        rho_post = rebuild_density_from_paulis(results[1:]) / post_prob
        psi_post = principal_statevector(rho_post)

        post_probs.append(post_prob)
        psi_posts.append(psi_post)

    return np.array(psi_posts), np.array(post_probs)


def plot_pde_vs_numerical(times, theory, numerical, post_probs, output_prefix=None):
    """Plot normalized PDE solution vs post-selected numerical results."""
    basis_labels = ["|00>", "|01>", "|10>", "|11>"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for idx, ax in enumerate(axes.flat):
        ax.plot(times, np.real(theory[:, idx]), "k--", label="theory")
        ax.plot(times, np.real(numerical[:, idx]), "r.-", label="numerical")
        ax.set_title(f"Component {basis_labels[idx]}")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()

    fig.suptitle("Heat equation: theoretical vs CV-DV (post-selected)")
    fig.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(times, post_probs, "b-", label="post-selection probability")
    ax2.set_title("Post-selection probability vs time")
    ax2.set_xlabel("time")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()

    if output_prefix:
        fig.savefig(f"{output_prefix}_pde_vs_numerical.png", dpi=200)
        fig2.savefig(f"{output_prefix}_post_selection.png", dpi=200)


@qml.qnode(DEV)
def cvdv_heat_gaussian(
    total_time=0.1,
    n_steps=10,
    init_qubits=(0, 0),
    r_prime=6.0,
    r_target=8.0,
    beta=0.8,
    use_displacement=False,
    mode="m0",
):
    """Gaussian-only CV state prep and post-selection for the CV-DV LCHS circuit.
    
    NOTE: This uses a Gaussian state (Squeezed state) which implements a Gaussian kernel
    exp(-L^2 * t^2) rather than the Heat Equation kernel exp(-L * t).
    """
    # --- initial DV state |q0 q1>
    if init_qubits is not None:
        if len(init_qubits) != 2:
            raise ValueError("init_qubits must specify two qubits for wires [0, 1].")
        for wire, bit in enumerate(init_qubits):
            if bit not in (0, 1):
                raise ValueError("init_qubits entries must be 0 or 1.")
            if bit:
                qml.PauliX(wire)

    # --- CV Gaussian state prep (approximate LCHS state) ---
    alpha_init = displacement_from_beta(beta, r_prime) if use_displacement else None
    apply_gaussian_prep(mode, r_prime, alpha=alpha_init)

    # --- Trotterization ---
    dt = total_time / n_steps
    for _ in range(n_steps):
        trotter_step(dt, mode)

    # --- Post-selection on phi_post = S(r_target) |0> (and optional displacement) ---
    alpha_post = alpha_init if use_displacement else None
    apply_inverse_gaussian_prep(mode, r_target, alpha=alpha_post)

    # HybridLane's bosonic-qiskit backend expects numpy scalar data for FockStateProjector.
    proj = hqml.FockStateProjector(np.array(0), wires=mode)
    measurements = [qml.expval(proj)]

    # Post-selected DV correlators: <phi| (sigma_a ⊗ sigma_b) |phi>
    for label_a, label_b in PAULI_COMBOS:
        obs = proj @ (pauli_observable(label_a, 0) @ pauli_observable(label_b, 1))
        measurements.append(qml.expval(obs))

    return tuple(measurements)


if __name__ == "__main__":
    total_time = 0.05
    n_steps = 50
    initial_qubits = (0, 1)
    r_prime = 6.0
    r_target = 8.0
    beta = 0.8

    results = cvdv_heat_gaussian(
        total_time=total_time,
        n_steps=n_steps,
        init_qubits=initial_qubits,
        r_prime=r_prime,
        r_target=r_target,
        beta=beta,
        use_displacement=False,
    )

    post_prob = results[0]
    pauli_expectations = results[1:]

    # The I ⊗ I entry should match the post-selection probability.
    post_prob_from_paulis = pauli_expectations[0]

    print("Post-selection probability:", post_prob)
    print("Post-selection prob (from I⊗I):", post_prob_from_paulis)

    if np.isclose(post_prob, 0.0):
        raise RuntimeError("Post-selection probability is zero; cannot normalize.")

    rho_post = rebuild_density_from_paulis(pauli_expectations) / post_prob
    psi_post = principal_statevector(rho_post)

    print("Post-selected DV density matrix:\n", rho_post)
    print("Principal DV statevector:\n", psi_post)

    plot_total_time = total_time
    plot_points = 25
    step_dt = total_time / n_steps

    times = np.linspace(0.0, plot_total_time, plot_points)
    theory = theoretical_heat_solution(times, initial_qubits)
    theory_norm = theory / np.linalg.norm(theory, axis=1, keepdims=True)

    psi_posts, post_probs = sweep_cvdv(
        times,
        step_dt,
        init_qubits=initial_qubits,
        r_prime=r_prime,
        r_target=r_target,
        beta=beta,
        use_displacement=False,
    )

    aligned = np.zeros_like(psi_posts)
    for idx, vec in enumerate(psi_posts):
        ref = theory_norm[idx]
        aligned[idx] = align_to_reference(ref, [vec])[0]

    numerical_norm = np.zeros_like(aligned)
    for idx, vec in enumerate(aligned):
        norm = np.linalg.norm(vec)
        if norm > 0:
            numerical_norm[idx] = vec / norm

    plot_pde_vs_numerical(
        times,
        theory_norm,
        numerical_norm,
        post_probs,
        output_prefix="heat_eq_gaussian",
    )
    plt.show()
