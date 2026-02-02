import numpy as np
import pennylane as qml
import hybridlane as hqml
from scipy.linalg import expm
from scipy.special import factorial, eval_hermite
from numpy.polynomial.hermite import hermgauss
from qutip import basis, squeeze, displace

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


def get_lchs_states(r_target, r_prime, n_dim, n_quad_points=100):
    """
    Calculates the LCHS coefficients C_n using Gauss-Hermite quadrature.
    Returns the CV initial state and the post-selection state in Fock basis.
    """
    print(f"Preparing LCHS states (N={n_dim}, r_target={r_target})...")

    cn_array = lchs_coefficients(r_target, r_prime, n_dim, n_quad_points=n_quad_points)
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


def gaussian_cv_state(n_dim, r_prime, beta, use_displacement=False):
    psi = basis(n_dim, 0)
    s_op = squeeze(n_dim, r_prime)
    psi = s_op * psi
    if use_displacement:
        alpha = displacement_from_beta(beta, r_prime)
        d_op = displace(n_dim, alpha)
        psi = d_op * psi
    return psi.unit().full().flatten()


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
        trotter_step(dt, mode)

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
    hqml.FockLadder(int(fock_n), wires=[fock_ancilla_wire, mode])
    if int(fock_n) % 2 == 1:
        qml.PauliX(fock_ancilla_wire)

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
        trotter_step(dt, mode)

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
    r_target = 2.0
    r_prime = 0.8
    beta = 0.9
    use_gaussian_prep = False
    use_displacement = True
    compute_gaussian_fidelity = True
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
    print("  beta:", beta)
    print("  use_gaussian_prep:", use_gaussian_prep)
    print("  use_displacement:", use_displacement)
    print("  use_fock_expansion:", use_fock_expansion)
    print("  fock_expansion_cutoff:", fock_expansion_cutoff)
    print("  max_fock_level:", MAX_FOCK_LEVEL)
    print("  hbar:", 2.0)

    psi_init = None
    psi_lchs_init = None
    if n_dim != MAX_FOCK_LEVEL:
        raise ValueError("n_dim must match MAX_FOCK_LEVEL for FockStateVector.")
    if compute_gaussian_fidelity or not use_gaussian_prep:
        psi_lchs_init, _ = get_lchs_states(r_target, r_prime, n_dim)
    if not use_gaussian_prep:
        psi_init = psi_lchs_init
    if compute_gaussian_fidelity:
        psi_gauss = gaussian_cv_state(n_dim, r_prime, beta, use_displacement=use_displacement)
        overlap = np.vdot(psi_lchs_init, psi_gauss)
        fid = float(np.abs(overlap) ** 2)
        print("CV state fidelity |<psi_lchs|psi_gauss>|^2 =", fid)

    if use_fock_expansion:
        if use_gaussian_prep:
            raise ValueError("use_fock_expansion=True is incompatible with use_gaussian_prep=True.")
        coeffs = lchs_coefficients(r_target, r_prime, n_dim)
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

    rho_post = rebuild_density_from_paulis(pauli_expectations) / post_prob
    psi_post = principal_statevector(rho_post)

    dv_gen = dv_generator_matrix()
    u_theory = expm(-alpha * total_time * dv_gen) @ initial_dv_state(initial_qubits)
    norm_theory = np.linalg.norm(u_theory)
    if np.isclose(norm_theory, 0.0):
        raise RuntimeError("Theoretical solution norm is zero; cannot normalize.")

    u_theory_norm = u_theory / norm_theory

    overlap = np.vdot(u_theory_norm, psi_post)
    psi_post_aligned = psi_post
    if np.abs(overlap) > 0:
        psi_post_aligned = psi_post * np.exp(-1j * np.angle(overlap))

    diff_norm = np.linalg.norm(u_theory_norm - psi_post_aligned)

    scale = np.vdot(u_theory, psi_post_aligned)
    diff_scaled = np.linalg.norm(u_theory - scale * psi_post_aligned)

    print("||u_theory||:", norm_theory)
    print("Norm diff (normalized vectors):", diff_norm)
    print("Norm diff (best-fit scaled):", diff_scaled)

    u_cvdv = np.sqrt(post_prob) * psi_post_aligned
    diff_unscaled = np.linalg.norm(u_theory - u_cvdv)
    print("Norm diff (unnormalized u_theory vs sqrt(p)*psi_post):", diff_unscaled)
