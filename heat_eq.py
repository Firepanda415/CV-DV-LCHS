import numpy as np
import pennylane as qml
import hybridlane as hqml
from hybridlane.drawer.draw import draw_mpl
from scipy.linalg import expm

# ---- Device: one qumode "m0" + two qubits (implicit) ----
# Use a power-of-two Fock truncation (8, 16, ...)
dev = qml.device("bosonicqiskit.hybrid", shots=None, max_fock_level=8, hbar=2.0)

# ---- Problem constants (you can change these) ----
h = 1.0                   # grid spacing
alpha = 1.0               # thermal diffusivity scaling; absorbed into A if desired
# Coefficients λ_j in L = A = (1/h^2) * [ 2 I - (I ⊗ X) - 1/2 (XX + YY) ]
lam_I    =  2.0 / h**2
lam_X1   = -1.0 / h**2
lam_XX   = -0.5 / h**2
lam_YY   = -0.5 / h**2


# ---- For classical sanity checking ---- 

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
    # Normalize to guard against numeric drift
    return vec / np.linalg.norm(vec)



# ---- CV-DV Hamiltonian Simulation ---- 


def cond_disp(control_qubit, amp, mode="m0"):
    """Apply a conditional displacement D(amp) on `mode` controlled by Z on `control_qubit`.
       In HybridLane: ConditionalDisplacement takes (r, phi, wires=[qubit, mode]),
       where r * e^{i phi} = alpha in the standard D(alpha).
    """
    hqml.ConditionalDisplacement(amp, 0.0, wires=[control_qubit, mode])

def term_I(amp, mode="m0"):
    """Unconditional displacement by amp (implements e^{-i amp p})"""
    hqml.Displacement(amp, 0.0, wires=mode)

def term_X1(amp, mode="m0"):
    """e^{-i amp (X1 ⊗ p)} via H to Z-basis, conditional displacement on qubit 1"""
    qml.H(1)
    cond_disp(1, amp, mode)
    qml.H(1)

def term_XX(amp, mode="m0"):
    """e^{-i amp (X0 X1 ⊗ p)} via H on both -> ZZ, parity trick with a single control."""
    qml.H(0); qml.H(1)
    qml.CNOT(wires=[0,1])
    cond_disp(1, amp, mode)       # now it's Z1 ⊗ p
    qml.CNOT(wires=[0,1])
    qml.H(0); qml.H(1)

def term_YY(amp, mode="m0"):
    """e^{-i amp (Y0 Y1 ⊗ p)} via S;H on both -> ZZ, parity trick, then undo."""
    # Map Y -> Z using (S;H)
    qml.RZ(np.pi/2, wires=0); qml.RZ(np.pi/2, wires=1)  # S on both
    qml.H(0); qml.H(1)
    qml.CNOT(wires=[0,1])
    cond_disp(1, amp, mode)       # Z1 ⊗ p
    qml.CNOT(wires=[0,1])
    qml.H(0); qml.H(1)
    qml.RZ(-np.pi/2, wires=0); qml.RZ(-np.pi/2, wires=1)  # S^\dagger

def trotter_step(dt, mode="m0"):
    """One first-order Trotter step for e^{-i dt (L ⊗ p)} with L from Sec. 3.4.2."""
    # Amplitudes equal θ for D(θ) when hbar=2 (since e^{-iθ p} = D(θ))
    term_I(lam_I * dt, mode)
    term_X1(lam_X1 * dt, mode)
    term_XX(lam_XX * dt, mode)
    term_YY(lam_YY * dt, mode)

@qml.qnode(dev)
def cvdv_heat_evolution(total_time=0.1, n_steps=10, init_qubits=(0,0), mode="m0"):
    """Run the CV–DV hybrid oracle U = exp(-i total_time (L ⊗ p)) via Trotterization.
       Returns a couple of sanity-check observables.
    """
    # --- initial DV state |q0 q1>
    if init_qubits is not None:
        if len(init_qubits) != 2:
            raise ValueError("init_qubits must specify two qubits for wires [0, 1].")
        for wire, bit in enumerate(init_qubits):
            if bit not in (0, 1):
                raise ValueError("init_qubits entries must be 0 or 1.")
            if bit:
                qml.PauliX(wire)  # flip into |1> when requested

    # The qumode starts in its vacuum by default (no CV state-prep yet).
    # --- Trotterization ---
    dt = total_time / n_steps
    for _ in range(n_steps):
        trotter_step(dt, mode)

    # --- Example measurements (feel free to change) ---
    # Mode energy proxy and a DV correlator to ensure we entangled the bus:
    measurements = [
        hqml.expval(hqml.NumberOperator(mode)),
        qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)),
    ]

    for label_a, label_b in PAULI_COMBOS:
        obs = pauli_observable(label_a, 0) @ pauli_observable(label_b, 1)
        measurements.append(qml.expval(obs))

    return tuple(measurements)


if __name__ == "__main__":
    # execute the circuit
    total_time = 0.05
    n_steps = 50
    initial_qubits = (0, 1)
    results = cvdv_heat_evolution(
        total_time=total_time,
        n_steps=n_steps,
        init_qubits=initial_qubits,
    )
    exp_n, exp_zz = results[0], results[1]
    pauli_expectations = results[2:]
    print("⟨n_m0⟩ =", exp_n, "   ⟨Z0Z1⟩ =", exp_zz)

    # Extract the DV component of the hybrid state and compare with the DV-only propagator.
    rho_sim = rebuild_density_from_paulis(pauli_expectations)
    psi_sim = principal_statevector(rho_sim)

    dv_generator = dv_generator_matrix()
    theoretical_state = expm(-1j * total_time * dv_generator) @ initial_dv_state(initial_qubits)
    theoretical_state /= np.linalg.norm(theoretical_state)

    overlap = np.vdot(theoretical_state, psi_sim)
    fidelity = np.abs(overlap) ** 2
    diff_norm = np.linalg.norm(psi_sim - theoretical_state * np.exp(1j * np.angle(overlap)))

    zz_matrix = np.kron(PAULI_Z, PAULI_Z)
    theoretical_exp_zz = np.real(np.vdot(theoretical_state, zz_matrix @ theoretical_state))

    print("‖|ψ_sim⟩ - |ψ_theory⟩‖ =", diff_norm)
    print("Fidelity(|ψ_sim⟩, |ψ_theory⟩) =", fidelity)
    print("Simulated DV density matrix:\n", rho_sim)
    print("Theoretical ⟨Z0Z1⟩ =", theoretical_exp_zz)
    print("Deviation |⟨Z0Z1⟩_sim - ⟨Z0Z1⟩_theory| =", abs(exp_zz - theoretical_exp_zz))

    # Draw the circuit using HybridLane's matplotlib helper
    circuit_visualizer = draw_mpl(
        cvdv_heat_evolution,
        show_all_wires=True,
        style="pennylane",
    )
    fig, ax = circuit_visualizer(
        total_time=total_time,
        n_steps=n_steps,
        init_qubits=initial_qubits,
    )
    fig.canvas.manager.set_window_title("HybridLane CV-DV Heat Equation Circuit")
    fig.savefig("HybridLane CV-DV Heat Equation Circuit.png")
