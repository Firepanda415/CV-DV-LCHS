import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import hybridlane as hqml
from numpy.polynomial.hermite import hermgauss
from scipy.linalg import expm
from scipy.special import eval_hermite, factorial


# Requested fixed cutoff
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

h = 1.0
lam_I = 2.0 / h**2
lam_X1 = -1.0 / h**2
lam_XX = -0.5 / h**2
lam_YY = -0.5 / h**2

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


@dataclass(frozen=True)
class ModelParams:
    kernel_beta: float = 0.8
    r_target: float = 2.0
    r_prime: float = 0.8
    alpha_disp: float = 1.0
    energy_shift: float = 0.0
    total_time: float = 0.4
    n_steps: int = 10
    init_qubits: tuple[int, int] = (0, 1)
    n_quad_points: int = 100
    fock_expansion_cutoff: float = 1e-2


def kernel_function(k_points: np.ndarray, kernel_beta: float) -> np.ndarray:
    # Shape-only kernel dependence (normalization cancels after state normalization).
    return np.exp((1.0 + 1.0j * k_points) ** kernel_beta) / (1.0 - 1.0j * k_points)


def lchs_coefficients(
    r_target: float,
    r_prime: float,
    n_dim: int,
    kernel_beta: float,
    n_quad_points: int = 100,
) -> np.ndarray:
    width_param = np.exp(r_prime)
    scale_factor = np.sqrt(2) * width_param
    gamma = np.exp(-2 * r_prime) - np.exp(-2 * r_target)
    if gamma <= 0:
        raise ValueError("Require r_prime < r_target so gamma > 0.")

    x_roots, weights = hermgauss(n_quad_points)
    k_points = x_roots * scale_factor
    g_part = kernel_function(k_points, kernel_beta)
    target_part = np.exp(-gamma * k_points**2) * g_part

    coeffs = []
    sqrt_pi = np.sqrt(np.pi)
    basis_prefactor = 1.0 / np.sqrt(sqrt_pi * width_param)
    for n in range(n_dim):
        fock_norm = 1.0 / np.sqrt((2**n) * factorial(n))
        h_val = eval_hermite(n, k_points / width_param)
        integrand = target_part * (basis_prefactor * fock_norm * h_val)
        coeffs.append(np.sum(weights * integrand) * scale_factor)

    coeffs = np.array(coeffs)
    norm = np.linalg.norm(coeffs)
    if np.isclose(norm, 0.0):
        raise RuntimeError("Computed LCHS coefficients have zero norm.")
    return coeffs / norm


def pauli_observable(label: str, wire: int):
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


def initial_dv_state(init_qubits: tuple[int, int]) -> np.ndarray:
    vec = np.zeros(4, dtype=complex)
    vec[init_qubits[0] * 2 + init_qubits[1]] = 1.0
    return vec


def dv_generator_matrix(alpha_disp: float, energy_shift: float) -> np.ndarray:
    # H' = H - E_s I, then globally scaled by alpha_disp.
    lam_I_shifted = lam_I - energy_shift
    base = (
        lam_I_shifted * np.kron(PAULI_I, PAULI_I)
        + lam_X1 * np.kron(PAULI_I, PAULI_X)
        + lam_XX * np.kron(PAULI_X, PAULI_X)
        + lam_YY * np.kron(PAULI_Y, PAULI_Y)
    )
    return alpha_disp * base


def cond_disp(control_qubit: int, amp: float, mode: str = "m0"):
    hqml.ConditionalDisplacement(amp, 0.0, wires=[control_qubit, mode])


def term_I(amp: float, mode: str = "m0"):
    hqml.Displacement(amp, 0.0, wires=mode)


def term_X1(amp: float, mode: str = "m0"):
    qml.H(1)
    cond_disp(1, amp, mode)
    qml.H(1)


def term_XX(amp: float, mode: str = "m0"):
    qml.H(0)
    qml.H(1)
    qml.CNOT(wires=[0, 1])
    cond_disp(1, amp, mode)
    qml.CNOT(wires=[0, 1])
    qml.H(0)
    qml.H(1)


def term_YY(amp: float, mode: str = "m0"):
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


def trotter_step(dt: float, alpha_disp: float, energy_shift: float, mode: str = "m0"):
    lam_I_shifted = lam_I - energy_shift
    scale = alpha_disp * dt
    term_I(lam_I_shifted * scale, mode)
    term_X1(lam_X1 * scale, mode)
    term_XX(lam_XX * scale, mode)
    term_YY(lam_YY * scale, mode)


@qml.qnode(DEV)
def cvdv_heat_postselect_fock_component(
    fock_n: int,
    total_time: float,
    n_steps: int,
    init_qubits: tuple[int, int],
    r_target: float,
    r_prime: float,
    alpha_disp: float,
    energy_shift: float,
    mode: str = "m0",
    fock_ancilla_wire: int = 0,
):
    # Backend-safe Fock preparation path (instead of unsupported FockStateVector).
    hqml.FockLadder(int(fock_n), wires=[fock_ancilla_wire, mode])
    if int(fock_n) % 2 == 1:
        qml.PauliX(fock_ancilla_wire)

    if not np.isclose(r_prime, 0.0):
        qml.Squeezing(r_prime, 0.0, wires=mode)

    for wire, bit in enumerate(init_qubits):
        if bit not in (0, 1):
            raise ValueError("init_qubits entries must be 0 or 1.")
        if bit:
            qml.PauliX(wire)

    dt = total_time / n_steps
    for _ in range(n_steps):
        trotter_step(dt, alpha_disp, energy_shift, mode=mode)

    if not np.isclose(r_target, 0.0):
        qml.Squeezing(-r_target, 0.0, wires=mode)

    proj = hqml.FockStateProjector(np.array(0), wires=mode)
    measurements = [qml.expval(proj)]
    for label_a, label_b in PAULI_COMBOS:
        obs = proj @ (pauli_observable(label_a, 0) @ pauli_observable(label_b, 1))
        measurements.append(qml.expval(obs))
    return tuple(measurements)


def evaluate_params(params: ModelParams) -> dict[str, float]:
    if params.n_steps <= 0:
        raise ValueError("n_steps must be > 0.")
    if params.r_prime >= params.r_target:
        raise ValueError("Require r_prime < r_target.")

    coeffs = lchs_coefficients(
        r_target=params.r_target,
        r_prime=params.r_prime,
        n_dim=MAX_FOCK_LEVEL,
        kernel_beta=params.kernel_beta,
        n_quad_points=params.n_quad_points,
    )

    weights = np.abs(coeffs) ** 2
    post_prob = 0.0
    pauli_accum = np.zeros(len(PAULI_COMBOS), dtype=float)

    for n, w in enumerate(weights):
        if w < params.fock_expansion_cutoff:
            continue
        res = cvdv_heat_postselect_fock_component(
            fock_n=n,
            total_time=params.total_time,
            n_steps=params.n_steps,
            init_qubits=params.init_qubits,
            r_target=params.r_target,
            r_prime=params.r_prime,
            alpha_disp=params.alpha_disp,
            energy_shift=params.energy_shift,
        )
        post_prob += float(w) * float(res[0])
        pauli_accum += float(w) * np.array(res[1:], dtype=float)

    if post_prob <= 0:
        return {
            "post_prob": post_prob,
            "fidelity": np.nan,
            "purity": np.nan,
        }

    rho_post = rebuild_density_from_paulis(pauli_accum) / post_prob
    purity = float(np.real(np.trace(rho_post @ rho_post)))

    theory = expm(
        -params.total_time * dv_generator_matrix(params.alpha_disp, params.energy_shift)
    ) @ initial_dv_state(params.init_qubits)
    norm_theory = np.linalg.norm(theory)
    if np.isclose(norm_theory, 0.0):
        fidelity = np.nan
    else:
        theory_norm = theory / norm_theory
        # Correct fidelity for pure target |u> and potentially mixed rho: F = <u|rho|u>.
        fidelity = float(np.real(np.vdot(theory_norm, rho_post @ theory_norm)))
        fidelity = float(np.clip(fidelity, 0.0, 1.0))

    return {
        "post_prob": float(post_prob),
        "fidelity": fidelity,
        "purity": purity,
    }


def run_sweep(
    param_name: str,
    values: np.ndarray,
    baseline: ModelParams,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for val in values:
        params = ModelParams(**{**baseline.__dict__, param_name: float(val)})
        if params.r_prime >= params.r_target:
            continue
        metrics = evaluate_params(params)
        rows.append({"value": float(val), **metrics})
        print(
            f"{param_name}={val:.4f}  post_prob={metrics['post_prob']:.6e}  "
            f"fidelity={metrics['fidelity']:.6e}  purity={metrics['purity']:.6e}"
        )
    return rows


def save_csv(path: Path, rows: list[dict[str, float]]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["value", "fidelity", "purity", "post_prob"]
        )
        writer.writeheader()
        writer.writerows(rows)


def save_csv_r_surface(path: Path, rows: list[dict[str, float]]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["r_target", "r_prime", "fidelity", "purity", "post_prob"]
        )
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path: Path, rows: list[dict[str, float]], param_label: str):
    x = np.array([r["value"] for r in rows], dtype=float)
    fidelity = np.array([r["fidelity"] for r in rows], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(x, fidelity, "o-", color="tab:blue")
    ax.set_xlabel(param_label)
    ax.set_ylabel("Fidelity")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Fidelity sensitivity: {param_label} (Fock cutoff={MAX_FOCK_LEVEL})")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_r_surface(
    r_target_values: np.ndarray,
    r_prime_values: np.ndarray,
    baseline: ModelParams,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for r_target in r_target_values:
        for r_prime in r_prime_values:
            if float(r_prime) >= float(r_target):
                continue
            params = ModelParams(
                **{
                    **baseline.__dict__,
                    "r_target": float(r_target),
                    "r_prime": float(r_prime),
                }
            )
            metrics = evaluate_params(params)
            rows.append(
                {
                    "r_target": float(r_target),
                    "r_prime": float(r_prime),
                    **metrics,
                }
            )
            print(
                f"r_target={r_target:.3f}, r_prime={r_prime:.3f}  "
                f"fidelity={metrics['fidelity']:.6e}  purity={metrics['purity']:.6e}"
            )
    return rows


def save_r_surface_plot(path: Path, rows: list[dict[str, float]]):
    x = np.array([r["r_target"] for r in rows], dtype=float)
    y = np.array([r["r_prime"] for r in rows], dtype=float)
    z = np.array([r["fidelity"] for r in rows], dtype=float)

    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(x, y, z, cmap="viridis", linewidth=0.2, antialiased=True)
    ax.scatter(x, y, z, color="k", s=12, alpha=0.5)
    ax.set_xlabel("r_target")
    ax.set_ylabel("r_prime")
    ax.set_zlabel("Fidelity")
    ax.set_zlim(0.0, 1.0)
    fig.colorbar(surf, ax=ax, shrink=0.7, pad=0.1, label="Fidelity")
    ax.set_title("Coupled sensitivity: fidelity vs (r_target, r_prime)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    output_dir = Path("sensitivity_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = ModelParams()
    # Denser point count for 2D single-parameter sensitivity plots.
    sweeps = {
        "kernel_beta": np.linspace(0.4, 1.2, 9),
        "r_target": np.linspace(1.2, 2.8, 9),
        "r_prime": np.linspace(0.3, 1.5, 9),
        "alpha_disp": np.linspace(0.6, 1.4, 9),
        "energy_shift": np.linspace(-1.0, 1.0, 9),
    }
    labels = {
        "kernel_beta": "Kernel hyperparameter beta",
        "r_target": "Post-selection squeezing r_target",
        "r_prime": "Prep-basis squeezing r_prime",
        "alpha_disp": "Controlled-displacement alpha",
        "energy_shift": "Energy shift E_s",
    }

    print("Running sensitivity sweeps...")
    for name, values in sweeps.items():
        print(f"\n--- Sweep: {name} ---")
        rows = run_sweep(name, values, baseline=baseline)
        if not rows:
            continue
        save_csv(output_dir / f"sensitivity_{name}.csv", rows)
        save_plot(output_dir / f"sensitivity_{name}.png", rows, labels[name])

    print("\n--- Coupled Sweep: (r_target, r_prime) ---")
    r_target_values = np.array([1.2, 1.6, 2.0, 2.4, 2.8])
    r_prime_values = np.array([0.3, 0.6, 0.9, 1.2, 1.5])
    r_surface_rows = run_r_surface(r_target_values, r_prime_values, baseline=baseline)
    if r_surface_rows:
        save_csv_r_surface(output_dir / "sensitivity_r_target_r_prime_surface.csv", r_surface_rows)
        save_r_surface_plot(
            output_dir / "sensitivity_r_target_r_prime_surface_3d.png",
            r_surface_rows,
        )

    print(f"\nDone. Outputs: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
