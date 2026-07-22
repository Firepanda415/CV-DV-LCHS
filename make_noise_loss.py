"""Oscillator photon-loss evaluation at the map level (shared operating point).

Model: a single lumped photon-loss channel Lambda_eta (transmissivity
eta = 1 - gamma) acting on the prepared kernel state before the joint
evolution. For the heat benchmarks (H = 0) this is exhaustive for oscillator
loss up to a deterministic, software-compensable time rescaling: within every
L eigenbranch the evolution is a displacement, and loss commutes with
displacements as Lambda o D(alpha) = D(sqrt(eta) alpha) o Lambda, so
mid-evolution loss reduces exactly to an endpoint channel plus a known kernel
rescaling (not applied here; we evaluate the uncompensated endpoint channel).

The Kraus operators act in the physical Fock basis,
    A_k = sqrt(gamma^k / k!) * eta^{n/2} * a^k,
so each noisy branch map is the ideal postselected map evaluated with the
branch oscillator vector A_k |psi_N> (unnormalized; weights carried by the
vectors themselves):
    rho_out(u0) = sum_k Ktilde_k u0 u0^dag Ktilde_k^dag,
    Ktilde_k = <phi_r'| U (A_k |psi_N> otimes I_D).

Validation gate: at gamma = 0 the branch-0 map must reproduce
clean_core.exact_truncated_cv_map_h0 to machine precision for every boundary.

Outputs results_revision_v2/noise_loss.csv with, per (boundary, gamma):
photon moments of the kernel state, success probability p and conditional
infidelity 1 - F of the benchmark input u(0) = |01>, the fixed-scale map error
of the coherent branch, the pure coherent distortion, its state-level bound
(gamma/2) sqrt(<n^2>), and the incoherent branch fraction.
"""

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from clean_core import (
    KernelSpec,
    PAPER_TO_CODE_L_SCALE,
    exact_reference_map,
    exact_truncated_cv_map_h0,
    position_operator,
    system_blocks,
    truncated_oscillator_states,
    zeroth_moment_scale,
)
from revision_eval import _heat_system, _load_frozen_coefficients

SELECTED = Path("results_revision/selected_params_v2.json")
COEFFS = Path(
    "results_revision_v2/coefficients_dirichlet_r1p6_rp0p25_b0p5_N32_nq240.json"
)
OUTPUT = Path("results_revision_v2/noise_loss.csv")
META = Path("results_revision_v2/noise_loss.meta.json")

GAMMAS = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
BOUNDARIES = ["dirichlet", "neumann", "periodic"]
KRAUS_TAIL = 1e-12
BENCH_INDEX = 1


def map_from_states(system, kernel, psi_vec, phi_vec):
    """Same construction as exact_truncated_cv_map_h0, arbitrary psi vector."""

    l_block, h_block = system_blocks(system)
    if np.linalg.norm(h_block) > 1e-12:
        raise ValueError("H = 0 required")
    x_hat = position_operator(kernel.n_fock)
    l_coupling = PAPER_TO_CODE_L_SCALE * l_block
    x_values, x_vectors = np.linalg.eigh(x_hat)
    l_values, l_vectors = np.linalg.eigh(l_coupling)
    psi_tilde = x_vectors.conj().T @ psi_vec
    phi_tilde = x_vectors.conj().T @ phi_vec
    phases = np.exp(-1.0j * system.total_time * np.outer(x_values, l_values))
    response = (np.conj(phi_tilde) * psi_tilde) @ phases
    return np.asarray((l_vectors * response) @ l_vectors.conj().T, dtype=complex)


def kraus_branches(psi_vec, gamma):
    """Unnormalized branch vectors A_k psi in the physical Fock basis."""

    eta = 1.0 - gamma
    n_fock = psi_vec.shape[0]
    levels = np.arange(n_fock)
    branches = []
    total = 0.0
    for k in range(n_fock):
        if k == 0:
            lowered = psi_vec.copy()
        else:
            from scipy.special import gammaln

            lowered = np.zeros_like(psi_vec)
            src = np.arange(k, n_fock)
            amp = np.exp(0.5 * (gammaln(src + 1) - gammaln(src - k + 1)))
            lowered[: n_fock - k] = amp * psi_vec[src]
        branch = math.sqrt(gamma**k / math.factorial(k)) * (eta ** (levels / 2.0)) * lowered
        weight = float(np.vdot(branch, branch).real)
        branches.append(branch)
        total += weight
        if 1.0 - total < KRAUS_TAIL:
            break
    return branches, total


def main():
    selected = json.loads(SELECTED.read_text())["boundaries"]["dirichlet"]
    kernel = KernelSpec(
        r_target=selected["r_target"],
        r_prime=selected["r_prime"],
        beta=selected["beta"],
        n_coeff=selected["n_coeff"],
        n_fock=selected["n_fock"],
        n_quad=selected["n_quad"],
    )
    coeffs = _load_frozen_coefficients(COEFFS, kernel)
    if coeffs is None:
        raise RuntimeError(f"coefficient artifact does not match kernel: {COEFFS}")
    psi, phi = truncated_oscillator_states(kernel, coeffs)
    levels = np.arange(kernel.n_fock)
    prob = np.abs(psi) ** 2
    n_mean = float(np.sum(levels * prob))
    n2_mean = float(np.sum(levels**2 * prob))
    print(f"kernel-state photon moments: <n>={n_mean:.4f}  sqrt(<n^2>)={math.sqrt(n2_mean):.4f}")

    rows = []
    for boundary in BOUNDARIES:
        system = _heat_system(boundary, 1.0)
        reference = exact_reference_map(system)
        alpha = zeroth_moment_scale(kernel, coeffs)
        ideal = map_from_states(system, kernel, psi, phi)
        gate = exact_truncated_cv_map_h0(system, kernel, seed_state=coeffs)
        gate_dev = float(np.max(np.abs(ideal - gate)))
        if gate_dev > 1e-13:
            raise RuntimeError(f"{boundary}: gamma=0 gate failed ({gate_dev:.3e})")
        eps_ideal = float(
            np.linalg.norm(alpha * ideal - reference) / np.linalg.norm(reference)
        )
        print(f"[{boundary}] gamma=0 gate ok (dev {gate_dev:.1e}); ideal epsF={eps_ideal*100:.4f}%")

        u0 = np.zeros(system.dv_dim, dtype=complex)
        u0[BENCH_INDEX] = 1.0
        target = reference @ u0
        target_hat = target / np.linalg.norm(target)

        for gamma in GAMMAS:
            branches, mass = kraus_branches(psi, gamma)
            maps = [map_from_states(system, kernel, b, phi) for b in branches]
            outputs = [m @ u0 for m in maps]
            p_total = float(sum(np.vdot(v, v).real for v in outputs))
            overlap = float(
                sum(np.abs(np.vdot(target_hat, v)) ** 2 for v in outputs)
            )
            fidelity = overlap / p_total
            coherent = maps[0]
            eps_coherent = float(
                np.linalg.norm(alpha * coherent - reference) / np.linalg.norm(reference)
            )
            distortion = float(
                np.linalg.norm(coherent - ideal) / np.linalg.norm(ideal)
            )
            state_dev = float(np.linalg.norm(branches[0] - psi))
            bound = 0.5 * gamma * math.sqrt(n2_mean)
            incoherent_fraction = float(
                1.0 - np.vdot(outputs[0], outputs[0]).real / p_total
            )
            rows.append(
                {
                    "boundary": boundary,
                    "gamma": gamma,
                    "n_mean": n_mean,
                    "n2_sqrt": math.sqrt(n2_mean),
                    "kraus_branches": len(branches),
                    "kraus_mass": mass,
                    "p_succ": p_total,
                    "one_minus_F": 1.0 - fidelity,
                    "eps_F_coherent": eps_coherent,
                    "coherent_distortion": distortion,
                    "state_deviation": state_dev,
                    "state_bound": bound,
                    "incoherent_fraction": incoherent_fraction,
                    "eps_F_ideal": eps_ideal,
                }
            )
            print(
                f"   gamma={gamma:<7} p={p_total*100:6.2f}%  1-F={1-fidelity:.3e}  "
                f"epsF_coh={eps_coherent*100:6.3f}%  distortion={distortion:.3e} "
                f"(bound {bound:.3e})  incoh={incoherent_fraction:.3e}",
                flush=True,
            )

    fieldnames = list(rows[0])
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    META.write_text(
        json.dumps(
            {
                "output": str(OUTPUT),
                "model": (
                    "single lumped photon-loss channel on the prepared kernel "
                    "state before the H=0 joint evolution (exhaustive for "
                    "oscillator loss in the heat benchmarks up to a "
                    "deterministic, compensable kernel rescaling)"
                ),
                "coefficients": str(COEFFS),
                "coefficients_sha256": hashlib.sha256(COEFFS.read_bytes()).hexdigest(),
                "gammas": GAMMAS,
                "benchmark_input_index": BENCH_INDEX,
                "kraus_tail": KRAUS_TAIL,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUTPUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
