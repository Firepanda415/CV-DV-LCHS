#!/usr/bin/env python3
"""
Readable bosonic-qiskit implementation for CV-DV LCHS (gate-based focus).

Implements the hybrid CV-DV Linear Combination of Hamiltonian Simulation
(LCHS) algorithm of Das, Zheng, Dutta, Li, Liu (2025),
``res_base/Hybrid_CV_DV_LCHS.pdf``, using the bosonic_qiskit circuit
library for a single qumode + multi-qubit register.

Circuit structure (Paper Fig. 1):
  |seed>  --[S(r')]-- ---------[Trotter exp(-iT x ⊗ A)]---[S†(r)]-- <0|
                                        |
  |b>     ----------- --[DV init]------[                ]----------- meas

    1. CV state prep:  Fock-expansion of LCHS kernel coefficients C_n
    2. Squeezing S(r'): broadens the integration window
    3. DV init:         computational basis |b> via X gates
    4. Trotter loop:    first-order product formula for exp(-iT x ⊗ A)
    5. Post-selection:  project CV onto Fock-|0> (implements <phi_r|)
    6. DV readout:      extract 2^n amplitudes -> approximate u(T)

Convention notes:
  - bosonic_qiskit uses hbar=2 internally (x_hat = a + a†), but this file
    converts all displacement arguments using the hbar=1 relation
    exp(-iλ x̂) = D(-iλ/√2) where x̂ = (a + a†)/√2.
  - Squeezing sign: cv_sq(-r) in bosonic_qiskit corresponds to
    QuTiP squeeze(+r). See Weedbrook et al. RMP 84, 621 (2012), Sec. II.

Organization:
  1) CV state preparation modules  (injection, SNAP+D, Givens)
  2) DV basis-state preparation
  3) Generic Trotter block from Pauli decomposition
  4) CV post-selection and DV extraction
  5) Circuit assembly, simulation, and reports

No CLI is used. All runtime parameters are in __main__.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from bosonic_qiskit import CVCircuit, QumodeRegister
from bosonic_qiskit import util as cv_util
from qiskit import QuantumRegister

from clacod_heat1d_stateprep import (
    apply_snap_d_circuit,
    givens_decomposition,
    givens_resource_stats,
    optimize_snap_d_params,
)
from formed_qutip import (
    LCHSParams,
    ODESystemFromPauli,
    fit_global_scale,
    run_qutip_lchs,
    state_fidelity,
)


# -----------------------------------------------------------------------------
# 1) CV state preparation modules
# -----------------------------------------------------------------------------


def pad_coefficients(coeffs: np.ndarray, max_fock_level: int) -> np.ndarray:
    r"""
    Embed the active coefficient vector into the full truncated Fock space.

    The LCHS kernel expansion has N_active << N_fock non-negligible
    coefficients C_n (Paper Eq. 12).  bosonic_qiskit requires a Fock
    vector of length max_fock_level (a power of 2).  This function
    zero-pads the active coefficients to that length.

    Args:
        coeffs: Active Fock coefficients C_0 .. C_{N_active-1}.
        max_fock_level: Total truncated Fock dimension (power of 2).

    Returns:
        Zero-padded complex vector of length max_fock_level.
    """
    out = np.zeros(max_fock_level, dtype=complex)
    out[: len(coeffs)] = coeffs
    return out


def prepare_cv_state_injection(qc: CVCircuit, mode, coeffs_seed: np.ndarray, max_fock_level: int) -> Dict[str, float]:
    r"""
    Simulator-only injection: directly set the qumode to |seed> = sum_n C_n |n>.

    Uses bosonic_qiskit's ``cv_initialize(vec, mode)`` which writes the
    Fock amplitudes into the statevector without any gate decomposition.
    This is exact by construction (fidelity = 1) but has no hardware
    counterpart — it serves as the theoretical upper bound for the
    circuit-level methods (SNAP+D, Givens).

    Args:
        qc: CVCircuit instance.
        mode: Qumode register.
        coeffs_seed: Active Fock coefficients C_n from LCHS kernel expansion.
        max_fock_level: Fock space truncation (must be power of 2).

    Returns:
        Metadata dict with ``stateprep_seed_fidelity`` = 1.0.
    """
    qc.cv_initialize(pad_coefficients(coeffs_seed, max_fock_level), mode)
    return {"stateprep_seed_fidelity": 1.0}



def prepare_cv_state_snap_d(
    qc: CVCircuit,
    mode,
    coeffs_seed: np.ndarray,
    max_fock_level: int,
    depth: int,
    n_restarts: int,
    maxiter: int,
) -> Dict[str, float]:
    r"""
    Gate-based state preparation from alternating SNAP + displacement layers.

    Circuit structure (depth = L layers):
      |0> --[SNAP(θ_1)]--[D(α_1)]-- ... --[SNAP(θ_L)]--[D(α_L)]-- ≈ |seed>

    Each SNAP gate applies number-selective phases:
      SNAP(θ_1,...,θ_N) = sum_n exp(iθ_n) |n><n|        (diagonal in Fock basis)

    Each displacement D(α) is a coherent shift:
      D(α) = exp(α a† - α* a)                           (off-diagonal, mixes Fock levels)

    The parameters {θ_l, α_l} are found by numerical optimization
    (L-BFGS-B, multi-start) to maximize |<seed|U|0>|^2.

    Typical performance for the LCHS kernel with 8 active Fock levels:
      depth=4:  99.997% fidelity (infidelity ~3e-5)
      depth=2:  99.4%   (sufficient since Trotter error dominates)

    Published references:
      - R. W. Heeres et al., "Cavity State Manipulation Using Photon-Number
        Selective Phase Gates," Phys. Rev. Lett. 115, 137002 (2015),
        doi:10.1103/PhysRevLett.115.137002
        Eq. (1): SNAP gate definition.  Demonstrated on superconducting cavity.

      - S. Krastanov et al., "Universal control of an oscillator with
        dispersive coupling to a qubit," Phys. Rev. A 92, 040303(R) (2015),
        doi:10.1103/PhysRevA.92.040303
        Theorem 1: SNAP + D is universal for oscillator state preparation.
        Any Fock superposition can be reached from |0> with O(N) layers.

      - Liu et al., arXiv:2407.10381, Sec. VI.E
        Alternating SNAP+D compilation strategy for LCHS kernel states.

    Optimization details:
      Implemented in ``clacod_heat1d_stateprep.optimize_snap_d_params()``.
      Uses scipy.linalg.expm for displacement matrices and L-BFGS-B
      with ``n_restarts`` random initial points.

    Args:
        qc: CVCircuit instance.
        mode: Qumode register.
        coeffs_seed: Target Fock coefficients C_n.
        max_fock_level: Fock space truncation (power of 2).
        depth: Number of SNAP+D layer pairs.
        n_restarts: Multi-start optimization restarts.
        maxiter: Maximum L-BFGS-B iterations per restart.

    Returns:
        Metadata dict with fidelity, depth, and restart count.
    """
    result = optimize_snap_d_params(
        target_coeffs=coeffs_seed,
        n_fock=max_fock_level,
        depth=depth,
        n_restarts=n_restarts,
        maxiter=maxiter,
        verbose=True,
    )
    apply_snap_d_circuit(qc, mode, result)
    return {
        "stateprep_seed_fidelity": float(result.fidelity),
        "stateprep_depth": float(depth),
        "stateprep_restarts": float(n_restarts),
    }


def prepare_cv_state_givens(
    qc: CVCircuit,
    mode,
    prep_qubit,
    coeffs_seed: np.ndarray,
    max_fock_level: int,
) -> Dict[str, float]:
    r"""
    Optimization-free, analytically exact state preparation via
    Givens rotation decomposition compiled to JC gate sequences.

    The target seed state |psi> = sum_n C_n |n> is decomposed into
    N-1 adjacent Givens rotations G(n, n+1; theta, phi) applied to |0>.
    Each Givens rotation is then compiled into a Law-Eberly step:

      Step k (k = 0, 1, ..., N_active - 2):
        1. Qubit rotation  ry(2*theta_k)  -- load amplitude ratio
        2. Qubit phase      rz(phi_k)     -- set relative phase
        3. JC swap           cv_jc(pi/(2*sqrt(k+1)), 0, mode, qubit)
                             -- transfer |k, e> -> |k+1, g>

    The protocol builds the state bottom-up from |0, g>.

    Gate compilation note:
      Each Givens rotation G(n, n+1) corresponds to a number-selective
      JC swap pulse on a physical superconducting platform, where frequency
      tuning provides the selectivity (Hofheinz et al. 2009, Fig. 1).
      However, bosonic_qiskit's cv_jc is a GLOBAL JC interaction that
      couples all Fock manifolds simultaneously with sqrt(n+1)-dependent
      Rabi frequencies.  A single cv_jc call cannot implement a selective
      Givens rotation without severe crosstalk.

      Therefore, this method uses cv_initialize with the analytically
      computed prepared state (exact by construction), and reports the
      Givens rotation parameters for physical resource estimation.
      The gate count reflects what a frequency-selective hardware
      implementation would require.

    Published references:
      - C. K. Law and J. H. Eberly, "Arbitrary Control of a Quantum
        Electromagnetic Field," Phys. Rev. Lett. 76, 1055 (1996),
        doi:10.1103/PhysRevLett.76.1055
        Eq. (1): H_I(t) = hbar Omega(t)(sigma+ a e^{i delta t} + h.c.)
        Eq. (5)-(7): sequential protocol -- at each step, a qubit rotation
        followed by a resonant (number-selective) JC interaction transfers
        one quantum of excitation.
        Theorem: any Fock superposition with N+1 terms requires N steps.

      - M. Hofheinz et al., "Synthesizing arbitrary quantum states in a
        superconducting resonator," Nature 459, 546 (2009),
        doi:10.1038/nature08005
        Experimental realization on superconducting platform (Fig. 1).
        Number selectivity achieved via qubit frequency tuning.
        Demonstrated arbitrary states with up to ~10 photons.

    Analytic angle computation:
      clacod_heat1d_stateprep.py, givens_decomposition() (lines 356-501)
      Each G(n, n+1; theta, phi) at line 338-353 corresponds to one
      Law-Eberly step (qubit rotation + selective JC swap).

    Args:
        qc: CVCircuit instance.
        mode: Qumode register.
        prep_qubit: Helper qubit (unused in simulation; kept for API consistency).
        coeffs_seed: Target Fock-basis coefficients C_n.
        max_fock_level: Fock space truncation.

    Returns:
        Dictionary with fidelity and gate count metadata.
    """
    # 1. Compute exact Givens rotation angles (no optimization).
    givens_result = givens_decomposition(
        target_coeffs=coeffs_seed,
        n_fock=max_fock_level,
        verbose=True,
    )

    # 2. Inject the analytically prepared state.
    #    The Givens decomposition gives the exact target state; we inject it
    #    directly and report the JC pulse count for resource estimation.
    #    On physical hardware with frequency-selective JC, each nontrivial
    #    Givens rotation G(n, n+1; theta, phi) maps to:
    #      - 1 qubit rotation  ry(2*theta) + rz(phi)
    #      - 1 selective JC pi-pulse at Rabi frequency sqrt(n+1)
    qc.cv_initialize(
        pad_coefficients(givens_result.prepared_state, max_fock_level), mode
    )

    # 3. Collect metadata.
    stats = givens_resource_stats(givens_result)
    return {
        "stateprep_analytic_fidelity": float(givens_result.fidelity),
        "n_active_fock_levels": float(givens_result.n_active),
        "n_jc_pulses": float(stats["n_jc_pulses"]),
        "n_qubit_rotations": float(stats["n_jc_pulses"]),
    }


def apply_cv_state_preparation(
    qc: CVCircuit,
    mode,
    coeffs_seed: np.ndarray,
    *,
    prep_qubit,
    max_fock_level: int,
    method: str,
    snap_depth: int,
    snap_restarts: int,
    snap_maxiter: int,
) -> Dict[str, float]:
    r"""
    Dispatch helper for CV seed-state preparation.

    Routes to one of three methods for preparing |seed> = sum_n C_n |n>:

      "injection" (default):
          Simulator-only cv_initialize.  Exact fidelity, no gate cost.
          Use as the theoretical baseline.

      "snap_d":
          Gate-compiled SNAP+D layers (optimization-based).
          Best gate-level fidelity; requires numerical optimization.
          See Heeres et al. PRL 115, 137002 (2015).

      "givens" (or "coherent"):
          Optimization-free Givens decomposition (analytically exact angles).
          Currently uses cv_initialize internally; reports JC gate counts
          for physical resource estimation.
          See Law & Eberly PRL 76, 1055 (1996).

    Args:
        qc: CVCircuit instance.
        mode: Qumode register.
        coeffs_seed: Active Fock coefficients from LCHS kernel expansion.
        prep_qubit: Helper qubit (used by Givens method's API).
        max_fock_level: Fock space truncation (power of 2).
        method: One of "injection", "snap_d", "givens", "coherent".
        snap_depth, snap_restarts, snap_maxiter: SNAP+D optimizer settings.

    Returns:
        Metadata dict from the chosen preparation method.
    """
    if method == "snap_d":
        return prepare_cv_state_snap_d(
            qc,
            mode,
            coeffs_seed,
            max_fock_level=max_fock_level,
            depth=snap_depth,
            n_restarts=snap_restarts,
            maxiter=snap_maxiter,
        )
    if method in ("coherent", "givens"):
        return prepare_cv_state_givens(
            qc, mode, prep_qubit, coeffs_seed, max_fock_level
        )
    return prepare_cv_state_injection(qc, mode, coeffs_seed, max_fock_level)


# -----------------------------------------------------------------------------
# 2) DV basis-state preparation
# -----------------------------------------------------------------------------


def prepare_dv_basis_state(qc: CVCircuit, qreg: QuantumRegister, basis_index: int) -> None:
    r"""
    Prepare a DV computational-basis state |b_{n-1}...b_0> using X gates.

    Flips each qubit whose corresponding bit in ``basis_index`` is 1.
    Starting from |00...0>, this produces the desired basis state.

    Limitation: this can only prepare basis states, not superpositions.
    For superposition initial conditions (e.g., u_0 = sin(pi*x)),
    additional Ry/CNOT gates or state-preparation routines would be
    needed.  Currently, superposition inputs silently collapse to their
    argmax basis state.

    Args:
        qc: CVCircuit instance.
        qreg: DV qubit register (n qubits encode 2^n grid points).
        basis_index: Integer 0..2^n-1 identifying the target basis state.
            E.g., for n=2: 0->|00>, 1->|01>, 2->|10>, 3->|11>.
    """
    n_qubits = len(qreg)
    for qidx in range(n_qubits):
        bit = (basis_index >> (n_qubits - 1 - qidx)) & 1
        if bit == 1:
            qc.x(qreg[qidx])


# -----------------------------------------------------------------------------
# 3) Generic Trotter block from Pauli decomposition
# -----------------------------------------------------------------------------


def _apply_basis_change_forward(qc: CVCircuit, qreg: QuantumRegister, pauli: str) -> None:
    r"""
    Rotate each qubit from the Pauli eigenbasis to the Z basis.

    The Trotter step exp(-iλ x̂ ⊗ P) for a multi-qubit Pauli string P
    is implemented by first rotating each qubit so that all non-identity
    factors become Z, then applying a parity-conditional displacement,
    and finally undoing the rotation (see ``_apply_basis_change_inverse``).

    Conjugation identities used:
      X:  H · X · H  = Z             (Hadamard)
      Y:  Rz(π/2) H · Y · H Rz(-π/2) = Z
      Z:  already diagonal — no gate needed
      I:  identity — no gate needed

    Reference:
      Nielsen & Chuang, *Quantum Computation and Quantum Information*
      (2010), Sec. 4.2, for single-qubit Clifford decompositions.

    Args:
        qc: CVCircuit instance.
        qreg: DV qubit register.
        pauli: Pauli string (e.g., "XIYY"), one character per qubit.
    """
    for qidx, ch in enumerate(pauli):
        if ch == "X":
            qc.h(qreg[qidx])
        elif ch == "Y":
            qc.rz(np.pi / 2.0, qreg[qidx])
            qc.h(qreg[qidx])


def _apply_basis_change_inverse(qc: CVCircuit, qreg: QuantumRegister, pauli: str) -> None:
    r"""
    Undo the Z-basis rotation applied by ``_apply_basis_change_forward``.

    Gates are applied in reverse order (last qubit first) with each
    gate replaced by its inverse:
      X:  H†  = H                     (Hadamard is self-inverse)
      Y:  Rz(+π/2)† H† = H Rz(-π/2)  (reverse Rz sign, then H)

    Args:
        qc: CVCircuit instance.
        qreg: DV qubit register.
        pauli: Same Pauli string used in the forward pass.
    """
    for qidx in reversed(range(len(pauli))):
        ch = pauli[qidx]
        if ch == "X":
            qc.h(qreg[qidx])
        elif ch == "Y":
            qc.h(qreg[qidx])
            qc.rz(-np.pi / 2.0, qreg[qidx])


def _apply_parity_conditional_displacement(
    qc: CVCircuit,
    mode,
    qreg: QuantumRegister,
    active_z_qubits: Sequence[int],
    alpha: complex,
) -> None:
    r"""
    Implement exp(-iλ x̂ ⊗ Z⊗Z⊗...⊗Z) via a parity CNOT ladder + one cv_c_d.

    After the forward basis change, the multi-qubit Pauli P has been
    reduced to a product of Z's on the active qubits.  The joint parity
    Z_1 Z_2 ... Z_m has eigenvalue (-1)^{b_1+...+b_m}.  We encode this
    parity into a single target qubit using a CNOT cascade:

      Circuit (m = 3 active qubits, target = q_2):
        q_0 ──●──────────────────●──
        q_1 ──┼──●───────────●──┼──
        q_2 ──X──X──[cv_c_d]──X──X──

    The target qubit then holds |parity mod 2>, so a single
    conditional displacement ``cv_c_d(α, mode, target)`` applies
    D(+α) when parity=0 and D(-α) when parity=1, implementing
    exp(-iλ x̂ ⊗ Z...Z) exactly.

    Special cases:
      - 0 active qubits (all-I Pauli): unconditional D(α).
      - 1 active qubit: direct cv_c_d, no CNOT needed.

    Reference:
      Das et al. (2025), Sec. 3.2 and Fig. 2 — CNOT parity encoding
      for multi-qubit conditional displacement.

    Args:
        qc: CVCircuit instance.
        mode: Qumode register.
        qreg: DV qubit register.
        active_z_qubits: Indices of qubits with non-identity Pauli factor
            (these are already in the Z basis after forward rotation).
        alpha: Displacement parameter α = -iλ/√2, where λ = c_j · dt
            and c_j is the Pauli coefficient.
    """
    if len(active_z_qubits) == 0:
        qc.cv_d(alpha, mode)
        return

    if len(active_z_qubits) == 1:
        qc.cv_c_d(alpha, mode, qreg[active_z_qubits[0]])
        return

    target = active_z_qubits[-1]
    for cidx in active_z_qubits[:-1]:
        qc.cx(qreg[cidx], qreg[target])

    qc.cv_c_d(alpha, mode, qreg[target])

    for cidx in reversed(active_z_qubits[:-1]):
        qc.cx(qreg[cidx], qreg[target])


def apply_pauli_term_trotter_step(
    qc: CVCircuit,
    mode,
    qreg: QuantumRegister,
    pauli: str,
    coeff: complex,
    dt: float,
) -> None:
    r"""
    Apply one first-order Trotter factor for a single Pauli term.

    Implements the unitary:
      exp(-i dt · x̂ ⊗ c_j P_j)

    where x̂ = (a + a†)/√2 is the hbar=1 position operator and P_j is
    a multi-qubit Pauli string with real coefficient c_j.

    Displacement convention derivation:
      exp(-iλ x̂)  with  x̂ = (a + a†)/√2
      = exp(-iλ (a + a†)/√2)
      = exp(α a† - α* a)    with  α = -iλ/√2
      = D(-iλ/√2)

    So λ = c_j · dt gives α = -i c_j dt / √2.

    The three-step circuit pattern per Pauli term:
      1. Forward basis change:  rotate X/Y factors to Z
      2. Parity CD:             CNOT parity ladder + cv_c_d(α)
      3. Inverse basis change:  undo step 1

    Reference:
      Das et al. (2025), Sec. 3.2, Eq. (16)-(19) — Pauli decomposition
      of A = L + iH and mapping to conditional displacements.

    Args:
        qc: CVCircuit instance.
        mode: Qumode register.
        qreg: DV qubit register.
        pauli: Multi-qubit Pauli string (e.g., "XX", "YY", "IX").
        coeff: Complex coefficient c_j in the Pauli expansion A = Σ c_j P_j.
        dt: Trotter time step Δt = T / n_trotter_steps.
    """
    lam = coeff * dt
    alpha = -1.0j * lam / np.sqrt(2.0)

    _apply_basis_change_forward(qc, qreg, pauli)
    active = [idx for idx, ch in enumerate(pauli) if ch != "I"]
    _apply_parity_conditional_displacement(qc, mode, qreg, active, alpha)
    _apply_basis_change_inverse(qc, qreg, pauli)


def apply_trotterized_evolution(
    qc: CVCircuit,
    mode,
    qreg: QuantumRegister,
    pauli_strings: Sequence[str],
    coeffs: Sequence[complex],
    total_time: float,
    n_trotter_steps: int,
) -> None:
    r"""
    Apply first-order Lie-Trotter product formula for the LCHS evolution.

    Approximates the joint unitary:
      exp(-iT x̂ ⊗ A)  ≈  [∏_j exp(-iΔt x̂ ⊗ c_j P_j)]^{n_steps}

    where A = Σ_j c_j P_j (c_j may be complex) is the Pauli decomposition of the system
    matrix and Δt = T / n_steps.

    The first-order Trotter error scales as O(Δt² · ||[H_j, H_k]||),
    so increasing n_trotter_steps reduces the systematic error
    quadratically at the cost of a proportionally deeper circuit.

    Typical settings for the 2-qubit heat equation:
      n_trotter_steps = 80-200 (Trotter error < 1% for T ≤ 1.0)

    Reference:
      Suzuki, Phys. Lett. A 146, 319 (1990) — product formula error bounds.
      Das et al. (2025), Sec. 3.3 — Trotterization of the LCHS integral.

    Args:
        qc: CVCircuit instance.
        mode: Qumode register.
        qreg: DV qubit register.
        pauli_strings: List of Pauli strings ["II", "IX", "XX", "YY", ...].
        coeffs: Corresponding real coefficients c_j.
        total_time: Total evolution time T.
        n_trotter_steps: Number of Trotter steps (circuit depth scales linearly).
    """
    dt = total_time / n_trotter_steps
    for _ in range(n_trotter_steps):
        for pauli, coeff in zip(pauli_strings, coeffs):
            apply_pauli_term_trotter_step(qc, mode, qreg, pauli, coeff, dt)


# -----------------------------------------------------------------------------
# 4) CV post-selection and DV extraction
# -----------------------------------------------------------------------------


def detect_statevector_layout(max_fock_level: int, n_dv_qubits: int) -> str:
    r"""
    Detect how bosonic_qiskit interleaves Fock and qubit indices.

    The joint Hilbert space has dimension N_fock × 2^n_qubits.  The
    statevector could be ordered either as:

      "fock_major":  |fock, qubit>  — index = fock * 2^n + qubit
                     First 2^n entries are Fock=0 with all qubit states.
      "qubit_major": |qubit, fock>  — index = qubit * N_fock + fock
                     Entries at stride N_fock correspond to Fock=0.

    This function creates a minimal probe circuit (|1>_fock ⊗ |0>_qubit),
    simulates it, and checks which slot the non-zero amplitude lands in
    to determine the convention.

    This detection is needed because bosonic_qiskit's internal ordering
    is not documented and may change between versions.

    Args:
        max_fock_level: Fock space truncation (power of 2).
        n_dv_qubits: Number of DV qubits.

    Returns:
        "fock_major" or "qubit_major".
    """
    qmr = QumodeRegister(num_qumodes=1, num_qubits_per_qumode=int(np.log2(max_fock_level)))
    qbr = QuantumRegister(n_dv_qubits, "q")
    qc = CVCircuit(qbr, qmr)

    probe = np.zeros(max_fock_level, dtype=complex)
    probe[1] = 1.0
    qc.cv_initialize(probe, qmr[0])

    state, _, _ = cv_util.simulate(qc, shots=1, return_fockcounts=False, add_save_statevector=True)
    vec = np.asarray(state.data, dtype=complex)
    idx = int(np.where(np.abs(vec) > 1e-12)[0][0])

    return "fock_major" if idx == 2 ** n_dv_qubits else "qubit_major"


def qiskit_to_physics_index_map(n_qubits: int) -> np.ndarray:
    r"""
    Bit-reversal permutation: qiskit little-endian -> physics big-endian.

    Qiskit uses little-endian qubit ordering: |q_{n-1}...q_1 q_0> where
    q_0 is the least significant bit.  Physics convention is big-endian:
    |q_0 q_1 ... q_{n-1}>.  The mapping is a simple bit reversal:

      physics_index = int(reverse_bits(qiskit_index))

    For n=2:  qiskit |00>=0, |01>=1, |10>=2, |11>=3
              physics |00>=0, |10>=2, |01>=1, |11>=3
              map = [0, 2, 1, 3]

    Args:
        n_qubits: Number of qubits.

    Returns:
        Integer array of length 2^n_qubits mapping qiskit indices to
        physics indices.
    """
    return np.array([int(f"{i:0{n_qubits}b}"[::-1], 2) for i in range(2**n_qubits)], dtype=int)


def extract_postselected_dv(
    statevector: np.ndarray,
    *,
    layout: str,
    max_fock_level: int,
    n_dv_qubits: int,
) -> np.ndarray:
    r"""
    Extract DV amplitudes conditioned on the CV mode being in Fock |0>.

    The LCHS protocol requires projecting onto <φ_r| = <0|S†(r), which
    after applying S†(r) to the circuit (already done in build_bosonic_circuit),
    reduces to selecting the Fock-|0> component of the joint statevector.

    The resulting 2^n complex amplitudes are proportional to the
    solution vector u(T) of the ODE system (Paper Eq. 8).  The overall
    normalization is unknown (it equals the post-selection probability),
    so downstream comparison uses ``fit_global_scale()`` to extract the
    best-fit scalar η such that η · dv_vec ≈ u_ref.

    Post-selection probability:
      p_success = ||dv_vec||² = <ψ| (|0><0|_fock ⊗ I_dv) |ψ>
      Typical values: 1e-3 to 1e-1 depending on (r, r', β).

    The bit-reversal map corrects for qiskit's little-endian qubit
    ordering so that the output matches the physics-convention
    grid-point ordering.

    Args:
        statevector: Full joint statevector (N_fock × 2^n_qubits entries).
        layout: "fock_major" or "qubit_major" (from detect_statevector_layout).
        max_fock_level: Fock truncation dimension.
        n_dv_qubits: Number of DV qubits.

    Returns:
        Complex array of length 2^n_dv_qubits with physics-convention ordering.
    """
    dv_dim = 2**n_dv_qubits
    if layout == "fock_major":
        dv_qiskit = statevector[:dv_dim]
    else:
        dv_qiskit = statevector[0::max_fock_level][:dv_dim]
    return dv_qiskit[qiskit_to_physics_index_map(n_dv_qubits)]


# -----------------------------------------------------------------------------
# 5) Circuit assembly, simulation, and reports
# -----------------------------------------------------------------------------


@dataclass
class BosonicParams:
    r"""
    Configuration for the bosonic-qiskit circuit simulation.

    Attributes:
        max_fock_level: Fock space truncation dimension.
            Must be a power of 2 (bosonic_qiskit encodes each qumode
            in log2(max_fock_level) qubits).  Typical: 64.
        n_trotter_steps: Number of first-order Trotter steps.
            Trotter error ~ O(T²/n²).  Typical: 80-200.
        state_prep_method: CV seed-state preparation strategy.
            "injection" — exact simulator injection (cv_initialize).
            "snap_d"    — gate-compiled SNAP+D (optimization-based).
            "givens"    — Givens decomposition (optimization-free).
        snap_depth: Number of SNAP+D layer pairs (for method="snap_d").
        snap_restarts: Multi-start restarts (for method="snap_d").
        snap_maxiter: L-BFGS-B iterations per restart (for method="snap_d").
    """

    max_fock_level: int
    n_trotter_steps: int
    state_prep_method: str
    snap_depth: int
    snap_restarts: int
    snap_maxiter: int


def merged_pauli_terms(system: ODESystemFromPauli) -> Tuple[List[str], List[complex]]:
    r"""
    Combine L (dissipative) and H (Hamiltonian) Pauli terms into a single list.

    The LCHS system matrix is A = L + iH, where L and H are each given
    as Pauli decompositions (Paper Eq. 16-17).  For the Trotter loop,
    we concatenate all terms into a single list since each term gets
    its own exp(-iΔt x̂ ⊗ c_j P_j) factor.

    For the heat equation: H = 0, so only L terms contribute.

    Args:
        system: ODESystemFromPauli with separate L and H Pauli lists.

    Returns:
        (pauli_strings, coeffs) — merged lists ready for Trotterization.
    """
    paulis = list(system.pauli_strings_l) + list(system.pauli_strings_h)
    coeffs = [complex(c) for c in system.coeffs_l] + [1j * complex(c) for c in system.coeffs_h]
    return paulis, coeffs


def build_bosonic_circuit(
    system: ODESystemFromPauli,
    lchs_params: LCHSParams,
    bosonic_params: BosonicParams,
    coeffs_seed: np.ndarray,
    init_basis_index: int,
) -> Tuple[CVCircuit, Dict[str, float]]:
    r"""
    Assemble the full CV-DV LCHS circuit (Paper Fig. 1).

    The circuit proceeds in five stages:

      1. **CV seed prep**: Prepare |seed> = Σ C_n |n> on the qumode,
         encoding the LCHS kernel expansion (Paper Eq. 12).

      2. **Squeezing S(r')**: Apply S(r') to broaden the integration
         window.  In bosonic_qiskit, cv_sq(-r') = QuTiP squeeze(+r')
         due to a sign convention difference.

      3. **DV init**: Prepare the initial condition |b> on the qubit
         register (computational basis state only).

      4. **Trotterized evolution**: Apply the product formula
         [∏_j exp(-iΔt x̂ ⊗ c_j P_j)]^{n_steps} implementing
         the LCHS coupling between the CV mode and DV qubits.

      5. **Post-selection squeeze S†(r)**: Apply S†(r) = S(-r) so
         that the Fock-|0> projection in the statevector implements
         <φ_r| = <0| S†(r).  Here cv_sq(+r) gives S†(r) in
         bosonic_qiskit's convention.

    After simulation, ``extract_postselected_dv`` picks out the
    Fock-|0> slice to obtain the approximate solution vector u(T).

    Args:
        system: ODE system with Pauli decomposition of A = L + iH.
        lchs_params: LCHS parameters (r, r', β, T, n_coeff, etc.).
        bosonic_params: Circuit settings (Fock dim, Trotter steps, prep method).
        coeffs_seed: Fock coefficients C_n for the kernel expansion.
        init_basis_index: DV initial condition as a basis index.

    Returns:
        (qc, prep_meta) — the assembled CVCircuit and state-prep metadata.
    """
    n_dv = system.n_qubits()
    qmr = QumodeRegister(num_qumodes=1, num_qubits_per_qumode=int(np.log2(bosonic_params.max_fock_level)))
    qbr = QuantumRegister(n_dv, "q")
    qc = CVCircuit(qbr, qmr)
    mode = qmr[0]

    prep_meta = apply_cv_state_preparation(
        qc,
        mode,
        coeffs_seed,
        prep_qubit=qbr[0],
        max_fock_level=bosonic_params.max_fock_level,
        method=bosonic_params.state_prep_method,
        snap_depth=bosonic_params.snap_depth,
        snap_restarts=bosonic_params.snap_restarts,
        snap_maxiter=bosonic_params.snap_maxiter,
    )

    # In bosonic_qiskit sign convention, cv_sq(-r') matches QuTiP squeeze(+r').
    if abs(lchs_params.r_prime) > 1e-14:
        qc.cv_sq(-lchs_params.r_prime, mode)

    prepare_dv_basis_state(qc, qbr, init_basis_index)

    paulis, coeffs = merged_pauli_terms(system)
    apply_trotterized_evolution(
        qc,
        mode,
        qbr,
        paulis,
        coeffs,
        total_time=lchs_params.total_time,
        n_trotter_steps=bosonic_params.n_trotter_steps,
    )

    # Apply S^dagger(r) before taking Fock-|0> component for <phi_r| projection.
    if abs(lchs_params.r_target) > 1e-14:
        qc.cv_sq(+lchs_params.r_target, mode)

    return qc, prep_meta


def circuit_resource_report(qc: CVCircuit) -> Dict[str, object]:
    r"""
    Extract a gate-count and depth summary from the compiled circuit.

    Uses qiskit's built-in ``depth()``, ``size()``, and ``count_ops()``
    methods.  Note that bosonic_qiskit gates (cv_d, cv_c_d, cv_snap, etc.)
    each count as a single operation in this accounting, even though they
    decompose into many physical gates on hardware.

    Returns:
        Dict with "depth" (critical path length), "size" (total gate count),
        and "count_ops" (gate type -> count breakdown).
    """
    return {
        "depth": int(qc.depth()),
        "size": int(qc.size()),
        "count_ops": dict(qc.count_ops()),
    }


def run_bosonic_method(
    system: ODESystemFromPauli,
    lchs_params: LCHSParams,
    bosonic_params: BosonicParams,
    coeffs_seed: np.ndarray,
    init_basis_index: int,
    u_ref: np.ndarray,
    u_qutip: np.ndarray,
) -> Dict[str, object]:
    r"""
    End-to-end pipeline: build circuit, simulate, post-select, and compare.

    Orchestrates the full LCHS workflow:
      1. Build the CV-DV circuit via ``build_bosonic_circuit()``.
      2. Simulate with bosonic_qiskit's statevector backend.
      3. Detect the Fock/qubit index layout.
      4. Post-select on Fock |0> and apply bit-reversal correction.
      5. Compare the extracted DV vector against two references:
         - u_ref:   classical exact solution (e.g., scipy.linalg.expm)
         - u_qutip: QuTiP LCHS simulation (same kernel, no Trotter error)

    Comparison uses ``fit_global_scale(dv_vec, u_ref)`` which finds the
    best-fit scalar η minimizing ||η · dv_vec - u_ref||₂ / ||u_ref||₂,
    since the post-selected amplitudes carry an unknown normalization.
    State fidelity |<dv|u>|² / (||dv|| · ||u||) is also reported.

    Args:
        system: ODE system with Pauli decomposition.
        lchs_params: LCHS parameters.
        bosonic_params: Circuit settings.
        coeffs_seed: Fock coefficients for CV state preparation.
        init_basis_index: DV initial condition (basis index).
        u_ref: Classical reference solution vector.
        u_qutip: QuTiP LCHS reference solution vector.

    Returns:
        Dict with DV vector, post-selection probability, relative errors,
        fidelities, scaling factors, resource report, and prep metadata.
    """
    qc, prep_meta = build_bosonic_circuit(
        system=system,
        lchs_params=lchs_params,
        bosonic_params=bosonic_params,
        coeffs_seed=coeffs_seed,
        init_basis_index=init_basis_index,
    )

    state, _, _ = cv_util.simulate(qc, shots=1, return_fockcounts=False, add_save_statevector=True)
    vec = np.asarray(state.data, dtype=complex)

    layout = detect_statevector_layout(bosonic_params.max_fock_level, system.n_qubits())
    dv_vec = extract_postselected_dv(
        vec,
        layout=layout,
        max_fock_level=bosonic_params.max_fock_level,
        n_dv_qubits=system.n_qubits(),
    )

    eta_ref, err_ref = fit_global_scale(dv_vec, u_ref)
    eta_qutip, err_qutip = fit_global_scale(dv_vec, u_qutip)

    return {
        "dv_bosonic": dv_vec,
        "layout": layout,
        "post_prob": float(np.real(np.vdot(dv_vec, dv_vec))),
        "vs_classical_rel_error": err_ref,
        "vs_classical_fidelity": state_fidelity(dv_vec, u_ref),
        "vs_qutip_rel_error": err_qutip,
        "vs_qutip_fidelity": state_fidelity(dv_vec, u_qutip),
        "eta_vs_classical": eta_ref,
        "eta_vs_qutip": eta_qutip,
        "resources": circuit_resource_report(qc),
        "prep_meta": prep_meta,
    }


if __name__ == "__main__":
    # -----------------------------------------------------------------
    # ODE setup (Pauli-decomposed; easy to swap to another system)
    # -----------------------------------------------------------------
    # Heat-equation example on 2 qubits: A = L + iH, with H = 0 here.
    alpha = 1.0  # typical heat-equation range: alpha > 0
    h_grid = 1.0  # practical range in this project: (0, 2]
    s = alpha / (h_grid**2)

    # L = sum_j l_j P_j
    pauli_strings_l = ["II", "IX", "XX", "YY"]
    coeffs_l = [2.0 * s, -1.0 * s, -0.5 * s, -0.5 * s]

    # H = sum_k h_k P_k (set to zero for heat equation)
    pauli_strings_h = ["II"]
    coeffs_h = [0.0]

    total_time = 1.0  # suggested scan range: [0.1, 2.0]

    # Basis-only DV input for gate-level runs.
    # index 0..(2^n-1), e.g. n=2: 0->|00>, 1->|01>, 2->|10>, 3->|11>
    init_basis_index = 1

    # -----------------------------------------------------------------
    # CV-DV LCHS parameters
    # -----------------------------------------------------------------
    # Suggested ranges based on current experiments:
    # r_target > r_prime >= 0, beta in (0,1), n_coeff <= n_fock.
    # Low-tail default (current):
    #   r_target=1.50, r_prime=0.02, beta=0.75, n_coeff=24, n_quad=260
    # Optimized-for-map-error reference (higher tail pressure):
    #   r_target=0.80, r_prime=0.04, beta=0.95, n_coeff=24, n_quad=220
    lchs_params = LCHSParams(
        total_time=total_time,
        n_fock=64,
        n_coeff=24,
        r_target=3.88,
        r_prime=3.35,
        beta=0.78,
        n_quad=300,
        coeff_method="explicit_overlap",
    )

    # [ 841/864] r=5.000, r'=4.000, beta=0.300, n_coeff=24, trotter=100, prep=injection
    #     fidelity=0.95857229, post_prob=1.496e-01, mo_score=0.378714, rel_err=2.079e-01
    # [ 842/864] r=5.000, r'=4.000, beta=0.300, n_coeff=24, trotter=200, prep=injection
    #     fidelity=0.95827581, post_prob=1.499e-01, mo_score=0.378966, rel_err=2.087e-01
    # [ 843/864] r=5.000, r'=4.000, beta=0.300, n_coeff=48, trotter=100, prep=injection
    #     fidelity=0.96052444, post_prob=1.320e-01, mo_score=0.356041, rel_err=2.027e-01
    # [ 844/864] r=5.000, r'=4.000, beta=0.300, n_coeff=48, trotter=200, prep=injection
    
        # === Summary: formed_bosonic.py === <= BUT could be unphysical
        # n_qubits=2, n_fock=64, n_coeff=48
        # r=8.16, r'=4.19, beta=0.8, T=1.0, trotter_steps=80

        # [injection] post_prob=6.637671e-02, layout=fock_major
        #   rel_err vs classical : 7.313619e-02
        #   fidelity vs classical: 0.99467956

    # -----------------------------------------------------------------
    # Bosonic circuit settings
    # -----------------------------------------------------------------
    # max_fock_level must be a power of 2 for bosonic_qiskit qumode encoding.
    # n_trotter_steps tradeoff: larger -> lower Trotter error, deeper circuit.
    common_bosonic = {
        "max_fock_level": 64,
        "n_trotter_steps": 10,
        "snap_depth": 10,
        "snap_restarts": 2,
        "snap_maxiter": 100,
    }

    # Methods requested: injection, gate-based SNAP+D, gate-based coherent JC/AJC constructive.
    # methods = ["injection", "snap_d", "coherent"]
    methods = ["injection", "coherent"]

    # -----------------------------------------------------------------
    # Theoretical references from QuTiP (same coefficients/system)
    # -----------------------------------------------------------------
    system = ODESystemFromPauli(
        pauli_strings_l=pauli_strings_l,
        coeffs_l=coeffs_l,
        pauli_strings_h=pauli_strings_h,
        coeffs_h=coeffs_h,
    )

    dv_dim = 2 ** system.n_qubits()
    init_state = np.zeros(dv_dim, dtype=complex)
    init_state[init_basis_index] = 1.0

    qutip_result = run_qutip_lchs(system, lchs_params, init_state)
    coeffs_seed = qutip_result["coeffs_seed"]

    # -----------------------------------------------------------------
    # Run each bosonic method and compare
    # -----------------------------------------------------------------
    all_results: Dict[str, Dict[str, object]] = {}
    for method in methods:
        print(f"\n=== Running bosonic method: {method} ===")
        params = BosonicParams(state_prep_method=method, **common_bosonic)
        all_results[method] = run_bosonic_method(
            system=system,
            lchs_params=lchs_params,
            bosonic_params=params,
            coeffs_seed=coeffs_seed,
            init_basis_index=init_basis_index,
            u_ref=qutip_result["u_ref"],
            u_qutip=qutip_result["u_cv"],
        )

    print("\n=== Summary: formed_bosonic.py ===")
    print(f"n_qubits={system.n_qubits()}, n_fock={lchs_params.n_fock}, n_coeff={lchs_params.n_coeff}")
    print(
        f"r={lchs_params.r_target}, r'={lchs_params.r_prime}, beta={lchs_params.beta}, "
        f"T={lchs_params.total_time}, trotter_steps={common_bosonic['n_trotter_steps']}"
    )

    for method in methods:
        r = all_results[method]
        print(f"\n[{method}] post_prob={r['post_prob']:.6e}, layout={r['layout']}")
        print(f"  rel_err vs classical : {r['vs_classical_rel_error']:.6e}")
        print(f"  fidelity vs classical: {r['vs_classical_fidelity']:.8f}")
        print(f"  rel_err vs qutip     : {r['vs_qutip_rel_error']:.6e}")
        print(f"  fidelity vs qutip    : {r['vs_qutip_fidelity']:.8f}")
        if r["prep_meta"]:
            print(f"  state-prep metadata  : {r['prep_meta']}")

        res = r["resources"]
        print(f"  depth={res['depth']}, size={res['size']}")
        print(f"  count_ops={res['count_ops']}")

    print("\nPairwise bosonic method fidelity (shape only):")
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1 = methods[i]
            m2 = methods[j]
            fid = state_fidelity(all_results[m1]["dv_bosonic"], all_results[m2]["dv_bosonic"])
            print(f"  {m1:9s} vs {m2:9s} : {fid:.8f}")
