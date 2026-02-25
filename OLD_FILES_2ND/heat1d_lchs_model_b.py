#!/usr/bin/env python3
r"""
Model B (coherent aggregation) for 1D heat-equation CV-DV LCHS.

Model B keeps Model A untouched and changes only the aggregation rule:

  Model A (incoherent):
    rho_post \propto \sum_n |C_n|^2 rho_n

  Model B (coherent):
    |\psi_post> = \sum_n C_n |\psi_n>,   rho_post = |\psi_post><psi_post| / p_post
    p_post = <\psi_post|\psi_post>

where |\psi_n> is the unnormalized postselected DV branch from initial Fock |n>.

Implementation detail:
- We execute each Fock branch with Bosonic-Qiskit CVCircuit using the same gate
  semantics as hybridlane's Bosonic-Qiskit translator.
- We then project onto CV Fock |0> after S(-r) by taking the mode=0 sector of
  the full statevector.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

from bosonic_qiskit import CVCircuit, QumodeRegister
from bosonic_qiskit import util as cv_util
import numpy as np
from qiskit import QuantumRegister
from scipy.linalg import expm

import heat1d_lchs_cv_dv as core


class Heat1DLCHSModeBSolver:
    """Coherent CV-state preparation and coherent branch aggregation."""

    def __init__(self, cfg: core.HeatEquationConfig):
        core.validate_config(cfg)
        self.cfg = cfg
        self.coeffs = core.pauli_coefficients_for_heat(cfg.h)
        self.num_mode_qubits = int(np.log2(cfg.max_fock_level))

    def _simulate_branch_dv_amplitude(
        self,
        fock_n: int,
        r_target: float,
        r_prime: float,
        disp_phase: float,
    ) -> np.ndarray:
        """
        Execute one Fock branch and return the unnormalized postselected DV amplitude.

        The circuit mirrors Model A gate-by-gate, with parameter mappings taken from
        hybridlane's Bosonic-Qiskit translation:
          qml.Squeezing(r, 0)   -> cv_sq(-r)
          qml.Squeezing(-r, 0)  -> cv_sq(+r)
          Displacement(a, phi)  -> cv_d(a * exp(i phi))
          ConditionalDisplacement(a, phi) -> cv_c_d(a * exp(i phi))
        """
        qmr = QumodeRegister(num_qumodes=1, num_qubits_per_qumode=self.num_mode_qubits)
        qbr = QuantumRegister(2, "q")
        qc = CVCircuit(qbr, qmr, force_parameterized_unitary_gate=False)
        mode = qmr[0]

        qc.cv_initialize(int(fock_n), mode)
        if not np.isclose(r_prime, 0.0):
            qc.cv_sq(-float(r_prime), mode)

        if int(self.cfg.init_qubits[0]) == 1:
            qc.x(qbr[0])
        if int(self.cfg.init_qubits[1]) == 1:
            qc.x(qbr[1])

        dt = float(self.cfg.total_time) / float(self.cfg.n_steps)
        phase_factor = np.exp(1j * float(disp_phase))
        lam_i = self.coeffs["I"] * dt
        lam_x1 = self.coeffs["IX"] * dt
        lam_xx = self.coeffs["XX"] * dt
        lam_yy = self.coeffs["YY"] * dt

        for _ in range(self.cfg.n_steps):
            qc.cv_d(lam_i * phase_factor, mode)

            qc.h(qbr[1])
            qc.cv_c_d(lam_x1 * phase_factor, mode, qbr[1])
            qc.h(qbr[1])

            qc.h(qbr[0])
            qc.h(qbr[1])
            qc.cx(qbr[0], qbr[1])
            qc.cv_c_d(lam_xx * phase_factor, mode, qbr[1])
            qc.cx(qbr[0], qbr[1])
            qc.h(qbr[0])
            qc.h(qbr[1])

            qc.rz(np.pi / 2, qbr[0])
            qc.rz(np.pi / 2, qbr[1])
            qc.h(qbr[0])
            qc.h(qbr[1])
            qc.cx(qbr[0], qbr[1])
            qc.cv_c_d(lam_yy * phase_factor, mode, qbr[1])
            qc.cx(qbr[0], qbr[1])
            qc.h(qbr[0])
            qc.h(qbr[1])
            qc.rz(-np.pi / 2, qbr[0])
            qc.rz(-np.pi / 2, qbr[1])

        # Postselection transform: qml.Squeezing(-r_target,0) maps to cv_sq(+r_target).
        if not np.isclose(r_target, 0.0):
            qc.cv_sq(float(r_target), mode)

        state, _, _ = cv_util.simulate(
            qc,
            shots=1,
            return_fockcounts=False,
            add_save_statevector=True,
        )
        if state is None:
            raise RuntimeError("Bosonic-Qiskit simulation did not return a statevector.")

        # Registers are [q0, q1, mode_bits...]. Mode=0 subspace corresponds to first 4 entries.
        vec = np.asarray(state.data, dtype=complex)
        expected_size = 4 * self.cfg.max_fock_level
        if vec.size != expected_size:
            raise RuntimeError(f"Unexpected statevector size {vec.size}; expected {expected_size}.")

        # First 4 entries are in qiskit qubit ordering (q1,q0). Convert to model ordering (q0,q1).
        dv_qiskit = vec[:4]
        dv_qml_order = dv_qiskit[[0, 2, 1, 3]]
        return dv_qml_order

    def collect_component_table(
        self,
        r_target: float,
        r_prime: float,
        disp_phase: float,
        levels: Optional[Iterable[int]] = None,
    ) -> Dict[str, np.ndarray]:
        n_dim = self.cfg.n_dim
        dv_amps = np.zeros((n_dim, 4), dtype=complex)
        post_probs = np.zeros(n_dim, dtype=float)

        if levels is None:
            level_iter = range(n_dim)
        else:
            level_iter = sorted(set(int(i) for i in levels if 0 <= int(i) < n_dim))

        for n in level_iter:
            dv_unnorm = self._simulate_branch_dv_amplitude(
                fock_n=int(n),
                r_target=float(r_target),
                r_prime=float(r_prime),
                disp_phase=float(disp_phase),
            )
            dv_amps[n, :] = dv_unnorm
            post_probs[n] = float(np.real(np.vdot(dv_unnorm, dv_unnorm)))

        return {"dv_amps": dv_amps, "post_probs": post_probs}

    def evaluate(
        self,
        params: core.CVLCHSParams,
        component_table: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, float]:
        coeffs = core.lchs_coefficients(
            r_target=params.r_target,
            r_prime=params.r_prime,
            n_dim=self.cfg.n_dim,
            kernel_beta=params.kernel_beta,
        )
        weights = np.abs(coeffs) ** 2
        weights = weights / float(np.sum(weights))

        if component_table is None:
            active_levels = np.where(np.abs(coeffs) >= self.cfg.fock_weight_cutoff)[0]
            component_table = self.collect_component_table(
                r_target=params.r_target,
                r_prime=params.r_prime,
                disp_phase=params.disp_phase,
                levels=active_levels,
            )

        dv_amps = np.asarray(component_table["dv_amps"], dtype=complex)
        if dv_amps.shape != (self.cfg.n_dim, 4):
            raise RuntimeError(
                f"Invalid component table shape {dv_amps.shape}; expected {(self.cfg.n_dim, 4)}."
            )

        # Coherent branch sum.
        dv_unnorm = np.tensordot(coeffs, dv_amps, axes=(0, 0))
        post_prob = float(np.real(np.vdot(dv_unnorm, dv_unnorm)))
        if np.isclose(post_prob, 0.0):
            raise RuntimeError("Post-selection probability is zero; cannot normalize.")

        rho_raw = np.outer(dv_unnorm, dv_unnorm.conj()) / post_prob
        rho_post, rho_info = core.sanitize_density_matrix(rho_raw)

        t_matrix = core.dv_generator_matrix(self.cfg.h)
        u0 = core.initial_dv_state(self.cfg.init_qubits)
        u_target = expm(-self.cfg.alpha * self.cfg.total_time * t_matrix) @ u0
        norm_target = float(np.linalg.norm(u_target))
        if np.isclose(norm_target, 0.0):
            raise RuntimeError("Classical target norm is zero.")

        u_hat = u_target / norm_target
        rho_target = np.outer(u_hat, u_hat.conj())

        fidelity = float(np.real(np.vdot(u_hat, rho_post @ u_hat)))
        fidelity = float(np.clip(fidelity, 0.0, 1.0))

        rho_delta = 0.5 * ((rho_post - rho_target) + (rho_post - rho_target).conj().T)
        eig_delta = np.linalg.eigvalsh(rho_delta)
        trace_distance = float(0.5 * np.sum(np.abs(eig_delta)))
        hs_distance = float(np.linalg.norm(rho_post - rho_target, ord="fro"))

        psi_principal = core.principal_statevector(rho_post)
        overlap = np.vdot(u_hat, psi_principal)
        if np.abs(overlap) > 0.0:
            psi_principal = psi_principal * np.exp(-1j * np.angle(overlap))

        u_cvdv = np.sqrt(post_prob) * psi_principal
        pde_error = float(np.linalg.norm(u_target - u_cvdv) / norm_target)

        pauli_target = []
        pauli_conditional = []
        for la, lb in core.PAULI_COMBOS:
            pmat = np.kron(core.PAULI_MATRICES[la], core.PAULI_MATRICES[lb])
            pauli_target.append(float(np.real(np.trace(rho_target @ pmat))))
            pauli_conditional.append(float(np.real(np.trace(rho_post @ pmat))))
        pauli_target = np.asarray(pauli_target, dtype=float)
        pauli_conditional = np.asarray(pauli_conditional, dtype=float)
        pauli_err = pauli_conditional - pauli_target
        pauli_rmse = float(np.sqrt(np.mean(pauli_err**2)))
        pauli_max_abs = float(np.max(np.abs(pauli_err)))

        gamma = core.coefficient_gamma(params.r_target, params.r_prime)
        n_eff = float(1.0 / np.sum(weights**2))
        tail_mass = core.estimate_tail_mass(
            r_target=params.r_target,
            r_prime=params.r_prime,
            kernel_beta=params.kernel_beta,
            n_dim=self.cfg.n_dim,
            ref_dim=max(96, 2 * self.cfg.n_dim),
        )
        c_comm = core.commutator_constant(self.cfg.h)
        trotter_bound = (self.cfg.alpha * self.cfg.total_time) ** 2 * c_comm / (2.0 * self.cfg.n_steps)

        return {
            "post_prob": post_prob,
            "pde_error": pde_error,
            "fidelity": fidelity,
            "infidelity": 1.0 - fidelity,
            "trace_distance": trace_distance,
            "hs_distance": hs_distance,
            "pauli_rmse": pauli_rmse,
            "pauli_max_abs": pauli_max_abs,
            "gamma": float(gamma),
            "n_eff": n_eff,
            "tail_mass_est": tail_mass,
            "trotter_bound_est": float(trotter_bound),
            "rho_trace_before": rho_info["trace_before"],
            "rho_trace_after": rho_info["trace_after"],
            "rho_min_eig_before": rho_info["min_eig_before"],
            "rho_purity": rho_info["purity"],
            "weights_l1": float(np.sum(np.abs(weights))),
            "weights_l2_sq": float(np.sum(weights**2)),
            "active_levels": float(np.sum(np.abs(coeffs) >= self.cfg.fock_weight_cutoff)),
            "post_prob_incoherent_baseline": float(np.dot(weights, component_table["post_probs"])),
        }

