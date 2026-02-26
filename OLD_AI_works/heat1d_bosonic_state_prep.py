#!/usr/bin/env python3
"""
State-preparation strategies for bosonic-qiskit heat-equation runs.

Current production path:
- compute C_n classically,
- inject the truncated CV superposition with cv_initialize,
- run PDE evolution on top of that injected state.

This module isolates preparation logic so gate-based preparation can be added
later without changing solver/sensitivity code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np
from bosonic_qiskit import CVCircuit


@dataclass(frozen=True)
class PreparedCVState:
    coeffs: np.ndarray
    injection_state: np.ndarray
    active_levels: int
    method: str


class CVStatePreparationStrategy(Protocol):
    name: str

    def prepare(
        self,
        qc: CVCircuit,
        mode: Sequence,
        *,
        r_target: float,
        r_prime: float,
        kernel_beta: float,
        n_dim: int,
        max_fock_level: int,
        fock_weight_cutoff: float,
    ) -> PreparedCVState:
        ...


class ClassicalInjectionStatePreparation:
    """Prepare CV state by direct injection of classically computed coefficients."""

    name = "classical_injection"

    def __init__(self, coefficient_builder: Callable[[float, float, int, float], np.ndarray]) -> None:
        self._coefficient_builder = coefficient_builder

    def prepare(
        self,
        qc: CVCircuit,
        mode: Sequence,
        *,
        r_target: float,
        r_prime: float,
        kernel_beta: float,
        n_dim: int,
        max_fock_level: int,
        fock_weight_cutoff: float,
    ) -> PreparedCVState:
        coeffs = np.asarray(
            self._coefficient_builder(
                float(r_target),
                float(r_prime),
                int(n_dim),
                float(kernel_beta),
            ),
            dtype=complex,
        )
        if coeffs.shape != (int(n_dim),):
            raise RuntimeError(
                f"Coefficient builder returned shape {coeffs.shape}; expected {(int(n_dim),)}."
            )

        inject = np.zeros(int(max_fock_level), dtype=complex)
        inject[: int(n_dim)] = coeffs
        qc.cv_initialize(inject, mode)

        # Keep the same squeezed-basis convention used in prior coherent runs.
        if not np.isclose(float(r_prime), 0.0):
            qc.cv_sq(-float(r_prime), mode)

        active_levels = int(np.sum(np.abs(coeffs) >= float(fock_weight_cutoff)))
        return PreparedCVState(
            coeffs=coeffs,
            injection_state=inject,
            active_levels=active_levels,
            method=self.name,
        )


class GateBasedStatePreparation:
    """Placeholder for future gate-based state preparation."""

    name = "gate_based"

    def prepare(
        self,
        qc: CVCircuit,
        mode: Sequence,
        *,
        r_target: float,
        r_prime: float,
        kernel_beta: float,
        n_dim: int,
        max_fock_level: int,
        fock_weight_cutoff: float,
    ) -> PreparedCVState:
        raise NotImplementedError(
            "Gate-based state preparation is not implemented yet. "
            "Use --state-prep classical-injection."
        )
