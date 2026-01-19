import sys
from pathlib import Path

import numpy as np
import pennylane as qml
import hybridlane as hqml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import heat_eq


def queued_operations(builder, amp: float):
    """Return the list of queued operations for a term helper."""
    with qml.queuing.AnnotatedQueue() as q:
        builder(amp, mode="m0")
    tape = qml.tape.QuantumScript.from_queue(q)
    return tape.operations


def assert_displacement(op, amp):
    assert isinstance(op, hqml.Displacement)
    assert op.wires.labels == ("m0",)
    assert np.isclose(op.parameters[0], amp)
    assert np.isclose(op.parameters[1], 0.0)


def assert_conditional_displacement(op, amp, control):
    assert isinstance(op, hqml.ConditionalDisplacement)
    assert op.wires.labels == (control, "m0")
    assert np.isclose(op.parameters[0], amp)
    assert np.isclose(op.parameters[1], 0.0)


def test_term_I_emits_single_mode_displacement():
    amp = 0.123
    ops = queued_operations(heat_eq.term_I, amp)
    assert len(ops) == 1
    assert_displacement(ops[0], amp)


def test_term_X1_wraps_conditional_disp_with_hadamards():
    amp = -0.42
    ops = queued_operations(heat_eq.term_X1, amp)
    assert [op.name for op in ops] == ["Hadamard", "ConditionalDisplacement", "Hadamard"]
    assert ops[0].wires.labels == (1,)
    assert_conditional_displacement(ops[1], amp, control=1)
    assert ops[2].wires.labels == (1,)


def test_term_XX_matches_expected_gate_pattern():
    amp = 0.77
    ops = queued_operations(heat_eq.term_XX, amp)
    expected_names = [
        "Hadamard",
        "Hadamard",
        "CNOT",
        "ConditionalDisplacement",
        "CNOT",
        "Hadamard",
        "Hadamard",
    ]
    assert [op.name for op in ops] == expected_names
    assert ops[0].wires.labels == (0,)
    assert ops[1].wires.labels == (1,)
    assert ops[2].wires.labels == (0, 1)
    assert_conditional_displacement(ops[3], amp, control=1)
    assert ops[4].wires.labels == (0, 1)
    assert ops[5].wires.labels == (0,)
    assert ops[6].wires.labels == (1,)


def test_term_YY_matches_expected_gate_pattern():
    amp = -0.13
    ops = queued_operations(heat_eq.term_YY, amp)
    expected_names = [
        "RZ",
        "RZ",
        "Hadamard",
        "Hadamard",
        "CNOT",
        "ConditionalDisplacement",
        "CNOT",
        "Hadamard",
        "Hadamard",
        "RZ",
        "RZ",
    ]
    assert [op.name for op in ops] == expected_names
    # Initial S rotations
    assert np.isclose(ops[0].parameters[0], np.pi / 2)
    assert ops[0].wires.labels == (0,)
    assert np.isclose(ops[1].parameters[0], np.pi / 2)
    assert ops[1].wires.labels == (1,)
    # Conditional displacement sandwiched by parity gadget
    assert_conditional_displacement(ops[5], amp, control=1)
    # Final S† rotations
    assert np.isclose(ops[-2].parameters[0], -np.pi / 2)
    assert ops[-2].wires.labels == (0,)
    assert np.isclose(ops[-1].parameters[0], -np.pi / 2)
    assert ops[-1].wires.labels == (1,)
