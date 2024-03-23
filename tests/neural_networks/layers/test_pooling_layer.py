# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Quantum Pooling Layer classes"""

from __future__ import annotations
import pytest
from pytest import raises
from qiskit.circuit import QuantumCircuit
from piqture.neural_networks.layers import (
    QuantumPoolingLayer2,
    QuantumPoolingLayer3,
)

# pylint: disable=not-context-manager


@pytest.fixture
def circuit1a():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(2, 2)
    circuit.h(1)
    circuit.measure(1, 1)
    with circuit.if_test((1, 1)):
        circuit.z(0)
    return circuit, [0]


@pytest.fixture
def circuit1b():
    """Fixture for fully connected layer circuit with False conditional"""
    circuit = QuantumCircuit(2, 2)
    circuit.cz(1, 0)
    return circuit, [0]


@pytest.fixture
def circuit2a():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(4, 4)
    circuit.h([1, 3])
    circuit.measure([1, 3], [1, 3])
    with circuit.if_test([1, 1]):
        circuit.z(0)
    with circuit.if_test([3, 1]):
        circuit.z(2)
    return circuit, [0, 2]


@pytest.fixture
def circuit2b():
    """Fixture for fully connected layer circuit with False conditional"""
    circuit = QuantumCircuit(4, 4)
    circuit.cz(1, 0)
    circuit.cz(3, 2)
    return circuit, [0, 2]


@pytest.fixture
def circuit3a():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(8, 8)
    circuit.h([3, 7])
    circuit.measure([3, 7], [3, 7])
    with circuit.if_test([3, 1]):
        circuit.z(2)
    with circuit.if_test([7, 1]):
        circuit.z(5)
    return circuit, [2, 5]


@pytest.fixture
def circuit3b():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(8, 8)
    circuit.cz(3, 2)
    circuit.cz(7, 5)
    return circuit, [2, 5]


@pytest.fixture
def circuit4a():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(6, 6)
    circuit.h([3])
    circuit.measure([3], [3])
    with circuit.if_test([3, 1]):
        circuit.z(2)
    return circuit, [2, 5]


@pytest.fixture
def circuit4b():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(6, 6)
    circuit.cz(3, 2)
    return circuit, [2, 5]


@pytest.fixture
def circuit5a():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(3, 3)
    circuit.h([0, 2])
    circuit.measure([0, 2], [0, 2])
    with circuit.if_test([0, 1]):
        with circuit.if_test([2, 1]):
            circuit.z(1)
    return circuit, [1]


@pytest.fixture
def circuit5b():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(3, 3)
    circuit.cz(0, 1)
    circuit.cz(2, 1)
    return circuit, [1]


@pytest.fixture
def circuit6a():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(6, 6)
    circuit.h([0, 2, 4])
    circuit.measure([0, 2, 4], [0, 2, 4])
    with circuit.if_test([0, 1]):
        with circuit.if_test([2, 1]):
            circuit.z(1)
    with circuit.if_test([2, 1]):
        with circuit.if_test([4, 1]):
            circuit.z(3)
    return circuit, [1, 3, 5]


@pytest.fixture
def circuit6b():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(6, 6)
    circuit.cz(0, 1)
    circuit.cz(2, 1)
    circuit.cz(2, 3)
    circuit.cz(4, 3)
    return circuit, [1, 3, 5]


@pytest.fixture
def circuit7a():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(8, 8)
    circuit.h([2, 5])
    circuit.measure([2, 5], [2, 5])
    with circuit.if_test([2, 1]):
        with circuit.if_test([5, 1]):
            circuit.z(3)
    return circuit, [3, 7]


@pytest.fixture
def circuit7b():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(8, 8)
    circuit.cz(2, 3)
    circuit.cz(5, 3)
    return circuit, [3, 7]


class TestQuantumPoolingLayer2:
    """Tests for Quantum Pooling Layer (2) class"""

    # pylint: disable=too-few-public-methods

    @pytest.mark.parametrize(
        "num_qubits, circuit, unmeasured_bits, conditional",
        [
            (2, None, None, None),
            (None, QuantumCircuit(2, 2), None, 2.5),
            (None, None, [2, 5], "abc"),
        ],
    )
    def test_conditional(self, num_qubits, circuit, unmeasured_bits, conditional):
        """Tests for the type of conditional input."""
        with raises(TypeError, match="The input conditional must be of the type bool."):
            _ = QuantumPoolingLayer2(num_qubits, circuit, unmeasured_bits, conditional)

    @pytest.mark.parametrize(
        "num_qubits, circuit, unmeasured_bits, conditional, resulting_circuit",
        [
            (2, None, None, False, "circuit1b"),
            (2, None, None, True, "circuit1a"),
            (None, QuantumCircuit(4, 4), None, False, "circuit2b"),
            (None, QuantumCircuit(4, 4), None, True, "circuit2a"),
            (None, None, [2, 3, 5, 7], False, "circuit3b"),
            (None, None, [2, 3, 5, 7], True, "circuit3a"),
            (None, None, [2, 3, 5], False, "circuit4b"),
            (None, None, [2, 3, 5], True, "circuit4a"),
        ],
    )
    def test_build_layer(
        self,
        request,
        num_qubits,
        circuit,
        unmeasured_bits,
        conditional,
        resulting_circuit,
    ):
        # pylint: disable=too-many-arguments
        """Tests the build_layer method of QuantumPoolingLayer2 class."""
        circuit, new_unmeasured_bits = QuantumPoolingLayer2(
            num_qubits,
            circuit,
            unmeasured_bits,
            conditional,
        ).build_layer()
        resulting_circuit, resulting_unmeasured_bits = request.getfixturevalue(
            resulting_circuit
        )
        print(circuit)
        print(resulting_circuit)
        assert circuit == resulting_circuit
        assert new_unmeasured_bits == resulting_unmeasured_bits


class TestQuantumPoolingLayer3:
    """Tests for Quantum Pooling Layer (3) class"""

    @pytest.mark.parametrize(
        "num_qubits, circuit, unmeasured_bits",
        [(2, None, None), (None, QuantumCircuit(2, 2), None), (None, None, [2, 3])],
    )
    def test_three_qubits(self, num_qubits, circuit, unmeasured_bits):
        """Tests for presence of at least 3 qubits in the circuit."""
        with raises(ValueError):
            _ = QuantumPoolingLayer3(num_qubits, circuit, unmeasured_bits)

    @pytest.mark.parametrize(
        "num_qubits, circuit, unmeasured_bits, conditional",
        [
            (3, None, None, None),
            (None, QuantumCircuit(3, 3), None, 2.5),
            (None, None, [2, 3, 5], "abc"),
        ],
    )
    def test_conditional(self, num_qubits, circuit, unmeasured_bits, conditional):
        """Tests for the type of conditional input."""
        with raises(TypeError, match="The input conditional must be of the type bool."):
            _ = QuantumPoolingLayer3(num_qubits, circuit, unmeasured_bits, conditional)

    @pytest.mark.parametrize(
        "num_qubits, circuit, unmeasured_bits, conditional, resulting_circuit",
        [
            (3, None, None, False, "circuit5b"),
            (3, None, None, True, "circuit5a"),
            (None, QuantumCircuit(6, 6), None, False, "circuit6b"),
            (None, QuantumCircuit(6, 6), None, True, "circuit6a"),
            (None, None, [2, 3, 5, 7], False, "circuit7b"),
            (None, None, [2, 3, 5, 7], True, "circuit7a"),
        ],
    )
    def test_build_layer(
        self,
        request,
        num_qubits,
        circuit,
        unmeasured_bits,
        conditional,
        resulting_circuit,
    ):
        # pylint: disable=too-many-arguments
        """Tests the build_layer method of QuantumPoolingLayer2 class."""
        circuit, new_unmeasured_bits = QuantumPoolingLayer3(
            num_qubits, circuit, unmeasured_bits, conditional
        ).build_layer()
        resulting_circuit, resulting_unmeasured_bits = request.getfixturevalue(
            resulting_circuit
        )
        print(circuit, new_unmeasured_bits)
        print(resulting_circuit)
        assert circuit == resulting_circuit
        assert new_unmeasured_bits == resulting_unmeasured_bits
