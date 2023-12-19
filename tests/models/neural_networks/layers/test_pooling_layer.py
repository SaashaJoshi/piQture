"""Tests for Quantum Pooling Layer classes"""
from __future__ import annotations
import pytest
from pytest import raises
from qiskit.circuit import QuantumCircuit
from quantum_image_processing.models.neural_networks.layers import (
    QuantumPoolingLayer2,
    QuantumPoolingLayer3,
)
# pylint: disable=not-context-manager


@pytest.fixture
def circuit1():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(2, 2)
    circuit.h(1)
    circuit.measure(1, 1)
    with circuit.if_test((1, 1)):
        circuit.z(0)
    return circuit


@pytest.fixture
def circuit2():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(4, 4)
    circuit.h([1, 3])
    circuit.measure([1, 3], [1, 3])
    with circuit.if_test([1, 1]):
        circuit.z(0)
    with circuit.if_test([3, 1]):
        circuit.z(2)
    return circuit


@pytest.fixture
def circuit3():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(8, 8)
    circuit.h([3, 7])
    circuit.measure([3, 7], [3, 7])
    with circuit.if_test([3, 1]):
        circuit.z(2)
    with circuit.if_test([7, 1]):
        circuit.z(5)
    return circuit


@pytest.fixture
def circuit4():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(6, 6)
    circuit.h([3])
    circuit.measure([3], [3])
    with circuit.if_test([3, 1]):
        circuit.z(2)
    return circuit


@pytest.fixture
def circuit5():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(3, 3)
    circuit.h([0, 2])
    circuit.measure([0, 2], [0, 2])
    with circuit.if_test([0, 1]):
        with circuit.if_test([2, 1]):
            circuit.z(1)
    return circuit


@pytest.fixture
def circuit6():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(6, 6)
    circuit.h([0, 2, 3, 5])
    circuit.measure([0, 2, 3, 5], [0, 2, 3, 5])
    with circuit.if_test([0, 1]):
        with circuit.if_test([2, 1]):
            circuit.z(1)
    with circuit.if_test([3, 1]):
        with circuit.if_test([5, 1]):
            circuit.z(4)
    return circuit


@pytest.fixture
def circuit7():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(8, 8)
    circuit.h([2, 5])
    circuit.measure([2, 5], [2, 5])
    with circuit.if_test([2, 1]):
        with circuit.if_test([5, 1]):
            circuit.z(3)
    return circuit


class TestQuantumPoolingLayer2:
    """Tests for Quantum Pooling Layer (2) class"""

    # pylint: disable=too-few-public-methods

    @pytest.mark.parametrize(
        "num_qubits, circuit, unmeasured_bits, resulting_circuit",
        [
            (2, None, None, "circuit1"),
            (None, QuantumCircuit(4, 4), None, "circuit2"),
            (None, None, [2, 3, 5, 7], "circuit3"),
            (None, None, [2, 3, 5], "circuit4"),
        ],
    )
    def test_build_layer(
            self, request, num_qubits, circuit, unmeasured_bits, resulting_circuit
    ):
        # pylint: disable=too-many-arguments
        """Tests the build_layer method of QuantumPoolingLayer2 class."""
        circuit, _ = QuantumPoolingLayer2(
            num_qubits, circuit, unmeasured_bits
        ).build_layer()
        resulting_circuit = request.getfixturevalue(resulting_circuit)
        assert circuit == resulting_circuit


class TestQuantumPoolingLayer3:
    """Tests for Quantum Pooling Layer (3) class"""

    @pytest.mark.parametrize(
        "num_qubits, circuit, unmeasured_bits",
        [
            (2, None, None),
            (None, QuantumCircuit(2, 2), None),
            (None, None, [2, 3])
        ],
    )
    def test_three_qubits(self, num_qubits, circuit, unmeasured_bits):
        """Tests for presence of at least 3 qubits in the circuit."""
        with raises(ValueError):
            _ = QuantumPoolingLayer3(num_qubits, circuit, unmeasured_bits)

    @pytest.mark.parametrize(
        "num_qubits, circuit, unmeasured_bits, resulting_circuit",
        [
            (3, None, None, "circuit5"),
            (None, QuantumCircuit(6, 6), None, "circuit6"),
            (None, None, [2, 3, 5, 7], "circuit7"),
        ],
    )
    def test_build_layer(
            self, request, num_qubits, circuit, unmeasured_bits, resulting_circuit
    ):
        # pylint: disable=too-many-arguments
        """Tests the build_layer method of QuantumPoolingLayer2 class."""
        circuit, _ = QuantumPoolingLayer3(
            num_qubits, circuit, unmeasured_bits
        ).build_layer()
        resulting_circuit = request.getfixturevalue(resulting_circuit)
        assert circuit == resulting_circuit
