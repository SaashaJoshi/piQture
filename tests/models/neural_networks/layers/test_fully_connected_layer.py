"""Tests for Fully Connected Layer"""
from __future__ import annotations
import pytest
from qiskit.circuit import QuantumCircuit
from quantum_image_processing.models.neural_networks.layers import FullyConnectedLayer


@pytest.fixture
def circuit1():
    circuit = QuantumCircuit(2, 2)
    circuit.cz(0, 1)
    return circuit


@pytest.fixture
def circuit2():
    circuit = QuantumCircuit(4, 4)
    circuit.cz(0, 1)
    circuit.cz(1, 2)
    circuit.cz(2, 3)
    return circuit


@pytest.fixture
def circuit3():
    circuit = QuantumCircuit(8, 8)
    circuit.cz(2, 3)
    circuit.cz(3, 5)
    circuit.cz(5, 7)
    return circuit


CIRCUIT_FIXTURE_LIST = [circuit1, circuit2, circuit3]


class TestFullyConnectedLayer:
    """Tests for Fully Connected Layer class"""

    @pytest.mark.parametrize(
        "num_qubits, circuit, unmeasured_bits, resulting_circuit",
        [
            (2, None, None, "circuit1"),
            (None, QuantumCircuit(4, 4), None, "circuit2"),
            (None, None, [2, 3, 5, 7], "circuit3"),
        ],
    )
    def test_build_layer(
        self, request, num_qubits, circuit, unmeasured_bits, resulting_circuit
    ):
        """Tests the build_layer method of FullyConnectedLayer class."""
        circuit, _ = FullyConnectedLayer(
            num_qubits, circuit, unmeasured_bits
        ).build_layer()

        resulting_circuit = request.getfixturevalue(resulting_circuit)
        assert circuit.data == resulting_circuit.data
