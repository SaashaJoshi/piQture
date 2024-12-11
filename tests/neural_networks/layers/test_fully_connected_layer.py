# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Fully Connected Layer"""

from __future__ import annotations

import pytest
from qiskit.circuit import QuantumCircuit

from piqture.neural_networks.layers import FullyConnectedLayer


@pytest.fixture
def circuit1():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(2, 2)
    circuit.cz(0, 1)
    return circuit


@pytest.fixture
def circuit2():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(4, 4)
    circuit.cz(0, 1)
    circuit.cz(1, 2)
    circuit.cz(2, 3)
    return circuit


@pytest.fixture
def circuit3():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(8, 8)
    circuit.cz(2, 3)
    circuit.cz(3, 5)
    circuit.cz(5, 7)
    return circuit


@pytest.fixture
def circuit4():
    """Fixture for fully connected layer circuit"""
    circuit = QuantumCircuit(4, 4)
    circuit.cz(2, 3)
    return circuit


class TestFullyConnectedLayer:
    """Tests for Fully Connected Layer class"""

    # pylint: disable=too-few-public-methods
    @pytest.mark.parametrize(
        "num_qubits, circuit, unmeasured_bits, resulting_circuit",
        [
            (2, None, None, "circuit1"),
            (None, QuantumCircuit(4, 4), None, "circuit2"),
            (None, None, [2, 3, 5, 7], "circuit3"),
            (6, None, [2, 3], "circuit4"),
        ],
    )

    # pylint: disable=R0917
    def test_build_layer(
        self, request, num_qubits, circuit, unmeasured_bits, resulting_circuit
    ):
        # pylint: disable=too-many-arguments
        """Tests the build_layer method of FullyConnectedLayer class."""
        circuit, _ = FullyConnectedLayer(
            num_qubits, circuit, unmeasured_bits
        ).build_layer()
        resulting_circuit = request.getfixturevalue(resulting_circuit)
        print(circuit)
        print(resulting_circuit)
        assert circuit.data == resulting_circuit.data
