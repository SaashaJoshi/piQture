# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Quantum Convolutionall Neural Network structure"""

from __future__ import annotations

import re
from unittest import mock

import pytest
from pytest import raises
from qiskit.circuit import QuantumCircuit

from piqture.neural_networks import QCNN
from piqture.neural_networks.layers import (
    BaseLayer,
    FullyConnectedLayer,
    QuantumConvolutionalLayer,
    QuantumPoolingLayer2,
    QuantumPoolingLayer3,
)
from piqture.tensor_networks import MERA


class TestQCNN:
    """Tests for QCNN class"""

    @pytest.mark.parametrize("num_qubits", [None, 3.5, "abc", {"mnp"}])
    def test_type_num_qubits(self, num_qubits):
        """Tests the type of num_qubits input."""
        with raises(TypeError, match="The input num_qubits must be of the type int."):
            _ = QCNN(num_qubits)

    @pytest.mark.parametrize("num_qubits", [0])
    def test_value_num_qubits(self, num_qubits):
        """Tests the value of num_qubits input."""
        with raises(ValueError, match="The input num_qubits must be at least 1."):
            _ = QCNN(num_qubits)

    @pytest.mark.parametrize("num_qubits", [2, 5])
    def test_circuit_property(self, num_qubits):
        """Tests the QCNN circuit."""
        qcnn = QCNN(num_qubits)
        assert qcnn.circuit == QuantumCircuit(num_qubits)

    @pytest.mark.parametrize(
        "num_qubits, operations",
        [(2, {}), (3, (QuantumConvolutionalLayer, {})), (4, None), (5, "abc")],
    )
    def test_operations(self, num_qubits, operations):
        """Tests the type of operations argument in sequence method of QCNN class."""
        with raises(TypeError, match="The input operations must be of the type list."):
            _ = QCNN(num_qubits).sequence(operations)

    @pytest.mark.parametrize(
        "num_qubits, operations",
        [
            (2, [{}]),
            (4, [[]]),
            (5, [[()]]),
        ],
    )
    def test_operation_tuple(self, num_qubits, operations):
        """Tests the operation tuple in operations list."""
        with raises(
            TypeError,
            match=re.escape(
                pattern="The input operations list must contain tuple[operation, params]."
            ),
        ):
            _ = QCNN(num_qubits).sequence(operations)

    @pytest.mark.parametrize(
        "num_qubits, operations",
        [
            (2, [([], {})]),
            (3, [(pytest.mark, {})]),
            (4, [(1, {})]),
            (5, [("abc", {})]),
            (2, [(print, {})]),
            (3, [(sum, {})]),
            (4, [(lambda x: x, {})]),
        ],
    )
    def test_operation_isclass(self, num_qubits, operations):
        """Tests the type of operation in operation tuple."""
        with raises(
            TypeError, match=r"Operation at index \d+\ must be a class, got .*?"
        ):
            _ = QCNN(num_qubits).sequence(operations)

    @pytest.mark.parametrize(
        "num_qubits, operations",
        [
            (2, [(QuantumPoolingLayer2, [])]),
            (3, [(QuantumConvolutionalLayer, ())]),
            (4, [(QuantumPoolingLayer3, 1)]),
            (5, [(FullyConnectedLayer, "abc")]),
        ],
    )
    def test_params(self, num_qubits, operations):
        """Tests the type of params in operation tuple."""
        with raises(
            TypeError,
            match=r"Parameters of operation at index \d+ must be in a dictionary, got .*?",
        ):
            _ = QCNN(num_qubits).sequence(operations)

    @pytest.mark.parametrize(
        "num_qubits, operations",
        [
            (2, [(QuantumPoolingLayer2, {})]),
            (3, [(QuantumConvolutionalLayer, {})]),
            (3, [(QuantumPoolingLayer3, {})]),
            (5, [(FullyConnectedLayer, {})]),
        ],
    )
    def test_sequence(self, num_qubits, operations):
        """Tests the sequence method of QCNN class."""

        with (
            mock.patch(
                "piqture.neural_networks.layers.QuantumConvolutionalLayer.build_layer"
            ) as mock_quantum_convolutional_layer,
            mock.patch(
                "piqture.neural_networks.layers.QuantumPoolingLayer2.build_layer"
            ) as mock_quantum_pooling_layer2,
            mock.patch(
                "piqture.neural_networks.layers.QuantumPoolingLayer3.build_layer"
            ) as mock_quantum_pooling_layer3,
            mock.patch(
                "piqture.neural_networks.layers.FullyConnectedLayer.build_layer"
            ) as mock_fully_connected_layer,
        ):

            mock_quantum_convolutional_layer.return_value = None, None
            mock_quantum_pooling_layer2.return_value = None, None
            mock_quantum_pooling_layer3.return_value = None, None
            mock_fully_connected_layer.return_value = None, None

            _ = QCNN(num_qubits).sequence(operations)

            assert (
                mock_quantum_convolutional_layer.call_count > 0
                or mock_quantum_pooling_layer2.call_count > 0
                or mock_quantum_pooling_layer3.call_count > 0
                or mock_fully_connected_layer.call_count > 0
            )

    @pytest.mark.parametrize(
        "num_qubits, operations",
        [
            (2, [(dict, {})]),
            (3, [(str, {})]),
            (4, [(object, {})]),
            (2, [(MERA, {})]),
            (3, [(QCNN, {})]),
        ],
    )
    def test_non_baselayer_operation(self, num_qubits, operations):
        """Tests if the layer operation inherits from the abstract BaseLayer class."""
        with raises(
            TypeError,
            match=r"Operation at index \d+\ must inherit from BaseLayer, got \*?",
        ):
            _ = QCNN(num_qubits).sequence(operations)

    @pytest.mark.parametrize(
        "num_qubits, operations",
        [(2, [(BaseLayer, {})])],
    )
    def test_baselayer(self, num_qubits, operations):
        """Tests if the layer operation is the abstract BaseLayer class."""
        with raises(
            TypeError,
            match=r"Operation at index \d+\ cannot be BaseLayer itself.",
        ):
            _ = QCNN(num_qubits).sequence(operations)
