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
        [(2, [([], {})]), (3, [(pytest.mark, {})]), (4, [(1, {})]), (5, [("abc", {})])],
    )
    def test_operation(self, num_qubits, operations):
        """Tests the type of operation in operation tuple."""
        # Construct the expected error message based on the input
        for idx, (layer, _) in enumerate(operations):
            # Determine the actual type of the invalid operation
            actual_type = type(layer).__name__

            # Define the expected error message
            expected_error_message = (
                f"Operation at index {idx} must be a class, got {actual_type}"
            )

            # Check if the error is raised and matches the dynamically constructed message
            with raises(TypeError, match=expected_error_message):
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
        for idx, (_, params) in enumerate(operations):
            # Determine the actual type of the invalid parameter
            actual_type = type(params).__name__

            # Construct the expected error message dynamically
            expected_error_message = (
                f"Parameters at index {idx} must be in a dictionary, got {actual_type}"
            )

            # Validate that the correct error message is raised
            with raises(TypeError, match=expected_error_message):
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

        with mock.patch(
            "piqture.neural_networks.layers.QuantumConvolutionalLayer.build_layer"
        ) as mock_quantum_convolutional_layer, mock.patch(
            "piqture.neural_networks.layers.QuantumPoolingLayer2.build_layer"
        ) as mock_quantum_pooling_layer2, mock.patch(
            "piqture.neural_networks.layers.QuantumPoolingLayer3.build_layer"
        ) as mock_quantum_pooling_layer3, mock.patch(
            "piqture.neural_networks.layers.FullyConnectedLayer.build_layer"
        ) as mock_fully_connected_layer:

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

    # Tests added for BaseLayer inheritance validation
    @pytest.mark.parametrize(
        "num_qubits, operations",
        [
            (2, [(print, {})]),
            (3, [(sum, {})]),
            (4, [(lambda x: x, {})]),
        ],
    )
    def test_non_class_operations(self, num_qubits, operations):
        """Tests that non-class callables are rejected."""
        with raises(TypeError, match="must be a class"):
            _ = QCNN(num_qubits).sequence(operations)

    @pytest.mark.parametrize(
        "num_qubits, operations",
        [
            (2, [(dict, {})]),
            (3, [(str, {})]),
            (4, [(object, {})]),
        ],
    )
    def test_non_baselayer_operations(self, num_qubits, operations):
        """Tests that classes not inheriting from BaseLayer are rejected."""
        with raises(TypeError, match="must inherit from BaseLayer"):
            _ = QCNN(num_qubits).sequence(operations)

    @pytest.mark.parametrize(
        "num_qubits, operations",
        [(2, [(BaseLayer, {})])],
    )
    def test_baselayer_itself(self, num_qubits, operations):
        """Tests that BaseLayer itself is rejected."""
        with raises(TypeError, match="cannot be BaseLayer itself"):
            _ = QCNN(num_qubits).sequence(operations)
