"""Tests for Quantum Convolutionall Neural Network structure"""
from __future__ import annotations
import re
import pytest
from pytest import raises
from qiskit.circuit import QuantumCircuit
from quantum_image_processing.models.neural_networks.qcnn import QCNN
from quantum_image_processing.models.neural_networks.layers import (
    QuantumConvolutionalLayer,
    QuantumPoolingLayer2,
    QuantumPoolingLayer3,
    FullyConnectedLayer,
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
        assert qcnn.circuit == QuantumCircuit(num_qubits, num_qubits)

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
        with raises(
            TypeError,
            match="Operation in input operations list must be Callable functions/classes.",
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
            match="Parameters of operation in input operations list must be in a dictionary.",
        ):
            _ = QCNN(num_qubits).sequence(operations)

    @pytest.mark.parametrize(
        "num_qubits, operations",
        [
            (2, [(QuantumPoolingLayer2, {})]),
            (3, [(QuantumConvolutionalLayer, {})]),
            (4, [(QuantumPoolingLayer3, {})]),
            (5, [(FullyConnectedLayer, {})]),
        ],
    )
    def test_sequence(self, num_qubits, operations):
        """Tests the sequence method of QCNN class."""

        assert False
