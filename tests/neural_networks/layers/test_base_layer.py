"""Tests for Base Layer abstract class"""

from __future__ import annotations
import pytest
from pytest import raises
from qiskit.circuit import QuantumCircuit
from quantum_image_processing.neural_networks.layers import (
    QuantumConvolutionalLayer,
)


class TestQuantumConvolutionalLayer:
    """Tests for Base Layer abstract class"""

    def test_args(self):
        """Test if all arguments are input to initialize the BaseLayer class."""
        with raises(
            ValueError,
            match="At least one of the inputs, num_qubits, circuit, "
            "or unmeasured_bits, must be provided.",
        ):
            _ = QuantumConvolutionalLayer()

    @pytest.mark.parametrize("num_qubits", ["abc", {"efg", 1}])
    def test_type_num_qubits(self, num_qubits):
        """Tests the type of num_qubits input."""
        with raises(TypeError, match="The input num_qubits must be of the type int."):
            _ = QuantumConvolutionalLayer(num_qubits)

    @pytest.mark.parametrize("num_qubits", [0, -3])
    def test_value_num_qubits(self, num_qubits):
        """Tests the value of num_qubits input."""
        with raises(
            ValueError, match="The input num_qubits must be greater than zero."
        ):
            _ = QuantumConvolutionalLayer(num_qubits)

    @pytest.mark.parametrize("circuit", [0, "abc", {"efg", 1}])
    def test_circuit(self, circuit):
        """Tests the type of circuit input."""
        with raises(
            TypeError, match="The input circuit must be of the type QuantumCircuit."
        ):
            _ = QuantumConvolutionalLayer(circuit=circuit)

    @pytest.mark.parametrize("unmeasured_bits", [0, "abc", {"efg", 1}])
    def test_type_unmeasured_bits(self, unmeasured_bits):
        """Tests the type of unmeasured_bits input."""
        with raises(TypeError, match="The input qubits must be of the type list."):
            _ = QuantumConvolutionalLayer(unmeasured_bits=unmeasured_bits)

    @pytest.mark.parametrize("unmeasured_bits", [["abc"], [None], [3.25]])
    def test_bits(self, unmeasured_bits):
        """Tests the type of bits in unmeasured_bits input."""
        with raises(
            TypeError,
            match="Indices inside the unmeasured_bits list must be of the type int.",
        ):
            _ = QuantumConvolutionalLayer(unmeasured_bits=unmeasured_bits)

    @pytest.mark.parametrize("num_qubits", [2, 4])
    def test_ynn(self, num_qubits):
        """
        Tests the values of circuit and unmeasured_bits
        when only num_qubits input is provided.
        """
        qc_layer = QuantumConvolutionalLayer(num_qubits)
        assert qc_layer.circuit == QuantumCircuit(num_qubits, num_qubits)
        assert qc_layer.unmeasured_bits == list(range(num_qubits))

    @pytest.mark.parametrize("circuit", [QuantumCircuit(3, 3)])
    def test_nyn(self, circuit):
        """
        Tests the values of num_qubits and unmeasured_bits
        when only circuit input is provided.
        """
        qc_layer = QuantumConvolutionalLayer(circuit=circuit)
        assert qc_layer.num_qubits == len(circuit.qubits)
        assert qc_layer.unmeasured_bits == list(range(len(circuit.qubits)))

    @pytest.mark.parametrize("unmeasured_bits", [[2, 3, 4]])
    def test_nny(self, unmeasured_bits):
        """
        Tests the values of num_qubits and circuit
        when only unmeasured_bits input is provided.
        """
        qc_layer = QuantumConvolutionalLayer(unmeasured_bits=unmeasured_bits)
        assert qc_layer.circuit == QuantumCircuit(
            max(unmeasured_bits) + 1, max(unmeasured_bits) + 1
        )
        assert qc_layer.num_qubits == max(unmeasured_bits) + 1
