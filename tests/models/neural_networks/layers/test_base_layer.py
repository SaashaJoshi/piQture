"""Tests for Base Layer abstract class"""
from __future__ import annotations
import pytest
from pytest import raises
from qiskit.circuit import QuantumCircuit
from quantum_image_processing.models.neural_networks.layers import (
    QuantumConvolutionalLayer,
)


class TestQuantumConvolutionalLayer:
    """Tests for Base Layer abstract class"""

    def test_args(self):
        with raises(
            ValueError,
            match="At least one of the inputs, num_qubits, circuit, "
            "or unmeasured_bits, must be provided.",
        ):
            _ = QuantumConvolutionalLayer()

    @pytest.mark.parametrize("num_qubits", ["abc", {"efg", 1}])
    def test_type_num_qubits(self, num_qubits):
        with raises(TypeError, match="The input num_qubits must be of the type int."):
            _ = QuantumConvolutionalLayer(num_qubits)

    @pytest.mark.parametrize("num_qubits", [0, -3])
    def test_value_num_qubits(self, num_qubits):
        with raises(
            ValueError, match="The input num_qubits must be greater than zero."
        ):
            _ = QuantumConvolutionalLayer(num_qubits)

    @pytest.mark.parametrize("circuit", [0, "abc", {"efg", 1}])
    def test_circuit(self, circuit):
        with raises(
            TypeError, match="The input circuit must be of the type QuantumCircuit."
        ):
            _ = QuantumConvolutionalLayer(circuit=circuit)

    @pytest.mark.parametrize("unmeasured_bits", [0, "abc", {"efg", 1}])
    def test_type_unmeasured_bits(self, unmeasured_bits):
        with raises(TypeError, match="The input qubits must be of the type list."):
            _ = QuantumConvolutionalLayer(unmeasured_bits=unmeasured_bits)

    @pytest.mark.parametrize("unmeasured_bits", [["abc"], [None], [3.25]])
    def test_bits(self, unmeasured_bits):
        with raises(
            TypeError,
            match="Indices inside the unmeasured_bits list must be of the type int.",
        ):
            _ = QuantumConvolutionalLayer(unmeasured_bits=unmeasured_bits)

    @pytest.mark.parametrize("num_qubits", [2, 4])
    def test_ynn(self, num_qubits):
        qc_layer = QuantumConvolutionalLayer(num_qubits)
        assert qc_layer.circuit == QuantumCircuit(num_qubits, num_qubits)
        assert qc_layer.unmeasured_bits == range(num_qubits)

    @pytest.mark.parametrize("circuit", [QuantumCircuit(3, 3)])
    def test_nyn(self, circuit):
        qc_layer = QuantumConvolutionalLayer(circuit=circuit)
        assert qc_layer.num_qubits == len(circuit.qubits)

    @pytest.mark.parametrize("unmeasured_bits", [[2, 3, 4]])
    def test_nny(self, unmeasured_bits):
        qc_layer = QuantumConvolutionalLayer(unmeasured_bits=unmeasured_bits)
        assert qc_layer.circuit == QuantumCircuit(
            len(unmeasured_bits), len(unmeasured_bits)
        )
        assert qc_layer.num_qubits == len(unmeasured_bits)
