"""Unit test for TTN class"""
from __future__ import annotations

import math
import re
from unittest import mock
import pytest
from pytest import raises
from qiskit.circuit import QuantumCircuit, ParameterVector
from quantum_image_processing.models.tensor_network_circuits.ttn import TTN


@pytest.fixture(name="ttn_simple_circuit")
def ttn_simple_circuit_fixture(real_simple_block):
    def _ttn_simple_circuit(img_dims, parameter_vector):
        test_circuit = QuantumCircuit(int(math.prod(img_dims)))
        if math.prod(img_dims) == 2:
            test_circuit.compose(real_simple_block(parameter_vector[:2]), qubits=[0, 1], inplace=True)
        elif math.prod(img_dims) == 4:
            test_circuit.compose(real_simple_block(parameter_vector[:2]), qubits=[0, 1], inplace=True)
            test_circuit.compose(real_simple_block(parameter_vector[2:4]), qubits=[2, 3], inplace=True)
            test_circuit.compose(real_simple_block(parameter_vector[4:6]), qubits=[1, 3], inplace=True)
        return test_circuit

    return _ttn_simple_circuit


class TestTTN:
    """Tests for TTN class"""

    @pytest.mark.parametrize("img_dims", [({"abc", "def"}), (2, 1.5), (None, None)])
    def test_img_dims(self, img_dims):
        """Tests the type of img_dims input."""
        with raises(
                TypeError,
                match=re.escape("Input img_dims must be of the type tuple[int, int]."),
        ):
            _ = TTN(img_dims)

    @pytest.mark.parametrize("img_dims", [(-3, 1), (2, 0)])
    def test_num_qubits(self, img_dims):
        """Tests the product of img_dims."""
        with raises(ValueError, match="Image dimensions cannot be zero or negative."):
            _ = TTN(img_dims)

    @pytest.mark.parametrize("img_dims", [(2, 4)])
    def test_circuit_property(self, img_dims):
        """Tests the TTN circuit initialization."""
        test_circuit = QuantumCircuit(math.prod(img_dims))
        assert test_circuit.data == TTN(img_dims).circuit.data

    @pytest.mark.parametrize("img_dims, complex_structure", [((2, 2), False)])
    def test_ttn_simple(self, img_dims, complex_structure, ttn_simple_circuit):
        """Tests the simple TTN circuit."""
        ttn_object = TTN(img_dims)
        mock_circuit = QuantumCircuit(int(math.prod(img_dims)))
        parameter_vector = ParameterVector(name="theta", length=2 * int(math.prod(img_dims)) - 1)
        test_circuit = ttn_simple_circuit(img_dims, parameter_vector)
        test_circuit.ry(parameter_vector[-1], int(math.prod(img_dims)) - 1)
        with mock.patch("quantum_image_processing.models.tensor_network_circuits.ttn.TTN.circuit",
                        new_callable=lambda: mock_circuit):
            ttn_object.ttn_simple(complex_structure)
            print(mock_circuit)
            print(test_circuit)
            # assert False
            assert mock_circuit.data == test_circuit.data
