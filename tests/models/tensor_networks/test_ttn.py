"""Unit test for TTN class"""
from __future__ import annotations

import math
import re
from unittest import mock
import pytest
from pytest import raises
from qiskit.circuit import QuantumCircuit
from quantum_image_processing.models.tensor_network_circuits.ttn import TTN


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

    @pytest.mark.parametrize(
        "img_dims, complex_structure", [((2, 2), False), ((4, 5), True)]
    )
    def test_ttn_simple(self, img_dims, complex_structure):
        # pylint: disable=line-too-long
        """Tests the ttn_backbone mehtod call via the ttn_simple function."""
        with mock.patch(
            "quantum_image_processing.models.tensor_network_circuits.ttn.TTN.ttn_backbone"
        ) as mock_ttn_simple:
            with mock.patch(
                "quantum_image_processing.gates.two_qubit_unitary.TwoQubitUnitary.simple_parameterization"
            ) as simple_parameterization:
                _ = TTN(img_dims).ttn_simple(complex_structure)
                mock_ttn_simple.assert_called_once_with(
                    simple_parameterization, mock.ANY, complex_structure
                )

    @pytest.mark.parametrize(
        "img_dims, complex_structure", [((2, 2), False), ((4, 5), True)]
    )
    def test_ttn_general(self, img_dims, complex_structure):
        # pylint: disable=line-too-long
        """Tests the ttn_backbone method call via the ttn_general function."""
        with mock.patch(
            "quantum_image_processing.models.tensor_network_circuits.ttn.TTN.ttn_backbone"
        ) as mock_ttn_simple:
            with mock.patch(
                "quantum_image_processing.gates.two_qubit_unitary.TwoQubitUnitary.general_parameterization"
            ) as general_parameterization:
                _ = TTN(img_dims).ttn_general(complex_structure)
                mock_ttn_simple.assert_called_once_with(
                    general_parameterization, mock.ANY, complex_structure
                )
