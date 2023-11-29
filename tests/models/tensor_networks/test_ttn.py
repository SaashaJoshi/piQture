"""Unit test for TTN class"""
from __future__ import annotations

import math
import re
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
