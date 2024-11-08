# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unit test for NEQR class"""

from __future__ import annotations

import math
from unittest import mock

import numpy as np
import pytest
from pytest import raises
from qiskit.circuit import QuantumCircuit

from piqture.embeddings.image_embeddings.neqr import NEQR

MAX_COLOR_INTENSITY = 255


@pytest.fixture(name="neqr_pixel_value")
def neqr_pixel_value_fixture():
    """Fixture for embedding pixel values."""

    def _circuit(img_dims, pixel_val, color_qubits):
        feature_dim = int(np.sqrt(math.prod(img_dims)))
        pixel_val_bin = f"{int(pixel_val):0>8b}"
        test_circuit = QuantumCircuit(int(math.prod(img_dims)) + color_qubits)

        # Add gates to test_circuit
        for index, color in enumerate(pixel_val_bin):
            if color == "1":
                test_circuit.mcx(list(range(feature_dim)), feature_dim + index)
        return test_circuit

    return _circuit


class TestNEQR:
    """Tests for FRQI image representation class"""

    @pytest.mark.parametrize(
        "img_dims, pixel_vals, max_color_intensity",
        [
            ((2, 2), [list(range(251, 255))], 300),
            ((2, 2), [list(range(251, 255))], -20),
        ],
    )
    def test_max_color_intensity(self, img_dims, pixel_vals, max_color_intensity):
        """Tests value of maximum color intensity."""
        with raises(
            ValueError,
            match=r"Maximum color intensity cannot be less than \d or greater than \d.",
        ):
            _ = NEQR(img_dims, pixel_vals, max_color_intensity)

    @pytest.mark.parametrize(
        "img_dims, pixel_vals, max_color_intensity",
        [((2, 2), [list(range(251, 255))], MAX_COLOR_INTENSITY)],
    )
    def test_circuit_property(self, img_dims, pixel_vals, max_color_intensity):
        """Tests the FRQI circuits initialization."""
        color_qubits = int(np.ceil(math.log(max_color_intensity, 2)))
        test_circuit = QuantumCircuit(
            np.ceil(np.sqrt(math.prod(img_dims))) + color_qubits
        )
        assert NEQR(img_dims, pixel_vals).circuit == test_circuit

    @pytest.mark.parametrize(
        "img_dims, pixel_vals, max_color_intensity",
        [((2, 2), [list(range(235, 239))], MAX_COLOR_INTENSITY)],
    )
    def test_pixel_value(
        self, img_dims, pixel_vals, max_color_intensity, neqr_pixel_value
    ):
        """Tests the circuit received after pixel value embedding."""
        neqr_object = NEQR(img_dims, pixel_vals, max_color_intensity)
        color_qubits = int(np.ceil(math.log(max_color_intensity, 2)))
        mock_circuit = QuantumCircuit(int(math.prod(img_dims)) + color_qubits)

        for _, pixel_val in enumerate(pixel_vals[0]):
            mock_circuit.clear()
            test_circuit = neqr_pixel_value(img_dims, pixel_val, color_qubits)

            with mock.patch(
                "piqture.embeddings.image_embeddings.neqr.NEQR.circuit",
                new_callable=lambda: mock_circuit,
            ):
                neqr_object.pixel_value(color_byte=f"{pixel_val:0>8b}")
                assert mock_circuit == test_circuit

    # pylint: disable=too-many-arguments
    @pytest.mark.parametrize(
        "img_dims, pixel_vals, max_color_intensity",
        [((2, 2), [list(range(1, 5))], MAX_COLOR_INTENSITY)],
    )

    # pylint: disable=R0917
    def test_neqr(
        self,
        img_dims,
        pixel_vals,
        max_color_intensity,
        circuit_pixel_position,
        neqr_pixel_value,
    ):
        """Tests the final NEQR circuit."""
        neqr_object = NEQR(img_dims, pixel_vals, max_color_intensity)
        color_qubits = int(np.ceil(math.log(max_color_intensity, 2)))
        mock_circuit = QuantumCircuit(int(math.prod(img_dims)) + color_qubits)

        test_circuit = QuantumCircuit(int(math.prod(img_dims)) + color_qubits)
        test_circuit.h(list(range(int(np.sqrt(math.prod(img_dims))))))
        for index, pixel_val in enumerate(pixel_vals[0]):
            pixel_pos_binary = f"{index:0>2b}"
            mock_circuit.clear()
            test_circuit.compose(
                circuit_pixel_position(img_dims, pixel_pos_binary), inplace=True
            )
            test_circuit.compose(
                neqr_pixel_value(img_dims, pixel_val, color_qubits), inplace=True
            )
            test_circuit.compose(
                circuit_pixel_position(img_dims, pixel_pos_binary), inplace=True
            )

        with mock.patch(
            "piqture.embeddings.image_embeddings.neqr.NEQR.circuit",
            new_callable=lambda: mock_circuit,
        ):
            neqr_object.neqr()
            assert mock_circuit == test_circuit
