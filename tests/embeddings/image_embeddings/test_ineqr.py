# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unit test for INEQR class"""

from __future__ import annotations

import math
from unittest import mock

import numpy as np
import pytest
from pytest import raises
from qiskit.circuit import QuantumCircuit

from piqture.embeddings.image_embeddings.ineqr import INEQR

MAX_COLOR_INTENSITY = 255
COLOR_QUBITS = int(np.ceil(math.log(MAX_COLOR_INTENSITY, 2)))


@pytest.fixture
def circuit_2_2():
    """Circuit fixture for img_dims (2, 2)"""
    circuit = QuantumCircuit(2 + COLOR_QUBITS)
    circuit.h(range(2))
    # Pixel vals = [[40, 128], [65, 2]]
    # Pixel-00
    circuit.x([0, 1])
    circuit.mcx(control_qubits=list(range(2)), target_qubit=2 + 2)
    circuit.mcx(control_qubits=list(range(2)), target_qubit=2 + 4)
    circuit.x([0, 1])
    # Pixel-01
    circuit.x(0)
    circuit.mcx(control_qubits=list(range(2)), target_qubit=2)
    circuit.x(0)
    # Pixel-10
    circuit.x(1)
    circuit.mcx(control_qubits=list(range(2)), target_qubit=2 + 1)
    circuit.mcx(control_qubits=list(range(2)), target_qubit=2 + 7)
    circuit.x(1)
    # Pixel-11
    circuit.mcx(control_qubits=list(range(2)), target_qubit=2 + 6)

    return circuit


@pytest.fixture
def circuit_4_2():
    """Circuit fixture for img_dims (4, 2)"""
    circuit = QuantumCircuit(3 + COLOR_QUBITS)
    circuit.h(range(3))
    # Pixel vals = [[128, 64, 1, 2], [0, 0, 0, 1]]
    # Pixel-000
    circuit.x([0, 1, 2])
    circuit.mcx(control_qubits=list(range(3)), target_qubit=3)
    circuit.x([0, 1, 2])
    # Pixel-001
    circuit.x([0, 1])
    circuit.mcx(control_qubits=list(range(3)), target_qubit=3 + 1)
    circuit.x([0, 1])
    # Pixel-010
    circuit.x([0, 2])
    circuit.mcx(control_qubits=list(range(3)), target_qubit=3 + 7)
    circuit.x([0, 2])
    # Pixel-011
    circuit.x(0)
    circuit.mcx(control_qubits=list(range(3)), target_qubit=3 + 6)
    circuit.x(0)
    # Pixel-100
    circuit.x([1, 2])
    circuit.x([1, 2])
    # Pixel-101
    circuit.x(1)
    circuit.x(1)
    # Pixel-110
    circuit.x(2)
    circuit.x(2)
    # Pixel-111
    circuit.mcx(control_qubits=list(range(3)), target_qubit=3 + 7)

    return circuit


class TestINEQR:
    """Tests for FRQI image representation class"""

    @pytest.mark.parametrize(
        "img_dims, pixel_vals",
        [((2, 4), [[list(range(251, 255)), list(range(251, 255))]])],
    )
    def test_circuit_property(self, img_dims, pixel_vals):
        """Tests the INEQR circuits initialization."""

        feature_dims = int(math.log(img_dims[1], 2)) + int(math.log(img_dims[0], 2))
        test_circuit = QuantumCircuit(feature_dims + COLOR_QUBITS)
        assert INEQR(img_dims, pixel_vals).circuit == test_circuit

    @pytest.mark.parametrize(
        "img_dims, pixel_vals",
        [((2, 4, 5), [[list(range(251, 255)), list(range(251, 255))]])],
    )
    def test_2d_image(self, img_dims, pixel_vals):
        """Tests if images are 2-dimensional"""
        with raises(ValueError, match=r"(.*) supports 2-dimensional images only."):
            _ = INEQR(img_dims, pixel_vals)

    @pytest.mark.parametrize(
        "img_dims, pixel_vals",
        [((3, 7), [[list(range(251, 255)), list(range(251, 255))]])],
    )
    def test_img_dim_power_of_2(self, img_dims, pixel_vals):
        """Tests if image dimensions are powers of 2."""
        with raises(ValueError, match="Image dimensions must be powers of 2."):
            _ = INEQR(img_dims, pixel_vals)

    @pytest.mark.parametrize(
        "img_dims, pixel_vals",
        [((4, 2), [[list(range(250, 255)), list(range(155, 160))]])],
    )
    def test_number_pixels(self, img_dims, pixel_vals):
        """Tests if the number of pixels is the same as the image dimension."""
        with raises(
            ValueError,
            match=r"No. of pixels \(\[\d+\]\) "
            r"in each pixel_lists in pixel_vals must be equal to the "
            r"product of image dimensions \d.",
        ):
            _ = INEQR(img_dims, pixel_vals)

    @pytest.mark.parametrize(
        "img_dims, pixel_vals",
        [((4, 2), [[[128, 64, 1, 2], [0, 0, 0, 1]]])],
    )
    def test_pixel_value(
        self,
        img_dims,
        pixel_vals,
    ):
        """Tests the circuit received after pixel value embedding."""
        ineqr_object = INEQR(img_dims, pixel_vals)
        feature_dims = int(math.log(img_dims[1], 2)) + int(math.log(img_dims[0], 2))

        mock_circuit = QuantumCircuit(feature_dims + COLOR_QUBITS)
        test_circuit = QuantumCircuit(feature_dims + COLOR_QUBITS)

        for _, y_val in enumerate(pixel_vals[0]):
            for _, x_val in enumerate(y_val):
                mock_circuit.clear()
                test_circuit.clear()

                pixel_val_bin = f"{int(x_val):0>8b}"
                for index, val in enumerate(pixel_val_bin):
                    if val == "1":
                        test_circuit.mcx(
                            control_qubits=list(range(3)), target_qubit=3 + index
                        )

                with mock.patch(
                    "piqture.embeddings.image_embeddings.ineqr.INEQR.circuit",
                    new_callable=lambda: mock_circuit,
                ):
                    ineqr_object.pixel_value(color_byte=f"{x_val:0>8b}")
                    assert mock_circuit == test_circuit

    @pytest.mark.parametrize(
        "img_dims, pixel_vals, resulting_circuit",
        [
            ((4, 2), [[[128, 64, 1, 2], [0, 0, 0, 1]]], "circuit_4_2"),
            ((2, 4), [[[128, 64], [1, 2], [0, 0], [0, 1]]], "circuit_4_2"),
            ((2, 2), [[[40, 128], [65, 2]]], "circuit_2_2"),
        ],
    )
    def test_ineqr(self, request, img_dims, pixel_vals, resulting_circuit):
        """Tests the final INEQR circuit."""
        ineqr_object = INEQR(img_dims, pixel_vals)
        feature_dims = int(math.log(img_dims[1], 2)) + int(math.log(img_dims[0], 2))
        mock_circuit = QuantumCircuit(feature_dims + COLOR_QUBITS)

        resulting_circuit = request.getfixturevalue(resulting_circuit)
        with mock.patch(
            "piqture.embeddings.image_embeddings.ineqr.INEQR.circuit",
            new_callable=lambda: mock_circuit,
        ):
            ineqr_object.ineqr()
            assert mock_circuit == resulting_circuit
