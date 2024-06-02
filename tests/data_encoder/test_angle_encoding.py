# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unit test for AngleEncoding class"""

from __future__ import annotations

import math
import numpy as np
import pytest
from pytest import raises
from qiskit.circuit import QuantumCircuit, ParameterVector
from piqture.data_encoder.angle_encoding import AngleEncoding


@pytest.fixture(name="circuit_embedding")
def circuit_fixture():
    """Circuit fixture for Angle Embedding."""

    def _circuit(img_dims, pixel_vals):
        img_dims = math.prod(img_dims)
        if pixel_vals is None:
            pixels = ParameterVector("Angle", img_dims)
        else:
            pixels = [pixel for pixel_list in pixel_vals for pixel in pixel_list]

        embedding_circuit = QuantumCircuit(img_dims)
        for qubit, pixel in enumerate(pixels):
            embedding_circuit.ry(pixel, qubit)

        return embedding_circuit

    return _circuit


class TestAngleEncoding:
    """Tests for AngleEncoding class"""

    @pytest.mark.parametrize(
        "img_dims, pixel_vals",
        [
            ((1, 2), [[215], [255], [209]]),
            ((3, 2), [[12.5, 98.2, 67.5]]),
            ((4, 1), [[12.5, 98.2, 34.9, 87.2], [12.5, 98.2, 34.9, 87.2]]),
            ((9, 8), [[1]]),
        ],
    )
    def test_pixel_lists(self, img_dims, pixel_vals):
        """Tests the number of pixel_lists in pixel_vals argument."""
        with raises(
            ValueError,
            match=r"No. of pixel_lists \(\d+\) must be equal "
            r"to the number of columns in the image \d+\.",
        ):
            _ = AngleEncoding(img_dims, pixel_vals)

    # Since lists are converted to array, pixel_lists cannot
    # have different number of pixels.
    # Figure the way to test this.
    @pytest.mark.parametrize(
        "img_dims, pixel_vals",
        [
            ((1, 2), [[215, 215], [255, 255]]),
            ((2, 1), [[215, 255, 234]]),
            ((3, 2), [[12.5, 98.2], [45, 34.9]]),
            ((4, 1), [[12.5, 98.2, 34.9, 87.2, 45.6]]),
        ],
    )
    def test_number_pixels(self, img_dims, pixel_vals):
        """Tests the number of pixels in each pixel_lists in pixel_vals argument."""
        with raises(
            ValueError,
            match=r"No. of pixels in each pixel_list in pixel_vals must "
            r"be equal to the number of rows in the image \d+\.",
        ):
            _ = AngleEncoding(img_dims, pixel_vals)

    @pytest.mark.parametrize(
        "img_dims, pixel_vals",
        [
            ((1, 2), [[215], [255]]),
            ((2, 1), [[215, 255]]),
            ((3, 2), [[12.5, 98.2, 67.5], [45, 34.9, 87.2]]),
            ((4, 1), [[12.5, 98.2, 34.9, 87.2]]),
            ((9, 8), None),
            ((2, 1), None),
        ],
    )
    def test_embedding(self, img_dims, pixel_vals, circuit_embedding):
        """Tests Angle embedding circuits."""
        test_circuit = circuit_embedding(img_dims, pixel_vals)
        resulting_circuit = AngleEncoding(img_dims, pixel_vals)

        if pixel_vals is None:
            pixel_vals = np.random.random(math.prod(img_dims))
            test_circuit.assign_parameters(pixel_vals, inplace=True)
            resulting_circuit.circuit.assign_parameters(pixel_vals, inplace=True)

        assert test_circuit == resulting_circuit.circuit
