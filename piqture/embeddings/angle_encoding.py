# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Angle Encoder"""
from __future__ import annotations
import math
from qiskit.circuit import QuantumCircuit
from piqture.embeddings.image_embedding import ImageEmbedding


class AngleEncoding(ImageEmbedding):
    """
    Implements an Angle encoding technique.
    """

    def __init__(self, img_dims: tuple[int, ...], pixel_vals: list[list] = None):
        ImageEmbedding.__init__(self, img_dims, pixel_vals)

        self.feature_dims = int(math.prod(self.img_dims))
        self._circuit = QuantumCircuit(self.feature_dims)
        self._qr = self._circuit.qubits

        # Performs encoding at instantiation.
        self.embedding()

    @property
    def circuit(self):
        """Returns angle embedding circuit."""
        return self._circuit

    def validate_image_dimensions(self, img_dims):
        """Validates img_dims input."""

    def validate_number_pixel_lists(self, pixel_vals):
        """
        Validates the number of pixel_lists in
        pixel_vals input.
        """
        if len(pixel_vals) != self.img_dims[1]:
            raise ValueError(
                f"No. of pixel_lists ({len(pixel_vals)}) must be equal "
                f"to the number of columns in the image {self.img_dims[1]}."
            )

    @staticmethod
    def validate_number_pixels(img_dims, pixel_vals):
        """
        Validates the number of pixels in pixel_lists
        in pixel_vals input.
        """
        if all(len(pixel_list) != img_dims[0] for pixel_list in pixel_vals):
            raise ValueError(
                f"No. of pixels in each pixel_list in pixel_vals must "
                f"be equal to the number of rows in the image {img_dims[0]}."
            )

    def pixel_position(self, pixel_pos_binary: str):
        """Embeds pixel positions on the qubits."""

    def pixel_value(self, *args, **kwargs):
        """Embeds pixel or color values on the qubits."""

    def embedding(self) -> QuantumCircuit:
        """Embeds data using Angle encoding technique."""
        for qubit in range(self.feature_dims):
            self.circuit.ry(self.parameters[qubit], qubit)
        return self.circuit
