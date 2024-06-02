# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Flexible Representation of Quantum Images (FRQI)"""

from __future__ import annotations
import math
import numpy as np
from qiskit.circuit import QuantumCircuit
from piqture.embeddings.image_embedding import (
    ImageEmbedding,
)
from piqture.mixin.image_embedding_mixin import ImageMixin


class FRQI(ImageEmbedding, ImageMixin):
    """
    Represents images in FRQI representation format
    """

    def __init__(self, img_dims: tuple[int, int], pixel_vals: list[list] = None):
        ImageEmbedding.__init__(self, img_dims, pixel_vals)

        # feature_dim = no. of qubits for pixel position embedding
        self.feature_dim = int(np.sqrt(math.prod(self.img_dims)))

        # FRQI circuit
        self._circuit = QuantumCircuit(self.feature_dim + 1)
        self.qr = self._circuit.qubits

    @property
    def circuit(self):
        """Returns the FRQI circuit."""
        return self._circuit

    def pixel_position(self, pixel_pos_binary: str):
        """Embeds pixel position values in a circuit."""
        ImageMixin.pixel_position(self.circuit, pixel_pos_binary)

    def pixel_value(self, *args, **kwargs):
        """Embeds pixel (color) values in a circuit"""
        pixel_pos = kwargs.get("pixel_pos")

        self.circuit.cry(
            self._parameters[pixel_pos],
            target_qubit=self.feature_dim,
            control_qubit=self.feature_dim - 2,
        )
        self.circuit.cx(0, 1)
        self.circuit.cry(
            -self._parameters[pixel_pos],
            target_qubit=self.feature_dim,
            control_qubit=self.feature_dim - 1,
        )
        self.circuit.cx(0, 1)
        self.circuit.cry(
            self._parameters[pixel_pos],
            target_qubit=self.feature_dim,
            control_qubit=self.feature_dim - 1,
        )

    def frqi(self) -> QuantumCircuit:
        # pylint: disable=duplicate-code
        """
        Builds the FRQI image representation on a circuit.

        Returns:
            QuantumCircuit: final circuit with the frqi image
            representation.
        """
        for i in range(self.feature_dim):
            self.circuit.h(i)

        # Supports grayscale images only.
        num_theta = math.prod(self.img_dims)
        for pixel in range(num_theta):
            pixel_pos_binary = f"{pixel:0>2b}"

            # Embed pixel position on qubits
            self.pixel_position(pixel_pos_binary)
            # Embed color information on qubits
            self.pixel_value(pixel_pos=pixel)
            # Remove pixel position embedding
            self.pixel_position(pixel_pos_binary)

        return self.circuit
