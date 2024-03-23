# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Novel Enhanced Quantum Representation (NEQR) of digital images"""

from __future__ import annotations
import math
import numpy as np
from qiskit.circuit import QuantumCircuit
from piqture.data_encoder.image_representations.image_embedding import (
    ImageEmbedding,
)
from piqture.mixin.image_embedding_mixin import ImageMixin


class NEQR(ImageEmbedding, ImageMixin):
    """Represents images in NEQR representation format."""

    def __init__(
        self,
        img_dims: tuple[int, int],
        pixel_vals: list[list],
        max_color_intensity: int = 255,
    ):
        ImageEmbedding.__init__(self, img_dims, pixel_vals, colored=False)

        if max_color_intensity < 0 or max_color_intensity > 255:
            raise ValueError(
                "Maximum color intensity cannot be less than 0 or greater than 255."
            )

        self.feature_dim = int(np.ceil(np.sqrt(math.prod(self.img_dims))))
        self.max_color_intensity = max_color_intensity + 1

        # number of qubits to encode color byte
        self.color_qubits = int(np.ceil(math.log(self.max_color_intensity, 2)))

        # NEQR circuit
        self._circuit = QuantumCircuit(self.feature_dim + self.color_qubits)
        self.qr = self._circuit.qubits

    @property
    def circuit(self):
        """Returns NEQR circuit."""
        return self._circuit

    def pixel_position(self, pixel_pos_binary: str):
        """Embeds pixel position values in a circuit."""
        ImageMixin.pixel_position(self.circuit, pixel_pos_binary)

    def pixel_value(self, *args, **kwargs):
        """
        Embeds pixel (color) values in a circuit
        """
        color_byte = kwargs.get("color_byte")
        control_qubits = list(range(self.feature_dim))
        for index, color in enumerate(color_byte):
            if color == "1":
                self.circuit.mct(
                    control_qubits=control_qubits, target_qubit=self.feature_dim + index
                )

    def neqr(self) -> QuantumCircuit:
        # pylint: disable=duplicate-code
        """
        Builds the NEQR image representation on a circuit.

        Returns:
            QuantumCircuit: final circuit with the frqi image
            representation.
        """
        self.pixel_vals = self.pixel_vals.flatten()
        for i in range(self.feature_dim):
            self.circuit.h(i)

        num_theta = math.prod(self.img_dims)
        for pixel in range(num_theta):
            pixel_pos_binary = f"{pixel:0>2b}"
            color_byte = f"{int(self.pixel_vals[pixel]):0>8b}"

            # Embed pixel position on qubits
            self.pixel_position(pixel_pos_binary)
            # Embed color information on qubits
            self.pixel_value(color_byte=color_byte)
            # Remove pixel position embedding
            self.pixel_position(pixel_pos_binary)

        return self.circuit
