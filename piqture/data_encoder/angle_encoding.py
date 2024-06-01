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
from qiskit.circuit import QuantumCircuit, ParameterVector
from piqture.data_encoder.image_embedding import ImageEmbedding


class AngleEncoding(ImageEmbedding):
    """
    Implements an Angle encoding technique.
    """

    def __init__(self, img_dims: tuple[int, ...], pixel_vals: list[list] = None):
        ImageEmbedding.__init__(self, img_dims, pixel_vals, color_channels=1)
        self.img_dims = img_dims
        self.feature_dims = int(math.prod(self.img_dims))

        if self.pixel_vals is None:
            self._parameters = ParameterVector("Angle", self.feature_dims)
        else:
            self._parameters = self.pixel_vals

        self._circuit = QuantumCircuit(self.feature_dims)
        self._qr = self._circuit.qubits

        # Performs encoding at instantiation.
        self.embedding()

    @property
    def parameters(self):
        """Returns parameters in angle embedding circuit."""
        return self._parameters

    @property
    def circuit(self):
        """Returns angle embedding circuit."""
        return self._circuit

    def validate_image_dimensions(self, img_dims):
        """Validates img_dims input."""

    def pixel_position(self, pixel_pos_binary: str):
        pass

    def pixel_value(self, *args, **kwargs):
        pass

    def embedding(self) -> QuantumCircuit:
        """Embeds data using Angle encoding technique."""
        for qubit in range(self.feature_dims):
            self.circuit.ry(self.parameters[qubit], qubit)
        return self.circuit
