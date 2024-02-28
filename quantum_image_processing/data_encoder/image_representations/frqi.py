"""Flexible Representation of Quantum Images (FRQI)"""

from __future__ import annotations
import math
import numpy as np
from qiskit.circuit import QuantumCircuit
from quantum_image_processing.data_encoder.image_representations.image_embedding import (
    ImageEmbedding,
)
from quantum_image_processing.mixin.image_embedding_mixin import ImageMixin


class FRQI(ImageEmbedding, ImageMixin):
    """
    Represents images in FRQI representation format
    """

    def __init__(self, img_dims: tuple[int, int], pixel_vals: list):
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
            self.pixel_vals[pixel_pos],
            target_qubit=self.feature_dim,
            control_qubit=self.feature_dim - 2,
        )
        self.circuit.cx(0, 1)
        self.circuit.cry(
            -self.pixel_vals[pixel_pos],
            target_qubit=self.feature_dim,
            control_qubit=self.feature_dim - 1,
        )
        self.circuit.cx(0, 1)
        self.circuit.cry(
            self.pixel_vals[pixel_pos],
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
