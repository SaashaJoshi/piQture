"""Flexible Representation of Quantum Images (FRQI)"""
from __future__ import annotations
import math
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from quantum_image_processing.data_encoder.image_representations.image_embedding import (
    ImageEmbedding,
)


class FRQI(ImageEmbedding):
    """
    Represents images in FRQI representation format
    """

    def __init__(self, img_dims: tuple[int, int], pixel_vals: list):
        ImageEmbedding.__init__(self, img_dims, pixel_vals)

        if len(set(img_dims)) > 1:
            raise ValueError(
                f"{self.__class__.__name__} supports square images only. "
                f"Input img_dims must have same dimensions."
            )

        if len(pixel_vals) != math.prod(self.img_dims):
            raise ValueError(
                f"No. of pixel values {len(pixel_vals)} must be equal to "
                f"the product of image dimensions {math.prod(self.img_dims)}."
            )

        for val in pixel_vals:
            if val < 0 or val > 255:
                raise ValueError("Pixel values cannot be less than 0 or greater than 255.")

        # feature_dim = no. of qubits for pixel position embedding
        self.feature_dim = int(np.sqrt(math.prod(self.img_dims)))

        # FRQI circuit
        self.qr = QuantumRegister(self.feature_dim + 1)
        self._circuit = QuantumCircuit(self.qr)

    @property
    def circuit(self):
        """Returns the FRQI circuit."""
        return self._circuit

    def pixel_position(self, pixel_pos_binary: str):
        """Embeds pixel position values in a circuit."""

        for index, value in enumerate(pixel_pos_binary):
            if value == "0":
                self.circuit.x(index)

    def pixel_value(self, pixel_pos: int):
        """Embeds pixel (color) values in a circuit"""

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
            self.pixel_value(pixel)
            # Remove pixel position embedding
            self.pixel_position(pixel_pos_binary)

        return self.circuit
