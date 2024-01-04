"""Improved Novel Enhanced Quantum Representation (INEQR) of digital images"""
from __future__ import annotations
import math
from typing import Optional
import numpy as np
from qiskit.circuit import QuantumCircuit
# from quantum_image_processing.data_encoder.image_representations.neqr import NEQR
from quantum_image_processing.data_encoder.image_representations.image_embedding import (
    ImageEmbedding,
)


class INEQR(ImageEmbedding):
    """Represents images in INEQR representation format."""

    def __init__(
        self,
        img_dims: tuple[int, int],
        pixel_vals: list,
        max_color_intensity: int = 255,
    ):
        ImageEmbedding.__init__(self, img_dims, pixel_vals)

        if len(set(img_dims)) > 2:
            raise ValueError(
                f"{self.__class__.__name__} supports 2-dimensional images only. "
            )

        # Determine number of qubits for position embedding
        x_coord = int(math.log(img_dims[0], 2))
        y_coord = int(math.log(img_dims[1], 2))
        self.feature_dim = x_coord + y_coord

        # Number of qubits to encode color byte
        if max_color_intensity < 0 or max_color_intensity > 255:
            raise ValueError(
                "Maximum color intensity cannot be less than 0 or greater than 255."
            )
        self.max_color_intensity = max_color_intensity + 1
        self.color_qubits = int(np.ceil(math.log(self.max_color_intensity, 2)))

        # INEQR circuit
        self._circuit = QuantumCircuit(self.feature_dim + self.color_qubits)
        self.qr = self._circuit.qubits
        print(f"Total qubits: {self.feature_dim + self.color_qubits}")

    @property
    def circuit(self):
        """Returns INEQR circuit."""
        return self._circuit

    def pixel_position(self, pixel_pos_binary: str):
        """Embeds pixel position values in a circuit."""

    def pixel_value(self, pixel_pos: int):
        """Embeds pixel (color) values in a circuit"""

    def ineqr(self) -> QuantumCircuit:
        """
        Builds the INEQR image representation on a circuit.

        Returns:
            QuantumCircuit: final circuit with the frqi image
            representation.
        """
