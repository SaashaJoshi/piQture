"""Improved Novel Enhanced Quantum Representation (INEQR) of digital images"""
from __future__ import annotations
import math
import numpy as np
from qiskit.circuit import QuantumCircuit

# from quantum_image_processing.data_encoder.image_representations.neqr import NEQR
from quantum_image_processing.data_encoder.image_representations.image_embedding import (
    ImageEmbedding,
)
from quantum_image_processing.mixin.image_embedding_mixin import ImageMixin


class INEQR(ImageEmbedding, ImageMixin):
    """
    Represents images in INEQR representation format.
    It is an enhanced version of the existing NEQR image
    representation method which helps in representing
    rectangular images on a quantum circuit.

    References:
        [1] N. Jiang and L. Wang, “Quantum image scaling
        using nearest neighbor interpolation,” Quantum
        Information Processing, vol. 14, no. 5, pp. 1559–1571,
        Sep. 2014, doi: https://doi.org/10.1007/s11128-014-0841-8.

    """

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
        self.x_coord = int(math.log(img_dims[0], 2))
        self.y_coord = int(math.log(img_dims[1], 2))
        self.feature_dim = self.x_coord + self.y_coord

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
        ImageMixin.pixel_position(self.circuit, pixel_pos_binary)

    def pixel_value(self, pixel_pos: int):
        """Embeds pixel (color) values in a circuit"""

    def ineqr(self) -> QuantumCircuit:
        """
        Builds the INEQR image representation on a circuit.

        Returns:
            QuantumCircuit: final circuit with the frqi image
            representation.
        """
        for y_index, y_val in enumerate(self.pixel_vals):
            for x_index, x_val in enumerate(y_val):
                self.circuit.barrier()
                pixel_pos_binary = (
                    f"{y_index:0>{self.y_coord}b}{x_index:0>{self.x_coord}b}"
                )

                # Embed pixel position on qubits
                self.pixel_position(pixel_pos_binary)
                # Embed color information on qubits
                self.pixel_value(x_val)
                # Remove pixel position embedding
                self.pixel_position(pixel_pos_binary)

        return self.circuit
