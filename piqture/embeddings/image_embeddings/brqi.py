"""Bitplane Representation of Quantum Images (BRQI)"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
from qiskit.circuit import QuantumCircuit

from piqture.embeddings.image_embedding import ImageEmbedding
from piqture.mixin.image_embedding_mixin import ImageMixin


class BRQI(ImageEmbedding, ImageMixin):
    """Represents images in BRQI representation format."""

    def __init__(
        self,
        img_dims: tuple[int, int],
        pixel_vals: Union[list[list], np.ndarray],
        max_color_intensity: int = 255,
    ):
        self.max_color_intensity = max_color_intensity
        self.validate_max_color_intensity()

        # Convert numpy array to list if necessary
        if isinstance(pixel_vals, np.ndarray):
            pixel_vals = [pixel_vals.flatten().tolist()]

        ImageEmbedding.__init__(self, img_dims, pixel_vals)
        self.color_qubits = int(np.ceil(np.log2(self.max_color_intensity + 1)))
        self.feature_dim = int(np.ceil(np.log2(math.prod(self.img_dims))))

        # Initialize the _circuit attribute
        self._circuit = QuantumCircuit(self.feature_dim + self.color_qubits)

    def validate_max_color_intensity(self):
        """Validate the maximum color intensity value.

        Raises:
            ValueError: If max_color_intensity is less than 0 or greater than 255
        """
        if self.max_color_intensity < 0 or self.max_color_intensity > 255:
            raise ValueError(
                "Maximum color intensity cannot be less than 0 or greater than 255."
            )

    def validate_number_pixel_lists(self, pixel_vals):
        if len(pixel_vals) > 1:
            raise ValueError(
                f"{self.__class__.__name__} supports grayscale images only. "
                f"No. of pixel_lists in pixel_vals must be maximum 1."
            )

    @property
    def circuit(self):
        """Returns BRQI circuit."""
        return self._circuit

    def pixel_position(self, pixel_pos_binary: str):
        """Embeds pixel position values in a circuit."""
        ImageMixin.pixel_position(self.circuit, pixel_pos_binary)

    def pixel_value(self, *args, **kwargs):
        """
        Embeds pixel (color) values in a circuit.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
                color_byte (str): Binary representation of the color value.
        """
        # Get the color byte from kwargs
        color_byte = kwargs.get("color_byte")
        if not color_byte:
            raise ValueError("color_byte must be provided")

        # Create control qubits list
        control_qubits = self._get_control_qubits()

        # Apply gates for each '1' in the color byte
        self._apply_color_gates(color_byte, control_qubits)

    def _get_control_qubits(self):
        """Helper method to get control qubits."""
        return list(range(self.feature_dim))

    def _apply_color_gates(self, color_byte: str, control_qubits: list):
        """
        Apply quantum gates for color encoding.

        Args:
            color_byte (str): Binary representation of the color value.
            control_qubits (list): List of control qubits.
        """
        for index, color in enumerate(color_byte):
            if color == "1":
                self.circuit.mcx(
                    control_qubits=control_qubits, target_qubit=self.feature_dim + index
                )

    def brqi(self) -> QuantumCircuit:
        """
        Builds the BRQI image representation on a circuit.

        Returns:
            QuantumCircuit: final circuit with the BRQI image
            representation.
        """
        self.pixel_vals = np.array(self.pixel_vals).flatten()
        for i in range(self.feature_dim):
            self.circuit.h(i)

        num_pixels = math.prod(self.img_dims)
        for pixel in range(num_pixels):
            color_byte = f"{int(self.pixel_vals[pixel]):0>{self.color_qubits}b}"
            self.pixel_value(color_byte=color_byte)

        # Add measurement to all qubits
        self.circuit.measure_all()

        return self.circuit
