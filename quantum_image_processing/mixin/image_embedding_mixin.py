"""Mixin class for Image Embedding methods"""

from __future__ import annotations
from qiskit.circuit import QuantumCircuit


class ImageMixin:
    """
    A mixin class for implementation of common
    image embedding methods.
    """

    @staticmethod
    def pixel_position(circuit: QuantumCircuit, pixel_pos_binary: str):
        """
        Embeds image pixel positions on the qubits.

        Args:
            circuit: input circuit on which pixel
            position is to be embedded.

            pixel_pos_binary (str): takes a binary
            representation of the pixel position.
        """
        for index, value in enumerate(pixel_pos_binary):
            if value == "0":
                circuit.x(index)

    def pixel_value(self, pixel_pos: int):
        """
        Embeds pixel or color values on the qubits.

        Args:
            pixel_pos (int): takes as an input
            the pixel position.
        """
