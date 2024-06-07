# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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

    @staticmethod
    def channel_index(
        circuit: QuantumCircuit, channel_index_binary: str, qubit_padding: int
    ):
        """
        Embeds channel indices on the qubits.
        """
        for index, value in enumerate(channel_index_binary):
            if value == "0":
                circuit.x(index + qubit_padding)
