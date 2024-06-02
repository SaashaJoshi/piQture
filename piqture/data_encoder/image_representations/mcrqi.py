"""Multi-Channel Representation of Quantum Image (MCRQI)"""

from __future__ import annotations
import math
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import MCMT, RYGate
from piqture.data_encoder.image_embedding import ImageEmbedding
from piqture.mixin.image_embedding_mixin import ImageMixin


class MCRQI(ImageEmbedding):
    """
    Represents image in RBG-alpha color space on
    a quantum circuit, where RBG is the color channel
    and alpha represents transparency in an image [1].

    References:
        [1] “A Multi-Channel Representation for images
        on quantum computers using the RGBα color space
        | IEEE Conference Publication | IEEE Xplore,”
        ieeexplore.ieee.org.
        https://ieeexplore.ieee.org/document/6051718.
    """

    def __init__(self, img_dims: tuple[int, int], pixel_vals: list[list]):
        ImageEmbedding.__init__(self, img_dims, pixel_vals, color_channels=4)

        self.feature_dim = int(np.ceil(np.sqrt(math.prod(self.img_dims))))
        # No. of qubits for RGB-alpha color channels
        self.color_channels = 1
        # No. of qubits for RGB-alpha color index
        self.channel_index_qubits = 2

        # MCRQI circuit
        self._circuit = QuantumCircuit(
            self.feature_dim + self.channel_index_qubits + self.color_channels
        )
        self.qr = self.circuit.qubits

    @property
    def circuit(self):
        """Returns MCRQI circuit."""
        return self._circuit

    def pixel_position(self, pixel_pos_binary: str):
        """Embeds pixel position values in a circuit."""
        ImageMixin.pixel_position(self.circuit, pixel_pos_binary)

    def channel_index(self, channel_index_binary: str, qubit_padding: int):
        """Embeds color index on the qubits."""
        ImageMixin.channel_index(self.circuit, channel_index_binary, qubit_padding)

    def pixel_value(self, *args, **kwargs):
        """Embeds pixel (color) values in a circuit"""
        pixel = kwargs.get("pixel")

        self.circuit.compose(
            MCMT(
                RYGate(2 * pixel),
                num_target_qubits=1,
                num_ctrl_qubits=self.feature_dim + self.channel_index_qubits,
            ),
            inplace=True,
        )

    def mcrqi(self) -> QuantumCircuit:
        """
        Builds the MCRQI image representation with RGB-alpha
        color channels on a circuit.

        Returns:
            QuantumCircuit: final circuit with the MCRQI image
            representation.
        """
        for i in range(self.feature_dim + self.channel_index_qubits):
            self.circuit.h(i)

        for channel, channel_pixels in enumerate(self.pixel_vals):
            for pixel_pos, pixel in enumerate(channel_pixels):
                pixel_pos_binary = f"{pixel_pos:0>{self.feature_dim}b}"
                channel_index_binary = f"{channel:0>2b}"

                # Embed pixel position and channel index on qubits
                self.pixel_position(pixel_pos_binary)
                self.channel_index(channel_index_binary, self.feature_dim)

                # Embed color information on qubits
                self.pixel_value(pixel=pixel)

                # Remove pixel position and channel index embedding
                self.pixel_position(pixel_pos_binary)
                self.channel_index(channel_index_binary, self.feature_dim)

        return self.circuit
