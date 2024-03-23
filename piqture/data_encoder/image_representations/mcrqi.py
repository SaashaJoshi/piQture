"""Multi-Channel Representation of Quantum Image (MCRQI)"""
from __future__ import annotations
import math
import numpy as np
from qiskit.circuit import QuantumCircuit
from piqture.data_encoder.image_representations.frqi import FRQI


class MCRQI(FRQI):
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

    def __init__(self, img_dims: tuple[int, int], pixel_vals: list):
        FRQI.__init__(self, img_dims, pixel_vals)

        self.feature_dim = int(np.ceil(np.sqrt(math.prod(self.img_dims))))
        # No. of qubits for RGB-alpha color channels
        self.color_channels = 3

        # MCRQI circuit
        self._circuit = QuantumCircuit(self.feature_dim + self.color_channels)
        self.qr = self.circuit.qubits

    @property
    def circuit(self):
        return self._circuit

    def pixel_value(self, pixel_pos: int):
        """Embeds pixel (color) values in a circuit"""

    def mcrqi(self) -> QuantumCircuit:
        """
        Builds the MCRQI image representation with RGB-alpha
        color channels on a circuit.

        Returns:
            QuantumCircuit: final circuit with the MCRQI image
            representation.
        """
        return self.frqi()
