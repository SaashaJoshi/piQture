"""Novel Enhanced Quantum Representation (NEQR) of digital images"""
from __future__ import annotations
import math
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from quantum_image_processing.data_encoder.image_representations.frqi import FRQI


class NEQR(FRQI):
    """Represents images in NEQR representation format."""

    def __init__(
        self, img_dims: tuple[int, int], pixel_vals: list[str], max_color: int = 255
    ):
        FRQI.__init__(self, img_dims, pixel_vals)

        self.feature_dim = int(np.sqrt(math.prod(self.img_dims)))
        self.max_color = max_color + 1
        # color/pixel value in binary
        self.color_binary = int(math.log(self.max_color, 2))

        # NEQR circuit
        self.qr = QuantumRegister(self.feature_dim + self.color_binary)
        self.circ = QuantumCircuit(self.qr)

    def pixel_value(self, pixel_pos: int):
        """Embeds pixel (color) values in a circuit"""
        color_binary = f"{int(self.pixel_vals[pixel_pos]):0>8b}"

        control_qubits = list(range(self.feature_dim))
        for index, color in enumerate(color_binary):
            if color == "1":
                self.circ.mct(
                    control_qubits=control_qubits, target_qubit=self.feature_dim + index
                )

    def neqr(self) -> QuantumCircuit:
        """
        Builds the NEQR image representation on a circuit.

        Returns:
            QuantumCircuit: final circuit with the frqi image
            representation.
        """
        return self.frqi()
