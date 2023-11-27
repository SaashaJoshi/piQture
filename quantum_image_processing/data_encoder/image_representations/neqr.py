"""Novel Enhanced Quantum Representation (NEQR) of digital images"""
from __future__ import annotations
import math
from typing import Optional
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from quantum_image_processing.data_encoder.image_representations.frqi import FRQI


class NEQR(FRQI):
    """Represents images in NEQR representation format."""

    def __init__(
        self,
        img_dims: tuple[int, int],
        pixel_vals: list,
        max_color_intensity: Optional[int] = 255,
    ):
        FRQI.__init__(self, img_dims, pixel_vals)

        if max_color_intensity < 0 or max_color_intensity > 255:
            raise ValueError(
                "Maximum color intensity cannot be less than 0 or greater than 255."
            )

        self.feature_dim = int(np.sqrt(math.prod(self.img_dims)))
        self.max_color_intensity = max_color_intensity + 1

        # number of qubits to encode color byte
        self.color_qubits = int(math.log(self.max_color_intensity, 2))

        # NEQR circuit
        self.qr = QuantumRegister(self.feature_dim + self.color_qubits)
        self._circuit = QuantumCircuit(self.qr)

    @property
    def circuit(self):
        return self._circuit

    def pixel_value(self, pixel_pos: int):
        """Embeds pixel (color) values in a circuit"""
        color_byte = f"{int(self.pixel_vals[pixel_pos]):0>8b}"

        control_qubits = list(range(self.feature_dim))
        for index, color in enumerate(color_byte):
            if color == "1":
                self.circuit.mct(
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
