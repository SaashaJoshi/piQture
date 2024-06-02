# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Improved Novel Enhanced Quantum Representation (INEQR) of digital images"""

from __future__ import annotations
import math
from qiskit.circuit import QuantumCircuit
from piqture.embeddings.image_embeddings.neqr import NEQR


class INEQR(NEQR):
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
        pixel_vals: list[list[list]],
        max_color_intensity: int = 255,
    ):
        NEQR.__init__(self, img_dims, pixel_vals, max_color_intensity)

        # Determine number of qubits for position embedding
        self.x_coord = int(math.log(img_dims[0], 2))
        self.y_coord = int(math.log(img_dims[1], 2))
        self.feature_dim = self.x_coord + self.y_coord

        # INEQR circuit
        self._circuit = QuantumCircuit(self.feature_dim + self.color_qubits)
        self.qr = self._circuit.qubits

    @property
    def circuit(self):
        """Returns INEQR circuit."""
        return self._circuit

    def validate_image_dimensions(self, img_dims):
        """Override existing method in ABC."""
        # Override validation of square images

        # Checks for 2-D images
        if len(set(img_dims)) > 2:
            raise ValueError(
                f"{self.__class__.__name__} supports 2-dimensional images only."
            )

        # Checks for img_dims as powers of 2.
        for dim in img_dims:
            if math.ceil(math.log(dim, 2)) != math.floor(math.log(dim, 2)):
                raise ValueError("Image dimensions must be powers of 2.")

    @staticmethod
    def validate_number_pixels(img_dims, pixel_vals):
        """
        Validates the number of pixels in pixel_lists
        in pixel_vals input.
        """
        # INEQR supports multi-dimensional lists to adjust for
        # unequal horizontal and vertical image dimensions.
        if all(
            len(pixel_lists.flatten()) != math.prod(img_dims)
            for pixel_lists in pixel_vals
        ):
            raise ValueError(
                f"No. of pixels ({[len(pixel_lists.flatten()) for pixel_lists in pixel_vals]}) "
                f"in each pixel_lists in pixel_vals must be equal to the "
                f"product of image dimensions {math.prod(img_dims)}."
            )

    def ineqr(self) -> QuantumCircuit:
        """
        Builds the INEQR image representation on a circuit.

        Returns:
            QuantumCircuit: final circuit with the INEQR image
            representation.
        """
        for i in range(self.feature_dim):
            self.circuit.h(i)

        for y_index, y_val in enumerate(self.pixel_vals[0]):
            for x_index, x_val in enumerate(y_val):
                pixel_pos_binary = (
                    f"{y_index:0>{self.y_coord}b}{x_index:0>{self.x_coord}b}"
                )
                color_byte = f"{x_val:0>8b}"

                # Embed pixel position on qubits
                self.pixel_position(pixel_pos_binary)
                # Embed color information on qubits
                self.pixel_value(color_byte=color_byte)
                # Remove pixel position embedding
                self.pixel_position(pixel_pos_binary)

        return self.circuit
