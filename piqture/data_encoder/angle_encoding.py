# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Angle Encoder"""
from __future__ import annotations
import math
from qiskit.circuit import QuantumCircuit, ParameterVector


class AngleEncoding:
    """
    Implements an Angle encoding technique.
    """

    def __init__(self, img_dims: tuple[int, ...]):
        if not isinstance(img_dims, tuple):
            raise TypeError("Input img_dims must be of the type tuple.")

        for dims in img_dims:
            print("Type of dims: ", isinstance(dims, int))

        if not all((isinstance(dims, int) for dims in img_dims)) or all(
            (isinstance(dims, bool) for dims in img_dims)
        ):
            raise TypeError("Input img_dims must be of the type tuple[int, ...].")
        self.img_dims = img_dims

        self.feature_dims = int(math.prod(self.img_dims))
        self._parameters = ParameterVector("Angle", self.feature_dims)
        self._circuit = QuantumCircuit(self.feature_dims)
        self._qr = self._circuit.qubits

        # Performs encoding at instantiation.
        self.embedding()

    @property
    def parameters(self):
        """Returns parameters in angle embedding circuit."""
        return self._parameters

    @property
    def circuit(self):
        """Returns angle embedding circuit."""
        return self._circuit

    def embedding(self) -> QuantumCircuit:
        """Embeds data using Angle encoding technique."""
        for qubit in range(self.feature_dims):
            self.circuit.ry(self.parameters[qubit], qubit)
        return self.circuit
