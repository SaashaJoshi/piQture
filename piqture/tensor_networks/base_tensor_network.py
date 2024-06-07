# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Base Tensor Network Circuit"""

from __future__ import annotations

from abc import ABC

from qiskit.circuit import QuantumCircuit

# pylint: disable=too-few-public-methods


class BaseTensorNetwork(ABC):
    """Abstract Base Class for Tensor Network Circuits"""

    def __init__(self, num_qubits: int):
        """
        Initializes a tensor network circuit.

        Args:
            num_qubits (int): number of pixels in the input image data.
        """
        if not isinstance(num_qubits, int):
            raise TypeError("Input num_qubits must be of the type int.")

        if num_qubits <= 0:
            raise ValueError("Number of qubits cannot be zero or negative.")

        self.num_qubits = num_qubits
        self._circuit = QuantumCircuit(self.num_qubits)
        self.q_reg = self._circuit.qubits

    @property
    def circuit(self):
        """Returns the tensor network circuit property."""
        return self._circuit
