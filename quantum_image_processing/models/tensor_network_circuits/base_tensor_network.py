"""Base Tensor Network Circuit"""
from __future__ import annotations
import math
from qiskit.circuit import QuantumCircuit

# pylint: disable=too-few-public-methods


class BaseTensorNetwork:
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
