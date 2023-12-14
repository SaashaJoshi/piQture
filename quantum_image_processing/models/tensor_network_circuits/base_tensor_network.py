"""Base Tensor Network Circuit"""
from __future__ import annotations
import math
from qiskit.circuit import QuantumCircuit


class BaseTensorNetwork:
    """Abstract Base Class for Tensor Network Circuits"""

    def __init__(self, img_dims: tuple[int, int]):
        """
        Initializes a tensor network circuit.

        Args:
            img_dims (int): dimensions of the input image data.
        """
        if not all((isinstance(dims, int) for dims in img_dims)) or not isinstance(
            img_dims, tuple
        ):
            raise TypeError("Input img_dims must be of the type tuple[int, int].")

        if math.prod(img_dims) <= 0:
            raise ValueError("Image dimensions cannot be zero or negative.")

        self.img_dims = img_dims
        self.num_qubits = int(math.prod(self.img_dims))
        self._circuit = QuantumCircuit(self.num_qubits)
        self.q_reg = self._circuit.qubits

    @property
    def circuit(self):
        """Returns the tensor network circuit property."""
        return self._circuit
