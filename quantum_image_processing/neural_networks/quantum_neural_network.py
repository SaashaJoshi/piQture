"""Neural Network Abstract Base Class"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable
from qiskit.circuit import QuantumCircuit

# pylint: disable=too-few-public-methods


class QuantumNeuralNetwork(ABC):
    """
    Abstract base class for all quantum neural network structures.

    These structures may consist of data encoding/embedding,
    model layers (consisting of layers such as convolutional,
    pooling, etc.), and a measurement stage.
    """

    def __init__(self, num_qubits: int):
        """
        Initializes a Quantum Neural Network circuit with the given
        number of qubits.
        """
        if not isinstance(num_qubits, int):
            raise TypeError("The input num_qubits must be of the type int.")

        if num_qubits <= 0:
            raise ValueError("The input num_qubits must be at least 1.")

        self.num_qubits = num_qubits
        self._circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
        self.qr = self._circuit.qubits
        self.cr = self._circuit.clbits

    @property
    def circuit(self):
        """Returns the QCNN circuit."""
        return self._circuit

    @abstractmethod
    def sequence(self, operations: list[tuple[Callable, dict]]):
        """
        Composes circuits with given list of operations.

        Args:
            operations (list[tuple[Callable, dict]]: a tuple
            of a Layer object and a dictionary of its arguments.
        """
        if not isinstance(operations, list):
            raise TypeError("The input operations must be of the type list.")

        if not all(isinstance(operation, tuple) for operation in operations):
            raise TypeError(
                "The input operations list must contain tuple[operation, params]."
            )

        if not callable(operations[0][0]):
            raise TypeError(
                "Operation in input operations list must be Callable functions/classes."
            )

        if not isinstance(operations[0][1], dict):
            raise TypeError(
                "Parameters of operation in input operations list must be in a dictionary."
            )
