"""Neural Network Abstract Base Class"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable

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
        self.num_qubits = num_qubits
        # self.qr = QuantumRegister(self.num_qubits)
        # self.cr = ClassicalRegister(self.num_qubits)
        # self.circuit = QuantumCircuit(self.qr, self.cr)

    @abstractmethod
    def sequence(self, operations: list[tuple[Callable, dict]]):
        """
        Composes circuits with given list of operations.

        Args:
            operations (list[tuple[Callable, dict]]: a tuple
            of a Layer object and a dictionary of its arguments.
        """
        return NotImplementedError
