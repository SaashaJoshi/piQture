"""Neural Network Abstract Base Class"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister


class QuantumNeuralNetwork(ABC):
    """
    Abstract base class for all quantum neural network structures.
    These structures consist of data encoding/embedding,
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
    def sequence(self, operations: Callable):
        """
        Composes circuits with given list of operations.
        """
        return NotImplementedError
