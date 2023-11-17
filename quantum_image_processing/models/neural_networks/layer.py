"""Abstract Base Class for QNN Layers"""

from __future__ import annotations
from abc import ABC, abstractmethod

# pylint: disable=too-few-public-methods


class Layer(ABC):
    """
    Abstract base class for layer structures in a quantum
    neural network.
    """

    def __init__(self, num_qubits: int):
        """
        Initializes a Layer circuit with the given number
        of qubits.
        """
        self.num_qubits = num_qubits

    @abstractmethod
    def build_layer(self):
        """
        Helps build a layer circuit.
        """
        return NotImplementedError
