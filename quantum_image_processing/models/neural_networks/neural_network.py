"""Neural Network Abstract Base Class"""

from __future__ import annotations
from abc import ABC, abstractmethod


class NeuralNetwork(ABC):
    """
    Abstract base class for all neural network structures.
    These structures consist of data encoding/embedding,
    forward pass (consisting of layers such as convolutional,
    pooling, etc.), backward pass (for training purposes),
    and finally a measurement stage.
    """

    @abstractmethod
    def forward_pass(self):
        """
        Implements a forward pass in a neural network.
        """
        return NotImplementedError

    @abstractmethod
    def backward_pass(self):
        """
        Implements a backward pass in a neural network.
        """
        return NotImplementedError
