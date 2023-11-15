"""Abstract Base Class for QNN Layers"""

from __future__ import annotations
from abc import ABC, abstractmethod
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister


class Layer(ABC):
    def __init__(self, num_qubits: int, **kwargs):
        self.num_qubits = num_qubits

    @abstractmethod
    def build_layer(self):
        return NotImplementedError
