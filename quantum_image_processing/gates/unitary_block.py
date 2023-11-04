from __future__ import annotations
from abc import ABC, abstractmethod
from qiskit.circuit import QuantumCircuit, ParameterVector


class Unitary(ABC):
    """
    Implements a two-qubit unitary block with real and complex implementations.
    This block can be implemented in 3 ways, as mentioned by Grant et al. (2018)
        - simple, general and auxiliary gate implementations.

    These implementations are called alternative parameterizations.
    """

    @abstractmethod
    def simple_parameterization(
        self,
        circuit: QuantumCircuit,
        qubits: list,
        parameter_vector: ParameterVector,
        complex_structure: bool = False,
    ):
        return NotImplementedError

    @abstractmethod
    def general_parameterization(
        self,
        circuit: QuantumCircuit,
        qubits: list,
        parameter_vector: ParameterVector,
        complex_structure: bool = True,
    ):
        return NotImplementedError

    @abstractmethod
    def auxiliary_parameterization(
        self,
        circuit: QuantumCircuit,
        qubits: list,
        parameter_vector: ParameterVector,
        complex_structure: bool = True,
    ):
        return NotImplementedError
