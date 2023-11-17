"""Unitary Gate class"""
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
        """
        Used to build a two-qubit unitary gate with real or complex
        simple parameterization.

        Args:
            circuit (QuantumCircuit): Circuit on which two-qubit
            unitary gate needs to be applied.

            qubits (list): list of qubits on which the gates need to
            be applied.

            parameter_vector (ParameterVector): list of parameters
            of the unitary gates.

            complex_structure (bool): If True, builds the unitary gate
            parameterization with complex unitary gates (e.g. RY, etc.)
        """
        return NotImplementedError

    @abstractmethod
    def general_parameterization(
        self,
        circuit: QuantumCircuit,
        qubits: list,
        parameter_vector: ParameterVector,
        complex_structure: bool = True,
    ):
        """
        Used to build a two-qubit unitary gate with real or complex
        general parameterization.

        Args:
            circuit (QuantumCircuit): Circuit on which two-qubit
            unitary gate needs to be applied.

            qubits (list): list of qubits on which the gates need to
            be applied.

            parameter_vector (ParameterVector): list of parameters
            of the unitary gates.

            complex_structure (bool): If True, builds the unitary gate
            parameterization with complex unitary gates (e.g. RY, etc.)
        """
        return NotImplementedError

    @abstractmethod
    def auxiliary_parameterization(
        self,
        circuit: QuantumCircuit,
        qubits: list,
        parameter_vector: ParameterVector,
        complex_structure: bool = True,
    ):
        """
        Used to build a two-qubit unitary gate parameterization
        with the help of an auxiliary qubit.

        Args:
            circuit (QuantumCircuit): Circuit on which two-qubit
            unitary gate needs to be applied.

            qubits (list): list of qubits on which the gates need to
            be applied.

            parameter_vector (ParameterVector): list of parameters
            of the unitary gates.

            complex_structure (bool): If True, builds the unitary gate
            parameterization with complex unitary gates (e.g. RY, etc.)
        """
        return NotImplementedError
