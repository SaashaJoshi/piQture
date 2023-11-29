"""Unitary Gate class"""
from __future__ import annotations
from abc import ABC, abstractmethod
from qiskit.circuit import QuantumCircuit, ParameterVector


class UnitaryBlock(ABC):
    """
    Implements a unitary block with real and complex implementations.
    This block can be implemented in 3 ways, as mentioned by Grant et al. (2018)
        - simple, general and auxiliary gate implementations.

    These implementations are called alternative parameterizations.
    """

    @abstractmethod
    def simple_parameterization(self, *args, **kwargs):
        """
        Used to build a unitary gate with real or complex
        simple parameterization.
        """

    @abstractmethod
    def general_parameterization(self, *args, **kwargs):
        """
        Used to build a unitary gate with real or complex
        general parameterization.
        """

    @abstractmethod
    def auxiliary_parameterization(self, *args, **kwargs):
        """
        Used to build a unitary gate parameterization
        with the help of an auxiliary qubit.
        """
