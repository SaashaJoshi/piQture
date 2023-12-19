"""Abstract Base Class for QNN Layers"""

from __future__ import annotations
from typing import Optional
from abc import ABC, abstractmethod
from qiskit.circuit import QuantumCircuit


# pylint: disable=too-few-public-methods


class BaseLayer(ABC):
    """
    Abstract base class for layer structure in a quantum
    neural network.
    """

    def __init__(
        self,
        num_qubits: Optional[int] = None,
        circuit: QuantumCircuit = None,
        unmeasured_bits: Optional[list] = None,
    ):
        """
        Initializes a Layer circuit with the given number
        of qubits.

        Args:
            num_qubits (int): number of qubits on which
            the layer is applied.

            unmeasured_bits (list): list of unmeasured qubits on
            which the layer is built.
        """
        # None of the inputs are given (NNN).
        if num_qubits is None and circuit is None and unmeasured_bits is None:
            raise ValueError(
                "At least one of the inputs, num_qubits, circuit, "
                "or unmeasured_bits, must be provided."
            )

        # If num_qubits is given.
        if num_qubits is not None:
            if not isinstance(num_qubits, int):
                raise TypeError("The input num_qubits must be of the type int.")

            if num_qubits <= 0:
                raise ValueError("The input num_qubits must be greater than zero.")
            self._num_qubits = num_qubits

            if circuit is None:
                self._circuit = QuantumCircuit(self._num_qubits, self._num_qubits)

        if circuit is not None:
            if not isinstance(circuit, QuantumCircuit):
                raise TypeError("The input circuit must be of the type QuantumCircuit.")
            self._circuit = circuit

        if unmeasured_bits is not None:
            self._validate_unmeasured_bits(unmeasured_bits)
            self._unmeasured_bits = unmeasured_bits

            if circuit is None:
                self._circuit = QuantumCircuit(
                    max(self._unmeasured_bits) + 1, max(self._unmeasured_bits) + 1
                )
            if num_qubits is None:
                self._num_qubits = len(self._circuit.qubits)
        else:
            self._unmeasured_bits = list(range(len(self._circuit.qubits)))

        if num_qubits is None:
            self._num_qubits = len(self._circuit.qubits)

    @property
    def circuit(self):
        """Returns a base_layer circuit."""
        return self._circuit

    @property
    def num_qubits(self):
        """Returns number of qubits in base_layer circuit."""
        return self._num_qubits

    @property
    def unmeasured_bits(self):
        """Returns number of unmeasured bits in base_layer circuit."""
        return self._unmeasured_bits

    @staticmethod
    def _validate_unmeasured_bits(unmeasured_bits: list):
        """Validates the input unmeasured_bits and index values in the list."""
        if not isinstance(unmeasured_bits, list):
            raise TypeError("The input qubits must be of the type list.")

        if not all(isinstance(index, int) for index in unmeasured_bits):
            raise TypeError(
                "Indices inside the unmeasured_bits list must be of the type int."
            )

    # @staticmethod
    # def _check_num_qubits(
    #     num_qubits: int,
    #     unmeasured_bits: list,
    # ):
    #     """Checks if inputs are of equal length."""
    #     if num_qubits != len(unmeasured_bits):
    #         raise ValueError(
    #             "The input num_qubits must be equal to the length of unmeasured_bits list."
    #         )

    @abstractmethod
    def build_layer(self):
        """
        Helps build the layer circuit.
        """
        return NotImplementedError
