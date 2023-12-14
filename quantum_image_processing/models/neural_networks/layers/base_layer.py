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
        if not num_qubits and circuit is None and unmeasured_bits is None:
            raise ValueError(
                "At least one of the inputs, num_qubits, circuit, or unmeasured_bits, must be provided."
            )

        # If num_qubits is given.
        if num_qubits:
            if not isinstance(num_qubits, int):
                raise TypeError("The input num_qubits must be of the type int.")
            self.num_qubits = num_qubits

            # YYN
            if circuit is not None:
                if not isinstance(circuit, QuantumCircuit):
                    raise TypeError(
                        "The input circuit must be of the type QuantumCircuit."
                    )
                # if circuit.qubits != num_qubits:
                #     self._check_num_qubits(self.num_qubits, circuit=circuit)
                self._circuit = circuit
                # self.q_reg = self._circuit.qubits
            else:
                # YNN
                self._circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
                # self.q_reg = self._circuit.qubits

            # YNY
            if unmeasured_bits is not None:
                self._validate_unmeasured_bits(unmeasured_bits)
                if num_qubits != len(unmeasured_bits):
                    self._check_num_qubits(
                        self.num_qubits, unmeasured_bits=unmeasured_bits
                    )
                self.unmeasured_bits = unmeasured_bits
            else:
                # YNN
                self.unmeasured_bits = self.circuit.qubits

        # If num_qubits is not given.
        else:
            # NYN
            if circuit is not None:
                self._circuit = circuit
                # self.q_reg = self._circuit.qubits
                self.unmeasured_bits = self.circuit.qubits
                self.num_qubits = len(self.circuit.qubits)

            # NNY
            if unmeasured_bits is not None:
                self.unmeasured_bits = unmeasured_bits
                self.num_qubits = len(self.unmeasured_bits)
                self._circuit = QuantumCircuit(self.num_qubits, self.num_qubits)
                # self.q_reg = self._circuit.qubits

    @property
    def circuit(self):
        return self._circuit

    @staticmethod
    def _validate_unmeasured_bits(unmeasured_bits: list):
        if not isinstance(unmeasured_bits, list):
            raise TypeError("The input qubits must be of the type list.")

        if not all(isinstance(index, int) for index in unmeasured_bits):
            raise TypeError(
                "Indices inside the unmeasured_bits list must be of the type int."
            )

    @staticmethod
    def _check_num_qubits(
        num_qubits: int,
        unmeasured_bits: list,
    ):
        # if num_qubits != len(circuit.qubits):
        #     raise ValueError("The input num_qubits must be equal to the number of qubits in input circuit.")

        if num_qubits != len(unmeasured_bits):
            raise ValueError(
                "The input num_qubits must be equal to the length of unmeasured_bits list."
            )

    # @staticmethod
    # def _check_unmeasured_bits(
    #         unmeasured_bits,
    #         circuit: Optional[QuantumCircuit] = None,
    # ):
    #     # Check if indices in unmeasured_bits are equal to indices of unmeasured qubits in the circuit.
    #     # IDK if such a check can be performed easily.
    #     # if len(unmeasured_bits) != len(circuit.qubits):
    #     raise ValueError(
    #         "The length of unmeasured_bits list must be equal to the number of
    #         unmeasured qubits in the circuit input.")

    @abstractmethod
    def build_layer(self):
        """
        Helps build the layer circuit.
        """
        return NotImplementedError
