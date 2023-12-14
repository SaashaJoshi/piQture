"""Tree Tensor Network (TTN)"""
from __future__ import annotations
from typing import Callable
import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from quantum_image_processing.gates.two_qubit_unitary import TwoQubitUnitary
from quantum_image_processing.models.tensor_network_circuits.base_tensor_network import (
    BaseTensorNetwork,
)


class TTN(BaseTensorNetwork):
    """
    Implements a Tree Tensor Network (TTN) structure class with
    alternative unitary qubit parameterization.

    References:
        [1] E. Grant et al., “Hierarchical quantum classifiers,”
        npj Quantum Information, vol. 4, no. 1, Dec. 2018,
        doi: https://doi.org/10.1038/s41534-018-0116-9.
    """

    def __init__(self, num_qubits: int):
        """
        Initializes the TTN class with given input variables.

        Args:
            num_qubits (int): number of qubits.
        """
        BaseTensorNetwork.__init__(self, num_qubits)

    def __repr__(self):
        """TTN class representation"""
        return f"TreeTensorNetwork(num_qubits={self.num_qubits})"

    def ttn_simple(self, complex_structure: bool = True) -> QuantumCircuit:
        """
        Implements a TTN network with simple alternative
        parameterization as given in [1].

        Args:
            complex_structure (default=True): boolean marker
            for real or complex gate parameterization.

        Returns:
            QuantumCircuit: quantum circuit with unitary gates
            represented by simple parameterization.
        """
        param_vector = ParameterVector("theta", 2 * self.num_qubits - 1)
        param_vector_copy = param_vector
        return self.ttn_backbone(
            TwoQubitUnitary().simple_parameterization,
            param_vector_copy,
            complex_structure,
        )

    def ttn_general(self, complex_structure: bool = True) -> QuantumCircuit:
        """
        Implements a TTN network with general alternative
        parameterization as given in [1].

        Args:
            complex_structure (default=True): boolean marker
            for real or complex gate parameterization.

        Returns:
            QuantumCircuit: quantum circuit with unitary gates
            represented by general parameterization.
        """
        # Check number of params here.
        if complex_structure:
            param_vector = ParameterVector("theta", 15 * self.num_qubits - 1)
            param_vector_copy = param_vector
        else:
            param_vector = ParameterVector("theta", 6 * self.num_qubits - 1)
            param_vector_copy = param_vector

        return self.ttn_backbone(
            TwoQubitUnitary().general_parameterization,
            param_vector_copy,
            complex_structure,
        )

    def ttn_with_aux(self, complex_structure: bool = True):
        """
        Implements a TTN network with alternative parameterization
        that requires an auxiliary qubit, as given in [1].

        Args:
            complex_structure (default=True): boolean marker for
            real or complex gate parameterization.

        Returns:
            QuantumCircuit: quantum circuit with unitary gates
            represented by general parameterization.
        """

    def ttn_backbone(
        self,
        gate_structure: Callable,
        param_vector_copy: ParameterVector,
        complex_structure: bool = True,
    ) -> QuantumCircuit:
        """
        Lays out the TTN structure by progressively building layers
        of unitary gates with their alternative parameterization.

        Args:
            gate_structure (Callable): a callable function that implements
            either one of the three available unitary gate parameterization -
            simple, general or auxiliary.

            param_vector_copy (ParameterVector): copy of the parameter
            vector list.

            complex_structure (default=True): boolean marker for
            real or complex gate parameterization.

        Returns:
            QuantumCircuit: quantum circuit with unitary gates
            represented by general parameterization.
        """
        # Layer = 0
        qubit_list = []
        for index in range(0, self.num_qubits, 2):
            if index == self.num_qubits - 1:
                qubit_list.append(self.q_reg[index])
            else:
                qubit_list.append(self.q_reg[index + 1])
                unitary_block, param_vector_copy = gate_structure(
                    parameter_vector=param_vector_copy,
                    complex_structure=complex_structure,
                )
                self.circuit.compose(
                    unitary_block,
                    qubits=[self.q_reg[index], self.q_reg[index + 1]],
                    inplace=True,
                )

        # Rest of the layers.
        for _ in range(int(np.sqrt(self.num_qubits))):
            temp_list = []
            for index in range(0, len(qubit_list) - 1, 2):
                unitary_block, param_vector_copy = gate_structure(
                    parameter_vector=param_vector_copy,
                    complex_structure=complex_structure,
                )
                self.circuit.compose(
                    unitary_block,
                    qubits=[qubit_list[index], qubit_list[index + 1]],
                    inplace=True,
                )
                temp_list.append(qubit_list[index + 1])

            if len(qubit_list) % 2:
                temp_list.append(qubit_list[-1])

            qubit_list = temp_list

        self.circuit.ry(param_vector_copy[0], qubit_list[-1])

        return self.circuit
