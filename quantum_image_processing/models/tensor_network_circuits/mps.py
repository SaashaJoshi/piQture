"""Matrix Product State Tensor Network (MPS)"""
from __future__ import annotations
from typing import Callable
from qiskit.circuit import QuantumCircuit, ParameterVector
from quantum_image_processing.models.tensor_network_circuits.base_tensor_network import (
    BaseTensorNetwork,
)
from quantum_image_processing.gates.two_qubit_unitary import TwoQubitUnitary


class MPS(BaseTensorNetwork):
    """
    Implements a Matrix Product State (MPS) tensor network
    structure class.

    References:
        [1] H.-M. Rieser, F. Köster, and A. P. Raulf, “Tensor
        networks for quantum machine learning,” Proceedings of
        the Royal Society A: Mathematical, Physical and Engineering
        Sciences, vol. 479, no. 2275, p. 20230218, Jul. 2023,
        doi: https://doi.org/10.1098/rspa.2023.0218.

        [2] D. Guala, S. Zhang, E. Cruz, C. A. Riofrío, J. Klepsch,
        and J. M. Arrazola, “Practical overview of image classification
        with tensor-network quantum circuits,” Scientific Reports,
        vol. 13, no. 1, p. 4427, Mar. 2023,
        doi: https://doi.org/10.1038/s41598-023-30258-y.
    """

    def __init__(self, num_qubits: int):
        """
        Initializes the MPS class with given input variables.

        Args:
            num_qubits (int): number of qubits.
        """
        BaseTensorNetwork.__init__(self, num_qubits)

    def __repr__(self):
        """MPS class representation"""
        return f"MatrixProductState(num_qubits={self.num_qubits})"

    def mps_simple(self, complex_structure: bool = True) -> QuantumCircuit:
        """
        Implements an MPS network, as given in [1], with simple
        alternative parameterization.

        Args:
            complex_structure (default=True): boolean marker
            for real or complex gate parameterization.

        Returns:
            QuantumCircuit: quantum circuit with unitary gates
            represented by simple parameterization.
        """
        param_vector = ParameterVector("theta", 2 * self.num_qubits - 2)
        param_vector_copy = param_vector
        return self.mps_backbone(
            TwoQubitUnitary().simple_parameterization,
            param_vector_copy,
            complex_structure,
        )

    def mps_general(self, complex_structure: bool = True) -> QuantumCircuit:
        """
        Implements an MPS network, as given in [1], with general
        alternative parameterization.

        Args:
            complex_structure (default=True): boolean marker
            for real or complex gate parameterization.

        Returns:
            QuantumCircuit: quantum circuit with unitary gates
            represented by general parameterization.
        """
        # Check number of params here.
        if complex_structure:
            param_vector = ParameterVector("theta", 15 * self.num_qubits - 2)
            param_vector_copy = param_vector
        else:
            param_vector = ParameterVector("theta", 6 * self.num_qubits - 2)
            param_vector_copy = param_vector

        return self.mps_backbone(
            TwoQubitUnitary().general_parameterization,
            param_vector_copy,
            complex_structure,
        )

    def mps_with_aux(self, complex_structure: bool = True):
        """
        Implements an MPS network, as given in [1], with
        alternative parameterization that requires an auxiliary qubit.

        Args:
            complex_structure (default=True): boolean marker for
            real or complex gate parameterization.

        Returns:
            QuantumCircuit: quantum circuit with unitary gates
            represented by general parameterization.
        """

    def mps_backbone(
        self,
        gate_structure: Callable,
        param_vector_copy: ParameterVector,
        complex_structure: bool = True,
    ) -> QuantumCircuit:
        """
        Lays out the MPS structure by progressively building layers
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
        for index in range(0, self.num_qubits - 1):
            unitary_block, param_vector_copy = gate_structure(
                param_vector_copy, complex_structure
            )
            self.circuit.compose(unitary_block, qubits=[index, index + 1], inplace=True)

        return self.circuit
