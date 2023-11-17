"""Tree Tensor Network (TTN)"""
from __future__ import annotations
from typing import Callable
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterVector
from quantum_image_processing.gates.two_qubit_unitary import TwoQubitUnitary


class TTN:
    """
    Implements a Tree Tensor Network (TTN) structure class with
    alternative unitary qubit parameterization.

    References:
        [1] E. Grant et al., “Hierarchical quantum classifiers,”
        npj Quantum Information, vol. 4, no. 1, Dec. 2018,
        doi: https://doi.org/10.1038/s41534-018-0116-9.
    """

    def __init__(self, img_dim: int):
        """
        Initializes the TTN class with given input variables.

        Args:
            img_dim (int): product of dimensions of the input data.
            For example,
                for a 2x2 image, img_dim = 4.
        """
        self.img_dim = img_dim

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
        param_vector = ParameterVector("theta", 2 * self.img_dim - 1)
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
            param_vector = ParameterVector("theta", 15 * self.img_dim - 1)
            param_vector_copy = param_vector
        else:
            param_vector = ParameterVector("theta", 6 * self.img_dim - 1)
            param_vector_copy = param_vector

        return self.ttn_backbone(
            TwoQubitUnitary().simple_parameterization,
            param_vector_copy,
            complex_structure,
        )

    def ttn_with_aux(self, complex_structure: bool = True):
        """
        TODO: Find the implementation procedure for this.
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
        ttn_qr = QuantumRegister(size=self.img_dim)
        ttn_circ = QuantumCircuit(ttn_qr)

        qubit_list = []
        for index in range(0, self.img_dim, 2):
            if index == self.img_dim - 1:
                qubit_list.append(ttn_qr[index])
            else:
                qubit_list.append(ttn_qr[index + 1])
                _, param_vector_copy = gate_structure(
                    circuit=ttn_circ,
                    qubits=[ttn_qr[index], ttn_qr[index + 1]],
                    parameter_vector=param_vector_copy,
                    complex_structure=complex_structure,
                )

        for _ in range(int(np.sqrt(self.img_dim))):
            temp_list = []
            for index in range(0, len(qubit_list) - 1, 2):
                _, param_vector_copy = gate_structure(
                    ttn_circ,
                    [qubit_list[index], qubit_list[index + 1]],
                    param_vector_copy,
                    complex_structure,
                )
                temp_list.append(qubit_list[index + 1])

            if len(qubit_list) % 2 != 0:
                temp_list.append(qubit_list[-1])

            qubit_list = temp_list

        ttn_circ.ry(param_vector_copy[0], qubit_list[-1])

        return ttn_circ
