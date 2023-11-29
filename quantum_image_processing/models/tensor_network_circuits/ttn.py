"""Tree Tensor Network (TTN)"""
from __future__ import annotations
from typing import Callable
import math
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterVector
from quantum_image_processing.gates.two_qubit_unitary import TwoQubitUnitary


class TTN(TwoQubitUnitary):
    """
    Implements a Tree Tensor Network (TTN) structure class with
    alternative unitary qubit parameterization.

    References:
        [1] E. Grant et al., “Hierarchical quantum classifiers,”
        npj Quantum Information, vol. 4, no. 1, Dec. 2018,
        doi: https://doi.org/10.1038/s41534-018-0116-9.
    """

    def __init__(self, img_dims: tuple[int, int]):
        """
        Initializes the TTN class with given input variables.

        Args:
            img_dims (int): dimensions of the input image data.
        """
        self.img_dims = img_dims

        if not all(isinstance(dim, int) for dim in img_dims):
            raise TypeError("Input img_dims must of the type tuple(int, int).")

        if math.prod(img_dims) <= 0:
            raise ValueError("Image dimensions cannot be zero or negative.")

        self.num_qubits = int(math.prod(img_dims))

        self.q_reg = QuantumRegister(self.num_qubits)
        self._circuit = QuantumCircuit(self.q_reg)

    @property
    def circuit(self):
        """Returns the TTN circuit."""
        return self._circuit

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
            self.simple_parameterization,
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
            self.simple_parameterization,
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
        qubit_list = []
        for index in range(0, self.num_qubits, 2):
            if index == self.num_qubits - 1:
                qubit_list.append(self.q_reg[index])
            else:
                qubit_list.append(self.q_reg[index + 1])
                _, param_vector_copy = gate_structure(
                    circuit=self.circuit,
                    qubits=[self.q_reg[index], self.q_reg[index + 1]],
                    parameter_vector=param_vector_copy,
                    complex_structure=complex_structure,
                )

        for _ in range(int(np.sqrt(self.num_qubits))):
            temp_list = []
            for index in range(0, len(qubit_list) - 1, 2):
                _, param_vector_copy = gate_structure(
                    self.circuit,
                    [qubit_list[index], qubit_list[index + 1]],
                    param_vector_copy,
                    complex_structure,
                )
                temp_list.append(qubit_list[index + 1])

            if len(qubit_list) % 2 != 0:
                temp_list.append(qubit_list[-1])

            qubit_list = temp_list

        self.circuit.ry(param_vector_copy[0], qubit_list[-1])

        return self.circuit
