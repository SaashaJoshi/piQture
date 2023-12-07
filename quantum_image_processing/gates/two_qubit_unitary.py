"""Two-Qubit Unitary Gate class"""
from __future__ import annotations
import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter
from quantum_image_processing.gates.unitary_block import UnitaryBlock


class TwoQubitUnitary(UnitaryBlock):
    """
    Implements two qubit unitary with alternative parameterizations.
    """

    @staticmethod
    def _validate_arguments(
        parameter_vector: ParameterVector,
        complex_structure: bool = True,
    ):
        """Validates the inputs for two-qubit parameterizations."""
        # if not isinstance(parameter_vector, list):
        #     raise TypeError(
        #         "Input parameter_vector must be of the type ParameterVector."
        #     )

        if not all(isinstance(vector, Parameter) for vector in parameter_vector):
            raise TypeError(
                "Vectors in parameter_vectors must be of the type Parameter."
            )

        if not isinstance(complex_structure, bool):
            raise TypeError(
                "Input complex_structure must be either True or False (bool)."
            )

    def simple_parameterization(
        self,
        parameter_vector: ParameterVector,
        complex_structure: bool = True,
    ) -> tuple[QuantumCircuit, ParameterVector]:
        self._validate_arguments(
            parameter_vector,
            complex_structure,
        )
        if complex_structure:
            return self.complex_simple_block(parameter_vector)
        return self.real_simple_block(parameter_vector)

    def general_parameterization(
        self,
        parameter_vector: ParameterVector,
        complex_structure: bool = True,
    ) -> tuple[QuantumCircuit, ParameterVector]:
        self._validate_arguments(
            parameter_vector,
            complex_structure,
        )
        if complex_structure:
            return self.complex_general_block(parameter_vector)
        return self.real_general_block(parameter_vector)

    def auxiliary_parameterization(
        self,
        parameter_vector: ParameterVector,
        complex_structure: bool = True,
    ):
        """
        Used to build a unitary gate parameterization
        with the help of an auxiliary qubit.
        """

    @staticmethod
    def real_simple_block(
        parameter_vector: ParameterVector,
    ) -> tuple[QuantumCircuit, ParameterVector]:
        """
        Builds a two-qubit unitary gate with simple parameterization,
        consisting of real two single-unitary gates followed by a CNOT
        gate as given in the following paper.

        References:
            [1] E. Grant et al., “Hierarchical quantum classifiers,”
            npj Quantum Information, vol. 4, no. 1, Dec. 2018,
            doi: https://doi.org/10.1038/s41534-018-0116-9.

        Args:
            parameter_vector (ParameterVector): list of parameters
            of the unitary gates.
        """
        block = QuantumCircuit(2)
        block.ry(parameter_vector[0], 0)
        block.ry(parameter_vector[1], 1)
        block.cx(0, 1)
        parameter_vector = parameter_vector[2:]

        return block, parameter_vector

    @staticmethod
    def complex_simple_block(
        parameter_vector: ParameterVector,
    ) -> tuple[QuantumCircuit, ParameterVector]:
        """
        Placeholder for complex simple box.
        """
        # Currently, does nothing.

    @staticmethod
    def real_general_block(
        parameter_vector: ParameterVector,
    ) -> tuple[QuantumCircuit, ParameterVector]:
        """
        Builds a two-qubit unitary gate with a general parameterization,
        consisting of real gates only, as given in the following reference paper.

        Reference:
            [1] F. Vatan and C. Williams, “Optimal quantum circuits for
            general two-qubit gates,” Physical Review A, vol. 69, no. 3,
            Mar. 2004, doi: https://doi.org/10.1103/physreva.69.032315.

        Args:
            parameter_vector (ParameterVector): list of parameters
            of the unitary gates.
        """
        block = QuantumCircuit(2)
        block.rz(np.pi / 2, 0)
        block.rz(np.pi / 2, 1)
        block.ry(np.pi / 2, 1)
        block.cnot(1, 0)

        block.rz(parameter_vector[0], 0)
        block.ry(parameter_vector[1], 0)
        block.rz(parameter_vector[2], 0)

        block.rz(parameter_vector[3], 1)
        block.ry(parameter_vector[4], 1)
        block.rz(parameter_vector[5], 1)

        block.cnot(1, 0)
        block.ry(-np.pi / 2, 1)
        block.rz(-np.pi / 2, 0)
        block.rz(-np.pi / 2, 1)

        parameter_vector = parameter_vector[6:]

        return block, parameter_vector

    @staticmethod
    def complex_general_block(
        parameter_vector: ParameterVector,
    ) -> tuple[QuantumCircuit, ParameterVector]:
        """
        Builds a two-qubit unitary gate with a general parameterization,
        consisting of complex gates, as given in the following reference paper.

        Reference:
            [1] F. Vatan and C. Williams, “Optimal quantum circuits for
            general two-qubit gates,” Physical Review A, vol. 69, no. 3,
            Mar. 2004, doi: https://doi.org/10.1103/physreva.69.032315.

        Args:
            parameter_vector (ParameterVector): list of parameters
            of the unitary gates.
        """
        block = QuantumCircuit(2)
        block.rz(parameter_vector[0], 0)
        block.ry(parameter_vector[1], 0)
        block.rz(parameter_vector[2], 0)

        block.rz(parameter_vector[3], 1)
        block.ry(parameter_vector[4], 1)
        block.rz(parameter_vector[5] + np.pi / 2, 1)
        block.cnot(1, 0)

        block.rz((2 * parameter_vector[6]) - np.pi / 2, 0)
        block.ry(np.pi / 2 - (2 * parameter_vector[7]), 1)
        block.cnot(0, 1)
        block.ry((2 * parameter_vector[8]) - np.pi / 2, 1)

        block.cnot(1, 0)
        block.rz(parameter_vector[9], 1)
        block.ry(parameter_vector[10], 1)
        block.rz(parameter_vector[11], 1)

        block.rz(parameter_vector[12] - np.pi / 2, 0)
        block.ry(parameter_vector[13], 0)
        block.rz(parameter_vector[14], 0)

        parameter_vector = parameter_vector[15:]

        return block, parameter_vector
