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
        circuit: QuantumCircuit,
        qubits: list,
        parameter_vector: ParameterVector,
        complex_structure: bool = True,
    ):
        """Validates the inputs for two-qubit parameterizations."""
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("Input circuit is not of the type QuantumCircuit.")

        if not isinstance(qubits, list):
            raise TypeError("Input qubits must be of the type list.")

        if len(qubits) == 0:
            raise ValueError("Input qubits list cannot be empty.")

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
        circuit: QuantumCircuit,
        qubits: list,
        parameter_vector: ParameterVector,
        complex_structure: bool = True,
    ) -> tuple[QuantumCircuit, ParameterVector]:
        self._validate_arguments(
            circuit,
            qubits,
            parameter_vector,
            complex_structure,
        )
        if complex_structure:
            return self._complex_simple_block(circuit, qubits, parameter_vector)
        return self._real_simple_block(circuit, qubits, parameter_vector)

    def general_parameterization(
        self,
        circuit: QuantumCircuit,
        qubits: list,
        parameter_vector: ParameterVector,
        complex_structure: bool = True,
    ) -> tuple[QuantumCircuit, ParameterVector]:
        self._validate_arguments(
            circuit,
            qubits,
            parameter_vector,
            complex_structure,
        )
        if complex_structure:
            return self._complex_general_block(circuit, qubits, parameter_vector)
        return self._real_general_block(circuit, qubits, parameter_vector)

    def auxiliary_parameterization(
        self,
        circuit: QuantumCircuit,
        qubits: list,
        parameter_vector: ParameterVector,
        complex_structure: bool = True,
    ):
        self._validate_arguments(
            circuit,
            qubits,
            parameter_vector,
            complex_structure,
        )
        if complex_structure:
            pass
        else:
            pass

    @staticmethod
    def _real_simple_block(
        circuit: QuantumCircuit, qubits: list, parameter_vector: ParameterVector
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
            circuit (QuantumCircuit): Circuit on which two-qubit
            unitary gate needs to be applied.

            qubits (list): list of qubits on which the gates need to
            be applied.

            parameter_vector (ParameterVector): list of parameters
            of the unitary gates.
        """
        block = circuit
        block.ry(parameter_vector[0], qubits[0])
        block.ry(parameter_vector[1], qubits[1])
        block.cx(qubits[0], qubits[1])

        parameter_vector = parameter_vector[2:]

        return block, parameter_vector

    @staticmethod
    def _complex_simple_block(
        circuit: QuantumCircuit, qubits: list, parameter_vector: ParameterVector
    ) -> tuple[QuantumCircuit, ParameterVector]:
        """
        Placeholder for complex simple box.
        """
        # Currently, does nothing.

    @staticmethod
    def _real_general_block(
        circuit: QuantumCircuit, qubits: list, parameter_vector: ParameterVector
    ) -> tuple[QuantumCircuit, ParameterVector]:
        """
        Builds a two-qubit unitary gate with a general parameterization,
        consisting of real gates only, as given in the following reference paper.

        Reference:
            [1] F. Vatan and C. Williams, “Optimal quantum circuits for
            general two-qubit gates,” Physical Review A, vol. 69, no. 3,
            Mar. 2004, doi: https://doi.org/10.1103/physreva.69.032315.

        Args:
            circuit (QuantumCircuit): Circuit on which two-qubit
            unitary gate needs to be applied.

            qubits (list): list of qubits on which the gates need to
            be applied.

            parameter_vector (ParameterVector): list of parameters
            of the unitary gates.
        """
        block = circuit

        block.rz(np.pi / 2, qubits[0])
        block.rz(np.pi / 2, qubits[1])
        block.ry(np.pi / 2, qubits[1])
        block.cnot(qubits[1], qubits[0])

        block.rz(parameter_vector[0], qubits[0])
        block.ry(parameter_vector[1], qubits[0])
        block.rz(parameter_vector[2], qubits[0])

        block.rz(parameter_vector[3], qubits[1])
        block.ry(parameter_vector[4], qubits[1])
        block.rz(parameter_vector[5], qubits[1])

        block.cnot(qubits[1], qubits[0])
        block.ry(-np.pi / 2, qubits[1])
        block.rz(-np.pi / 2, qubits[0])
        block.rz(-np.pi / 2, qubits[1])

        parameter_vector = parameter_vector[6:]

        return block, parameter_vector

    @staticmethod
    def _complex_general_block(
        circuit: QuantumCircuit, qubits: list, parameter_vector: ParameterVector
    ) -> tuple[QuantumCircuit, ParameterVector]:
        """
        Builds a two-qubit unitary gate with a general parameterization,
        consisting of complex gates, as given in the following reference paper.

        Reference:
            [1] F. Vatan and C. Williams, “Optimal quantum circuits for
            general two-qubit gates,” Physical Review A, vol. 69, no. 3,
            Mar. 2004, doi: https://doi.org/10.1103/physreva.69.032315.

        Args:
            circuit (QuantumCircuit): Circuit on which two-qubit
            unitary gate needs to be applied.

            qubits (list): list of qubits on which the gates need to
            be applied.

            parameter_vector (ParameterVector): list of parameters
            of the unitary gates.
        """
        block = circuit
        block.rz(parameter_vector[0], qubits[0])
        block.ry(parameter_vector[1], qubits[0])
        block.rz(parameter_vector[2], qubits[0])

        block.rz(parameter_vector[3], qubits[1])
        block.ry(parameter_vector[4], qubits[1])
        block.rz(parameter_vector[5] + np.pi / 2, qubits[1])
        block.cnot(qubits[1], qubits[0])

        block.rz((2 * parameter_vector[6]) - np.pi / 2, qubits[0])
        block.ry(np.pi / 2 - (2 * parameter_vector[7]), qubits[1])
        block.cnot(qubits[0], qubits[1])
        block.ry((2 * parameter_vector[8]) - np.pi / 2, qubits[1])

        block.cnot(qubits[1], qubits[0])
        block.rz(parameter_vector[9], qubits[1])
        block.ry(parameter_vector[10], qubits[1])
        block.rz(parameter_vector[11], qubits[1])

        block.rz(parameter_vector[12] - np.pi / 2, qubits[0])
        block.ry(parameter_vector[13], qubits[0])
        block.rz(parameter_vector[14], qubits[0])

        parameter_vector = parameter_vector[15:]

        return block, parameter_vector
