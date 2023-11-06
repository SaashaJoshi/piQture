from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector
from quantum_image_processing.gates.unitary_block import Unitary


class TwoQubitUnitary(Unitary):
    """
    Implements two qubit unitary with alternative parameterizations.
    """

    def simple_parameterization(
        self,
        circuit: QuantumCircuit,
        qubits: list,
        parameter_vector: ParameterVector,
        complex_structure: bool = True,
    ) -> tuple[QuantumCircuit, ParameterVector]:
        if complex_structure:
            pass
        else:
            return self._real_simple_block(circuit, qubits, parameter_vector)

    def general_parameterization(
        self,
        circuit: QuantumCircuit,
        qubits: list,
        parameter_vector: ParameterVector,
        complex_structure: bool = True,
    ) -> tuple[QuantumCircuit, ParameterVector]:
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
        if complex_structure:
            pass
        else:
            pass

    @staticmethod
    def _real_simple_block(
        circuit: QuantumCircuit, qubits: list, parameter_vector: ParameterVector
    ) -> tuple[QuantumCircuit, ParameterVector]:
        block = circuit
        block.ry(parameter_vector[0], qubits[0])
        block.ry(parameter_vector[1], qubits[1])
        block.cx(qubits[0], qubits[1])

        parameter_vector = parameter_vector[2:]

        return block, parameter_vector

    @staticmethod
    def _real_general_block(
        circuit: QuantumCircuit, qubits: list, parameter_vector: ParameterVector
    ) -> tuple[QuantumCircuit, ParameterVector]:
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
