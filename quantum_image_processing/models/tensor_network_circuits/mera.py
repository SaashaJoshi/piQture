"""Multiscale Entanglement Renormalization Ansatz (MERA) Tensor Network"""
from __future__ import annotations
import uuid
from typing import Callable
import math
import numpy as np
from qiskit.circuit import (
    QuantumCircuit,
    QuantumRegister,
    ClassicalRegister,
    ParameterVector,
)
from quantum_image_processing.gates.two_qubit_unitary import TwoQubitUnitary


class MERA(TwoQubitUnitary):
    """
    Implements a Multiscale Entanglement Renormalization Ansatz
    (MERA) tensor network structure as given by [2].

    References:
        [1] E. Grant et al., “Hierarchical quantum classifiers,”
        npj Quantum Information, vol. 4, no. 1, Dec. 2018,
        doi: https://doi.org/10.1038/s41534-018-0116-9.

        [2] G. Vidal, “Class of Quantum Many-Body States That
        Can Be Efficiently Simulated,” Physical Review Letters,
        vol. 101, no. 11, Sep. 2008,
        doi: https://doi.org/10.1103/physrevlett.101.110501.
    """

    def __init__(self, img_dims: tuple[int, int], layer_depth: type(None) = None):
        self.img_dims = img_dims
        self.num_qubits = int(math.prod(img_dims))
        if layer_depth is None:
            self.layer_depth = int(np.ceil(np.sqrt(self.num_qubits)))
        else:
            self.layer_depth = layer_depth

    def mera_simple(self, complex_structure: bool = True) -> QuantumCircuit:
        """
        Builds a MERA circuit with a simple unitary gate
        parameterization.

        Args:
            complex_structure (bool): If True, builds the MERA
            structure with complex unitary gates (e.g. RY, etc.)

        Returns:
            circuit (QuantumCircuit): Returns the MERA circuit
            generated with the help of the input arguments.
        """
        param_vector = ParameterVector(
            f"theta_{str(uuid.uuid4())[:5]}",
            int(self.num_qubits / 2 * (self.num_qubits / 2 + 1)) + 3,
        )
        param_vector_copy = param_vector
        return self.mera_backbone(
            self.simple_parameterization,
            param_vector_copy,
            complex_structure,
        )

    # Check number of params here.
    def mera_general(self, complex_structure: bool = True) -> QuantumCircuit:
        """
        Builds a MERA circuit with a general unitary gate
        parameterization. Refer [1].

        Args:
            complex_structure (bool): If True, builds the MERA
            structure with complex unitary gates (e.g. RY, etc.)

        Returns:
            circuit (QuantumCircuit): Returns the MERA circuit
            generated with the help of the input arguments.
        """
        if complex_structure:
            param_vector = ParameterVector(
                f"theta_{str(uuid.uuid4())[:5]}", 20 * self.num_qubits - 1
            )
            param_vector_copy = param_vector
        else:
            param_vector = ParameterVector(
                f"theta_{str(uuid.uuid4())[:5]}", 10 * self.num_qubits - 1
            )
            param_vector_copy = param_vector
        return self.mera_backbone(
            self.general_parameterization,
            param_vector_copy,
            complex_structure,
        )

    # pylint: disable=too-many-branches
    def mera_backbone(
        self,
        gate_structure: Callable,
        param_vector_copy: ParameterVector,
        complex_structure: bool = True,
    ) -> QuantumCircuit:
        """
        Lays out the backbone structure of a MERA circuit onto
        which the unitary gates are applied.

        Args:
            gate_structure (Callable): calls the function with
            the required unitary gate parameterization structure.

            param_vector_copy (ParameterVector): list of unitary
            gate parameters to be used in the circuit.

            complex_structure (bool): If True, builds the MERA
            structure with complex unitary gates (e.g. RY, etc.)

        Returns:
            circuit (QuantumCircuit): Returns the MERA circuit
            generated with the help of the input arguments.
        """
        mera_qr = QuantumRegister(size=self.num_qubits)
        mera_cr = ClassicalRegister(size=self.num_qubits)
        mera_circ = QuantumCircuit(mera_qr, mera_cr)

        # Make recursive layer structure using a staticmethod i.e. convert code to staticmethod.
        # This should solve R0912: Too many branches (13/12) (too-many-branches)
        qubit_list = []
        for layer in range(self.layer_depth):
            if layer == 0:
                # D unitary blocks
                for index in range(1, self.num_qubits, 2):
                    if index == self.num_qubits - 1:
                        break

                    _, param_vector_copy = gate_structure(
                        circuit=mera_circ,
                        qubits=[mera_qr[index], mera_qr[index + 1]],
                        parameter_vector=param_vector_copy,
                        complex_structure=complex_structure,
                    )

                # U unitary blocks
                mera_circ.barrier()
                for index in range(0, self.num_qubits, 2):
                    if index == self.num_qubits - 1:
                        qubit_list.append(mera_qr[index])
                    else:
                        qubit_list.append(mera_qr[index + 1])
                        _, param_vector_copy = gate_structure(
                            circuit=mera_circ,
                            qubits=[mera_qr[index], mera_qr[index + 1]],
                            parameter_vector=param_vector_copy,
                            complex_structure=complex_structure,
                        )

            else:
                temp_list = []
                # D unitary blocks
                mera_circ.barrier()
                for index in range(1, len(qubit_list), 2):
                    if len(qubit_list) == 2 or index == len(qubit_list) - 1:
                        break

                    _, param_vector_copy = gate_structure(
                        circuit=mera_circ,
                        qubits=[qubit_list[index], qubit_list[index + 1]],
                        parameter_vector=param_vector_copy,
                        complex_structure=complex_structure,
                    )

                # U unitary blocks
                mera_circ.barrier()
                for index in range(0, len(qubit_list) - 1, 2):
                    _, param_vector_copy = gate_structure(
                        circuit=mera_circ,
                        qubits=[qubit_list[index], qubit_list[index + 1]],
                        parameter_vector=param_vector_copy,
                        complex_structure=complex_structure,
                    )
                    temp_list.append(qubit_list[index + 1])

                if len(qubit_list) % 2 != 0:
                    temp_list.append(qubit_list[-1])
                qubit_list = temp_list

                if len(qubit_list) == 1:
                    mera_circ.ry(param_vector_copy[0], mera_qr[-1])

        return mera_circ
