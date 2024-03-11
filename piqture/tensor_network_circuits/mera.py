# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Multiscale Entanglement Renormalization Ansatz (MERA) Tensor Network"""

from __future__ import annotations
import uuid
from typing import Callable, Optional
import numpy as np
from qiskit.circuit import (
    QuantumCircuit,
    ParameterVector,
)
from piqture.gates.two_qubit_unitary import TwoQubitUnitary
from piqture.tensor_network_circuits.base_tensor_network import (
    BaseTensorNetwork,
)


class MERA(BaseTensorNetwork):
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

    def __init__(self, num_qubits: int, layer_depth: Optional[int] = None):
        """
        Initializes the MERA class with given input variables.

        Args:
            num_qubits (int): number of qubits.
            layer_depth (int): number of MERA layers to be built in a circuit.
        """
        BaseTensorNetwork.__init__(self, num_qubits)

        if not isinstance(layer_depth, int) and layer_depth is not None:
            raise TypeError("The input layer_depth must be of the type int or None.")

        # Should there be a max cap on the value of layer_depth?
        if isinstance(layer_depth, int) and layer_depth < 1:
            raise ValueError("The input layer_depth must be at least 1.")

        if layer_depth is None:
            self.layer_depth = int(np.ceil(np.sqrt(self.num_qubits)))
        else:
            self.layer_depth = layer_depth

    def __repr__(self):
        """MERA class representation"""
        return (
            f"MultiScaleEntanglementRenormalizationAnsatz("
            f"num_qubits={self.num_qubits}, layer_depth={self.layer_depth}"
            f")"
        )

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
        # Check params here.
        param_vector = ParameterVector(
            f"theta_{str(uuid.uuid4())[:5]}",
            10 * self.num_qubits - 1,
        )
        param_vector_copy = param_vector
        return self.mera_backbone(
            TwoQubitUnitary().simple_parameterization,
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
            TwoQubitUnitary().general_parameterization,
            param_vector_copy,
            complex_structure,
        )

    def mera_backbone(
        self,
        gate_structure: Callable,
        param_vector_copy: ParameterVector,
        complex_structure: bool = True,
    ) -> QuantumCircuit:
        # pylint: disable=duplicate-code
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
        qubit_list = []
        # Layer = 0
        for index in range(1, self.num_qubits, 2):
            if index == self.num_qubits - 1:
                break

            # D unitary blocks
            unitary_block, param_vector_copy = gate_structure(
                parameter_vector=param_vector_copy,
                complex_structure=complex_structure,
            )
            self.circuit.compose(
                unitary_block,
                qubits=[self.q_reg[index], self.q_reg[index + 1]],
                inplace=True,
            )

        # U unitary blocks
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
        temp_list = qubit_list.copy()
        if self.layer_depth > 1:
            while len(temp_list) > 1:
                # D unitary blocks
                for index in range(1, len(qubit_list), 2):
                    if len(temp_list) == 2 or index == len(temp_list) - 1:
                        break

                    unitary_block, param_vector_copy = gate_structure(
                        parameter_vector=param_vector_copy,
                        complex_structure=complex_structure,
                    )
                    self.circuit.compose(
                        unitary_block,
                        qubits=[qubit_list[index], qubit_list[index + 1]],
                        inplace=True,
                    )

                # U unitary blocks
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
                    temp_list.pop(0)
                qubit_list = temp_list

        if len(qubit_list) == 1:
            self.circuit.ry(param_vector_copy[0], self.q_reg[-1])

        return self.circuit
