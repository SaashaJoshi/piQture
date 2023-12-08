"""Multiscale Entanglement Renormalization Ansatz (MERA) Tensor Network"""
from __future__ import annotations
import uuid
from typing import Callable, Optional
import math
import numpy as np
from qiskit.circuit import (
    QuantumCircuit,
    ParameterVector,
)
from quantum_image_processing.gates.two_qubit_unitary import TwoQubitUnitary
from quantum_image_processing.models.tensor_network_circuits.ttn import TTN


class MERA(TTN):
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

    def __init__(
        self, img_dims: tuple[int, int], layer_depth: Optional[int] | None = None
    ):
        """
        Initializes the MERA class with given input variables.

        Args:
            img_dims (int): dimensions of the input image data.
            layer_depth (int): number of MERA layers to be built in a circuit.
        """
        TTN.__init__(self, img_dims)
        self.num_qubits = int(math.prod(img_dims))

        if not isinstance(layer_depth, int) and layer_depth is not None:
            raise TypeError("The input layer_depth must be of the type int or None.")

        # Should there be a max cap on the value of layer_depth?
        if isinstance(layer_depth, int) and layer_depth < 1:
            raise ValueError("The input layer_depth must be at least 1.")

        if layer_depth is None:
            self.layer_depth = int(np.ceil(np.sqrt(self.num_qubits)))
        else:
            self.layer_depth = layer_depth

        self._circuit = QuantumCircuit(self.num_qubits)
        self.mera_qr = self._circuit.qubits

    @property
    def circuit(self):
        """Returns the MERA circuit."""
        return self._circuit

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
                qubits=[self.mera_qr[index], self.mera_qr[index + 1]],
                inplace=True,
            )

        # U unitary blocks
        self.circuit.barrier()
        for index in range(0, self.num_qubits, 2):
            if index == self.num_qubits - 1:
                qubit_list.append(self.mera_qr[index])
            else:
                qubit_list.append(self.mera_qr[index + 1])
                unitary_block, param_vector_copy = gate_structure(
                    parameter_vector=param_vector_copy,
                    complex_structure=complex_structure,
                )
                self.circuit.compose(
                    unitary_block,
                    qubits=[self.mera_qr[index], self.mera_qr[index + 1]],
                    inplace=True,
                )

        # Rest of the layers.
        if self.layer_depth > 1:
            temp_list = []
            # D unitary blocks
            self.circuit.barrier()
            for index in range(1, len(qubit_list), 2):
                if len(qubit_list) == 2 or index == len(qubit_list) - 1:
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
            self.circuit.barrier()
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

            if len(qubit_list) % 2 != 0:
                temp_list.append(qubit_list[-1])
            qubit_list = temp_list

            if len(qubit_list) == 1:
                self.circuit.ry(param_vector_copy[0], self.mera_qr[-1])

        return self.circuit
