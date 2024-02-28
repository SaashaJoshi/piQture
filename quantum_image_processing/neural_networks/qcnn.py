# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum Convolutional Neural Network"""

from __future__ import annotations
from typing import Callable
from qiskit.circuit import QuantumCircuit
from quantum_image_processing.neural_networks.quantum_neural_network import (
    QuantumNeuralNetwork,
)

# pylint: disable=too-few-public-methods


class QCNN(QuantumNeuralNetwork):
    """
    Builds a Quantum Neural Network circuit with the help of
    convolutional, pooling or fully-connected layers.

    References:
        [1] I. Cong, S. Choi, and M. D. Lukin, “Quantum
        convolutional neural networks,” Nature Physics,
        vol. 15, no. 12, pp. 1273–1278, Aug. 2019,
        doi: https://doi.org/10.1038/s41567-019-0648-8.
    """

    def __init__(self, num_qubits: int):
        """
        Initializes a Quantum Neural Network circuit with the given
        number of qubits.

        Args:
            num_qubits (int): builds a quantum convolutional neural
            network circuit with the given number of qubits or image
            dimensions.
        """
        QuantumNeuralNetwork.__init__(self, num_qubits)

    def sequence(self, operations: list[tuple[Callable, dict]]) -> QuantumCircuit:
        """
        Builds a QNN circuit by composing the circuit with given
        sequence of list of operations.

        Args:
            operations (list[tuple[Callable, dict]]: a tuple
            of a Layer object and a dictionary of its arguments.

        Returns:
            circuit (QuantumCircuit): final QNN circuit with all the
            layers.
        """
        if not isinstance(operations, list):
            raise TypeError("The input operations must be of the type list.")

        if not all(isinstance(operation, tuple) for operation in operations):
            raise TypeError(
                "The input operations list must contain tuple[operation, params]."
            )

        if not callable(operations[0][0]):
            raise TypeError(
                "Operation in input operations list must be Callable functions/classes."
            )

        if not isinstance(operations[0][1], dict):
            raise TypeError(
                "Parameters of operation in input operations list must be in a dictionary."
            )

        unmeasured_bits = list(range(self.num_qubits))
        for layer, params in operations:
            layer_instance = layer(
                num_qubits=self.num_qubits,
                circuit=self.circuit,
                unmeasured_bits=unmeasured_bits,
                **params,
            )
            # Optionally collect circuit since it is
            # composed in place.
            _, unmeasured_bits = layer_instance.build_layer()

        return self.circuit
