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

from typing import Type
from inspect import isclass
from qiskit.circuit import QuantumCircuit
from piqture.neural_networks.quantum_neural_network import QuantumNeuralNetwork
from piqture.neural_networks.layers.base_layer import BaseLayer

class QCNN(QuantumNeuralNetwork):
    def __init__(self, num_qubits: int):
        """
        Initializes a Quantum Neural Network circuit with the given number of qubits.

        Args:
            num_qubits (int): builds a quantum convolutional neural network circuit 
                             with the given number of qubits or image dimensions.
        """
        QuantumNeuralNetwork.__init__(self, num_qubits)

    def sequence(self, operations: list[tuple[Type[BaseLayer], dict]]) -> QuantumCircuit:
        """
        Builds a QNN circuit by composing the circuit with given sequence of operations.

        Args:
            operations (list[tuple[Type[BaseLayer], dict]]): a tuple of a Layer class 
                      that inherits from BaseLayer and a dictionary of its arguments.

        Returns:
            circuit (QuantumCircuit): final QNN circuit with all the layers.

        Raises:
            TypeError: If operations format is invalid or if any operation doesn't
                      inherit from BaseLayer
            ValueError: If operations list is empty
        """
        # Validate operations list
        if not isinstance(operations, list):
            raise TypeError("The input operations must be of the type list.")

        if not operations:
            raise ValueError("The operations list cannot be empty.")

        if not all(isinstance(operation, tuple) for operation in operations):
            raise TypeError(
                "The input operations list must contain tuple[operation, params]."
            )

        # Validate each operation
        for idx, (layer, params) in enumerate(operations):
            # Check if it's a class and inherits from BaseLayer
            if not isclass(layer):
                raise TypeError(
                    f"Operation at index {idx} must be a class, got {type(layer).__name__}"
                )
            
            if not issubclass(layer, BaseLayer):
                raise TypeError(
                    f"Operation at index {idx} must inherit from BaseLayer, "
                    f"got {layer.__name__}"
                )

            # Prevent using BaseLayer itself
            if layer is BaseLayer:
                raise TypeError(
                    f"Operation at index {idx} cannot be BaseLayer itself, "
                    "must be a concrete implementation"
                )

            # Validate parameters
            if not isinstance(params, dict):
                raise TypeError(
                    f"Parameters at index {idx} must be in a dictionary, "
                    f"got {type(params).__name__}"
                )

        # Build the circuit
        unmeasured_bits = list(range(self.num_qubits))
        for layer, params in operations:
            layer_instance = layer(
                num_qubits=self.num_qubits,
                circuit=self.circuit,
                unmeasured_bits=unmeasured_bits,
                **params,
            )
            _, unmeasured_bits = layer_instance.build_layer()

        return self.circuit