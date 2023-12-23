"""Quanvolutional Neural Network structure"""
from __future__ import annotations
import math
from typing import Optional, Callable
from qiskit.circuit import QuantumCircuit
from quantum_image_processing.neural_networks.layers import QuanvolutionalLayer
from quantum_image_processing.neural_networks.quantum_neural_network import (
    QuantumNeuralNetwork,
)


class QuanvolutionalNeuralNetwork(QuantumNeuralNetwork):
    """
    Builds a Quanvolutional Neural Network structure [1].

    References:
        [1] M. Henderson, S. Shakya, S. Pradhan, and
        T. Cook, “Quanvolutional Neural Networks:
        Powering Image Recognition with Quantum Circuits,”
        arXiv:1904.04767 [quant-ph], Apr. 2019,
        Available: https://arxiv.org/abs/1904.04767
    """

    def __init__(self, num_qubits: int):
        """
        Initializes a Quanvolutional Neural Network
        circuit with the given number of qubits.

        Args:
            num_qubits (int): builds a quantum convolutional neural
            network circuit with the given number of qubits or image
            dimensions.
        """
        QuantumNeuralNetwork.__init__(self, num_qubits)

    def pre_quanvolutional_layer(self, params) -> list[QuantumCircuit]:
        """
        Helps in setting up the circuits for quanvolutional layers.
        """
        # Produce n sub_circuits.

        sub_circuits = []
        layer_instance = QuanvolutionalLayer(
            num_qubits=,
            circuit=,
            unmeasured_bits=unmeasured_bits,
            **params,
        )
        # Optionally collect circuit since it is
        # composed in place.
        sub_circuit, _ = layer_instance.build_layer()
        sub_circuits.append(sub_circuit)
        print(sub_circuits)

        return sub_circuits

    def post_quanvolutional_layer(self):
        pass

    def sequence(self, operations: list[tuple[Callable, dict]]):
        """
        Builds a QuNN circuit by composing the circuit with given
        sequence of list of operations, including a Quanvolutional Layer.

        This NN structure particularly differs from QCNN in terms of
        handling measurement results after the application of a
        Quanvolutional Layer and then re-embedding the data received
        from measurements to perform further quantum processes.

        Args:
            operations (list[tuple[Callable, dict]]: a tuple
            of a Layer object and a dictionary of its arguments.

        Returns:
            circuit (QuantumCircuit): final QNN circuit with all the
            layers.
        """
        super().sequence(operations)

        # Put a stop when QuanvLayer is followed by other quantum layers.
        # This situation will require data processing before embedding.
        unmeasured_bits = list(range(self.num_qubits))
        for layer, params in operations:
            # print(layer)
            if layer is QuanvolutionalLayer:
                print("Yes!", layer, params)

                self.pre_quanvolutional_layer(params)

            else:
                layer_instance = layer(
                    num_qubits=self.num_qubits,
                    circuit=self.circuit,
                    unmeasured_bits=unmeasured_bits,
                    **params,
                )
                # Optionally collect circuit since it is
                # composed in place.
                _, unmeasured_bits = layer_instance.build_layer()


        # Continue normally (similar to QCNN) when QuanvLayer is the
        # only layer or the last layer in operations.
        # unmeasured_bits = list(range(self.num_qubits))
        # for layer, params in operations:
        #     layer_instance = layer(
        #         num_qubits=self.num_qubits,
        #         circuit=self.circuit,
        #         unmeasured_bits=unmeasured_bits,
        #         **params,
        #     )
        #     # Optionally collect circuit since it is
        #     # composed in place.
        #     _, unmeasured_bits = layer_instance.build_layer()

        return self.circuit
