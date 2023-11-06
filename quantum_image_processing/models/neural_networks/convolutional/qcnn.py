"""Quantum Convolutional Neural Network"""
# QCNN has 4 types of layers: data encoding, convolutional layer (MERA),
# Pooling layer, FC layer, and measurement.

from __future__ import annotations
import numpy as np
from qiskit.circuit import QuantumCircuit, ClassicalRegister
from quantum_image_processing.models.tensor_network_circuits.mera import MERA


class QuantumConvolutionalLayer(MERA):
    """
    Builds a convolutional layer in the neural network
    with the help of the MERA tensor network.

    MERA and the convolutional layer in a QCNN run in the
    opposite directions. However, for the sake of simplicity,
    this repository builds both in the same direction.

    References:
        [1] I. Cong, S. Choi, and M. D. Lukin, “Quantum
        convolutional neural networks,” Nature Physics,
        vol. 15, no. 12, pp. 1273–1278, Aug. 2019,
        doi: https://doi.org/10.1038/s41567-019-0648-8.
    """

    def __init__(self, circuit, *args, **kwargs):
        """
        Initializes the convolutional layer to a MERA
        tensor network structure.

        Args:
            circuit (QuantumCircuit): Takes quantum circuit with an
            existing convolutional or pooling layer as an input,
            and applies an/additional convolutional layer over it.

            *args and **kwargs (like img_dims and layer_depth): inputs
            as inherited from parent class, MERA.
        """
        MERA.__init__(self, *args, **kwargs)
        self.circuit = circuit

    def convolutional_layer(
        self, mera_instance: int, complex_structure: bool
    ) -> QuantumCircuit:
        """
        Implements the MERA tensor network with a restriction
        on the depth of a convolutional layer, specified by a
        hyperparameter, `layer_depth`.

        Args:
            mera_instance (int): integer to denote a structural choice
            for unitary gate parameterization.
            For example, [0, 1, 2] == [real, general, aux]

            complex_structure (bool)(default=True): boolean marker
            for real or complex gate parameterization.
        """
        instance_mapping = {
            0: self.mera_simple,
            1: self.mera_general,
            2: None,
        }
        if mera_instance in instance_mapping:
            method = instance_mapping[mera_instance]
            if callable(method):
                self.circuit.append(method(complex_structure))

        return self.circuit


# pylint: disable=too-few-public-methods
class QuantumPoolingLayer:
    """
    Builds a pooling layer in the neural network
    with the help of controlled phase flip gates.

    References:
        [1] I. Cong, S. Choi, and M. D. Lukin, “Quantum
        convolutional neural networks,” Nature Physics,
        vol. 15, no. 12, pp. 1273–1278, Aug. 2019,
        doi: https://doi.org/10.1038/s41567-019-0648-8.
    """

    def __init__(self, circuit: QuantumCircuit):
        """
        Initializes the pooling layer with the preceding
        convolutional layer.

        Args:
            circuit (QuantumCircuit): Takes quantum circuit with an
            existing convolutional or pooling layer as an input,
            and applies an/additional pooling layer over it.
        """
        self.circuit = circuit

    def pooling_layer(self) -> QuantumCircuit:
        """
        Implements a pooling layer with alternating phase flips on
        qubits when their adjacent qubits result in X = -1, when
        measured in X-basis.
        """
        classical_reg = ClassicalRegister(np.ceil(len(self.circuit.qubits) / 2))
        self.circuit.add_register(classical_reg)

        self.circuit.barrier()
        for qubit in range(len(self.circuit.qubits)):
            if qubit % 2 == 0 and qubit != len(self.circuit.qubits) - 1:
                # Measurement in X-basis.
                self.circuit.h(qubit + 1)
                self.circuit.measure(qubit + 1, classical_reg[int(qubit / 2)])
                # Dynamic circuit
                with self.circuit.if_test((classical_reg[int(qubit / 2)], 1)):
                    self.circuit.z(qubit)

        return self.circuit


# class FullyConnectedLayer:
#     def __init__(self):
#         pass


# class QCNN(MERA):
#     def __init__(self, img_dim):
#         super(MERA, self).__init__(img_dim)
#
#     def qcnn_backbone(self):
#         pass
