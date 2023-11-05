"""Quantum Convolutional Neural Network"""
# QCNN has 4 types of layers: data encoding, convolutional layer (MERA), Pooling layer, FC layer, and measurement.

from __future__ import annotations
from typing import Callable
from qiskit.circuit import QuantumCircuit
from quantum_image_processing.models.tensor_network_circuits.mera import MERA


class QuantumConvolutionalLayer(MERA):
    """
    Builds the convolutional layer in the neural network
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

    def __init__(self, *args, **kwargs):
        """
        Initializes the convolutional layer to a MERA
        tensor network structure.
        """
        MERA.__init__(self, *args, **kwargs)

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

            complex_structure (bool):

        TODO: Make cases to apply different MERA parameterization.
        """
        instance_mapping = {
            0: self.mera_simple,
            1: self.mera_general,
            2: None,
        }
        if mera_instance in instance_mapping:
            method = instance_mapping[mera_instance]
            if callable(method):
                return method(complex_structure)


class QuantumPoolingLayer:
    def __init__(self):
        pass


class FullyConnectedLayer:
    def __init__(self):
        pass


class QCNN(MERA):
    def __init__(self, img_dim):
        super(MERA, self).__init__(img_dim)

    def qcnn_backbone(self):
        pass
