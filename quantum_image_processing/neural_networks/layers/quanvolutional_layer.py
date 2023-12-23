"""Quanvolutional Layer structure"""
from __future__ import annotations
import math
from typing import Optional
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from quantum_image_processing.neural_networks.layers.base_layer import BaseLayer


class QuanvolutionalLayer(BaseLayer):
    """
    Builds a Quanvolutional Layer [1] with a non-trainable
    random quantum circuit.

    References:
        [1] M. Henderson, S. Shakya, S. Pradhan, and
        T. Cook, “Quanvolutional Neural Networks:
        Powering Image Recognition with Quantum Circuits,”
        arXiv:1904.04767 [quant-ph], Apr. 2019,
        Available: https://arxiv.org/abs/1904.04767
    """

    def __init__(
        self,
        img_dims: tuple[int, int],
        filter_size: tuple[int, int],
        stride: Optional[int] = 1,
        random: Optional[bool] = True,
    ):
        """
        Initializes a Quanvolutional Layer circuit
        with the given number of qubits.

        Args:
            num_qubits (int): builds a quantum convolutional neural
            network circuit with the given number of qubits or image
            dimensions.

            circuit (QuantumCircuit): Takes quantum circuit with/without
            an existing layer as an input, and applies a quanvolutional
            layer over it.

            unmeasured_bits (dict): a dictionary of unmeasured qubits
            and classical bits in the circuit.
        """
        if not all((isinstance(dims, int) for dims in img_dims)) or not isinstance(
            img_dims, tuple
        ):
            raise TypeError("Input img_dims must be of the type tuple[int, ...].")

        self.img_dims = img_dims
        num_qubits = int(math.prod(self.img_dims))
        BaseLayer.__init__(self, num_qubits)

        if not isinstance(filter_size, tuple) or not all(
            isinstance(size, int) for size in filter_size
        ):
            raise TypeError(
                "The input filter_size must be of the type tuple[int, int]."
            )

        if filter_size[0] > min(self.img_dims):
            raise ValueError("The filter_size must be less than or equal "
                             "to the minimum image dimension.")
        self.filter_size = filter_size

        if not isinstance(stride, int):
            raise TypeError("The input stride must be of the type int.")

        if stride <= 0 or stride > min(img_dims):
            raise ValueError("The input stride must be at least 1 and less than "
                             "or equal to the minimum image dimension.")
        self.stride = stride

        if not isinstance(random, bool):
            raise TypeError("The input random must be of the type bool.")
        self.random = random

    def build_layer(self) -> tuple[QuantumCircuit, list]:
        """
        Builds the Quanvolutional layer circuit

        Returns:
            circuit (QuantumCircuit): circuit with a quanvolutional layer.

            unmeasured_bits (dict): a dictionary of unmeasured qubits
            and classical bits in the circuit.
        """
        sub_image_qubits = int(math.prod(self.filter_size))
        # Apply random circuit with measurements
        if self.random:
            sub_circuit = random_circuit(
                num_qubits=sub_image_qubits,
                depth=2,
                max_operands=3,
                measure=True,
            )
            return sub_circuit, []
