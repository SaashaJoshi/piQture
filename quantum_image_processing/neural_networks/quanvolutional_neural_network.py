"""Quanvolutional Neural Network"""
from __future__ import annotations
from typing import Callable
from qiskit.circuit import QuantumCircuit
from quantum_image_processing.models.neural_networks.quantum_neural_network import QuantumNeuralNetwork


class QuanvolutionalNeuralNetwork(QuantumNeuralNetwork):
    """
    Builds a Quanvolutional Neural Network [1].

    References:
        [1] M. Henderson, S. Shakya, S. Pradhan, and
        T. Cook, “Quanvolutional Neural Networks:
        Powering Image Recognition with Quantum Circuits,”
        arXiv:1904.04767 [quant-ph], Apr. 2019,
        Available: https://arxiv.org/abs/1904.04767
    """

    def __init__(self, num_qubits: int):
        """
        Initializes a Quanvolutional Neural Network circuit
        with the given number of qubits.

        Args:
            num_qubits (int): builds a quantum convolutional neural
            network circuit with the given number of qubits or image
            dimensions.
        """
        QuantumNeuralNetwork.__init__(self, num_qubits)

    def sequence(self, operations: list[tuple[Callable, dict]]) -> QuantumCircuit:
        """
        Builds a Quanvolutional NN circuit by composing the circuit
        with given sequence of list of operations.

        Args:
            operations (list[tuple[Callable, dict]]: a tuple
            of a Layer object and a dictionary of its arguments.

        Returns:
            circuit (QuantumCircuit): final QNN circuit with all the
            layers.
        """