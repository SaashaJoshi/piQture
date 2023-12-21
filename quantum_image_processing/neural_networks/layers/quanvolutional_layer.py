"""Quanvolutional Layer structure"""
from __future__ import annotations
from qiskit.circuit import QuantumCircuit
from quantum_image_processing.neural_networks.layers.base_layer import BaseLayer


class QuanvolutionalLayer(BaseLayer):
    """
    Builds a Quanvolutional Layer [1].

    References:
        [1] M. Henderson, S. Shakya, S. Pradhan, and
        T. Cook, “Quanvolutional Neural Networks:
        Powering Image Recognition with Quantum Circuits,”
        arXiv:1904.04767 [quant-ph], Apr. 2019,
        Available: https://arxiv.org/abs/1904.04767
    """

    def __init__(self, num_qubits: int, circuit: QuantumCircuit, unmeasured_bits: list):
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
        BaseLayer.__init__(self, num_qubits, circuit, unmeasured_bits)

    def build_layer(self) -> tuple[QuantumCircuit, list]:
        """
        Builds the Quanvolutional layer circuit

        Returns:
            circuit (QuantumCircuit): circuit with a quanvolutional layer.
            unmeasured_bits (dict): a dictionary of unmeasured qubits
            and classical bits in the circuit.
        """
