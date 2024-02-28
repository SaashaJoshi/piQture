# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum Fully Connected Layer Structure"""

from __future__ import annotations
from qiskit.circuit import QuantumCircuit
from quantum_image_processing.neural_networks.layers.base_layer import BaseLayer


class FullyConnectedLayer(BaseLayer):
    """
    Builds a fully-connected layer in the neural network
    with the help of controlled phase gates.

    References:
        [1] I. Cong, S. Choi, and M. D. Lukin, “Quantum
        convolutional neural networks,” Nature Physics,
        vol. 15, no. 12, pp. 1273–1278, Aug. 2019,
        doi: https://doi.org/10.1038/s41567-019-0648-8.
    """

    def __init__(self, num_qubits: int, circuit: QuantumCircuit, unmeasured_bits: list):
        """
        Initializes a fully connected layer object.

        Args:
            num_qubits (int): inputs number of qubits required
            in the circuit or the image dimensions.

            circuit (QuantumCircuit): Takes quantum circuit with an
            existing convolutional or pooling layer as an input,
            and applies an/additional convolutional layer over it.

            unmeasured_bits (list): Takes into consideration
            the unmeasured qubits in the preceding circuit. Only these
            qubits are used to create the FC layer.
        """
        BaseLayer.__init__(self, num_qubits, circuit, unmeasured_bits)

    def build_layer(self) -> tuple[QuantumCircuit, list]:
        """
        Implements a fully connected layer with controlled phase
        gates on adjacent qubits followed by a measurement in X-basis.

        Returns:
            circuit (QuantumCircuit): circuit with a fully connected
            layer.

            unmeasured_bits (dict): a dictionary of unmeasured qubits
            and classical bits in the circuit.
        """
        for index, qubit in enumerate(self.unmeasured_bits[:-1]):
            self.circuit.cz(
                qubit,
                self.unmeasured_bits[index + 1],
            )
        return self.circuit, self.unmeasured_bits
