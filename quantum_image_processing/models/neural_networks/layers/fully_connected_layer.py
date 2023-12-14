"""Quantum Fully Connected Layer Structure"""
from __future__ import annotations
from qiskit.circuit import QuantumCircuit
from quantum_image_processing.models.neural_networks.layers.base_layer import BaseLayer


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

    def __init__(self, num_qubits: int, circuit: QuantumCircuit, unmeasured_bits: dict):
        """
        Initializes a fully connected layer object.

        Args:
            num_qubits (int): inputs number of qubits required
            in the circuit or the image dimensions.

            circuit (QuantumCircuit): Takes quantum circuit with an
            existing convolutional or pooling layer as an input,
            and applies an/additional convolutional layer over it.

            unmeasured_bits (dict): Takes into consideration
            the unmeasured qubits in the preceding circuit. Only these
            qubits are used to create the FC layer.
        """
        BaseLayer.__init__(self, num_qubits)
        self.circuit = circuit
        self.unmeasured_bits = unmeasured_bits

    def build_layer(self) -> tuple[QuantumCircuit, dict]:
        """
        Implements a fully connected layer with controlled phase
        gates on adjacent qubits followed by a measurement in X-basis.

        Returns:
            circuit (QuantumCircuit): circuit with a fully connected
            layer.

            unmeasured_bits (dict): a dictionary of unmeasured qubits
            and classical bits in the circuit.
        """
        self.circuit.barrier()
        unmeasured_bits: dict = {"qubits": [], "clbits": []}
        for index, qubit in enumerate(self.unmeasured_bits["qubits"][:-1]):
            self.circuit.cz(
                qubit,
                self.unmeasured_bits["qubits"][index + 1],
            )
        # Comment next line to skip implicit measurement.
        self.final_measurement()
        return self.circuit, unmeasured_bits

    def final_measurement(self) -> QuantumCircuit:
        """
        Implements a measurement in X-basis on the remaining qubits
        after a fully connected layer is implemented.

        Returns:
            circuit (QuantumCircuit): final circuit with measurements
        """
        self.circuit.barrier()
        # Measurement in X-basis
        self.circuit.h(self.unmeasured_bits["qubits"])
        self.circuit.measure(
            self.unmeasured_bits["qubits"], self.unmeasured_bits["clbits"]
        )
        return self.circuit
