"""Quantum Pooling Layer Structure"""
from __future__ import annotations
from qiskit.circuit import QuantumCircuit
from quantum_image_processing.models.neural_networks.layers.base_layer import BaseLayer


class QuantumPoolingLayer(BaseLayer):
    """
    Builds a pooling layer in the neural network
    with the help of controlled phase flip gates.

    References:
        [1] I. Cong, S. Choi, and M. D. Lukin, “Quantum
        convolutional neural networks,” Nature Physics,
        vol. 15, no. 12, pp. 1273–1278, Aug. 2019,
        doi: https://doi.org/10.1038/s41567-019-0648-8.
    """

    def __init__(
        self,
        num_qubits: int,
        circuit: QuantumCircuit,
        unmeasured_bits: dict,
    ):
        """
        Initializes a pooling layer object.

        Args:
            circuit (QuantumCircuit): Takes quantum circuit with an
            existing convolutional or pooling layer as an input,
            and applies an/additional pooling layer over it.

            unmeasured_bits (dict): a dictionary of unmeasured qubits
            and classical bits in the circuit.
        """
        BaseLayer.__init__(self, num_qubits)
        self.circuit = circuit
        self.unmeasured_bits = unmeasured_bits

    def build_layer(self) -> tuple[QuantumCircuit, dict]:
        """
        Implements a pooling layer with alternating phase flips on
        qubits when the adjacent qubits measured in X-basis result
        in X = -1.

        Returns:
            circuit (QuantumCircuit): circuit with a pooling layer.

            unmeasured_bits (dict): a dictionary of unmeasured qubits
            and classical bits in the circuit.
        """
        unmeasured_bits: dict = {"qubits": [], "clbits": []}
        self.circuit.barrier()
        for index in range(0, len(self.unmeasured_bits["qubits"][:-1]), 2):
            unmeasured_bits["qubits"].append(self.unmeasured_bits["qubits"][index])
            unmeasured_bits["clbits"].append(self.unmeasured_bits["clbits"][index])

            # Measurement in X-basis.
            self.circuit.h(self.unmeasured_bits["qubits"][index + 1].index)
            self.circuit.measure(
                self.unmeasured_bits["qubits"][index + 1].index,
                self.unmeasured_bits["clbits"][index + 1].index,
            )
            # Dynamic circuit - cannot be composed with another circuit if
            # using context manager form (e.g. with).
            # Also, dynamic circuits don't work with runtime primitives.
            with self.circuit.if_test(
                (self.unmeasured_bits["clbits"][index + 1].index, 1)
            ):
                self.circuit.z(self.unmeasured_bits["qubits"][index].index)

            # Without if_test.
            # self.circuit.cz(
            #     self.unmeasured_bits["qubits"][index + 1].index,
            #     self.unmeasured_bits["qubits"][index].index,
            # )

        return self.circuit, unmeasured_bits
