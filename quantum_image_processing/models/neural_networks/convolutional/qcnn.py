"""Quantum Convolutional Neural Network"""
from __future__ import annotations
from typing import Callable, Optional
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from quantum_image_processing.models.neural_networks.layer import Layer
from quantum_image_processing.models.neural_networks.quantum_neural_network import (
    QuantumNeuralNetwork,
)
from quantum_image_processing.models.tensor_network_circuits.mera import MERA

# pylint: disable=too-few-public-methods


class QuantumConvolutionalLayer(Layer, MERA):
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

    def __init__(
        self,
        num_qubits: int,
        circuit: QuantumCircuit,
        layer_depth: int,
        mera_instance: int,
        complex_structure: bool = True,
        unmeasured_bits: Optional[dict] = None,
    ):
        """
        Initializes a convolutional layer from the MERA
        tensor network structure.

        Args:
            num_qubits (int): inputs number of qubits required
            in the circuit or the image dimensions.

            circuit (QuantumCircuit): Takes quantum circuit with an
            existing convolutional or pooling layer as an input,
            and applies an/additional convolutional layer over it.

            layer_depth (int): mentions the depth of convolutional
            layer (MERA) required.

            complex_structure (bool): If True, builds the layer with
            complex unitary gates (e.g. RY, etc.)

            unmeasured_bits (dict): a dictionary of unmeasured qubits
            and classical bits in the circuit.
        """
        Layer.__init__(self, num_qubits)
        MERA.__init__(self, num_qubits, layer_depth)

        self.circuit = circuit
        self.mera_instance = mera_instance
        self.complex_structure = complex_structure

        self.unmeasured_bits = {}
        if unmeasured_bits is None:
            self.unmeasured_bits["qubits"] = self.circuit.qubits
            self.unmeasured_bits["clbits"] = self.circuit.clbits
        else:
            self.unmeasured_bits = unmeasured_bits

    def build_layer(self) -> tuple[QuantumCircuit, dict]:
        """
        Implements the MERA tensor network with a restriction
        on the depth of the layers, specified by a
        hyperparameter, `layer_depth`.

        Returns:
            circuit (QuantumCircuit): circuit with a convolutional
            layer.

            unmeasured_bits (dict): a dictionary of unmeasured qubits
            and classical bits in the circuit.
        """
        self.circuit.barrier()
        instance_mapping = {
            0: self.mera_simple,
            1: self.mera_general,
            2: None,
        }
        if self.mera_instance in instance_mapping:
            method = instance_mapping[self.mera_instance]
            if callable(method):
                self.circuit.compose(
                    method(self.complex_structure),
                    qubits=self.circuit.qubits,
                    clbits=self.circuit.clbits,
                )
        else:
            raise ValueError(f"Invalid mera_instance value: {self.mera_instance}")

        return self.circuit, self.unmeasured_bits


class QuantumPoolingLayer(Layer):
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
        Layer.__init__(self, num_qubits)
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
            # Dynamic circuit - cannot be composed if using context manager form (e.g. with)
            with self.circuit.if_test(
                (self.unmeasured_bits["clbits"][index + 1].index, 1)
            ):
                self.circuit.z(self.unmeasured_bits["qubits"][index].index)

        return self.circuit, unmeasured_bits


class FullyConnectedLayer(Layer):
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
        Layer.__init__(self, num_qubits)
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
        for index in range(len(self.unmeasured_bits["qubits"]) - 1):
            self.circuit.cz(
                self.unmeasured_bits["qubits"][index],
                self.unmeasured_bits["qubits"][index + 1],
            )
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


class QCNN(QuantumNeuralNetwork):
    """
    Builds a Quantum Neural Network circuit with the help of
    convolutional, pooling or fully-connected layers.

    References:
        [1] I. Cong, S. Choi, and M. D. Lukin, “Quantum
        convolutional neural networks,” Nature Physics,
        vol. 15, no. 12, pp. 1273–1278, Aug. 2019,
        doi: https://doi.org/10.1038/s41567-019-0648-8.
    """

    def __init__(self, num_qubits: int):
        """
        Initializes a Quantum Neural Network circuit with the given
        number of qubits.

        Args:
            num_qubits (int): builds a quantum convolutional neural
            network circuit with the given number of qubits or image
            dimensions.
        """
        QuantumNeuralNetwork.__init__(self, num_qubits)
        self.qreg = QuantumRegister(self.num_qubits)
        self.creg = ClassicalRegister(self.num_qubits)
        self.circuit = QuantumCircuit(self.qreg, self.creg)

    def sequence(self, operations: list[tuple[Callable, dict]]) -> QuantumCircuit:
        """
        Builds a QNN circuit by composing the circuit with given
        sequence of list of operations.

        Args:
            operations (list[tuple[Callable, dict]]: a tuple
            of a Layer object and a dictionary of its arguments.

        Returns:
            circuit (QuantumCircuit): final QNN circuit with all the
            layers.
        """
        unmeasured_bits = {"qubits": self.circuit.qubits, "clbits": self.circuit.clbits}
        for layer, params in operations:
            layer_instance = layer(
                circuit=self.circuit,
                num_qubits=self.num_qubits,
                unmeasured_bits=unmeasured_bits,
                **params,
            )
            self.circuit, unmeasured_bits = layer_instance.build_layer()

        return self.circuit
