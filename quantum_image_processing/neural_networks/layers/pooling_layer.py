"""Quantum Pooling Layer Structure"""
from __future__ import annotations
import itertools
from qiskit.circuit import QuantumCircuit
from quantum_image_processing.neural_networks.layers.base_layer import BaseLayer

# pylint: disable = too-few-public-methods


class QuantumPoolingLayer2(BaseLayer):
    """
    Builds a Pooling Layer, performing measurements on
    one of the two adjacent qubits, with the help of
    controlled-phase gates (dynamic circuits).

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
        unmeasured_bits: list,
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
        BaseLayer.__init__(self, num_qubits, circuit, unmeasured_bits)

    def build_layer(self) -> tuple[QuantumCircuit, list]:
        """
        Implements a pooling layer with alternating phase flips on
        qubits when the adjacent qubits measured in X-basis result
        in X = -1.

        Returns:
            circuit (QuantumCircuit): circuit with a pooling layer.

            unmeasured_bits (dict): a dictionary of unmeasured qubits
            and classical bits in the circuit.
        """
        unmeasured_bits = self.unmeasured_bits.copy()
        for phase_bit, measure_bit in zip(
            itertools.islice(self.unmeasured_bits, 0, None, 2),
            itertools.islice(self.unmeasured_bits, 1, None, 2),
        ):
            unmeasured_bits.remove(measure_bit)

            # Measurement in X-basis.
            self.circuit.h(measure_bit)
            self.circuit.measure(measure_bit, measure_bit)

            # # Dynamic circuit - cannot be composed with another circuit if
            # # using context manager form (e.g. with).
            # # Also, dynamic circuits don't work with runtime primitives.
            with self.circuit.if_test((measure_bit, 1)):
                self.circuit.z(phase_bit)

            # Without if_test.
            # self.circuit.cz(
            #     measure_bit,
            #     phase_bit,
            # )

        return self.circuit, unmeasured_bits


class QuantumPoolingLayer3(BaseLayer):
    """
    Builds a Pooling Layer, performing measurements on
    two of the three adjacent qubits, with the help of
    controlled-phase gates (dynamic circuits).

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
        unmeasured_bits: list,
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
        BaseLayer.__init__(self, num_qubits, circuit, unmeasured_bits)

        if (
            self.num_qubits < 3
            or len(self.circuit.qubits) < 3
            or len(self.unmeasured_bits) < 3
        ):
            raise ValueError(
                "The value of input num_qubits must be at least 3 or there must be "
                "at least 3 qubits in the provided circuit input or there must be at "
                "least 3 unmeasured bits in the circuit. "
            )

    def build_layer(self) -> tuple[QuantumCircuit, list]:
        """
        Implements a pooling layer with alternating phase flips on
        qubits when the adjacent qubits measured in X-basis result
        in X = -1.

        Returns:
            circuit (QuantumCircuit): circuit with a pooling layer.

            unmeasured_bits (dict): a dictionary of unmeasured qubits
            and classical bits in the circuit.
        """
        unmeasured_bits = self.unmeasured_bits.copy()
        for phase_bit, measure_bit1, measure_bit2 in zip(
            itertools.islice(self.unmeasured_bits, 1, None, 3),
            itertools.islice(self.unmeasured_bits, 0, None, 3),
            itertools.islice(self.unmeasured_bits, 2, None, 3),
        ):
            unmeasured_bits.remove(measure_bit1)
            unmeasured_bits.remove(measure_bit2)

            # Measurement in X-basis.
            self.circuit.h([measure_bit1, measure_bit2])
            self.circuit.measure(
                [measure_bit1, measure_bit2], [measure_bit1, measure_bit2]
            )

            # # Dynamic circuit - cannot be composed with another circuit if
            # # using context manager form (e.g. with).
            # # Also, dynamic circuits don't work with runtime primitives.
            with self.circuit.if_test((measure_bit1, 1)):
                with self.circuit.if_test((measure_bit2, 1)):
                    self.circuit.z(phase_bit)

        return self.circuit, unmeasured_bits
