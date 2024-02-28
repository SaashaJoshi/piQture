"""Quantum Convolutional Layer Structure"""

from __future__ import annotations
from typing import Optional
from qiskit.circuit import QuantumCircuit
from quantum_image_processing.neural_networks.layers.base_layer import BaseLayer
from quantum_image_processing.tensor_network_circuits.mera import MERA


class QuantumConvolutionalLayer(BaseLayer):
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
        num_qubits: Optional[int] = None,
        circuit: Optional[QuantumCircuit] = None,
        unmeasured_bits: Optional[list] = None,
        mera_args: Optional[dict] = None,
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

            mera_args (dict): dictionary of arguments used to build
            a MERA layer.

            unmeasured_bits (dict): a dictionary of unmeasured qubits
            and classical bits in the circuit.
        """
        BaseLayer.__init__(self, num_qubits, circuit, unmeasured_bits)

        if mera_args is None:
            mera_args = {}

        if "mera_instance" not in mera_args:
            mera_args["mera_instance"] = 0
        else:
            if not isinstance(mera_args["mera_instance"], int):
                raise TypeError(
                    "The value corresponding to mera_instance key in mera_args "
                    "dictionary input must be of the type int."
                )

            if mera_args["mera_instance"] < 0 or mera_args["mera_instance"] > 2:
                raise ValueError(
                    "The value corresponding to mera_instance key in mera_args "
                    "dictionary input must be in range(0, 2)."
                )
        if "layer_depth" not in mera_args:
            mera_args["layer_depth"] = 1
        if "complex_structure" not in mera_args:
            mera_args["complex_structure"] = False

        self.mera_args = mera_args

    def build_layer(self) -> tuple[QuantumCircuit, list]:
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
        mera = MERA(self.num_qubits, self.mera_args["layer_depth"])

        mera_instance_mapping = {
            0: mera.mera_simple,
            1: mera.mera_general,
            2: None,
        }
        method = mera_instance_mapping[self.mera_args["mera_instance"]]

        if callable(method):
            self.circuit.compose(
                method(self.mera_args["complex_structure"]),
                qubits=self.circuit.qubits,
                inplace=True,
            )
        return self.circuit, self.unmeasured_bits
