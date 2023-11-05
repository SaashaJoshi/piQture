import numpy as np
from typing import Callable
from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterVector
from quantum_image_processing.gates.two_qubit_unitary import TwoQubitUnitary


class MERA(TwoQubitUnitary):
    """
    Implements QCNN structure by Cong et al. (2019), replicating
    the architecture described by MERA - Multiscale Entanglement
    Renormalization Ansatz, given by Vidal et al. (2008).

    The decomposition of MERA architecture takes from the paper
    by Grant et al. (2018).

    NOTE: Remember QCNN and MERA have opposite directions!!
    # Args:
    #     layer_depth (int): hyperparameter that restricts
    #     the depth of a convolutional layer to a specified
    #     number of unitary gate layers.
    """

    def __init__(self, img_dim: int, layer_depth: type(None) = None):
        self.img_dim = img_dim
        if layer_depth is None:
            self.layer_depth = int(np.ceil(np.sqrt(self.img_dim)))
        else:
            self.layer_depth = layer_depth

    def mera_simple(self, complex_structure: bool = True) -> QuantumCircuit:
        param_vector = ParameterVector(
            "theta",
            int(self.img_dim / 2 * (self.img_dim / 2 + 1)) + 3,
        )
        param_vector_copy = param_vector
        return self.mera_backbone(
            self.simple_parameterization,
            param_vector_copy,
            complex_structure,
        )

    # Check number of params here.
    def mera_general(self, complex_structure: bool = True) -> QuantumCircuit:
        if complex_structure:
            param_vector = ParameterVector("theta", 20 * self.img_dim - 1)
            param_vector_copy = param_vector
        else:
            param_vector = ParameterVector("theta", 10 * self.img_dim - 1)
            param_vector_copy = param_vector
        return self.mera_backbone(
            self.general_parameterization,
            param_vector_copy,
            complex_structure,
        )

    def mera_backbone(
        self,
        gate_structure: Callable,
        param_vector_copy: ParameterVector,
        complex_structure: bool = True,
    ) -> QuantumCircuit:
        mera_qr = QuantumRegister(size=self.img_dim)
        mera_circ = QuantumCircuit(mera_qr)

        # TODO: Make recursive layer structure using a static method.
        for layer in range(self.layer_depth):
            if layer == 0:
                # D unitary blocks
                for index in range(1, self.img_dim, 2):
                    if index == self.img_dim - 1:
                        break

                    _, param_vector_copy = gate_structure(
                        circuit=mera_circ,
                        qubits=[mera_qr[index], mera_qr[index + 1]],
                        parameter_vector=param_vector_copy,
                        complex_structure=complex_structure,
                    )

                # U unitary blocks
                mera_circ.barrier()
                qubit_list = []
                for index in range(0, self.img_dim, 2):
                    if index == self.img_dim - 1:
                        qubit_list.append(mera_qr[index])
                    else:
                        qubit_list.append(mera_qr[index + 1])
                        _, param_vector_copy = gate_structure(
                            circuit=mera_circ,
                            qubits=[mera_qr[index], mera_qr[index + 1]],
                            parameter_vector=param_vector_copy,
                            complex_structure=complex_structure,
                        )

            else:
                temp_list = []
                # D unitary blocks
                mera_circ.barrier()
                for index in range(1, len(qubit_list), 2):
                    if len(qubit_list) == 2 or index == len(qubit_list) - 1:
                        break

                    _, param_vector_copy = gate_structure(
                        circuit=mera_circ,
                        qubits=[qubit_list[index], qubit_list[index + 1]],
                        parameter_vector=param_vector_copy,
                        complex_structure=complex_structure,
                    )

                # U unitary blocks
                mera_circ.barrier()
                for index in range(0, len(qubit_list) - 1, 2):
                    block, param_vector_copy = gate_structure(
                        circuit=mera_circ,
                        qubits=[qubit_list[index], qubit_list[index + 1]],
                        parameter_vector=param_vector_copy,
                        complex_structure=complex_structure,
                    )
                    temp_list.append(qubit_list[index + 1])

                if len(qubit_list) % 2 != 0:
                    temp_list.append(qubit_list[-1])
                qubit_list = temp_list

        mera_circ.ry(param_vector_copy[0], mera_qr[-1])

        return mera_circ
