from __future__ import annotations
from typing import Callable
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumCircuit, QuantumRegister, ParameterVector
from quantum_image_processing.gates.two_qubit_unitary import TwoQubitUnitary


class TTN:
    """
    Implements a Tree Tensor Network (TTN) as given by
    Grant et al. (2018).

    The model architecture only consists of a hierarchical
    TTN model. It cannot be classified as a QCNN since
    there is no distinction between conv and pooling layers.
    """

    def __init__(self, img_dim):
        self.img_dim = img_dim

    def ttn_simple(self, complex_structure: bool = True) -> QuantumCircuit:
        """
        Rotations here can be either real or complex.

        For real rotations only RY gates are used since
        the gate has no complex rotations involved.
        For complex rotations, a combination of RZ and RY
        gates are used.

        I HAVE NO IDEA WHY I CHOSE THESE. THE SELECTION
        OF UNITARY GATES IS COMPLETELY VOLUNTARY.

        PennyLane implements a TTN template with only RX gates.

        :return:
        """
        param_vector = ParameterVector("theta", 2 * self.img_dim - 1)
        param_vector_copy = param_vector
        return self.ttn_backbone(
            TwoQubitUnitary().simple_parameterization, param_vector_copy, complex_structure
        )

    def ttn_general(self, complex_structure: bool = True) -> QuantumCircuit:
        """
        Two qubit gates built from referencing a paper by
        Vatan et al. (2004).

        As stated in the paper:
        "A general two-qubit quantum computation, up to a global phase,
        can be constructed using at most 3 CNOT gates and 15 elementary
        one-qubit gates from the family {Ry , Rz}."
        and

        Theorem 3. Every two-qubit quantum gate in SO(4) (i.e. real gate)
        can be realized by a circuit consisting of 12 elementary
        one-qubit gates and 2 CNOT gates.

        Theorem 4. Every two-qubit quantum gate in O(4) with determinant
        equal to −1 can be realized by a circuit consisting of 12 elementary
        gates and 2 CNOT gates and one SWAP gate
        :return:
        """
        # Check number of params here.
        if complex_structure:
            param_vector = ParameterVector("theta", 15 * self.img_dim - 1)
            param_vector_copy = param_vector
        else:
            param_vector = ParameterVector("theta", 6 * self.img_dim - 1)
            param_vector_copy = param_vector

        return self.ttn_backbone(
            TwoQubitUnitary().simple_parameterization, param_vector_copy, complex_structure
        )

    def ttn_with_aux(self):
        """
        TODO: Find the implementation procedure for this.
        """

    def ttn_backbone(
        self,
        gate_structure: Callable,
        param_vector_copy: ParameterVector,
        complex_structure: bool = True,
    ) -> QuantumCircuit:
        ttn_qr = QuantumRegister(size=self.img_dim)
        ttn_circ = QuantumCircuit(ttn_qr)

        qubit_list = []
        for index in range(0, self.img_dim, 2):
            if index == self.img_dim - 1:
                qubit_list.append(ttn_qr[index])
            else:
                qubit_list.append(ttn_qr[index + 1])
                _, param_vector_copy = gate_structure(
                    circuit=ttn_circ,
                    qubits=[ttn_qr[index], ttn_qr[index + 1]],
                    parameter_vector=param_vector_copy,
                    complex_structure=complex_structure,
                )

        for _ in range(int(np.sqrt(self.img_dim))):
            temp_list = []
            for index in range(0, len(qubit_list) - 1, 2):
                _, param_vector_copy = gate_structure(
                    ttn_circ,
                    [qubit_list[index], qubit_list[index + 1]],
                    param_vector_copy,
                    complex_structure,
                )
                temp_list.append(qubit_list[index + 1])

            if len(qubit_list) % 2 != 0:
                temp_list.append(qubit_list[-1])

            qubit_list = temp_list

        ttn_circ.ry(param_vector_copy[0], qubit_list[-1])

        return ttn_circ
