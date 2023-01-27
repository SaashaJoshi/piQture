import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector


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

    def _apply_real_simple_block(self, qubits, param_vector_copy):
        block = QuantumCircuit(self.img_dim)

        block.ry(param_vector_copy[0], qubits[0])
        block.ry(param_vector_copy[1], qubits[1])
        block.cx(qubits[0], qubits[1])

        param_vector_copy = param_vector_copy[2:]

        return block, param_vector_copy

    def _apply_complex_general_block(self, qubits, param_vector_copy):
        block = QuantumCircuit(self.img_dim)

        block.rz(param_vector_copy[0], qubits[0])
        block.ry(param_vector_copy[1], qubits[0])
        block.rz(param_vector_copy[2], qubits[0])

        block.rz(param_vector_copy[3], qubits[1])
        block.ry(param_vector_copy[4], qubits[1])
        block.rz(param_vector_copy[5] + np.pi / 2, qubits[1])
        block.cnot(qubits[1], qubits[0])

        block.rz((2 * param_vector_copy[6]) - np.pi / 2, qubits[0])
        block.ry(np.pi / 2 - (2 * param_vector_copy[7]), qubits[1])
        block.cnot(qubits[0], qubits[1])
        block.ry((2 * param_vector_copy[8]) - np.pi / 2, qubits[1])

        block.cnot(qubits[1], qubits[0])
        block.rz(param_vector_copy[9], qubits[1])
        block.ry(param_vector_copy[10], qubits[1])
        block.rz(param_vector_copy[11], qubits[1])

        block.rz(param_vector_copy[12] - np.pi / 2, qubits[0])
        block.ry(param_vector_copy[13], qubits[0])
        block.rz(param_vector_copy[14], qubits[0])

        param_vector_copy = param_vector_copy[15:]

        return block, param_vector_copy

    def _apply_real_general_block(self, qubits, param_vector_copy):
        block = QuantumCircuit(self.img_dim)

        block.rz(np.pi / 2, qubits[0])
        block.rz(np.pi / 2, qubits[1])
        block.ry(np.pi / 2, qubits[1])
        block.cnot(qubits[1], qubits[0])

        block.rz(param_vector_copy[0], qubits[0])
        block.ry(param_vector_copy[1], qubits[0])
        block.rz(param_vector_copy[2], qubits[0])

        block.rz(param_vector_copy[3], qubits[1])
        block.ry(param_vector_copy[4], qubits[1])
        block.rz(param_vector_copy[5], qubits[1])

        block.cnot(qubits[1], qubits[0])
        block.ry(-np.pi / 2, qubits[1])
        block.rz(-np.pi / 2, qubits[0])
        block.rz(-np.pi / 2, qubits[1])

        param_vector_copy = param_vector_copy[6:]

        return block, param_vector_copy

    def ttn_simple(self, complex_struct=True):
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
        param_vector = ParameterVector('theta', 2 * self.img_dim - 1)
        param_vector_copy = param_vector

        if complex_struct:
            pass
        else:
            return self.ttn_backbone(self._apply_real_simple_block, param_vector_copy)

    def ttn_general(self, complex_struct=True):
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
        if complex_struct:
            param_vector = ParameterVector('theta', 15 * self.img_dim - 1)
            param_vector_copy = param_vector
            return self.ttn_backbone(self._apply_complex_general_block, param_vector_copy)
        else:
            param_vector = ParameterVector('theta', 6 * self.img_dim - 1)
            param_vector_copy = param_vector
            return self.ttn_backbone(self._apply_real_general_block, param_vector_copy)

    def ttn_with_aux(self):
        pass

    def ttn_backbone(self, gate_structure, param_vector_copy):
        ttn_circ = QuantumCircuit(self.img_dim)

        qubit_list = []
        for qubits in range(0, self.img_dim, 2):
            if qubits == self.img_dim - 1:
                qubit_list.append(qubits)
            else:
                qubit_list.append(qubits + 1)
                block, param_vector_copy = gate_structure(
                    qubits=[qubits, qubits + 1],
                    param_vector_copy=param_vector_copy,
                )
                ttn_circ = ttn_circ.compose(block, range(self.img_dim))

        # The layers might not be symmetric right now.
        for layer in range(int(np.sqrt(self.img_dim))):
            temp_list = []

            for index in range(0, len(qubit_list) - 1, 2):
                block, param_vector_copy = gate_structure(
                    qubits=[qubit_list[index], qubit_list[index + 1]],
                    param_vector_copy=param_vector_copy,
                )
                ttn_circ = ttn_circ.compose(block, range(self.img_dim))
                temp_list.append(qubit_list[index + 1])

            if len(qubit_list) % 2 != 0:
                temp_list.append(qubit_list[-1])

            qubit_list = temp_list

        ttn_circ.ry(param_vector_copy[0], qubit_list[-1])

        return ttn_circ
