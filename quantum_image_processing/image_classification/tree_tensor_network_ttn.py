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
        self.param_vector = ParameterVector('theta', 2 * self.img_dim - 1)
        self.param_vector_copy = self.param_vector

    def _apply_simple_block(self, qubits):
        block = QuantumCircuit(self.img_dim)
        block.ry(self.param_vector_copy[0], qubits[0])
        block.ry(self.param_vector_copy[1], qubits[1])
        block.cx(qubits[0], qubits[1])
        self.param_vector_copy = self.param_vector_copy[2:]

        return block

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
        ttn_circ = QuantumCircuit(self.img_dim)

        if complex_struct:
            pass
        else:
            qubit_list = []
            for qubits in range(0, self.img_dim, 2):
                if qubits == self.img_dim - 1:
                    qubit_list.append(qubits)
                else:
                    qubit_list.append(qubits + 1)
                    block = self._apply_simple_block(qubits=[qubits, qubits + 1])
                    ttn_circ = ttn_circ.compose(block, range(self.img_dim))

        # The layers might not be symmetric right now.
        for layer in range(int(np.sqrt(self.img_dim))):
            temp_list = []

            for index in range(0, len(qubit_list) - 1, 2):
                block = self._apply_simple_block(qubits=[qubit_list[index], qubit_list[index + 1]])
                ttn_circ = ttn_circ.compose(block, range(self.img_dim))
                temp_list.append(qubit_list[index + 1])

            if len(qubit_list) % 2 != 0:
                temp_list.append(qubit_list[-1])

            qubit_list = temp_list

        ttn_circ.ry(self.param_vector_copy[0], qubit_list[-1])

        return ttn_circ

    def ttn_general(self):
        pass

    def ttn_with_aux(self):
        pass
