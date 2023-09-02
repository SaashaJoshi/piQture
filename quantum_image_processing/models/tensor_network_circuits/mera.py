import numpy as np
from qiskit.circuit import QuantumCircuit, ParameterVector


# Can TTN be an ABC for MERA? Good thought!
class MERA:
    """
    Implements QCNN structure by Cong et al. (2019), replicating
    the architecture described by MERA - Multiscale Entanglement
    Renormalization Ansatz, given by Vidal et al. (2008).

    The decomposition of MERA architecture takes from the paper
    by Grant et al. (2018).
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

    def mera_simple(self, complex_struct=True):
        param_vector = ParameterVector(
            "theta",
            int(self.img_dim / 2 * (self.img_dim / 2 + 1)) + 3,
        )
        param_vector_copy = param_vector

        if complex_struct:
            pass
        else:
            return self.mera_backbone(self._apply_real_simple_block, param_vector_copy)

    # Check number of params here.
    def mera_general(self, complex_struct=True):
        if complex_struct:
            param_vector = ParameterVector("theta", 20 * self.img_dim - 1)
            param_vector_copy = param_vector
            return self.mera_backbone(
                self._apply_complex_general_block, param_vector_copy
            )
        else:
            param_vector = ParameterVector("theta", 10 * self.img_dim - 1)
            param_vector_copy = param_vector
            return self.mera_backbone(self._apply_real_general_block, param_vector_copy)

    def mera_backbone(self, gate_structure, param_vector_copy):
        mera_circ = QuantumCircuit(self.img_dim)

        for qubits in range(1, self.img_dim, 2):
            if qubits == self.img_dim - 1:
                break

            block, param_vector_copy = gate_structure(
                qubits=[qubits, qubits + 1],
                param_vector_copy=param_vector_copy,
            )
            mera_circ = mera_circ.compose(block, range(self.img_dim))

        mera_circ.barrier()
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
                mera_circ = mera_circ.compose(block, range(self.img_dim))

        # The layers might not be symmetric right now.
        for layer in range(int(np.sqrt(self.img_dim))):
            temp_list = []

            mera_circ.barrier()
            for index in range(1, len(qubit_list), 2):
                if len(qubit_list) == 2 or index == len(qubit_list) - 1:
                    break

                block, param_vector_copy = gate_structure(
                    qubits=[qubit_list[index], qubit_list[index + 1]],
                    param_vector_copy=param_vector_copy,
                )
                mera_circ = mera_circ.compose(block, range(self.img_dim))

            mera_circ.barrier()

            for index in range(0, len(qubit_list) - 1, 2):
                block, param_vector_copy = gate_structure(
                    qubits=[qubit_list[index], qubit_list[index + 1]],
                    param_vector_copy=param_vector_copy,
                )
                mera_circ = mera_circ.compose(block, range(self.img_dim))
                temp_list.append(qubit_list[index + 1])

            if len(qubit_list) % 2 != 0:
                temp_list.append(qubit_list[-1])

            qubit_list = temp_list

        mera_circ.ry(param_vector_copy[0], qubit_list[-1])

        return mera_circ
