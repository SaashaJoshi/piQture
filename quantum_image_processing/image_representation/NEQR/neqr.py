from __future__ import annotations
import math
import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_histogram


class NEQR:

    def __init__(self, image_size: tuple[int, int], color_vals: list[str], max_color: int = 255):
        self.image_size = image_size
        self.color_vals = color_vals
        self.feature_dim = int(np.sqrt(math.prod(self.image_size)))
        self.max_color = max_color + 1
        self.q = int(math.log(self.max_color, 2))

    def pixel_position(self, pixel_pos_binary: str):

        circ = QuantumCircuit(self.feature_dim)

        for index, value in enumerate(pixel_pos_binary):
            if value == '0':
                circ.x(index)

        return circ

    def color_info(self, pixel_pos: int):

        qr = QuantumRegister(self.feature_dim + self.q)
        circ = QuantumCircuit(qr)

        color_binary = "{0:0>8b}".format(int(self.color_vals[pixel_pos]))

        control_qubits = list(range(self.feature_dim))
        for index, color in enumerate(color_binary):
            if color == "1":
                circ.mct(control_qubits=control_qubits, target_qubit=self.feature_dim + index)

        return circ

    def measure_circ(self, circ):

        # Append measurement gates to the circuit
        qr = QuantumRegister(self.feature_dim + self.q)
        cr = ClassicalRegister(self.feature_dim + self.q)

        meas_circ = QuantumCircuit(qr, cr)
        meas_circ.measure(
            [i for i in range(self.feature_dim + self.q)],
            [i for i in range(self.feature_dim + self.q)],
        )
        meas_circ = meas_circ.compose(circ, range(self.feature_dim + self.q), front=True)

        return meas_circ

    def image_encoding(self, measure=True):

        qr = QuantumRegister(self.feature_dim + self.q)
        circ = QuantumCircuit(qr)

        for i in range(self.feature_dim):
            circ.h(i)

        num_theta = math.prod(self.image_size)
        for pixel in range(num_theta):
            pixel_pos_binary = "{0:0>2b}".format(pixel)

            # Embed pixel position on qubits
            circ.compose(
                self.pixel_position(pixel_pos_binary),
                [qr[i] for i in range(self.feature_dim)],
            )

            # Embed color information on qubits
            circ.compose(
                self.color_info(pixel),
                [qr[i] for i in range(self.feature_dim + self.q)],
            )

            # Remove pixel position embedding
            circ.compose(
                self.pixel_position(pixel_pos_binary),
                [qr[i] for i in range(self.feature_dim)],
            )

        if measure:
            circ = self.measure_circ(circ)

        return circ

    @staticmethod
    def get_simulator_result(
            circ,
            backend: str = 'qasm_simulator',
            shots: int = 1024,
            plot_counts=True,
    ):
        backend = Aer.get_backend(backend)
        job = execute(circ, backend=backend, shots=shots)
        results = job.result()
        counts = results.get_counts()

        if plot_counts:
            plot_histogram(counts)

        return counts

    def qic(self):
        pass

    # TODO: Implement Color Operations - CC, PC and CS from NEQR paper by Zhang, Yi et al.
    def cc_operation(self, circ):
        # Complete Color Operation
        pass

    def pc_operation(self, circ):
        # Partial Color Operation
        pass

    def cs_operation(self, circ):
        # Color Statistical Operation
        pass

# if __name__ == '__main__':
#     # color_palette = ["0000", "0001", "0010", "0011",
#     #               "0100", "0101", "0110", "0111",
#     #               "1000", "1001", "1010", "1011",
#     #               "1100", "1101", "1110", "1111"]
#
#     # pixel_vals = ["1010", "1111", "0110", "0001"]
#     pixel_vals = [10, 15, 6, 1]
#     image_size = (2, 2)
#
#     circ = NEQR(image_size=image_size, color_vals=pixel_vals, max_color=15)
#     circ = circ.image_encoding(measure=True)
#     circ.decompose().draw('mpl')
#     plt.show()
