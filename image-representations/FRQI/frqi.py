from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.visualization import plot_histogram


class FRQI:

    def __init__(self, image_size: tuple[int, int], color_vals: list):
        self.image_size = image_size
        self.color_vals = color_vals
        self.feature_dimen = int(np.sqrt(math.prod(self.image_size)))

    def pixel_position(self, pixel_pos_binary: str):

        circ = QuantumCircuit(self.feature_dimen)

        for index, value in enumerate(pixel_pos_binary):
            if value == '0':
                circ.x(index)

        return circ

    def color_info(self, pixel_pos: int):

        qr = QuantumRegister(self.feature_dimen + 1)
        circ = QuantumCircuit(qr)

        circ.cry(
            self.color_vals[pixel_pos],
            target_qubit=self.feature_dimen,
            control_qubit=self.feature_dimen - 2,
        )
        circ.cx(0, 1)
        circ.cry(
            -self.color_vals[pixel_pos],
            target_qubit=self.feature_dimen,
            control_qubit=self.feature_dimen - 1,
        )
        circ.cx(0, 1)
        circ.cry(
            self.color_vals[pixel_pos],
            target_qubit=self.feature_dimen,
            control_qubit=self.feature_dimen - 1,
        )

        return circ

    def measure_circ(self, circ):

        # Append measurement gates to the circuit
        qr = QuantumRegister(self.feature_dimen + 1)
        cr = ClassicalRegister(self.feature_dimen + 1)

        meas_circ = QuantumCircuit(qr, cr)
        meas_circ.measure(
            [i for i in range(self.feature_dimen + 1)],
            [i for i in range(self.feature_dimen + 1)],
        )
        meas_circ = meas_circ.compose(circ, range(self.feature_dimen + 1), front=True)

        return meas_circ

    def image_encoding(self, measure=True):

        qr = QuantumRegister(self.feature_dimen + 1)
        circ = QuantumCircuit(qr)

        for i in range(self.feature_dimen):
            circ.h(i)

        num_theta = math.prod(self.image_size)
        for pixel in range(num_theta):
            pixel_pos_binary = "{0:0>2b}".format(pixel)

            # Embed pixel position on qubits
            circ.append(
                self.pixel_position(pixel_pos_binary),
                [qr[i] for i in range(self.feature_dimen)],
            )

            # Embed color information on qubits
            circ.append(
                self.color_info(pixel),
                [qr[i] for i in range(self.feature_dimen + 1)],
            )

            # Remove pixel position embedding
            circ.append(
                self.pixel_position(pixel_pos_binary),
                [qr[i] for i in range(self.feature_dimen)],
            )

        if measure:
            circ = self.measure_circ(circ)

        return circ

    @staticmethod
    def get_simulator_result(
            circ,
            backend: str='qasm_simulator',
            shots: int=1024,
            plot_counts=True,
    ):
        backend = Aer.get_backend(backend)
        job = execute(circ, backend=backend, shots=shots)
        results = job.result()
        counts = results.get_counts()

        if plot_counts:
            plot_histogram(counts)
            plt.show()

        return counts

    def qic(self):
        pass

    # TODO: Implement Geometric Transforms - G1, G2, and G3 from FRQI paper by Le, PQ. et al.
    def g_1_tranform(self, circ):
        # G1 transform for color only: color shift (S)
        # Apply U to color qubit.
        pass

    def g_2_tranform(self, circ):
        # G2 transform
        # Apply U to color qubit based on position qubit.
        pass

    def g_2_tranform(self, circ):
        # G3 transform
        # Apply U to color and position qubit.
        pass


# if __name__ == '__main__':
#     pixel_vals = np.zeros(4)
#     image_size = (2, 2)
#
#     circ = FRQI(image_size=image_size, color_vals=pixel_vals)
#     circ = circ.image_encoding()
#     circ.decompose().draw('mpl')
#     plt.show()
#     plt.savefig('foo.pdf')

