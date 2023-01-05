# Imports
import math
import numpy as np
import matplotlib.pyplot as plt
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit, QuantumRegister, \
    ClassicalRegister, ParameterVector
from qiskit.visualization import plot_histogram


class FRQI:

    def __init__(self, image_size, color_vals):
        self.image_size = image_size
        self.color_vals = color_vals
        self.feature_dimen = int(np.sqrt(len(color_vals)))

    def pixel_position(self, pixel_pos_binary):

        circ = QuantumCircuit(self.feature_dimen)

        for index, value in enumerate(pixel_pos_binary):
            if value == '1':
                circ.x(index)

        return circ

    def color_info(self, pixel_pos):

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

    def image_encoding(self, measure=False):

        qr = QuantumRegister(self.feature_dimen + 1)
        circ = QuantumCircuit(qr)

        for i in range(self.feature_dimen):
            circ.h(i)

        num_theta = math.prod(self.image_size)
        for pixel in range(num_theta):
            pixel_pos_binary = "{0:2b}".format(pixel)

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

            # Remove pixel embedding
            circ.append(
                self.pixel_position(pixel_pos_binary),
                [qr[i] for i in range(self.feature_dimen)],
            )

        if measure:
            circ = self.measure_circ(circ)

        return circ

    def qic(self):
        pass

    def g_1_tranform(self, circ):
        # G1 transform for color only: color shift (S)
        # Apply U to color qubit.

        pass

# if __name__ == '__main__':
#     pixel_vals = np.zeros(784)
#     feature_dimen = 28
#     image_size = (28, 28)
#
#     circ = FRQI()
#     circ = circ.frqi_image_encoding(image_size, feature_dimen, pixel_vals)
    # circ.draw('mpl')
    # plt.savefig('foo.pdf')

