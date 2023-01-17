import matplotlib.pyplot as plt
import numpy as np
from qiskit.utils import algorithm_globals
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.algorithms.optimizers import ADAM
from quantum_image_processing.data_loader.mnist_data_loader import load_mnist_data


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

            for index, _ in enumerate(qubit_list[:-1]):
                block = self._apply_simple_block(qubits=[qubit_list[index], qubit_list[index + 1]])
                ttn_circ = ttn_circ.compose(block, range(self.img_dim))

            ttn_circ.ry(self.param_vector_copy[0], qubit_list[-1])
            # ttn_circ.measure(qubit_list[-1], [0])

        return ttn_circ

    def ttn_general(self):
        pass

    def ttn_with_aux(self):
        pass


def data_embedding():
    """
    Embeds data using Qubit/Angle encoding for a
    single feature.
    Does not use QIR techniques.
    Complexity: O(1); requires 1 qubit for 1 feature.
    :return:
    """

    params = ParameterVector('img_data', 4)
    embedding = QuantumCircuit(4)
    for i in range(4):
        embedding.ry(params[i], i)

    return embedding


if __name__ == "__main__":
    _, __, train, test = load_mnist_data()
    train_data = iter(train)
    train_img, train_labels = next(train_data)
    test_data = iter(test)
    test_img, test_label = next(test_data)

    print(train_img.shape, train_labels.shape)

    # flatten_img_dim = train_img[0].reshape(-1, 4)
    # params_list = flatten_img_dim.tolist()[0]
    # # print(params_list)

    train_img = np.array(train_img.reshape(-1, 4))
    test_img = np.array(test_img.reshape(-1, 4))
    print(train_img.shape, test_img.shape)

    embedding_circ = data_embedding()
    embedding_circ.barrier()
    ttn = TTN(img_dim=4).ttn_simple(complex_struct=False)
    circ = embedding_circ.compose(ttn, range(4))
    circ.decompose().decompose().draw("mpl")
    # plt.show()

    from qiskit.quantum_info import SparsePauliOp

    observable = SparsePauliOp.from_list([("Z" * 4, 1)])

    def callback_graph(weights, obj_func_eval):
        # clear_output(wait=True)
        objective_func_vals.append(obj_func_eval)
        plt.title("Objective function value against iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Objective function value")
        plt.plot(range(len(objective_func_vals)), objective_func_vals)
        plt.savefig("graph.pdf")
        plt.show()

    estimator_qnn = EstimatorQNN(
        circuit=circ,
        observables=observable,
        input_params=embedding_circ.parameters,
        weight_params=ttn.parameters,
    )

    estimator_qnn.forward(
        input_data=train_img,
        weights=algorithm_globals.random.random(estimator_qnn.num_weights),
    )

    estimator_classifier = NeuralNetworkClassifier(
        estimator_qnn,
        optimizer=ADAM(maxiter=10),
        callback=callback_graph,
    )

    objective_func_vals = []
    estimator_classifier.fit(train_img, np.array(train_labels))
    estimator_classifier.score(train_img, np.array(train_labels))

    test_predict = estimator_classifier.predict(test_img)
    print(test_label, test_predict)
