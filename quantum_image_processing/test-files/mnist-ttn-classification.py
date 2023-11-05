import matplotlib.pyplot as plt
import numpy as np
from qiskit.utils import algorithm_globals
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit.algorithms.optimizers import ADAM, COBYLA
from qiskit.quantum_info import SparsePauliOp
from quantum_image_processing.data_loader.mnist_data_loader import load_mnist_data
from quantum_image_processing.image_classification.tree_tensor_network_ttn import TTN


def data_embedding(img_dim):
    """
    Embeds data using Qubit/Angle encoding for a
    single feature.
    Does not use QIR techniques.
    Complexity: O(1); requires 1 qubit for 1 feature.
    :return:
    """

    params = ParameterVector("img_data", img_dim)
    embedding = QuantumCircuit(img_dim)
    for i in range(img_dim):
        embedding.ry(params[i], i)

    return embedding


if __name__ == "__main__":
    _, __, train, test = load_mnist_data()
    train_data = iter(train)
    train_img, train_labels = next(train_data)
    test_data = iter(test)
    test_img, test_label = next(test_data)

    print(train_img.shape, train_labels.shape)

    train_labels = np.array(train_labels)
    for i in range(len(train_labels)):
        if train_labels[i] == 9:
            train_labels[i] = -1
        elif train_labels[i] == 2:
            train_labels[i] = 1

    train_img = np.array(train_img.reshape(-1, 4))
    test_img = np.array(test_img.reshape(-1, 4))
    print(train_img.shape, test_img.shape)

    img_dim = 4
    embedding_circ = data_embedding(img_dim=img_dim)
    embedding_circ.barrier()
    ttn = TTN(img_dim=img_dim).ttn_simple(complex_struct=False)
    circ = embedding_circ.compose(ttn, range(img_dim))
    circ.decompose().decompose().draw("mpl")
    plt.show()

    observable = SparsePauliOp.from_list([("Z" + "I" * 3, 1)])
    # Shouldn't this be I*3 + Z (IIIZ) because we just measure the last qubit
    # Shit it would be the exact opposite since qiskit ordering of qubit is diff.
    # But would it be opposite really?

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
        optimizer=COBYLA(maxiter=1),
        callback=callback_graph,
    )

    objective_func_vals = []
    estimator_classifier.fit(train_img, np.array(train_labels))
    estimator_classifier.score(train_img, np.array(train_labels))

    test_predict = estimator_classifier.predict(test_img)
    # print(test_label, test_predict)

    test_label = np.array(test_label)
    for i in range(len(test_label)):
        if test_label[i] == 9:
            test_label[i] = -1
        elif test_label[i] == 2:
            test_label[i] = 1

    count = 0
    for i in range(len(test_predict)):
        if test_predict[i] == test_label[i]:
            count += 1
    print(count)
