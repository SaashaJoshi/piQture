import mlflow
import numpy as np
import matplotlib.pyplot as plt
from qiskit_ibm_provider import IBMProvider
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit.primitives import BackendEstimator
from qiskit_machine_learning.neural_networks.estimator_qnn import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_algorithms.optimizers.cobyla import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from IPython.display import clear_output
from quantum_image_processing.data_loader.mnist_data_loader import load_mnist_data
from quantum_image_processing.data_encoder.angle_encoder import angle_encoding
from quantum_image_processing.neural_networks.layers import (
    QuantumConvolutionalLayer,
    QuantumPoolingLayer2,
    FullyConnectedLayer,
)
from quantum_image_processing.neural_networks import QCNN

global objective_func_vals


def callback_graph(weights, obj_func_eval):
    # clear_output(wait=False)
    objective_func_vals.append(obj_func_eval)


# Set Mlflow URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("qcnn_estimator")

with mlflow.start_run():
    seed = 100
    # Load data
    _, __, train, test = load_mnist_data()
    train_data = iter(train)
    train_img, train_labels = next(train_data)
    test_data = iter(test)
    test_img, test_label = next(test_data)

    # print(train_img.shape, train_labels.shape)
    # print("Length of data:", len(train_img), len(train_labels))

    train_labels = np.array(train_labels)
    for i in range(len(train_labels)):
        if train_labels[i] == 1:
            train_labels[i] = -1
        elif train_labels[i] == 7:
            train_labels[i] = 1

    train_img = np.array(train_img.reshape(-1, 4))
    test_img = np.array(test_img.reshape(-1, 4))

    # Log image data
    # ??

    # Build QCNN
    img_dim = 4
    qcnn_circ = QCNN(num_qubits=img_dim)
    mlflow.log_param("num_qubits", img_dim)
    mlflow.log_param("initial qcnn circuit data", qcnn_circ.circuit.data)

    # Data Embedding - Angle Embedding/Encoding
    embedding, feature_params = angle_encoding(img_dim)
    mlflow.log_param("Embedding circuit data", embedding.data)

    # CNN layers
    # Gathering parameters for layer objects.
    mera_params = {"layer_depth": 1, "mera_instance": 0, "complex_structure": False}
    convolutional_params = {"mera_args": mera_params}
    mlflow.log_params(mera_params)
    mlflow.log_params(convolutional_params)

    # Build QCNN circuit.
    qcnn_circuit = qcnn_circ.sequence(
        [
            (QuantumConvolutionalLayer, convolutional_params),
            (QuantumPoolingLayer2, {}),
            (FullyConnectedLayer, {})
        ]
    )
    final_circuit = embedding.compose(qcnn_circuit, qubits=range(img_dim))
    observable = SparsePauliOp(["IZIZ"])
    mlflow.log_param("final qcnn circuit data", final_circuit.data)
    mlflow.log_param("Observable", observable)

    options = {}
    # IBMProvider.save_account()
    provider = IBMProvider()
    # provider = IBMProvider('pinq-quebec-hub/ecole-dhiver/qml-workshop')
    ibm_simulator = provider.get_backend('ibmq_qasm_simulator')
    estimator = BackendEstimator(backend=ibm_simulator, options=options)
    estimator_qcnn = EstimatorQNN(
        estimator=estimator,
        circuit=final_circuit,
        observables=observable,
        input_params=feature_params.params,
        weight_params=qcnn_circuit.parameters,
    )
    mlflow.log_param("input_params", estimator_qcnn.input_params)
    mlflow.log_param("weight_params", estimator_qcnn.weight_params)

    # weights = algorithm_globals.random.random(estimator_qcnn.num_weights)
    initial_point = np.random.random((len(qcnn_circuit.parameters),))
    classifier = NeuralNetworkClassifier(
        neural_network=estimator_qcnn,
        optimizer=COBYLA(maxiter=1),
        callback=callback_graph,
        initial_point=initial_point,
    )
    mlflow.log_param("initial_point", classifier.initial_point)
    mlflow.log_param("optimizer_max_iterations", 20)

    objective_func_vals = []
    classifier.fit(train_img, train_labels)
    train_score = classifier.score(train_img, train_labels)
    mlflow.log_param("Train Score", train_score)
    mlflow.log_param("Loss", objective_func_vals)

    y_predict = classifier.predict(test_img)
    test_score = classifier.score(test_img, test_label)
    mlflow.log_param("Test Score", test_score)

    # plt.title("Objective function value against iteration")
    # plt.xlabel("Iteration")
    # plt.ylabel("Objective function value")
    # plt.plot(range(len(objective_func_vals)), objective_func_vals)
    # plt.show()
