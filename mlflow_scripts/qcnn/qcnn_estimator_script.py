"""MLflow script for running a QCNN model."""

import math
import mlflow
import numpy as np
import matplotlib.pyplot as plt

# from qiskit_aer import AerSimulator
from qiskit_ibm_provider import IBMProvider
from qiskit.quantum_info import SparsePauliOp

# from qiskit.primitives import BackendEstimator
from qiskit_machine_learning.neural_networks.estimator_qnn import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_algorithms.optimizers.cobyla import COBYLA
from quantum_image_processing.data_loader.mnist_data_loader import load_mnist_data
from quantum_image_processing.data_encoder.angle_encoder import angle_encoding
from quantum_image_processing.neural_networks.layers import (
    QuantumConvolutionalLayer,
    QuantumPoolingLayer2,
    FullyConnectedLayer,
)
from quantum_image_processing.neural_networks import QCNN
from mlflow_scripts.py_wrapper import QuantumModel

global objective_func_vals

# Set Mlflow URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("qcnn_estimator")


def callback_graph(_, obj_func_eval):
    """Appends losses after each epoch to a list."""
    # clear_output(wait=False)
    objective_func_vals.append(obj_func_eval)


if __name__ == "__main__":

    with mlflow.start_run():
        # Load data
        _, __, train, test = load_mnist_data()
        train_data = iter(train)
        train_img, train_labels = next(train_data)
        test_data = iter(test)
        test_img, test_label = next(test_data)

        # print(train_img.shape, train_labels.shape)
        # print("Length of data:", len(train_img), len(train_labels))

        train_labels = np.array(train_labels)
        for i, _ in enumerate(train_labels):
            if train_labels[i] == 1:
                train_labels[i] = -1
            elif train_labels[i] == 7:
                train_labels[i] = 1

        train_img = np.array(train_img.reshape(-1, 4))
        test_img = np.array(test_img.reshape(-1, 4))

        # Log image data
        # ??

        # Build QCNN
        img_dim = (2, 2)
        num_qubits = int(math.prod(img_dim))
        qcnn_circ = QCNN(num_qubits)
        mlflow.log_param("image dimensions", img_dim)
        mlflow.log_param("num_qubits", num_qubits)
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
                (FullyConnectedLayer, {}),
            ]
        )
        final_circuit = embedding.compose(qcnn_circuit, qubits=range(num_qubits))
        observable = SparsePauliOp(["IZIZ"])
        mlflow.log_param("final qcnn circuit data", final_circuit.data)
        mlflow.log_param("Observable", observable)

        options = {}
        # IBMProvider.save_account()
        provider = IBMProvider()
        # provider = IBMProvider('pinq-quebec-hub/ecole-dhiver/qml-workshop')
        # ibm_simulator = provider.get_backend('ibmq_qasm_simulator')
        # estimator = BackendEstimator(backend=ibm_simulator, options=options)
        estimator_qcnn = EstimatorQNN(
            # estimator=estimator,
            circuit=final_circuit,
            observables=observable,
            input_params=feature_params.params,
            weight_params=qcnn_circuit.parameters,
        )
        estimator_qcnn.circuit.draw("mpl")
        plt.savefig("estimator_qcnn_circuit.png")
        mlflow.log_param("input_params", estimator_qcnn.input_params)
        mlflow.log_param("weight_params", estimator_qcnn.weight_params)

        # weights = algorithm_globals.random.random(estimator_qcnn.num_weights)
        initial_point = np.random.random((len(qcnn_circuit.parameters),))
        classifier = NeuralNetworkClassifier(
            neural_network=estimator_qcnn,
            optimizer=COBYLA(maxiter=10),
            callback=callback_graph,
            initial_point=initial_point,
        )
        mlflow.log_param("initial_point", classifier.initial_point)
        mlflow.log_param("optimizer_max_iterations", 20)

        objective_func_vals = []
        classifier.fit(train_img, train_labels)
        train_score = classifier.score(train_img, train_labels)
        mlflow.log_metric("Train Score", train_score)
        mlflow.log_param("Training Loss", objective_func_vals)

        # Save model
        python_classifier = QuantumModel(classifier)
        # classifier.save(file_name="qcnn_model.pkl")
        # classifier.save(file_name="qcnn_model.json")

        # Save the quantum circuit structure as an artifact
        mlflow.log_artifact("estimator_qcnn_circuit.png")
        mlflow.pyfunc.save_model(
            path="qcnn_trained_model", python_model=python_classifier
        )
        mlflow.pyfunc.log_model(
            "model",
            python_model=python_classifier,
            artifacts={"model_path": "qcnn_trained_model"},
        )

        # Predictions from trained model
        loaded_model = mlflow.pyfunc.load_model("qcnn_trained_model")
        y_predict = loaded_model.predict(test_img)
        mlflow.log_param("Prediction", y_predict)
        test_score = loaded_model.score(test_img, test_label)
        mlflow.log_metric("Test Score", test_score)

        # plt.title("Objective function value against iteration")
        # plt.xlabel("Iteration")
        # plt.ylabel("Objective function value")
        # plt.plot(range(len(objective_func_vals)), objective_func_vals)
        # plt.show()
