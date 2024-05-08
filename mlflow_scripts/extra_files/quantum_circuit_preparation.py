"""MLflow script for running a QCNN model."""
import math
import click
import mlflow
from piqture.data_encoder.angle_encoding import AngleEncoding
from piqture.neural_networks import QCNN
from piqture.neural_networks.layers import (
    QuantumConvolutionalLayer,
    QuantumPoolingLayer2,
    FullyConnectedLayer,
)


@click.command()
@click.option("--img-dim", type=tuple)
@click.option("--conv-params", default=None, type=dict)
@click.option("--mera-params", default=None, type=dict)
def quantum_circuit_preparation(img_dim, conv_params, mera_params):
    with mlflow.start_run():
        # Data Embedding: Angle Encoding
        embedding = AngleEncoding(img_dim)
        mlflow.log_param("Embedding Parameters", embedding.circuit.parameters)

        # Model Selection: QCNN
        num_qubits = int(math.prod(img_dim))
        qcnn = QCNN(num_qubits)
        mlflow.log_param("Number of qubits", num_qubits)

        # FIXME: CHECK THIS
        # Build QCNN layers
        if conv_params is None and mera_params is not None:
            conv_params = {"mera_args": mera_params}

        qcnn_circuit = qcnn.sequence(
            [
                (QuantumConvolutionalLayer, conv_params),
                (QuantumPoolingLayer2, {}),
                (FullyConnectedLayer, {}),
            ]
        )
        mlflow.log_param("QCNN Circuit Parameters", qcnn_circuit.parameters)

        # Final Quantum Circuit
        final_circuit = embedding.compose(qcnn_circuit, qubits=range(num_qubits))
        mlflow.log_param("Final Circuit Data", final_circuit.data)


if __name__ == "__main__":
    quantum_circuit_preparation()