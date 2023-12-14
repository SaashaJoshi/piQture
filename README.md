# Quantum Image Processing Library (QIPL)

<p align="center">
    <img src="graphics/QuIPL-svg.svg" alt="QIPL-logo" width="350"/>
</p>
<p align="center">
    <img alt="Static Badge" src="https://img.shields.io/badge/license-Apache_2.0-bright_green">
    <img alt="Static Badge" src="https://img.shields.io/badge/build_status-passing-blue?link=https%3A%2F%2Fgithub.com%2FSaashaJoshi%2Fquantum-image-processing%2Factions">
    <img alt="Static Badge" src="https://img.shields.io/badge/release-0.0.0-orange?link=https%3A%2F%2Fgithub.com%2FSaashaJoshi%2Fquantum-image-processing%2Factions">
    <img alt="Static Badge" src="https://img.shields.io/badge/coverage-91%25-yellow?link=https%3A%2F%2Fgithub.com%2FSaashaJoshi%2Fquantum-image-processing%2Factions">
</p>

**Quantum Image Processing Library (QIPL)** is an open-source Python toolkit designed to simplify the development, execution, and training of Quantum Machine Learning (QML) models tailored for image processing tasks. This library seamlessly integrates with the Qiskit SDK, providing a convenient and user-friendly workflow for leveraging the potential of quantum computing for advanced image processing.

## Installation

### Dependencies

### Getting Started

Let's build a Quantum Convolutional Neural Network (QCNN) with Convolutional, Pooling, and Fully-Connected layers.

```python
from quantum_image_processing.models.neural_networks.layers import (
    QuantumConvolutionalLayer,
    QuantumPoolingLayer2,
    FullyConnectedLayer,
)
from quantum_image_processing.models.neural_networks.qcnn import QCNN

# Initializing a QCNN circuit with given image dimensions.
image_dimensions = 4
qcnn_circuit = QCNN(image_dimensions)

# Gathering parameters for layer objects.
mera_params = {"layer_depth": 1, "mera_instance": 0, "complex_structure": False}
convolutional_params = {"mera_args": mera_params}

# Build QCNN circuit.
qcnn_circuit = qcnn_circuit.sequence(
    [
        (QuantumConvolutionalLayer, convolutional_params),
        (QuantumPoolingLayer2, {}),
        (FullyConnectedLayer, {})
    ]
)

# Display circuit.
qcnn_circuit.draw()
```

### Further examples

## Contribution Guidelines

We welcome contributions! Whether you're a quantum enthusiast or a Python developer, your input is valuable. Check out our Contribution Guidelines to get started.

## Authors and Citation

Saasha Joshi

## License

This project is licensed under the Apache License - see the [LICENSE](https://github.com/SaashaJoshi/quantum-image-processing/blob/main/LICENSE) file for details.

## Acknowledgments

