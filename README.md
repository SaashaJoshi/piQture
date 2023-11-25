# Quantum Image Processing Library (QIPL)

**Quantum Image Processing Library (QIPL)** is an open-source Python toolkit designed to simplify the development, execution, and training of Quantum Machine Learning (QML) models tailored for image processing tasks. This library seamlessly integrates with the Qiskit SDK, providing a convenient and user-friendly workflow integration for leveraging the potential of quantum computing for advanced image processing.

## Installation

### Dependencies

### Getting Started

Let's build a Quantum Convolutional Neural Network (QCNN) with Convolutional, Pooling, and Fully-Connected layers.

```python
from quantum_image_processing.models.neural_networks.convolutional.qcnn import (
    QCNN,
    QuantumConvolutionalLayer,
    QuantumPoolingLayer,
    FullyConnectedLayer,
)

# Initializing a QCNN circuit with given image dimensions.
image_dimensions = 4
qcnn_circuit = QCNN(image_dimensions)

# Gathering parameters for layer objects.
mera_params = {"layer_depth": 1, "mera_instance": 0, "complex_structure": False}
convolutional_params = {"mera_args": mera_parameters}

# Build QCNN circuit.
qcnn_circuit = qcnn_circuit.sequence(
  [
    (QuantumConvolutionalLayer, convolutional_params),
    (QuantumPoolingLayer, {}),
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

