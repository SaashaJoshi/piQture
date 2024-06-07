# _**piQture**_: QML Library for Image Processing

[//]: # (<p align="center">)

[//]: # (    <img src="graphics/QuIPL-svg.svg" alt="QIPL-logo" width="350"/>)

[//]: # (</p>)
<p>
    <img alt="Static Badge" src="https://img.shields.io/badge/license-Apache_2.0-yellow?label=license&color=yellow&link=https%3A%2F%2Fgithub.com%2FSaashaJoshi%2FpiQture%2FLICENSE">
    <img alt="Static Badge" src="https://img.shields.io/badge/build_status-passing-blue?link=https%3A%2F%2Fgithub.com%2FSaashaJoshi%2Fquantum-image-processing%2Factions">
    <img alt="Static Badge" src="https://img.shields.io/badge/release-v0.1-orange?label=release&color=orange&link=https%3A%2F%2Fgithub.com%2FSaashaJoshi%2FpiQture%2Fpiqture%2Fversion.txt">
    <a href='https://coveralls.io/github/SaashaJoshi/piQture?branch=main'><img src='https://coveralls.io/repos/github/SaashaJoshi/piQture/badge.svg?branch=main' alt='Coverage Status' /></a>
</p>

_**piQture**_ is an open-source Python toolkit designed to simplify the development, execution, and training of Quantum Machine Learning (QML) models tailored for image processing tasks. This library seamlessly integrates with the Qiskit SDK, providing a convenient and user-friendly workflow for leveraging the potential of quantum computing for advanced image processing.


## Getting Started

### Setup


Begin by creating a new Python environment or activating an existing one for working with the `piQture` library. You set up a Python virtual environment `venv` or a Conda environment and use `pip` or `conda` to install the `piQture` package.

Here's how you can create a conda environment and manage a Python environment:

```bash
# Create a new conda environment
conda create -n piqture_env python=3.x

# Activate the conda environment
conda activate piqture_env
```

### Installation

Once the Python environment is activated, the required `piQture` package can be installed using `pip`. You can install the latest version directly from PyPI.

```bash
pip install piqture
```

To create a development environment, and install `piQture` from source, you can refer to section `Installation from Source`.


### Installation from Source

To set up a development environment and install `piQture` from source, follow these steps:

1. Start by cloning the `piQture` repository from GitHub.

```bash
# Clone the GitHub repository.
git clone https://github.com/SaashaJoshi/piQture.git
```

2. Activate the Python environment and navigate to the `piQture` repository directory. Then, inside the Python environment, install the required dependencies from the `requirements.txt` configuration file.

```bash
# Install the required dependencies
pip install -r requirements.txt
```

3. Install `piQture` in editable mode to make changes to the source code.

```bash
# Install from source in editable mode
pip install -e .
```

Your development environment is set up, and `piQture` is installed from source. You can now start making changes to the code, running tests, and contributing to the project as a developer.



### First program with _piQture_

Let's build a Quantum Image Representation with the `Improved Novel Enhanced Quantum Representation (INEQR)` encoding method.

```python
# INEQR Encoding Method
import torch.utils.data
from piqture.data_loader.mnist_data_loader import load_mnist_dataset
from piqture.embeddings.image_embeddings.ineqr import INEQR

# Load MNIST dataset
train_dataset, test_dataset = load_mnist_dataset()

# Retrieve a single image from the dataset
image, label = train_dataset[0]
image_size = tuple(image.squeeze().size())

# Change pixel values from float to integer
pixel_vals = (image * 255).round().to(torch.uint8)
pixel_vals = pixel_vals.tolist()

embedding = INEQR(image_size, pixel_vals).ineqr()

# Display circuit.
embedding.draw()
```

### Further examples

Let's build a Quantum Convolutional Neural Network (QCNN) with Convolutional, Pooling, and Fully-Connected layers.

```python
from piqture.neural_networks.layers import (
    QuantumConvolutionalLayer,
    QuantumPoolingLayer2,
    FullyConnectedLayer,
)
from piqture.neural_networks import QCNN

# Initializing a QCNN circuit with given image dimensions.
image_dims = 4
qcnn_circuit = QCNN(image_dims)

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


## Contribution Guidelines

We welcome contributions! Whether you're a quantum enthusiast or a Python developer, your input is valuable. Check out our Contribution Guidelines to get started.

## Authors and Citation

Saasha Joshi

## License

This project is licensed under the Apache License - see the [LICENSE](https://github.com/SaashaJoshi/quantum-image-processing/blob/main/LICENSE) file for details.

## Acknowledgments
