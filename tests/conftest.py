# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""conftest.py"""


import math

import pytest
from qiskit.circuit import ParameterVector, QuantumCircuit

from piqture.gates.two_qubit_unitary import TwoQubitUnitary


@pytest.fixture(name="circuit_pixel_position")
def circuit_pixel_position_fixture():
    """Fixture for embedding pixel position."""

    def _circuit(img_dims, pixel_pos_binary):
        test_circuit = QuantumCircuit(int(math.prod(img_dims)))
        index = [index for index, val in enumerate(pixel_pos_binary) if val == "0"]
        if len(index):
            test_circuit.x(index)
        return test_circuit

    return _circuit


@pytest.fixture(name="parameterization_mapper")
def parameterization_mapper_fixture():
    """Fixture for parameterization mapper dictionary."""

    def _mapper(num_qubits):
        parameterization_mapper = {
            "real_simple": [
                ParameterVector("test", 2 * num_qubits - 1),
                TwoQubitUnitary().simple_parameterization,
            ],
            "complex_simple": [
                ParameterVector("test", 2 * num_qubits - 1),
                TwoQubitUnitary().simple_parameterization,
            ],
            "real_general": [
                ParameterVector("test", 6 * num_qubits - 1),
                TwoQubitUnitary().general_parameterization,
            ],
            "complex_general": [
                ParameterVector("test", 15 * num_qubits - 1),
                TwoQubitUnitary().general_parameterization,
            ],
        }
        return parameterization_mapper

    return _mapper
