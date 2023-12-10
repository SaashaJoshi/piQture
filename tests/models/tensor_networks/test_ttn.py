"""Unit test for TTN class"""
from __future__ import annotations

import math
from unittest import mock
import pytest
from qiskit.circuit import QuantumCircuit, ParameterVector
from quantum_image_processing.models.tensor_network_circuits.ttn import TTN
from quantum_image_processing.gates.two_qubit_unitary import TwoQubitUnitary


@pytest.fixture(name="ttn_circuit")
def ttn_circuit_fixture():
    """Fixture to replicate a real simple two-qubit unitary block."""

    # pylint: disable=duplicate-code
    def _ttn_circuit(img_dims, parameter_vector, parameterization):
        test_circuit = QuantumCircuit(int(math.prod(img_dims)))

        parameterization_callable = {
            "real_simple": [TwoQubitUnitary().real_simple_block, 2],
            "real_general": [TwoQubitUnitary().real_general_block, 6],
            "complex_general": [TwoQubitUnitary().complex_general_block, 15],
        }

        mapper = parameterization_callable[parameterization]
        if math.prod(img_dims) == 2:
            test_circuit.compose(
                mapper[0](parameter_vector[: mapper[1]])[0],
                qubits=[0, 1],
                inplace=True,
            )
            test_circuit.ry(parameter_vector[mapper[1]], len(test_circuit.qubits) - 1)
        elif math.prod(img_dims) == 3:
            test_circuit.compose(
                mapper[0](parameter_vector[: mapper[1]])[0],
                qubits=[0, 1],
                inplace=True,
            )
            test_circuit.compose(
                mapper[0](parameter_vector[mapper[1] : 2 * mapper[1]])[0],
                qubits=[1, 2],
                inplace=True,
            )
            test_circuit.ry(
                parameter_vector[2 * mapper[1]], len(test_circuit.qubits) - 1
            )
        elif math.prod(img_dims) == 4:
            test_circuit.compose(
                mapper[0](parameter_vector[: mapper[1]])[0],
                qubits=[0, 1],
                inplace=True,
            )
            test_circuit.compose(
                mapper[0](parameter_vector[mapper[1] : 2 * mapper[1]])[0],
                qubits=[2, 3],
                inplace=True,
            )
            test_circuit.compose(
                mapper[0](parameter_vector[2 * mapper[1] : 3 * mapper[1]])[0],
                qubits=[1, 3],
                inplace=True,
            )
            test_circuit.ry(
                parameter_vector[3 * mapper[1]], len(test_circuit.qubits) - 1
            )
        return test_circuit

    return _ttn_circuit


class TestTTN:
    """Tests for TTN class"""

    @pytest.mark.parametrize("img_dims", [(2, 4)])
    def test_circuit_property(self, img_dims):
        """Tests the TTN circuit initialization."""
        test_circuit = QuantumCircuit(math.prod(img_dims))
        assert test_circuit.data == TTN(img_dims).circuit.data

    @pytest.mark.parametrize(
        "img_dims, complex_structure", [((2, 2), False), ((4, 5), True)]
    )
    def test_ttn_simple(self, img_dims, complex_structure):
        # pylint: disable=line-too-long
        """Tests the ttn_backbone method call via the ttn_simple function."""
        with mock.patch(
            "quantum_image_processing.models.tensor_network_circuits.ttn.TTN.ttn_backbone"
        ) as mock_ttn_simple:
            with mock.patch(
                "quantum_image_processing.gates.two_qubit_unitary.TwoQubitUnitary.simple_parameterization"
            ) as simple_parameterization:
                _ = TTN(img_dims).ttn_simple(complex_structure)
                mock_ttn_simple.assert_called_once_with(
                    simple_parameterization, mock.ANY, complex_structure
                )

    @pytest.mark.parametrize(
        "img_dims, complex_structure", [((2, 2), False), ((4, 5), True)]
    )
    def test_ttn_general(self, img_dims, complex_structure):
        # pylint: disable=line-too-long
        """Tests the ttn_backbone method call via the ttn_general function."""
        with mock.patch(
            "quantum_image_processing.models.tensor_network_circuits.ttn.TTN.ttn_backbone"
        ) as mock_ttn_general:
            with mock.patch(
                "quantum_image_processing.gates.two_qubit_unitary.TwoQubitUnitary.general_parameterization"
            ) as general_parameterization:
                _ = TTN(img_dims).ttn_general(complex_structure)
                mock_ttn_general.assert_called_once_with(
                    general_parameterization, mock.ANY, complex_structure
                )

    @pytest.mark.parametrize(
        "img_dims, complex_structure, parameterization",
        [
            ((1, 2), False, "real_general"),
            ((1, 3), False, "real_general"),
            ((1, 4), False, "real_general"),
            ((2, 1), False, "real_simple"),
            ((1, 3), False, "real_simple"),
            ((2, 2), False, "real_simple"),
            ((1, 2), True, "complex_general"),
            ((1, 3), True, "complex_general"),
            ((1, 4), True, "complex_general"),
        ],
    )
    def test_ttn_backbone(
        self, img_dims, complex_structure, parameterization, ttn_circuit
    ):
        """Tests the ttn_backbone circuit with real and complex parameterization."""
        parameterization_mapper = {
            "real_simple": [
                ParameterVector("test", length=2 * int(math.prod(img_dims)) - 1),
                TwoQubitUnitary().simple_parameterization,
            ],
            "real_general": [
                ParameterVector("test", 6 * int(math.prod(img_dims)) - 1),
                TwoQubitUnitary().general_parameterization,
            ],
            "complex_general": [
                ParameterVector("test", 15 * int(math.prod(img_dims)) - 1),
                TwoQubitUnitary().general_parameterization,
            ],
        }

        test_circuit = ttn_circuit(
            img_dims, parameterization_mapper[parameterization][0], parameterization
        )
        circuit = TTN(img_dims).ttn_backbone(
            parameterization_mapper[parameterization][1],
            parameterization_mapper[parameterization][0],
            complex_structure,
        )
        assert circuit.data == test_circuit.data
