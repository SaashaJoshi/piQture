"""Unit test for MERA class"""
from __future__ import annotations
import math
from unittest import mock
import pytest
from pytest import raises
from qiskit.circuit import QuantumCircuit, ParameterVector
from quantum_image_processing.models.tensor_network_circuits.mera import MERA
from quantum_image_processing.gates.two_qubit_unitary import TwoQubitUnitary


@pytest.fixture(name="mera_circuit")
def mera_circuit_fixture():
    """Fixture to replicate a real simple two-qubit unitary block."""

    # pylint: disable=duplicate-code
    def _mera_circuit(img_dims, parameter_vector, parameterization):
        test_circuit = QuantumCircuit(int(math.prod(img_dims)))
        parameterization_callable = {
            "real_simple": [TwoQubitUnitary().real_simple_block, 2],
            "complex_simple": [TwoQubitUnitary().complex_simple_block, 2],
            "real_general": [TwoQubitUnitary().real_general_block, 6],
            "complex_general": [TwoQubitUnitary().complex_general_block, 15],
        }

        mapper = parameterization_callable[parameterization]
        if math.prod(img_dims) == 2:
            # D-block doesn't exist
            # U-block implementation
            test_circuit.compose(
                mapper[0](parameter_vector[: mapper[1]])[0],
                qubits=[0, 1],
                inplace=True,
            )
            test_circuit.ry(parameter_vector[mapper[1]], len(test_circuit.qubits) - 1)
        elif math.prod(img_dims) == 3:
            # D-block
            test_circuit.compose(
                mapper[0](parameter_vector[: mapper[1]])[0],
                qubits=[1, 2],
                inplace=True,
            )
            # U-block
            test_circuit.compose(
                mapper[0](parameter_vector[mapper[1] : 2 * mapper[1]])[0],
                qubits=[0, 1],
                inplace=True,
            )
            # No D-block now. Skip to U-block
            test_circuit.compose(
                mapper[0](parameter_vector[2 * mapper[1] : 3 * mapper[1]])[0],
                qubits=[1, 2],
                inplace=True,
            )
            test_circuit.ry(
                parameter_vector[3 * mapper[1]], len(test_circuit.qubits) - 1
            )
        elif math.prod(img_dims) == 4:
            # D-block
            test_circuit.compose(
                mapper[0](parameter_vector[: mapper[1]])[0],
                qubits=[1, 2],
                inplace=True,
            )
            # U-block
            test_circuit.compose(
                mapper[0](parameter_vector[mapper[1] : 2 * mapper[1]])[0],
                qubits=[0, 1],
                inplace=True,
            )
            test_circuit.compose(
                mapper[0](parameter_vector[2 * mapper[1] : 3 * mapper[1]])[0],
                qubits=[2, 3],
                inplace=True,
            )
            # U-block again.
            test_circuit.compose(
                mapper[0](parameter_vector[3 * mapper[1] : 4 * mapper[1]])[0],
                qubits=[1, 3],
                inplace=True,
            )
            test_circuit.ry(
                parameter_vector[4 * mapper[1]], len(test_circuit.qubits) - 1
            )
        elif math.prod(img_dims) == 5:
            # D-block
            test_circuit.compose(
                mapper[0](parameter_vector[: mapper[1]])[0],
                qubits=[1, 2],
                inplace=True,
            )
            test_circuit.compose(
                mapper[0](parameter_vector[mapper[1] : 2 * mapper[1]])[0],
                qubits=[3, 4],
                inplace=True,
            )
            # U-block
            test_circuit.compose(
                mapper[0](parameter_vector[2 * mapper[1] : 3 * mapper[1]])[0],
                qubits=[0, 1],
                inplace=True,
            )
            test_circuit.compose(
                mapper[0](parameter_vector[3 * mapper[1] : 4 * mapper[1]])[0],
                qubits=[2, 3],
                inplace=True,
            )
            # D-block
            test_circuit.compose(
                mapper[0](parameter_vector[4 * mapper[1] : 5 * mapper[1]])[0],
                qubits=[3, 4],
                inplace=True,
            )
            # U-block
            test_circuit.compose(
                mapper[0](parameter_vector[5 * mapper[1] : 6 * mapper[1]])[0],
                qubits=[1, 3],
                inplace=True,
            )
            # U-block again.
            test_circuit.compose(
                mapper[0](parameter_vector[6 * mapper[1] : 7 * mapper[1]])[0],
                qubits=[3, 4],
                inplace=True,
            )
            test_circuit.ry(
                parameter_vector[7 * mapper[1]], len(test_circuit.qubits) - 1
            )
        return test_circuit

    return _mera_circuit


class TestMERA:
    """Tests for MERA class"""

    @pytest.mark.parametrize("img_dims, layer_depth", [((3, 1), -0.9), ((2, 1), "abc")])
    def test_layer_depth(self, img_dims, layer_depth):
        """Tests the type of layer_depth input."""
        with raises(
            TypeError, match="The input layer_depth must be of the type int or None."
        ):
            _ = MERA(img_dims, layer_depth)

    @pytest.mark.parametrize("img_dims, layer_depth", [((3, 1), 0)])
    def test_layer_depth_value(self, img_dims, layer_depth):
        """Tests the value of layer_depth input."""
        with raises(ValueError, match="The input layer_depth must be at least 1."):
            _ = MERA(img_dims, layer_depth)

    @pytest.mark.parametrize("img_dims", [(2, 4)])
    def test_circuit_property(self, img_dims):
        """Tests the MERA circuit initialization."""
        test_circuit = QuantumCircuit(math.prod(img_dims))
        assert test_circuit.data == MERA(img_dims).circuit.data

    @pytest.mark.parametrize(
        "img_dims, layer_depth, complex_structure",
        [((2, 2), 1, False), ((4, 5), None, True)],
    )
    def test_mera_simple(self, img_dims, layer_depth, complex_structure):
        # pylint: disable=line-too-long
        """Tests the mera_backbone method call via the mera_simple function."""
        with mock.patch(
            "quantum_image_processing.models.tensor_network_circuits.mera.MERA.mera_backbone"
        ) as mock_mera_simple:
            with mock.patch(
                "quantum_image_processing.gates.two_qubit_unitary.TwoQubitUnitary.simple_parameterization"
            ) as simple_parameterization:
                _ = MERA(img_dims, layer_depth).mera_simple(complex_structure)
                mock_mera_simple.assert_called_once_with(
                    simple_parameterization, mock.ANY, complex_structure
                )

    @pytest.mark.parametrize(
        "img_dims, layer_depth, complex_structure",
        [((2, 2), 1, False), ((4, 5), None, True)],
    )
    def test_mera_general(self, img_dims, layer_depth, complex_structure):
        # pylint: disable=line-too-long
        """Tests the mera_backbone method call via the mera_general function."""
        with mock.patch(
            "quantum_image_processing.models.tensor_network_circuits.mera.MERA.mera_backbone"
        ) as mock_mera_general:
            with mock.patch(
                "quantum_image_processing.gates.two_qubit_unitary.TwoQubitUnitary.general_parameterization"
            ) as general_parameterization:
                _ = MERA(img_dims, layer_depth).mera_general(complex_structure)
                mock_mera_general.assert_called_once_with(
                    general_parameterization, mock.ANY, complex_structure
                )

    @pytest.mark.parametrize(
        "img_dims, layer_depth, complex_structure, parameterization",
        [
            ((1, 2), None, False, "real_general"),
            ((1, 3), None, False, "real_general"),
            ((1, 4), None, False, "real_general"),
            ((2, 1), None, False, "real_simple"),
            ((1, 3), None, False, "real_simple"),
            ((2, 2), None, False, "real_simple"),
            ((1, 2), None, True, "complex_general"),
            ((1, 3), None, True, "complex_general"),
            ((1, 4), None, True, "complex_general"),
            ((1, 5), None, False, "real_general"),
            ((5, 1), None, False, "real_simple"),
            ((1, 5), None, True, "complex_general"),
            ((1, 5), None, True, "complex_simple"),
        ],
    )
    def test_mera_backbone(
        self, img_dims, layer_depth, complex_structure, parameterization, mera_circuit
    ):
        # pylint: disable=too-many-arguments
        """Tests the mera_backbone circuit with real and complex parameterization."""
        # Add test cases when layer_depth is not None.
        num_qubits = int(math.prod(img_dims))
        parameterization_mapper = {
            "real_simple": [
                ParameterVector("test", 10 * num_qubits - 1),
                TwoQubitUnitary().simple_parameterization,
            ],
            "complex_simple": [
                ParameterVector("test", 10 * num_qubits - 1),
                TwoQubitUnitary().simple_parameterization,
            ],
            "real_general": [
                ParameterVector("test", 20 * num_qubits - 1),
                TwoQubitUnitary().general_parameterization,
            ],
            "complex_general": [
                ParameterVector("test", 15 * 10 * num_qubits - 1),
                TwoQubitUnitary().general_parameterization,
            ],
        }

        test_circuit = mera_circuit(
            img_dims, parameterization_mapper[parameterization][0], parameterization
        )
        circuit = MERA(img_dims, layer_depth).mera_backbone(
            parameterization_mapper[parameterization][1],
            parameterization_mapper[parameterization][0],
            complex_structure,
        )
        assert circuit.data == test_circuit.data
