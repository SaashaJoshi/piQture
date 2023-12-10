"""Unit test for MPS class"""
from __future__ import annotations

import math
import re
from unittest import mock
import pytest
from pytest import raises
from qiskit.circuit import QuantumCircuit, ParameterVector
from quantum_image_processing.models.tensor_network_circuits.mps import MPS
from quantum_image_processing.gates.two_qubit_unitary import TwoQubitUnitary


@pytest.fixture(name="mps_circuit")
def mps_circuit_fixture():
    """Fixture to replicate a real simple two-qubit unitary block."""

    def _mps_circuit(img_dims, parameter_vector, parameterization):
        test_circuit = QuantumCircuit(int(math.prod(img_dims)))

        parameterization_callable = {
            "real_simple": [TwoQubitUnitary().real_simple_block, 2],
            "real_general": [TwoQubitUnitary().real_general_block, 6],
            "complex_general": [TwoQubitUnitary().complex_general_block, 15],
        }

        mapper = parameterization_callable[parameterization]
        if math.prod(img_dims) >= 2:
            test_circuit.compose(
                mapper[0](parameter_vector[: mapper[1]])[0],
                qubits=[0, 1],
                inplace=True,
            )
            if math.prod(img_dims) >= 3:
                test_circuit.compose(
                    mapper[0](parameter_vector[mapper[1] : 2 * mapper[1]])[0],
                    qubits=[1, 2],
                    inplace=True,
                )
                if math.prod(img_dims) >= 4:
                    test_circuit.compose(
                        mapper[0](parameter_vector[2 * mapper[1] : 3 * mapper[1]])[0],
                        qubits=[2, 3],
                        inplace=True,
                    )
        return test_circuit

    return _mps_circuit


class TestMPS:
    """Tests for MPS class"""

    @pytest.mark.parametrize("img_dims", [({"abc", "def"}), (2, 1.5), (None, None)])
    def test_img_dims(self, img_dims):
        """Tests the type of img_dims input."""
        with raises(
            TypeError,
            match=re.escape("Input img_dims must be of the type tuple[int, int]."),
        ):
            _ = MPS(img_dims)

    @pytest.mark.parametrize("img_dims", [(-3, 1), (2, 0)])
    def test_num_qubits(self, img_dims):
        """Tests the product of img_dims."""
        with raises(ValueError, match="Image dimensions cannot be zero or negative."):
            _ = MPS(img_dims)

    @pytest.mark.parametrize("img_dims", [(2, 4)])
    def test_circuit_property(self, img_dims):
        """Tests the MPS circuit initialization."""
        test_circuit = QuantumCircuit(math.prod(img_dims))
        assert test_circuit.data == MPS(img_dims).circuit.data

    @pytest.mark.parametrize(
        "img_dims, complex_structure", [((2, 2), False), ((4, 5), True)]
    )
    def test_mps_simple(self, img_dims, complex_structure):
        # pylint: disable=line-too-long
        """Tests the mps_backbone method call via the mps_simple function."""
        with mock.patch(
            "quantum_image_processing.models.tensor_network_circuits.mps.MPS.mps_backbone"
        ) as mock_mps_simple:
            with mock.patch(
                "quantum_image_processing.gates.two_qubit_unitary.TwoQubitUnitary.simple_parameterization"
            ) as simple_parameterization:
                _ = MPS(img_dims).mps_simple(complex_structure)
                mock_mps_simple.assert_called_once_with(
                    simple_parameterization, mock.ANY, complex_structure
                )

    @pytest.mark.parametrize(
        "img_dims, complex_structure", [((2, 2), False), ((4, 5), True)]
    )
    def test_mps_general(self, img_dims, complex_structure):
        # pylint: disable=line-too-long
        """Tests the mps_backbone method call via the mps_general function."""
        with mock.patch(
            "quantum_image_processing.models.tensor_network_circuits.mps.MPS.mps_backbone"
        ) as mock_mps_general:
            with mock.patch(
                "quantum_image_processing.gates.two_qubit_unitary.TwoQubitUnitary.general_parameterization"
            ) as general_parameterization:
                _ = MPS(img_dims).mps_general(complex_structure)
                mock_mps_general.assert_called_once_with(
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
    def test_mps_backbone(
        self, img_dims, complex_structure, parameterization, mps_circuit
    ):
        """Tests the mps_backbone circuit with real and complex parameterization."""
        parameterization_mapper = {
            "real_simple": [
                ParameterVector("test", 2 * int(math.prod(img_dims)) - 2),
                TwoQubitUnitary().simple_parameterization,
            ],
            "real_general": [
                ParameterVector("test", 6 * int(math.prod(img_dims)) - 2),
                TwoQubitUnitary().general_parameterization,
            ],
            "complex_general": [
                ParameterVector("test", 15 * int(math.prod(img_dims)) - 2),
                TwoQubitUnitary().general_parameterization,
            ],
        }

        test_circuit = mps_circuit(
            img_dims, parameterization_mapper[parameterization][0], parameterization
        )
        circuit = MPS(img_dims).mps_backbone(
            parameterization_mapper[parameterization][1],
            parameterization_mapper[parameterization][0],
            complex_structure,
        )
        assert circuit.data == test_circuit.data
