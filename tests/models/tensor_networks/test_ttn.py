"""Unit test for TTN class"""
from __future__ import annotations

import math
import re
from unittest import mock
import pytest
from pytest import raises
from qiskit.circuit import QuantumCircuit, ParameterVector
from quantum_image_processing.models.tensor_network_circuits.ttn import TTN
from quantum_image_processing.gates.two_qubit_unitary import TwoQubitUnitary


@pytest.fixture(name="ttn_simple_circuit")
def ttn_simple_circuit_fixture():
    """Fixture to replicate a real simple two-qubit unitary block."""
    def _ttn_simple_circuit(img_dims, parameter_vector):
        test_circuit = QuantumCircuit(int(math.prod(img_dims)))
        if math.prod(img_dims) == 2:
            test_circuit.compose(
                TwoQubitUnitary().real_simple_block(parameter_vector[:2])[0],
                qubits=[0, 1],
                inplace=True,
            )
        elif math.prod(img_dims) == 3:
            test_circuit.compose(
                TwoQubitUnitary().real_simple_block(parameter_vector[:2])[0],
                qubits=[0, 1],
                inplace=True,
            )
            test_circuit.compose(
                TwoQubitUnitary().real_simple_block(parameter_vector[2:4])[0],
                qubits=[1, 2],
                inplace=True,
            )
        elif math.prod(img_dims) == 4:
            test_circuit.compose(
                TwoQubitUnitary().real_simple_block(parameter_vector[:2])[0],
                qubits=[0, 1],
                inplace=True,
            )
            test_circuit.compose(
                TwoQubitUnitary().real_simple_block(parameter_vector[2:4])[0],
                qubits=[2, 3],
                inplace=True,
            )
            test_circuit.compose(
                TwoQubitUnitary().real_simple_block(parameter_vector[4:6])[0],
                qubits=[1, 3],
                inplace=True,
            )
        test_circuit.ry(parameter_vector[-1], len(test_circuit.qubits) - 1)
        return test_circuit

    return _ttn_simple_circuit


class TestTTN:
    """Tests for TTN class"""

    @pytest.mark.parametrize("img_dims", [({"abc", "def"}), (2, 1.5), (None, None)])
    def test_img_dims(self, img_dims):
        """Tests the type of img_dims input."""
        with raises(
            TypeError,
            match=re.escape("Input img_dims must be of the type tuple[int, int]."),
        ):
            _ = TTN(img_dims)

    @pytest.mark.parametrize("img_dims", [(-3, 1), (2, 0)])
    def test_num_qubits(self, img_dims):
        """Tests the product of img_dims."""
        with raises(ValueError, match="Image dimensions cannot be zero or negative."):
            _ = TTN(img_dims)

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
        """Tests the ttn_backbone mehtod call via the ttn_simple function."""
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
        ) as mock_ttn_simple:
            with mock.patch(
                "quantum_image_processing.gates.two_qubit_unitary.TwoQubitUnitary.general_parameterization"
            ) as general_parameterization:
                _ = TTN(img_dims).ttn_general(complex_structure)
                mock_ttn_simple.assert_called_once_with(
                    general_parameterization, mock.ANY, complex_structure
                )

    @pytest.mark.parametrize(
        "img_dims, complex_structure",
        [((1, 3), False), ((2, 2), False), ((1, 2), False)],
    )
    def test_ttn_backbone(self, img_dims, complex_structure, ttn_simple_circuit):
        """
        Tests the ttn_backbone circuit.
        Currently, tests only for simple real parameterization, i.e. complex_structure = False.
        """
        parameter_vector = ParameterVector(
            "test", length=2 * int(math.prod(img_dims)) - 1
        )
        test_circuit = ttn_simple_circuit(img_dims, parameter_vector)
        circuit = TTN(img_dims).ttn_backbone(
            TwoQubitUnitary().simple_parameterization,
            parameter_vector,
            complex_structure,
        )
        assert circuit.data == test_circuit.data
