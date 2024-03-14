# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unit test for TTN class"""

from __future__ import annotations
from unittest import mock
import pytest
from qiskit.circuit import QuantumCircuit
from piqture.tensor_network_circuits.ttn import TTN
from piqture.gates.two_qubit_unitary import TwoQubitUnitary


@pytest.fixture(name="ttn_circuit")
def ttn_circuit_fixture():
    """Fixture to replicate a real simple two-qubit unitary block."""

    # pylint: disable=duplicate-code
    def _ttn_circuit(num_qubits, parameter_vector, parameterization):
        test_circuit = QuantumCircuit(num_qubits)

        parameterization_callable = {
            "real_simple": [TwoQubitUnitary().real_simple_block, 2],
            "complex_simple": [TwoQubitUnitary().complex_simple_block, 2],
            "real_general": [TwoQubitUnitary().real_general_block, 6],
            "complex_general": [TwoQubitUnitary().complex_general_block, 15],
        }

        mapper = parameterization_callable[parameterization]
        if num_qubits == 2:
            test_circuit.compose(
                mapper[0](parameter_vector[: mapper[1]])[0],
                qubits=[0, 1],
                inplace=True,
            )
            test_circuit.ry(parameter_vector[mapper[1]], len(test_circuit.qubits) - 1)
        elif num_qubits == 3:
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
        elif num_qubits == 4:
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

    @pytest.mark.parametrize("num_qubits", [8])
    def test_repr(self, num_qubits):
        """Tests the representation of the TTN class."""
        test_representation = f"TreeTensorNetwork(num_qubits={num_qubits})"
        assert test_representation == repr(TTN(num_qubits))

    @pytest.mark.parametrize("num_qubits", [8])
    def test_circuit_property(self, num_qubits):
        """Tests the TTN circuit initialization."""
        test_circuit = QuantumCircuit(num_qubits)
        assert test_circuit == TTN(num_qubits).circuit

    @pytest.mark.parametrize("num_qubits, complex_structure", [(4, False), (12, True)])
    def test_ttn_simple(self, num_qubits, complex_structure):
        # pylint: disable=line-too-long
        """Tests the ttn_backbone method call via the ttn_simple function."""
        with mock.patch(
            "piqture.tensor_network_circuits.ttn.TTN.ttn_backbone"
        ) as mock_ttn_simple:
            with mock.patch(
                "piqture.gates.two_qubit_unitary.TwoQubitUnitary.simple_parameterization"
            ) as simple_parameterization:
                _ = TTN(num_qubits).ttn_simple(complex_structure)
                mock_ttn_simple.assert_called_once_with(
                    simple_parameterization, mock.ANY, complex_structure
                )

    @pytest.mark.parametrize("num_qubits, complex_structure", [(4, False), (20, True)])
    def test_ttn_general(self, num_qubits, complex_structure):
        # pylint: disable=line-too-long
        """Tests the ttn_backbone method call via the ttn_general function."""
        with mock.patch(
            "piqture.tensor_network_circuits.ttn.TTN.ttn_backbone"
        ) as mock_ttn_general:
            with mock.patch(
                "piqture.gates.two_qubit_unitary.TwoQubitUnitary.general_parameterization"
            ) as general_parameterization:
                _ = TTN(num_qubits).ttn_general(complex_structure)
                mock_ttn_general.assert_called_once_with(
                    general_parameterization, mock.ANY, complex_structure
                )

    @pytest.mark.parametrize(
        "num_qubits, complex_structure, parameterization",
        [
            (2, False, "real_general"),
            (3, False, "real_general"),
            (4, False, "real_general"),
            (2, False, "real_simple"),
            (3, False, "real_simple"),
            (4, False, "real_simple"),
            (2, True, "complex_simple"),
            (2, True, "complex_general"),
            (3, True, "complex_general"),
            (4, True, "complex_general"),
        ],
    )
    def test_ttn_backbone(
        self,
        num_qubits,
        complex_structure,
        parameterization,
        ttn_circuit,
        parameterization_mapper,
    ):
        # pylint: disable=too-many-arguments
        """Tests the ttn_backbone circuit with real and complex parameterization."""
        parameterization_mapper = parameterization_mapper(num_qubits)
        test_circuit = ttn_circuit(
            num_qubits, parameterization_mapper[parameterization][0], parameterization
        )
        circuit = TTN(num_qubits).ttn_backbone(
            parameterization_mapper[parameterization][1],
            parameterization_mapper[parameterization][0],
            complex_structure,
        )
        assert circuit == test_circuit
