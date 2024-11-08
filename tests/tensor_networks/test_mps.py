# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unit test for MPS class"""

from __future__ import annotations

from unittest import mock

import pytest
from pytest import raises
from qiskit.circuit import QuantumCircuit

from piqture.gates.two_qubit_unitary import TwoQubitUnitary
from piqture.tensor_networks import MPS


@pytest.fixture(name="mps_circuit")
def mps_circuit_fixture():
    """Fixture to replicate a real simple two-qubit unitary block."""

    # pylint: disable=duplicate-code
    def _mps_circuit(num_qubits, parameter_vector, parameterization):
        test_circuit = QuantumCircuit(num_qubits)

        parameterization_callable = {
            "real_simple": [TwoQubitUnitary().real_simple_block, 2],
            "complex_simple": [TwoQubitUnitary().complex_simple_block, 2],
            "real_general": [TwoQubitUnitary().real_general_block, 6],
            "complex_general": [TwoQubitUnitary().complex_general_block, 15],
        }

        mapper = parameterization_callable[parameterization]
        if num_qubits >= 2:
            test_circuit.compose(
                mapper[0](parameter_vector[: mapper[1]])[0],
                qubits=[0, 1],
                inplace=True,
            )
            if num_qubits >= 3:
                test_circuit.compose(
                    mapper[0](parameter_vector[mapper[1] : 2 * mapper[1]])[0],
                    qubits=[1, 2],
                    inplace=True,
                )
                if num_qubits >= 4:
                    test_circuit.compose(
                        mapper[0](parameter_vector[2 * mapper[1] : 3 * mapper[1]])[0],
                        qubits=[2, 3],
                        inplace=True,
                    )
        return test_circuit

    return _mps_circuit


class TestMPS:
    """Tests for MPS class"""

    @pytest.mark.parametrize("num_qubits", [8])
    def test_repr(self, num_qubits):
        """Tests the representation of the MPS class."""
        test_representation = f"MatrixProductState(num_qubits={num_qubits})"
        assert test_representation == repr(MPS(num_qubits))

    @pytest.mark.parametrize("num_qubits", [None, {"abc"}, "pqr"])
    def test_type_num_qubits(self, num_qubits):
        """Tests the type of input num_qubit."""
        with raises(TypeError, match="Input num_qubits must be of the type int."):
            _ = MPS(num_qubits)

    @pytest.mark.parametrize("num_qubits", [0, -20])
    def test_value_num_qubits(self, num_qubits):
        """Tests value of input num_qubits."""
        with raises(ValueError, match="Number of qubits cannot be zero or negative."):
            _ = MPS(num_qubits)

    @pytest.mark.parametrize("num_qubits", [8])
    def test_circuit_property(self, num_qubits):
        """Tests the MPS circuit initialization."""
        test_circuit = QuantumCircuit(num_qubits)
        assert test_circuit == MPS(num_qubits).circuit

    @pytest.mark.parametrize("num_qubits, complex_structure", [(4, False), (12, True)])
    def test_mps_simple(self, num_qubits, complex_structure):
        # pylint: disable=line-too-long
        """Tests the mps_backbone method call via the mps_simple function."""
        with mock.patch(
            "piqture.tensor_networks.mps.MPS.mps_backbone"
        ) as mock_mps_simple:
            with mock.patch(
                "piqture.gates.two_qubit_unitary.TwoQubitUnitary.simple_parameterization"
            ) as simple_parameterization:
                _ = MPS(num_qubits).mps_simple(complex_structure)
                mock_mps_simple.assert_called_once_with(
                    simple_parameterization, mock.ANY, complex_structure
                )

    @pytest.mark.parametrize("num_qubits, complex_structure", [(4, False), (20, True)])
    def test_mps_general(self, num_qubits, complex_structure):
        # pylint: disable=line-too-long
        """Tests the mps_backbone method call via the mps_general function."""
        with mock.patch(
            "piqture.tensor_networks.mps.MPS.mps_backbone"
        ) as mock_mps_general:
            with mock.patch(
                "piqture.gates.two_qubit_unitary.TwoQubitUnitary.general_parameterization"
            ) as general_parameterization:
                _ = MPS(num_qubits).mps_general(complex_structure)
                mock_mps_general.assert_called_once_with(
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

    # pylint: disable=R0917
    def test_mps_backbone(
        self,
        num_qubits,
        complex_structure,
        parameterization,
        mps_circuit,
        parameterization_mapper,
    ):
        # pylint: disable=too-many-arguments
        """Tests the mps_backbone circuit with real and complex parameterization."""
        parameterization_mapper_dict = parameterization_mapper(num_qubits)
        test_circuit = mps_circuit(
            num_qubits,
            parameterization_mapper_dict[parameterization][0],
            parameterization,
        )
        circuit = MPS(num_qubits).mps_backbone(
            parameterization_mapper_dict[parameterization][1],
            parameterization_mapper_dict[parameterization][0],
            complex_structure,
        )
        assert circuit == test_circuit
