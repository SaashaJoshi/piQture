# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unit test for MERA class"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
from pytest import raises
from qiskit.circuit import ParameterVector, QuantumCircuit

from piqture.gates.two_qubit_unitary import TwoQubitUnitary
from piqture.tensor_networks import MERA


@pytest.fixture(name="mera_circuit")
def mera_circuit_fixture():
    """Fixture to replicate a real simple two-qubit unitary block."""

    # pylint: disable=duplicate-code
    def _mera_circuit(num_qubits, parameter_vector, parameterization):
        test_circuit = QuantumCircuit(num_qubits)
        parameterization_callable = {
            "real_simple": [TwoQubitUnitary().real_simple_block, 2],
            "complex_simple": [TwoQubitUnitary().complex_simple_block, 2],
            "real_general": [TwoQubitUnitary().real_general_block, 6],
            "complex_general": [TwoQubitUnitary().complex_general_block, 15],
        }

        mapper = parameterization_callable[parameterization]
        if num_qubits == 2:
            # D-block doesn't exist
            # U-block implementation
            test_circuit.compose(
                mapper[0](parameter_vector[: mapper[1]])[0],
                qubits=[0, 1],
                inplace=True,
            )
            test_circuit.ry(parameter_vector[mapper[1]], len(test_circuit.qubits) - 1)
        elif num_qubits == 3:
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
        elif num_qubits == 4:
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
        elif num_qubits == 5:
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

    @pytest.mark.parametrize("num_qubits, layer_depth", [(8, 2), (6, None)])
    def test_repr(self, num_qubits, layer_depth):
        """Tests the representation of the MERA class."""
        representation = repr(MERA(num_qubits, layer_depth))
        if not layer_depth:
            layer_depth = int(np.ceil(np.sqrt(num_qubits)))
        test_representation = (
            f"MultiScaleEntanglementRenormalizationAnsatz("
            f"num_qubits={num_qubits}, layer_depth={layer_depth}"
            f")"
        )
        assert test_representation == representation

    @pytest.mark.parametrize("num_qubits, layer_depth", [(3, -0.9), (2, "abc")])
    def test_layer_depth(self, num_qubits, layer_depth):
        """Tests the type of layer_depth input."""
        with raises(
            TypeError, match="The input layer_depth must be of the type int or None."
        ):
            _ = MERA(num_qubits, layer_depth)

    @pytest.mark.parametrize("num_qubits, layer_depth", [(3, 0)])
    def test_layer_depth_value(self, num_qubits, layer_depth):
        """Tests the value of layer_depth input."""
        with raises(ValueError, match="The input layer_depth must be at least 1."):
            _ = MERA(num_qubits, layer_depth)

    @pytest.mark.parametrize("num_qubits", [8])
    def test_circuit_property(self, num_qubits):
        """Tests the MERA circuit initialization."""
        test_circuit = QuantumCircuit(num_qubits)
        assert test_circuit == MERA(num_qubits).circuit

    @pytest.mark.parametrize(
        "num_qubits, layer_depth, complex_structure",
        [(4, 1, False), (20, None, True)],
    )
    def test_mera_simple(self, num_qubits, layer_depth, complex_structure):
        # pylint: disable=line-too-long
        """Tests the mera_backbone method call via the mera_simple function."""
        with mock.patch(
            "piqture.tensor_networks.mera.MERA.mera_backbone"
        ) as mock_mera_simple:
            with mock.patch(
                "piqture.gates.two_qubit_unitary.TwoQubitUnitary.simple_parameterization"
            ) as simple_parameterization:
                _ = MERA(num_qubits, layer_depth).mera_simple(complex_structure)
                mock_mera_simple.assert_called_once_with(
                    simple_parameterization, mock.ANY, complex_structure
                )

    @pytest.mark.parametrize(
        "num_qubits, layer_depth, complex_structure",
        [(4, 1, False), (12, None, True)],
    )
    def test_mera_general(self, num_qubits, layer_depth, complex_structure):
        # pylint: disable=line-too-long
        """Tests the mera_backbone method call via the mera_general function."""
        with mock.patch(
            "piqture.tensor_networks.mera.MERA.mera_backbone"
        ) as mock_mera_general:
            with mock.patch(
                "piqture.gates.two_qubit_unitary.TwoQubitUnitary.general_parameterization"
            ) as general_parameterization:
                _ = MERA(num_qubits, layer_depth).mera_general(complex_structure)
                mock_mera_general.assert_called_once_with(
                    general_parameterization, mock.ANY, complex_structure
                )

    @pytest.mark.parametrize(
        "num_qubits, layer_depth, complex_structure, parameterization",
        [
            (2, None, False, "real_general"),
            (3, None, False, "real_general"),
            (4, None, False, "real_general"),
            (2, None, False, "real_simple"),
            (3, None, False, "real_simple"),
            (4, None, False, "real_simple"),
            (2, None, True, "complex_general"),
            (3, None, True, "complex_general"),
            (4, None, True, "complex_general"),
            (5, None, False, "real_general"),
            (5, None, False, "real_simple"),
            (5, None, True, "complex_general"),
            (5, None, True, "complex_simple"),
        ],
    )
    def test_mera_backbone(
        self, num_qubits, layer_depth, complex_structure, parameterization, mera_circuit
    ):
        # pylint: disable=too-many-arguments
        """Tests the mera_backbone circuit with real and complex parameterization."""
        # Add test cases when layer_depth is not None.
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
            num_qubits, parameterization_mapper[parameterization][0], parameterization
        )
        circuit = MERA(num_qubits, layer_depth).mera_backbone(
            parameterization_mapper[parameterization][1],
            parameterization_mapper[parameterization][0],
            complex_structure,
        )
        assert circuit == test_circuit
