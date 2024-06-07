# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for Quantum Convolutional Layer"""

from __future__ import annotations

import re
from unittest import mock

import pytest
from pytest import raises
from qiskit.circuit import QuantumCircuit

from piqture.neural_networks.layers import QuantumConvolutionalLayer


class TestQuantumConvolutionalLayer:
    """Tests for Quantum Convolutional Layer class"""

    @pytest.mark.parametrize("num_qubits, mera_instance", [(3, 3.25), (5, "abc")])
    def test_type_mera_instance(self, num_qubits, mera_instance):
        """Tests the type of mera_instance value in mera_args dictionary input."""
        with raises(
            TypeError,
            match="The value corresponding to mera_instance key in mera_args "
            "dictionary input must be of the type int.",
        ):
            _ = QuantumConvolutionalLayer(
                num_qubits, mera_args={"mera_instance": mera_instance}
            )

    @pytest.mark.parametrize("num_qubits, mera_instance", [(2, -3), (3, 4)])
    def test_value_mera_instance(self, num_qubits, mera_instance):
        """Tests the range of mera_instance value in mera_args dictionary input."""
        with raises(
            ValueError,
            match=re.escape(
                "The value corresponding to mera_instance key in mera_args "
                "dictionary input must be in range(0, 2)."
            ),
        ):
            _ = QuantumConvolutionalLayer(
                num_qubits, mera_args={"mera_instance": mera_instance}
            )

    @pytest.mark.parametrize(
        "num_qubits, circuit, unmeasured_bits, mera_args",
        [
            (2, None, None, None),
            (None, QuantumCircuit(3, 3), None, {"mera_instance": 0}),
            (None, None, [2, 3, 4, 5], {"layer_depth": 1}),
            (3, QuantumCircuit(3, 3), None, {"complex_structure": True}),
            (4, None, [1, 2, 3], {"layer_depth": 1, "complex_structure": False}),
        ],
    )
    def test_build_layer(self, num_qubits, circuit, unmeasured_bits, mera_args):
        """Tests the build_layer method of QuantumConvolutionalLayer class."""
        qc_layer = QuantumConvolutionalLayer(
            num_qubits=num_qubits,
            circuit=circuit,
            unmeasured_bits=unmeasured_bits,
            mera_args=mera_args,
        )
        mock_circuit = QuantumCircuit(qc_layer.num_qubits)
        with mock.patch(
            "piqture.tensor_networks.mera.MERA.mera_backbone",
            return_value=mock_circuit,
        ) as mock_mera:
            _, _ = qc_layer.build_layer()
            mock_mera.assert_called_once()
