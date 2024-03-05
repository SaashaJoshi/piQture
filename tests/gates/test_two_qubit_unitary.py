# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unit test for TwoQubitUnitary class"""

from __future__ import annotations
import re
from unittest import mock
import pytest
from pytest import raises
from qiskit.circuit import ParameterVector
from quantum_image_processing.gates.two_qubit_unitary import TwoQubitUnitary


class TestTwoQubitUnitary:
    """Tests for TwoQubitUnitary class"""

    @pytest.mark.parametrize(
        "parameter_vector, complex_structure",
        [([23.9, 1.4], False)],
    )
    def test_validate_vectors(self, parameter_vector, complex_structure):
        """Tests the type of vectors in parameter_vector input."""
        with raises(
            TypeError,
            match="Vectors in parameter_vector must be of the type Parameter.",
        ):
            _ = TwoQubitUnitary().simple_parameterization(
                parameter_vector, complex_structure
            )

    @pytest.mark.parametrize(
        "parameter_vector, complex_structure",
        [(ParameterVector("theta", 2), None)],
    )
    def test_validate_complex_structure(self, parameter_vector, complex_structure):
        """Tests the type of complex_structure input."""
        with raises(
            TypeError,
            match=re.escape(
                pattern="Input complex_structure must be either True or False (bool)."
            ),
        ):
            _ = TwoQubitUnitary().simple_parameterization(
                parameter_vector, complex_structure
            )

    @pytest.mark.parametrize(
        "parameter_vector, complex_structure",
        [(ParameterVector("theta", 2), False)],
    )
    def test_real_simple_parameterization(
        self,
        parameter_vector,
        complex_structure,
    ):
        """Tests the complex simple parameterization method call."""
        with mock.patch(
            "quantum_image_processing.gates.two_qubit_unitary.TwoQubitUnitary.real_simple_block",
        ) as mock_real:
            _ = TwoQubitUnitary().simple_parameterization(
                parameter_vector, complex_structure
            )
            mock_real.assert_called_once_with(parameter_vector)

    @pytest.mark.parametrize(
        "parameter_vector, complex_structure",
        [(ParameterVector("theta", 2), True)],
    )
    def test_complex_simple_parameterization(self, parameter_vector, complex_structure):
        # pylint: disable=line-too-long
        """Tests the complex simple parameterization method call."""
        with mock.patch(
            "quantum_image_processing.gates.two_qubit_unitary.TwoQubitUnitary.complex_simple_block",
        ) as mock_complex:
            _ = TwoQubitUnitary().simple_parameterization(
                parameter_vector, complex_structure
            )
            mock_complex.assert_called_once_with(parameter_vector)

    @pytest.mark.parametrize(
        "parameter_vector, complex_structure",
        [(ParameterVector("theta", 2), False)],
    )
    def test_real_general_parameterization(self, parameter_vector, complex_structure):
        """Tests the real general parameterization method call."""
        with mock.patch(
            "quantum_image_processing.gates.two_qubit_unitary.TwoQubitUnitary.real_general_block",
        ) as mock_real:
            _ = TwoQubitUnitary().general_parameterization(
                parameter_vector, complex_structure
            )
            mock_real.assert_called_once_with(parameter_vector)

    @pytest.mark.parametrize(
        "parameter_vector, complex_structure",
        [(ParameterVector("theta", 2), True)],
    )
    def test_complex_general_parameterization(
        self, parameter_vector, complex_structure
    ):
        # pylint: disable=line-too-long
        """Tests the complex general parameterization method call."""
        with mock.patch(
            "quantum_image_processing.gates.two_qubit_unitary.TwoQubitUnitary.complex_general_block",
        ) as mock_complex:
            _ = TwoQubitUnitary().general_parameterization(
                parameter_vector, complex_structure
            )
            mock_complex.assert_called_once_with(parameter_vector)
