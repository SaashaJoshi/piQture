"""Unit test for TwoQubitUnitary class"""
from __future__ import annotations
import pytest
import re
from pytest import raises
from qiskit.circuit import QuantumCircuit, ParameterVector, Qubit
from quantum_image_processing.gates.two_qubit_unitary import TwoQubitUnitary


class TestTwoQubitUnitary:
    """Tests for TwoQubitUnitary class"""

    @pytest.mark.parametrize(
        "circuit, qubits, parameter_vector, complex_structure",
        [(None, [0, 1], [12.8, 45.9], False)],
    )
    def test_validate_circuit(
        self, circuit, qubits, parameter_vector, complex_structure
    ):
        with raises(
            TypeError, match="Input circuit is not of the type QuantumCircuit."
        ):
            _ = TwoQubitUnitary().simple_parameterization(
                circuit, qubits, parameter_vector, complex_structure
            )

    @pytest.mark.parametrize(
        "circuit, qubits, parameter_vector, complex_structure",
        [(QuantumCircuit(2), None, [12.8, 45.9], False)],
    )
    def test_validate_qubits_type(
        self, circuit, qubits, parameter_vector, complex_structure
    ):
        with raises(TypeError, match="Input qubits must be of the type list."):
            _ = TwoQubitUnitary().simple_parameterization(
                circuit, qubits, parameter_vector, complex_structure
            )

    @pytest.mark.parametrize(
        "circuit, qubits, parameter_vector, complex_structure",
        [(QuantumCircuit(2), [], [12.8, 45.9], False)],
    )
    def test_validate_qubits_arg(
        self, circuit, qubits, parameter_vector, complex_structure
    ):
        with raises(ValueError, match="Input qubits list cannot be empty."):
            _ = TwoQubitUnitary().simple_parameterization(
                circuit, qubits, parameter_vector, complex_structure
            )

    @pytest.mark.parametrize(
        "circuit, qubits, parameter_vector, complex_structure",
        [(QuantumCircuit(2), [0, 1], [None, None], False)],
    )
    def test_validate_parameter_vector(
        self, circuit, qubits, parameter_vector, complex_structure
    ):
        with raises(
            TypeError, match="Vectors in parameters must be of the type Number."
        ):
            _ = TwoQubitUnitary().simple_parameterization(
                circuit, qubits, parameter_vector, complex_structure
            )

    @pytest.mark.parametrize(
        "circuit, qubits, parameter_vector, complex_structure",
        [(QuantumCircuit(2), [0, 1], [12.8, 45.9], None)],
    )
    def test_validate_complex_structure(
        self, circuit, qubits, parameter_vector, complex_structure
    ):
        with raises(
            TypeError,
            match=re.escape(
                pattern="Input complex_structure must be either True or False (bool)."
            ),
        ):
            _ = TwoQubitUnitary().simple_parameterization(
                circuit, qubits, parameter_vector, complex_structure
            )
