"""Unit test for TwoQubitUnitary class"""
from __future__ import annotations
import re
import pytest
from pytest import raises
from qiskit.circuit import QuantumCircuit, ParameterVector
from quantum_image_processing.gates.two_qubit_unitary import TwoQubitUnitary


class TestTwoQubitUnitary:
    """Tests for TwoQubitUnitary class"""

    @pytest.mark.parametrize(
        "circuit, qubits, parameter_vector, complex_structure",
        [(None, [0, 1], ParameterVector("theta", 2), False)],
    )
    def test_validate_circuit(
        self, circuit, qubits, parameter_vector, complex_structure
    ):
        """Tests the type of circuit input."""
        with raises(
            TypeError, match="Input circuit is not of the type QuantumCircuit."
        ):
            _ = TwoQubitUnitary().simple_parameterization(
                circuit, qubits, parameter_vector, complex_structure
            )

    @pytest.mark.parametrize(
        "circuit, qubits, parameter_vector, complex_structure",
        [(QuantumCircuit(2), None, ParameterVector("theta", 2), False)],
    )
    def test_validate_qubits_type(
        self, circuit, qubits, parameter_vector, complex_structure
    ):
        """Tests the type of qubits input."""
        with raises(TypeError, match="Input qubits must be of the type list."):
            _ = TwoQubitUnitary().simple_parameterization(
                circuit, qubits, parameter_vector, complex_structure
            )

    @pytest.mark.parametrize(
        "circuit, qubits, parameter_vector, complex_structure",
        [(QuantumCircuit(2), [], ParameterVector("theta", 2), False)],
    )
    def test_validate_qubits_arg(
        self, circuit, qubits, parameter_vector, complex_structure
    ):
        """Tests the type of elements in qubits input."""
        with raises(ValueError, match="Input qubits list cannot be empty."):
            _ = TwoQubitUnitary().simple_parameterization(
                circuit, qubits, parameter_vector, complex_structure
            )

    @pytest.mark.parametrize(
        "circuit, qubits, parameter_vector, complex_structure",
        [(QuantumCircuit(2), [0, 1], [23.9, 1.4], False)],
    )
    def test_validate_parameter_vector(
        self, circuit, qubits, parameter_vector, complex_structure
    ):
        """Tests the type of parameter_vector input."""
        with raises(
            TypeError,
            match="Vectors in parameter_vectors must be of the type Parameter.",
        ):
            _ = TwoQubitUnitary().simple_parameterization(
                circuit, qubits, parameter_vector, complex_structure
            )

    @pytest.mark.parametrize(
        "circuit, qubits, parameter_vector, complex_structure",
        [(QuantumCircuit(2), [0, 1], ParameterVector("theta", 2), None)],
    )
    def test_validate_complex_structure(
        self, circuit, qubits, parameter_vector, complex_structure
    ):
        """Tests the type of complex_structure input."""
        with raises(
            TypeError,
            match=re.escape(
                pattern="Input complex_structure must be either True or False (bool)."
            ),
        ):
            _ = TwoQubitUnitary().simple_parameterization(
                circuit, qubits, parameter_vector, complex_structure
            )

    # def test_real_simple_block(self):
    #     """Tests the real simple parameterization circuit."""
    #     pass
