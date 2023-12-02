"""conftest.py"""
import math

# pylint: disable=unused-import
import pytest
import uuid
from qiskit.circuit import QuantumCircuit, ParameterVector, Parameter
from tests.data_encoder.image_representations.test_frqi import (
    circuit_pixel_position_fixture,
)


@pytest.fixture(name="real_simple_block")
def real_simple_block_fixture():
    """Fixture for real simple parameterization block."""

    def _real_block(parameter_vector) -> QuantumCircuit:
        block = QuantumCircuit(2)
        block.ry(parameter_vector[0], 0)
        block.ry(parameter_vector[1], 1)
        block.cx(0, 1)
        return block

    return _real_block
