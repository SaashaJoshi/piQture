
"""Unit test for BRQI class"""

from __future__ import annotations

import math
from unittest import mock

import numpy as np
import pytest
from pytest import raises
from qiskit.circuit import QuantumCircuit

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)
print(f"Updated Python path: {sys.path}")

# Now try to import BRQI
try:
    from piqture.embeddings.image_embeddings.brqi import BRQI
    print("Successfully imported BRQI")
except ImportError as e:
    print(f"Error importing BRQI: {e}")
from piqture.embeddings.image_embeddings.brqi import BRQI


MAX_COLOR_INTENSITY = 255


@pytest.fixture(name="brqi_pixel_value")
def brqi_pixel_value_fixture():
    """Fixture for embedding pixel values."""

    def _circuit(img_dims, pixel_val, color_qubits):
        feature_dim = int(np.ceil(np.log2(math.prod(img_dims))))
        pixel_val_bin = f"{int(pixel_val):0>{color_qubits}b}"
        test_circuit = QuantumCircuit(feature_dim + color_qubits)

        # Add gates to test_circuit
        for index, color in enumerate(pixel_val_bin):
            if color == "1":
                test_circuit.mcx(list(range(feature_dim)), feature_dim + index)
        return test_circuit

    return _circuit


class TestBRQI:
    """Tests for BRQI image representation class"""

    @pytest.mark.parametrize(
        "img_dims, pixel_vals, max_color_intensity",
        [
            ((2, 2), [list(range(251, 255))], 300),
            ((2, 2), [list(range(251, 255))], -20),
        ],
    )
    def test_max_color_intensity(self, img_dims, pixel_vals, max_color_intensity):
        """Tests value of maximum color intensity."""
        with raises(
            ValueError,
            match=r"Maximum color intensity cannot be less than \d or greater than \d.",
        ):
            _ = BRQI(img_dims, pixel_vals, max_color_intensity)

    @pytest.mark.parametrize(
        "img_dims, pixel_vals, max_color_intensity",
        [((2, 2), [list(range(251, 255))], MAX_COLOR_INTENSITY)],
    )
    def test_circuit_property(self, img_dims, pixel_vals, max_color_intensity):
        """Tests the BRQI circuits initialization."""
        feature_dim = int(np.ceil(np.log2(math.prod(img_dims))))
        color_qubits = int(np.ceil(np.log2(max_color_intensity + 1)))
        test_circuit = QuantumCircuit(feature_dim + color_qubits)
        assert BRQI(img_dims, pixel_vals).circuit == test_circuit

    @pytest.mark.parametrize(
        "img_dims, pixel_vals, max_color_intensity",
        [((2, 2), [list(range(235, 239))], MAX_COLOR_INTENSITY)],
    )
    def test_pixel_value(
        self, img_dims, pixel_vals, max_color_intensity, brqi_pixel_value
    ):
        """Tests the circuit received after pixel value embedding."""
        brqi_object = BRQI(img_dims, pixel_vals, max_color_intensity)
        color_qubits = int(np.ceil(np.log2(max_color_intensity + 1)))
        feature_dim = int(np.ceil(np.log2(math.prod(img_dims))))
        mock_circuit = QuantumCircuit(feature_dim + color_qubits)

        for _, pixel_val in enumerate(pixel_vals[0]):
            mock_circuit.clear()
            test_circuit = brqi_pixel_value(img_dims, pixel_val, color_qubits)

            with mock.patch(
                "piqture.embeddings.image_embeddings.brqi.BRQI.circuit",
                new_callable=lambda: mock_circuit,
            ):
                brqi_object.pixel_value(color_byte=f"{pixel_val:0>{color_qubits}b}")
                assert mock_circuit == test_circuit

    @pytest.mark.parametrize(
        "img_dims, pixel_vals, max_color_intensity",
        [((2, 2), [list(range(1, 5))], MAX_COLOR_INTENSITY)],
    )
    def test_brqi(
        self,
        img_dims,
        pixel_vals,
        max_color_intensity,
        brqi_pixel_value,
    ):
        """Tests the final BRQI circuit."""
        brqi_object = BRQI(img_dims, pixel_vals, max_color_intensity)
        color_qubits = int(np.ceil(np.log2(max_color_intensity + 1)))
        feature_dim = int(np.ceil(np.log2(math.prod(img_dims))))
        mock_circuit = QuantumCircuit(feature_dim + color_qubits)

        test_circuit = QuantumCircuit(feature_dim + color_qubits)
        test_circuit.h(list(range(feature_dim)))
        for index, pixel_val in enumerate(pixel_vals[0]):
            pixel_pos_binary = f"{index:0>{feature_dim}b}"
            mock_circuit.clear()
            test_circuit.compose(
                brqi_pixel_value(img_dims, pixel_val, color_qubits), inplace=True
            )

        with mock.patch(
            "piqture.embeddings.image_embeddings.brqi.BRQI.circuit",
            new_callable=lambda: mock_circuit,
        ):
            brqi_object.brqi()
            assert mock_circuit == test_circuit

    @pytest.mark.parametrize(
        "img_dims, pixel_vals, max_color_intensity",
        [((2, 2), np.array([[1, 2], [3, 4]]), MAX_COLOR_INTENSITY)],
    )
    def test_brqi_with_numpy_array(self, img_dims, pixel_vals, max_color_intensity):
        """Tests BRQI with NumPy array input."""
        brqi_object = BRQI(img_dims, pixel_vals, max_color_intensity)
        circuit = brqi_object.brqi()
        
        # Check if the circuit is generated correctly
        assert isinstance(circuit, QuantumCircuit)
        assert circuit.num_qubits == int(np.ceil(np.log2(math.prod(img_dims)))) + int(np.ceil(np.log2(max_color_intensity + 1)))
