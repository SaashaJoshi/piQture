"""Unit test for INEQR class"""
from __future__ import annotations
import math
from unittest import mock
import numpy as np
import pytest
# from pytest import raises
from qiskit.circuit import QuantumCircuit
from quantum_image_processing.data_encoder.image_representations.ineqr import INEQR

MAX_COLOR_INTENSITY = 255


@pytest.fixture(name="ineqr_pixel_value")
def ineqr_pixel_value_fixture():
    """Fixture for embedding pixel values."""

    def _circuit(img_dims, pixel_val, color_qubits):
        feature_dim = int(math.log(img_dims[1], 2)) + int(math.log(img_dims[0], 2))
        pixel_val_bin = f"{int(pixel_val):0>8b}"
        test_circuit = QuantumCircuit(feature_dim + color_qubits)

        # Add gates to test_circuit
        for index, color in enumerate(pixel_val_bin):
            if color == "1":
                test_circuit.mct(list(range(feature_dim)), feature_dim + index)
        return test_circuit

    return _circuit


class TestINEQR:
    """Tests for FRQI image representation class"""

    @staticmethod
    def setup_mock_circuit(img_dims, max_color_intensity):
        """Setups params for mock circuit."""
        color_qubits = int(np.ceil(math.log(max_color_intensity, 2)))
        feature_dims = int(math.log(img_dims[1], 2)) + int(math.log(img_dims[0], 2))
        mock_circuit = QuantumCircuit(feature_dims + color_qubits)

        return mock_circuit, feature_dims, color_qubits

    @pytest.mark.parametrize(
        "img_dims, pixel_vals, max_color_intensity",
        [((2, 4), [list(range(251, 255)), list(range(251, 255))], MAX_COLOR_INTENSITY)],
    )
    def test_circuit_property(self, img_dims, pixel_vals, max_color_intensity):
        """Tests the INEQR circuits initialization."""
        color_qubits = int(np.ceil(math.log(max_color_intensity, 2)))
        feature_dims = int(math.log(img_dims[1], 2)) + int(math.log(img_dims[0], 2))
        test_circuit = QuantumCircuit(feature_dims + color_qubits)
        assert INEQR(img_dims, pixel_vals).circuit == test_circuit

    @pytest.mark.parametrize(
        "img_dims, pixel_vals, max_color_intensity",
        [((2, 4), [list(range(251, 255)), list(range(251, 255))], MAX_COLOR_INTENSITY)],
    )
    def test_pixel_value(
        self, img_dims, pixel_vals, max_color_intensity, ineqr_pixel_value
    ):
        """Tests the circuit received after pixel value embedding."""
        ineqr_object = INEQR(img_dims, pixel_vals, max_color_intensity)
        mock_circuit, _, color_qubits = self.setup_mock_circuit(
            img_dims, max_color_intensity
        )

        for _, y_val in enumerate(pixel_vals):
            for _, x_val in enumerate(y_val):
                mock_circuit.clear()
                test_circuit = ineqr_pixel_value(img_dims, x_val, color_qubits)

                with mock.patch(
                    "quantum_image_processing.data_encoder.image_representations."
                    "ineqr.INEQR.circuit",
                    new_callable=lambda: mock_circuit,
                ):
                    ineqr_object.pixel_value(color_byte=f"{x_val:0>8b}")
                    assert mock_circuit == test_circuit

    @pytest.mark.parametrize(
        "img_dims, pixel_vals, max_color_intensity",
        [((2, 4), [list(range(1, 5)), list(range(1, 5))], MAX_COLOR_INTENSITY)],
    )
    def test_ineqr(
        self,
        img_dims,
        pixel_vals,
        max_color_intensity,
        circuit_pixel_position,
        ineqr_pixel_value,
    ):
        # pylint: disable=too-many-arguments
        """Tests the final INEQR circuit."""
        ineqr_object = INEQR(img_dims, pixel_vals, max_color_intensity)
        mock_circuit, feature_dims, color_qubits = self.setup_mock_circuit(
            img_dims, max_color_intensity
        )

        test_circuit = QuantumCircuit(feature_dims + color_qubits)
        test_circuit.h(list(range(feature_dims)))
        for y_index, y_val in enumerate(pixel_vals):
            for x_index, x_val in enumerate(y_val):
                pixel_pos_binary = (
                    f"{y_index:0>{int(math.log(img_dims[1], 2))}b}"
                    f"{x_index:0>{int(math.log(img_dims[0], 2))}b}"
                )
                mock_circuit.clear()
                test_circuit.compose(
                    circuit_pixel_position(img_dims, pixel_pos_binary), inplace=True
                )
                test_circuit.compose(
                    ineqr_pixel_value(img_dims, x_val, color_qubits), inplace=True
                )
                test_circuit.compose(
                    circuit_pixel_position(img_dims, pixel_pos_binary), inplace=True
                )

        with mock.patch(
            "quantum_image_processing.data_encoder.image_representations.ineqr.INEQR.circuit",
            new_callable=lambda: mock_circuit,
        ):
            ineqr_object.ineqr()
            assert mock_circuit == test_circuit
