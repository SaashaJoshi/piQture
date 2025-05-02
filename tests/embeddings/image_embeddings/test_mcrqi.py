"""Unit test for MCRQI class"""

from __future__ import annotations

import math
from unittest import mock

import numpy as np
import pytest
from pytest import raises
from qiskit.circuit import QuantumCircuit

from piqture.embeddings.image_embeddings.mcrqi import MCRQI


class TestMCRQI:
    """Tests for MCRQI image representation class"""

    @pytest.mark.parametrize("img_dims, pixel_vals", [((2, 2), [list(range(4))])])
    def test_circuit_init(self, img_dims, pixel_vals):
        """Tests MCRQI circuit initialization."""
        feature_dim = int(np.ceil(np.sqrt(math.prod(img_dims))))
        test_circuit = QuantumCircuit(feature_dim + 3)
        assert MCRQI(img_dims, pixel_vals).circuit == test_circuit
