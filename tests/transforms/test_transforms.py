# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unit test for transforms"""

from __future__ import annotations
import numpy as np
import pytest
import torch
from pytest import raises
from piqture.transforms.transforms import MinMaxNormalization


class TestMinMaxNormalization:
    """Test class for MinMaxNormalization transform."""

    @pytest.mark.parametrize(
        "normalize_min, normalize_max", [(0, 1), (-np.pi, np.pi), (0, np.pi / 2)]
    )
    def test_repr(self, normalize_min, normalize_max):
        """Tests MinMaxNormalization class representation."""
        result = f"MinMaxNormalization(normalize_min={normalize_min}, normalize_max={normalize_max})"
        assert result == repr(MinMaxNormalization(normalize_min, normalize_max))

    @pytest.mark.parametrize(
        "normalize_min, normalize_max",
        [(None, None), ({}, []), ("12", "abc"), (True, False)],
    )
    def test_min(self, normalize_min, normalize_max):
        """Tests the normalize_min inputs"""
        with raises(
            TypeError, match="The input normalize_max must be of the type int or float."
        ):
            _ = MinMaxNormalization(1, normalize_max)

        with raises(
            TypeError, match="The input normalize_min must be of the type int or float."
        ):
            _ = MinMaxNormalization(normalize_min, 2.3)

    @pytest.mark.parametrize(
        "normalize_min, normalize_max, x, output",
        [
            (0, 1, torch.Tensor([1, 2, 3, 4]), torch.Tensor([0, 0.3333, 0.6667, 1])),
            (
                -np.pi,
                np.pi,
                torch.Tensor([251, 252, 253, 254]),
                torch.Tensor([-np.pi, -np.pi / 3, np.pi / 3, np.pi]),
            ),
            (
                0,
                np.pi / 2,
                torch.Tensor([1.8, 2.1, 3.2, 4.5]),
                torch.Tensor([0, np.pi / 18, (7 * np.pi) / 27, np.pi / 2]),
            ),
        ],
    )
    def test_minmax_transform(self, normalize_min, normalize_max, x, output):
        """Tests the transform output."""
        transform = MinMaxNormalization(normalize_min, normalize_max)
        result = transform(x)
        assert torch.allclose(result, output, atol=1e-5, rtol=1e-4)
