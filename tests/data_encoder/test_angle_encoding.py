# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unit test for AngleEncoding class"""

from __future__ import annotations
import re
import numpy as np
import pytest
from pytest import raises
from piqture.data_encoder.angle_encoding import AngleEncoding


class TestAngleEncoding:
    """Tests for AngleEncoding class"""

    @pytest.mark.parametrize(
        "img_dims", [{1: 1}, [1, 2, 3], [[10, 23.5], [14.2, 98.6]]]
    )
    def test_img_dims(self, img_dims):
        """Tests type of img_dims input."""
        with raises(TypeError, match=r"Input img_dims must be of the type tuple."):
            _ = AngleEncoding(img_dims)

    # Why does it not raise error with boolean values?
    @pytest.mark.parametrize(
        "img_dims",
        [
            (None, None, False),
            (1.5, 3.4),
            (1.5, 3.4, 7.3),
            (np.pi, 3.14),
            (False, True),
        ],
    )
    def test_dims(self, img_dims):
        """Tests the type of dims in img_dims"""
        with raises(
            TypeError,
            match=re.escape("Input img_dims must be of the type tuple[int, ...]."),
        ):
            _ = AngleEncoding(img_dims)
