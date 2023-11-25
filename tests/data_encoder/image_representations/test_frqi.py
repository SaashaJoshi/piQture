"""Unit test for FRQI class"""
from __future__ import annotations
import re
import pytest
from pytest import raises
from quantum_image_processing.data_encoder.image_representations.frqi import FRQI


class TestFRQI:
    """Tests for FRQI image representation class"""

    @pytest.mark.parametrize(
        "img_dims, pixel_vals",
        [((2.5, 2.5), list(range(6))), ({"abc", "def"}, list(range(6)))],
    )
    def test_abc_type_image_dims(self, img_dims, pixel_vals):
        """Tests the type of img_dims input."""
        pattern = re.escape("Input img_dims must be of the type tuple[int, ...].")
        with raises(TypeError, match=pattern):
            _ = FRQI(img_dims, pixel_vals)

    @pytest.mark.parametrize(
        "img_dims, pixel_vals",
        [((2, 2), tuple(range(4))), ((2, 2), {1.0, 2.35, 4.5, 8.9})],
    )
    def test_abc_type_pixel_vals(self, img_dims, pixel_vals):
        """Tests the type of pixel_vals input."""
        with raises(TypeError, match=r"Input pixel_vals must be of the type list."):
            _ = FRQI(img_dims, pixel_vals)

    @pytest.mark.parametrize("img_dims, pixel_vals", [((2, 3), [range(6)])])
    def test_init_square_images(self, img_dims, pixel_vals):
        """Tests if the input img_dims represents a square image."""
        with raises(
            ValueError,
            match=r".* supports square images only. "
            r"Input img_dims must have same dimensions.",
        ):
            _ = FRQI(img_dims, pixel_vals)

    @pytest.mark.parametrize("img_dims, pixel_vals", [((2, 2), [1, 2, 3])])
    def test_init_len_pixel_values(self, img_dims, pixel_vals):
        """Tests if the length of pixel_vals input is the same as the image dimension."""
        with raises(
            ValueError,
            match=r"No. of pixel values \d must "
            r"be equal to the product of image dimensions \d.",
        ):
            _ = FRQI(img_dims, pixel_vals)
