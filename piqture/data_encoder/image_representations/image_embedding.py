# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Abstract Base Class for Image Embedding"""

from __future__ import annotations
from abc import ABC, abstractmethod
import math
import numpy as np


class ImageEmbedding(ABC):
    """
    Abstract Base Class for embedding image data
    on a quantum circuit. It consists of two components:
    - Pixel position embedding
    - Pixel value (color) embedding
    """

    def __init__(
        self,
        img_dims: tuple[int, ...],
        pixel_vals: list[list],
        color_channels: int = 1,
    ):
        if not all((isinstance(dims, int) for dims in img_dims)) or not isinstance(
            img_dims, tuple
        ):
            raise TypeError("Input img_dims must be of the type tuple[int, ...].")
        self.validate_image_dimensions(img_dims)

        if not all(isinstance(pixels, list) for pixels in pixel_vals) or not isinstance(
            pixel_vals, list
        ):
            raise TypeError("Input pixel_vals must be of the type list[list].")
        pixel_vals = np.array(pixel_vals)

        self.color_channels = color_channels
        self.validate_number_pixel_lists(pixel_vals)
        self.validate_number_pixels(img_dims, pixel_vals)

        for val in pixel_vals.flatten():
            if val < 0 or val > 255:
                raise ValueError(
                    "Pixel values cannot be less than 0 or greater than 255."
                )

        self.img_dims = img_dims
        self.pixel_vals = pixel_vals

    def validate_image_dimensions(self, img_dims):
        """
        Validates img_dims input.

        Here, checks for square images. This
        function can be overriden.
        """
        if len(set(img_dims)) > 1:
            raise ValueError(
                f"{self.__class__.__name__} supports square images only. "
                f"Input img_dims must have same dimensions."
            )

    def validate_number_pixel_lists(self, pixel_vals):
        """
        Validates the number of pixel_lists in
        pixel_vals input.
        """
        if self.color_channels == 1:
            # For grayscale images.
            if len(pixel_vals) > 1:
                raise ValueError(
                    f"{self.__class__.__name__} supports grayscale images only. "
                    f"No. of pixel_lists in pixel_vals must be maximum 1."
                )
        else:
            if len(pixel_vals) > self.color_channels:
                raise ValueError(
                    f"{self.__class__.__name__} supports colored images. "
                    f"No. of pixel_lists in pixel_vals must be maximum "
                    f"{self.color_channels}."
                )

    @staticmethod
    def validate_number_pixels(img_dims, pixel_vals):
        """
        Validates the number of pixels in pixel_lists
        in pixel_vals input.
        """
        if all(len(pixel_lists) != math.prod(img_dims) for pixel_lists in pixel_vals):
            raise ValueError(
                f"No. of pixels ({[len(pixel_lists) for pixel_lists in pixel_vals]}) "
                f"in each pixel_lists in pixel_vals must be equal to the "
                f"product of image dimensions {math.prod(img_dims)}."
            )

    @abstractmethod
    def pixel_position(self, pixel_pos_binary: str):
        """
        Embeds pixel positions on the qubits.

        Args:
            pixel_pos_binary (str): takes a binary
            representation of the pixel position.
        """

    @abstractmethod
    def pixel_value(self, *args, **kwargs):
        """
        Embeds pixel or color values on the qubits.

        """
