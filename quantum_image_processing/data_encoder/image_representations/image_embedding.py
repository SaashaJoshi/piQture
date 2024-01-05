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

    def __init__(self, img_dims: tuple[int, ...], pixel_vals: list):
        if not all((isinstance(dims, int) for dims in img_dims)) or not isinstance(
            img_dims, tuple
        ):
            raise TypeError("Input img_dims must be of the type tuple[int, ...].")

        if not isinstance(pixel_vals, list):
            raise TypeError("Input pixel_vals must be of the type list.")
        pixel_vals = np.array(pixel_vals)

        # pixel_vals can be a multi-dimensional list.
        if len(pixel_vals.flatten()) != math.prod(img_dims):
            raise ValueError(
                f"No. of pixel values {len(pixel_vals)} must be equal to "
                f"the product of image dimensions {math.prod(img_dims)}."
            )

        for val in pixel_vals.flatten():
            if val < 0 or val > 255:
                raise ValueError(
                    "Pixel values cannot be less than 0 or greater than 255."
                )

        self.img_dims = img_dims
        self.pixel_vals = pixel_vals

    def validate_square_images(self):
        """
        Validates img_dims input for square images.
        """
        if len(set(self.img_dims)) > 1:
            raise ValueError(
                f"{self.__class__.__name__} supports square images only. "
                f"Input img_dims must have same dimensions."
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
    def pixel_value(self, pixel_pos: int):
        """
        Embeds pixel or color values on the qubits.

        Args:
            pixel_pos (int): takes as an input
            the pixel position.
        """
