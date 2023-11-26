"""Abstract Base Class for Image Embedding"""
from __future__ import annotations
from abc import ABC, abstractmethod


class ImageEmbedding(ABC):
    """
    Abstract Base Class for embedding image data
    on a quantum circuit. It consists of two components:
    - Pixel position embedding
    - Pixel value (color) embedding
    """

    def __init__(self, img_dims: tuple[int, ...], pixel_vals: list):
        if not all([isinstance(dims, int) for dims in img_dims]) or not isinstance(
            img_dims, tuple
        ):
            raise TypeError("Input img_dims must be of the type tuple[int, ...].")

        if not isinstance(pixel_vals, list):
            raise TypeError("Input pixel_vals must be of the type list.")

        self.img_dims = img_dims
        self.pixel_vals = pixel_vals

    @abstractmethod
    def pixel_position(self, pixel_pos_binary: str):
        """
        Embeds pixel positions on the qubits.

        Args:
            pixel_pos_binary (str): takes a binary
            representation of the pixel position.
        """
        return NotImplementedError

    @abstractmethod
    def pixel_value(self, pixel_pos: int):
        """
        Embeds pixel or color values on the qubits.

        Args:
            pixel_pos (int): takes as an input
            the pixel position.
        """
        return NotImplementedError
