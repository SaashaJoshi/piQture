"""Abstract Base Class for Image Embedding"""
from abc import ABC, abstractmethod


class ImageEmbedding(ABC):
    """
    Abstract Base Class for embedding image data
    on a quantum circuit. It consists of two components:
    - Pixel position embedding
    - Pixel value (color) embedding
    """

    def __init__(self, img_dims, pixel_vals):
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
