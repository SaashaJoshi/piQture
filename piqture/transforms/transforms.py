# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Data transforms for Pytorch datasets."""

from typing import Union

import torch
from torch import Tensor


# pylint: disable=too-few-public-methods
class MinMaxNormalization:
    """Normalizes input values in range [min, max]."""

    def __init__(
        self, normalize_min: Union[int, float], normalize_max: Union[int, float]
    ):
        # Check if normalize_min and max are int or float.
        if not isinstance(normalize_max, (int, float)) or isinstance(
            normalize_max, bool
        ):
            raise TypeError("The input normalize_max must be of the type int or float.")
        if not isinstance(normalize_min, (int, float)) or isinstance(
            normalize_min, bool
        ):
            raise TypeError("The input normalize_min must be of the type int or float.")
        self.min = normalize_min
        self.max = normalize_max

    def __repr__(self):
        """MinMaxNormalization transform representation."""
        return (
            f"{__class__.__name__}(normalize_min={self.min}, normalize_max={self.max})"
        )

    def __call__(self, data: Tensor) -> Tensor:
        """Normalizes data to a range [min, max]."""
        return self.min + (
            (data - torch.min(data))
            * (self.max - self.min)
            / (torch.max(data) - torch.min(data))
        )
