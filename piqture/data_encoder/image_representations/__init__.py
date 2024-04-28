# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Quantum Image Representations (module: piqture.data_encoder.image_representations)
"""

from .frqi import FRQI
from .neqr import NEQR
from .ineqr import INEQR
from .mcrqi import MCRQI

__all__ = [
    "FRQI",
    "NEQR",
    "INEQR",
    "MCRQI",
]
