# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""conftest.py"""

# pylint: disable=unused-import
from tests.embeddings.image_embeddings.test_frqi import (
    circuit_pixel_position_fixture,
)
from tests.tensor_networks.test_mps import parameterization_mapper_fixture
