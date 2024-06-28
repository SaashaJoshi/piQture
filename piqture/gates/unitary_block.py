# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Unitary Gate class"""

from __future__ import annotations

from abc import ABC, abstractmethod


class UnitaryBlock(ABC):
    """
    Implements a unitary block with real and complex implementations.
    This block can be implemented in 3 ways, as mentioned by Grant et al. (2018)
        - simple, general and auxiliary gate implementations.

    These implementations are called alternative parameterizations.
    """

    @abstractmethod
    def simple_parameterization(
        self,
        parameter_vector: list,
        complex_structure: bool = True,
    ):
        """
        Used to build a unitary gate with real or complex
        simple parameterization.
        """

    @abstractmethod
    def general_parameterization(
        self,
        parameter_vector: list,
        complex_structure: bool = True,
    ):
        """
        Used to build a unitary gate with real or complex
        general parameterization.
        """

    @abstractmethod
    def auxiliary_parameterization(
        self,
        parameter_vector: list,
        complex_structure: bool = True,
    ):
        """
        Used to build a unitary gate parameterization
        with the help of an auxiliary qubit.
        """
