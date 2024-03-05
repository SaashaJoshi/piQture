# (C) Copyright SaashaJoshi 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Angle Encoder"""

from qiskit.circuit import QuantumCircuit, ParameterVector


def angle_encoding(img_dim: int) -> QuantumCircuit:
    """
    Embeds data using Qubit/Angle encoding for a
    single feature.
    Does not use QIR techniques.
    Complexity: O(1); requires 1 qubit for 1 feature.
    :return:
    """

    params = ParameterVector("img_data", img_dim)
    embedding = QuantumCircuit(img_dim)
    for i in range(img_dim):
        embedding.ry(params[i], i)

    return embedding
