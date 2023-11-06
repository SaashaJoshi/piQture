"""
Gate implementations, including unitary gates and their alternative parameterization.
(module: quantum_image_processing.gates)
"""

from .unitary_block import Unitary
from .two_qubit_unitary import TwoQubitUnitary

__all__ = [
    "Unitary",
    "TwoQubitUnitary",
]
