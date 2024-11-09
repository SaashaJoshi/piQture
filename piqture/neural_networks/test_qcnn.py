# test file to check is the issue is resolved

import unittest
from qiskit.circuit import QuantumCircuit
from piqture.neural_networks.layers.base_layer import BaseLayer
from piqture.neural_networks.qcnn import QCNN

# Minimal valid subclass of BaseLayer for testing
class MockValidLayer(BaseLayer):
    def build_layer(self):
        # For simplicity, return the circuit and unmeasured_bits unchanged
        return self.circuit, self.unmeasured_bits

# Minimal regular class for invalid layer testing
class RegularClass:
    pass

class TestQCNNMinimal(unittest.TestCase):
    def setUp(self):
        self.num_qubits = 4
        self.qcnn = QCNN(num_qubits=self.num_qubits)
        self.valid_params = {}

    def test_valid_layer(self):
        """Test that a single valid layer subclass is accepted."""
        operations = [(MockValidLayer, self.valid_params)]
        try:
            self.qcnn.sequence(operations)
        except Exception as e:
            self.fail(f"Valid layer raised an unexpected exception: {e}")

    def test_regular_class_rejected(self):
        """Test that a regular class not inheriting from BaseLayer is rejected."""
        operations = [(RegularClass, self.valid_params)]
        with self.assertRaises(TypeError) as context:
            self.qcnn.sequence(operations)
        self.assertIn("must inherit from BaseLayer", str(context.exception))

if __name__ == '__main__':
    unittest.main()
