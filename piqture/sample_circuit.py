from piqture.tensor_network_circuits import TTN
from qiskit_ibm_provider import IBMProvider
# import matplotlib.pyplot as plt

if __name__ == "__main__":
    num_qubits = 4
    circuit = TTN(num_qubits).ttn_simple(complex_structure=False)
    # circuit.draw("mpl")
    # plt.show()

    provider = IBMProvider()
    print(provider.backend)
    backend = provider.get_backend("ibm_qasm_simulator")
    job = backend.run(circuit, shots=10)
    result = job.result()
    counts = result.get_counts()
    print(counts)
