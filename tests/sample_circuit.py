from piqture.tensor_network_circuits.ttn import TTN
from qiskit_ibm_provider import IBMProvider
# import matplotlib.pyplot as plt
import click


def main():
    num_qubits = 4
    circuit = TTN(num_qubits).ttn_simple(complex_structure=False)
    # circuit.draw("mpl")
    # plt.show()

    # provider = IBMProvider(token="IBMQ_API_TOKEN")
    provider = IBMProvider()
    print(provider.backend)
    backend = provider.get_backend("ibm_qasm_simulator")
    job = backend.run(circuit, shots=10)
    result = job.result()
    counts = result.get_counts()
    print(counts)


if __name__ == "__main__":
    main()