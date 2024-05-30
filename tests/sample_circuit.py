from qiskit_ibm_provider import IBMProvider
from qiskit.circuit import QuantumCircuit

# import matplotlib.pyplot as plt
# import click


def main():
    circuit = QuantumCircuit(2, 2)
    circuit.measure([0, 1], [0, 1])
    circuit.draw("mpl")
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
