import os
from qiskit_ibm_provider import IBMProvider
from qiskit.circuit import QuantumCircuit

# import matplotlib.pyplot as plt
# import click


def main():
    ibm_token = os.getenv('IBM_API_TOKEN')
    print(ibm_token)

    circuit = QuantumCircuit(2, 2)
    circuit.measure([0, 1], [0, 1])
    # circuit.draw("mpl")
    # plt.show()


    IBMProvider.save_account(token=ibm_token)
    provider = IBMProvider()
    print(provider.backend)
    backend = provider.get_backend("ibm_qasm_simulator")
    job = backend.run(circuit, shots=10)
    result = job.result()
    counts = result.get_counts()
    print(counts)


if __name__ == "__main__":
    main()
