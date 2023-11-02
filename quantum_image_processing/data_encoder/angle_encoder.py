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
