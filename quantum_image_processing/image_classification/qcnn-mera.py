class QCNN_MERA:
    """
    Implements QCNN structure by Cong et al. (2019), replicating
    the architecture described by MERA - Multiscale Entanglement
    Renormalization Ansatz, given by Vidal et al. (2008).

    The decomposition of MERA architecture takes from the paper
    by Grant et al. (2018).
    """

    def __init__(self):
        pass

    def data_embedding(self):
        """
        Embeds data using Amplitude encoding/Threshold Encoding.
        Does not use QIR techniques.
        :return:
        """
        pass

    def conv_layer(self):
        pass

    def pooling_layer(self):
        """
        Non-dynamic implementation of pooling layer.
        Utilizes two-qubit gates.
        :return:
        """
        pass

    def dynamic_pooling_layer(self):
        """
        Uses a dynamic circuit instead of two-qubit gates.
        :return:
        """
        pass

    def fully_con_layer(self):
        pass

    def measurement(self):
        pass


def data_loading():
    pass
