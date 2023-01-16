'''This whole thing makes no sense to me now.'''
def neqr_dynamic_circuit():
    '''
    Implements an NEQR image representation with the help
    using dynamic circuits. Since we are going to start with
    a toy image dataset, we are going to encode a 2x2 image with
    2 qubits but measure dynamically after encoding each pixel
    position and its corresponding color information.

    Motive of doing this:
        1. Trying synamic circuits
        2. See if there is a difference in results when
            the circuit is run on a real hardware.
            Specifically, see if there is reduction in
            noise due to circuits being shortened dynamically.
        3. Does use of dynamic circuits iteratively mean that
            we could change probabilistic results to deterministic
            because if we create the same circuit again and run
            it, say 1000 times, in one shot, we are basically
            making the aspect/use of shots to go redundant. (No, this is wrong thinking.)
    '''

