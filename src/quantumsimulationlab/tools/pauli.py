import numpy as np

from quantumsimulationlab.enums import PauliEnum

def identity():
    return np.array([[1, 0], [0, 1]], dtype=complex)

def sigma_x():
    return np.array([[0, 1], [1, 0]], dtype=complex)

def sigma_y():
    return np.array([[0, -1j], [1j, 0]], dtype=complex)

def sigma_z():
    return np.array([[1, 0], [0, -1]], dtype=complex)

def hadamard():
    return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

pauli = {
    PauliEnum.IDENTITY: identity(),
    PauliEnum.SIGMAX: sigma_x(),
    PauliEnum.SIGMAY: sigma_y(),
    PauliEnum.SIGMAZ: sigma_z(),
    PauliEnum.HADAMARD: hadamard()
}