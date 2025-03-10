import numpy as np
from scipy.special import factorial

from quantumsimulationlab.enums import CoherentOrientationEnum, GateTypeEnum, PauliEnum
from quantumsimulationlab.tools.pauli import pauli


def annihilation(hilbert_dimension: int):
    a = np.zeros((hilbert_dimension, hilbert_dimension), dtype=complex)
    for i in range(1, hilbert_dimension):
        a[i - 1, i] = np.sqrt(i)
    return a


def basis(space_size: int, selected_dimension: int):
    if selected_dimension > space_size:
        raise ValueError("The selected dimension is greater than the space size.")
    basis = np.zeros((space_size, 1), dtype=complex)
    basis[selected_dimension] = 1.0
    return basis


def coherent(hilbert_dimension: int, alpha, t: float, orientation):
    state = np.zeros((hilbert_dimension, 1), dtype=complex)
    match orientation:
        case CoherentOrientationEnum.UP:
            for n in range(hilbert_dimension):
                state[n] = (alpha(t) ** n / np.sqrt(factorial(n))) * np.exp(
                    -(abs(alpha(t)) ** 2) / 2
                )
            return state / np.linalg.norm(state)

        case CoherentOrientationEnum.DOWN:
            for n in range(hilbert_dimension):
                state[n] = ((-alpha(t)) ** n / np.sqrt(factorial(n))) * np.exp(
                    -(abs(-alpha(t)) ** 2) / 2
                )
            return state / np.linalg.norm(state)

        case _:
            raise ValueError("The orientation must use CoherentOrientationEnum.")


def density_matrix(state):
    density_matrix = np.outer(state, state.conj())
    trace = np.abs(np.trace(density_matrix))
    if not np.isclose(trace, 1.0, rtol=1e-10):
        print(f"Warning: initial density matrix trace = {trace}.")
    return density_matrix / trace


def gate_application(hilbert_dimension: int, density_matrix, gate_type):
    match gate_type:
        case GateTypeEnum.H:
            gate = pauli[PauliEnum.HADAMARD]
        case GateTypeEnum.Z:
            gate = pauli[PauliEnum.SIGMAZ]
        case GateTypeEnum.Id:
            gate = pauli[PauliEnum.IDENTITY]
        case _:
            raise ValueError("Invalid gate type. Choose 'H', 'I', or 'Z'.")
    full_gate = np.kron(gate, np.eye(hilbert_dimension))
    return full_gate @ density_matrix @ full_gate.conjugate().transpose()
