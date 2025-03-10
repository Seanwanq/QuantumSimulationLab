import numpy as np

from quantumsimulationlab.tools.tools import basis, coherent
from quantumsimulationlab.tools.pauli import pauli
from quantumsimulationlab.enums import (
    PreparedHamiltonianEnum,
    PauliEnum,
    CoherentOrientationEnum,
)


def origin(t: float = 0):
    return np.complex128(0.0 + 0.0j)


def time_dependent_cat_state(hilbert_dimension: int, alpha, t: float):
    coherent_plus = np.kron(
        basis(2, 0), coherent(hilbert_dimension, alpha, t, CoherentOrientationEnum.UP)
    )
    coherent_minus = np.kron(
        basis(2, 1), coherent(hilbert_dimension, alpha, t, CoherentOrientationEnum.DOWN)
    )
    cat_state = coherent_plus + coherent_minus
    return cat_state / np.linalg.norm(cat_state)


def prepared_initial(hilbert_dimension: int, alpha):
    spin_down = basis(2, 1)
    vacuum = coherent(hilbert_dimension, alpha, 0, CoherentOrientationEnum.UP)
    initial = np.kron(spin_down, vacuum)
    hadamard = pauli[PauliEnum.HADAMARD]
    identity = pauli[PauliEnum.IDENTITY]
    sigma_z = pauli[PauliEnum.SIGMAZ]
    sigma_x = pauli[PauliEnum.SIGMAX]
    combined = sigma_z @ hadamard
    total = np.kron(combined, np.eye(hilbert_dimension))
    final = total @ initial
    return final / np.linalg.norm(final)


def pre_hamiltonian(
    hilbert_dimension,
    omega_b,
    omega_r,
    a,
    adag,
    gamma_t,
    t,
    index,
    custom_hamiltonian=None,
):
    """Notice:
    a and adag should be prepared first.
    """
    match index:
        case PreparedHamiltonianEnum.RSIGMAX:
            hamiltonian = gamma_t(t) * np.kron(
                (
                    np.cos(omega_b * t) * pauli[PauliEnum.SIGMAX]
                    - np.sin(omega_b * t) * pauli[PauliEnum.SIGMAY]
                ),
                (a + adag),
            )
            return hamiltonian
        case PreparedHamiltonianEnum.RSIGMAZ:
            hamiltonian = gamma_t(t) * np.kron(pauli[PauliEnum.SIGMAZ], (a + adag))
            return hamiltonian
        case PreparedHamiltonianEnum.SIGMAX:
            hamiltonian = (
                omega_b * np.kron(pauli[PauliEnum.SIGMAZ], np.eye(hilbert_dimension))
                + omega_r * np.kron(np.eye(2), adag @ a)
                + gamma_t(t) * np.kron(pauli[PauliEnum.SIGMAX], (a + adag))
            )
            return hamiltonian
        case PreparedHamiltonianEnum.SIGMAZ:
            hamiltonian = (
                omega_b * np.kron(pauli[PauliEnum.SIGMAZ], np.eye(hilbert_dimension))
                + omega_r * np.kron(np.eye(2), adag @ a)
                + gamma_t(t) * np.kron(pauli[PauliEnum.SIGMAZ], (a + adag))
            )
            return hamiltonian
        case PreparedHamiltonianEnum.NONE:
            if custom_hamiltonian is None:
                raise ValueError(
                    "custom_hamiltonian cannot be None when ",
                    "PreparedHamiltonianEnum.NONE is selected.",
                )
            hamiltonian = custom_hamiltonian(
                hilbert_dimension=hilbert_dimension,
                omega_b=omega_b,
                omega_r=omega_r,
                a=a,
                adag=adag,
                gamma_t=gamma_t,
                t=t,
            )
            return hamiltonian
        case _:
            raise IndexError(
                "Invalid Hamiltonian index. Please use PrepareHamiltonianEnum."
            )
