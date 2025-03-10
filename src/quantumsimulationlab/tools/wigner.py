import numpy as np
import qutip as qt
from qutip import Qobj


def wigner_block(density_matrix_block, xvec, pvec):
    return qt.wigner(density_matrix_block, xvec, pvec)


def wigner_assembly(density_matrix, xvec, pvec):
    size = density_matrix.shape[0]
    if size % 2 != 0:
        raise ValueError("The density matrix size must be even.")
    N = size // 2

    rho_uu = Qobj(density_matrix[0:N, 0:N])
    rho_ud = Qobj(density_matrix[0:N, N : 2 * N])
    rho_du = Qobj(density_matrix[N : 2 * N, 0:N])
    rho_dd = Qobj(density_matrix[N : 2 * N, N : 2 * N])
    wigner_uu = wigner_block(rho_uu, xvec, pvec)
    wigner_ud = wigner_block(rho_ud, xvec, pvec)
    wigner_du = wigner_block(rho_du, xvec, pvec)
    wigner_dd = wigner_block(rho_dd, xvec, pvec)
    wigner_sum = wigner_uu + wigner_dd + np.real(wigner_ud) + np.real(wigner_du)  # type: ignore
    return wigner_sum
