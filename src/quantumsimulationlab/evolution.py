from quantumsimulationlab.tools import (
    annihilation,
    basis,
    gate_application,
    density_matrix,
)
from quantumsimulationlab.prepared import (
    pre_hamiltonian,
    prepared_initial,
    alpha,
    time_dependent_cat_state,
)
from quantumsimulationlab.enums.gate_type_enum import GateTypeEnum

import numpy as np
from tqdm import tqdm


class TimeDependentStateEvolution:
    def __init__(
        self,
        time_total,
        time_steps,
        omega_b,
        omega_r,
        gamma_t,
        prepared_hamiltonian_index,
        gate_frequency=0,
        kappa=0,
        custom_hamiltonian=None,
        hilbert_dimension=100,
        with_phase: bool = False,
    ):
        self.hilbert_dimension = hilbert_dimension
        self.time_total = time_total
        self.time_steps = time_steps
        self.dt = time_total / time_steps
        self.kappa = kappa
        self.omega_b = omega_b
        self.omega_r = omega_r
        self.gamma_t = gamma_t
        self.a = annihilation(self.hilbert_dimension)
        self.adag = self.a.conj().T
        self.state_plus = basis(2, 0)
        self.state_down = basis(2, 1)
        self.gate_frequency = gate_frequency
        self.with_phase: bool = with_phase
        self.prepared_hamiltonian_index = prepared_hamiltonian_index
        self.custom_hamiltonian = custom_hamiltonian

    def _lindblad_dissipator(self, t, density_matrix):
        identity = np.eye(2, dtype=complex)
        a = np.kron(identity, self.a)
        adag = a.conj().T
        if self.with_phase:
            phase = np.exp(-1j * t * self.omega_r)
            a = a * phase
            adag = adag * phase.conj()
        term1 = a @ density_matrix @ adag
        term2 = 0.5 * adag @ a @ density_matrix
        term3 = 0.5 * density_matrix @ adag @ a
        return term1 - term2 - term3

    def _hamiltonian_density_maxtrix_commutator(self, t, density_matrix):
        a = self.a
        adag = self.adag
        if self.with_phase:
            phase = np.exp(-1j * t * self.omega_r)
            a = a * phase
            adag = adag * phase.conj()
        hamiltonian = pre_hamiltonian(
            hilbert_dimension=self.hilbert_dimension,
            omega_b=self.omega_b,
            omega_r=self.omega_r,
            a=a,
            adag=adag,
            gamma_t=self.gamma_t,
            t=t,
            index=self.prepared_hamiltonian_index,
            custom_hamiltonian=self.custom_hamiltonian,
        )
        return hamiltonian @ density_matrix - density_matrix @ hamiltonian

    def simulation_evolution(self):
        last_gate_time = 0
        gate_period = 0
        gate_sequence = []
        current_gate_index = 0

        times = np.linspace(0, self.time_total, self.time_steps)
        density_matrix_initial = density_matrix(
            prepared_initial(self.hilbert_dimension, alpha)
        )
        # density_matrix_initial = density_matrix(time_dependent_cat_state(self.hilbert_dimension, alpha, 0))
        dt = self.dt
        density_matrix_history = [density_matrix_initial.copy()]

        if self.gate_frequency < 0:
            raise ValueError("Gate frequency must be greater than or equal to 0.")

        if self.gate_frequency > 0:
            gate_period = 1 / self.gate_frequency
            last_gate_time = 0
            gate_sequence = [GateTypeEnum.H, GateTypeEnum.Z]
            current_gate_index = 1

        def check_density_matrix(density_matrix, step):
            if np.any(np.isnan(density_matrix)):
                raise ValueError(f"NaN detected at step {step}.")

            trace = np.abs(np.trace(density_matrix))
            if not 0.99 < trace < 1.01:
                density_matrix = density_matrix / trace

            if not np.allclose(density_matrix, density_matrix.conj().T):
                raise ValueError(f"Non-Hermitian density matrix at step {step}.")

        for i in tqdm(range(self.time_steps)):
            t = times[i]

            if self.gate_frequency > 0:
                current_time = times[i]

                if current_time - last_gate_time >= gate_period:
                    gate_type = gate_sequence[current_gate_index]
                    density_matrix_i = gate_application(
                        self.hilbert_dimension, density_matrix_history[i], gate_type
                    )
                    current_gate_index = (current_gate_index + 1) % len(gate_sequence)

                    gate_type = gate_sequence[current_gate_index]
                    density_matrix_i = gate_application(
                        self.hilbert_dimension, density_matrix_history[i], gate_type
                    )
                    last_gate_time = current_time

            try:
                k1 = dt * (
                    -1j
                    * self._hamiltonian_density_maxtrix_commutator(
                        t, density_matrix_history[i]
                    )
                    + self.kappa
                    * self._lindblad_dissipator(t, density_matrix_history[i])
                )
                k2 = dt * (
                    -1j
                    * self._hamiltonian_density_maxtrix_commutator(
                        t + 0.5 * dt, density_matrix_history[i] + 0.5 * k1
                    )
                    + self.kappa
                    * self._lindblad_dissipator(
                        t + 0.5 * dt, density_matrix_history[i] + 0.5 * k1
                    )
                )
                k3 = dt * (
                    -1j
                    * self._hamiltonian_density_maxtrix_commutator(
                        t + 0.5 * dt, density_matrix_history[i] + 0.5 * k2
                    )
                    + self.kappa
                    * self._lindblad_dissipator(
                        t + 0.5 * dt, density_matrix_history[i] + 0.5 * k2
                    )
                )
                k4 = dt * (
                    -1j
                    * self._hamiltonian_density_maxtrix_commutator(
                        t + dt, density_matrix_history[i] + k3
                    )
                    + self.kappa
                    * self._lindblad_dissipator(t + dt, density_matrix_history[i] + k3)
                )
                density_matrix_i = (
                    density_matrix_history[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
                )
                check_density_matrix(density_matrix_i, i)
                density_matrix_history.append(density_matrix_i.copy())
            except ValueError as e:
                print(f"Simulation failed at step {i}, time {t}")
                print(f"Error: {str(e)}")
                print(f"Last valid trace: {np.trace(density_matrix_history[-1])}")
                break
        return density_matrix_history, times
