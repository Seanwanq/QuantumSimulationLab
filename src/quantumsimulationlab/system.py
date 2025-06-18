from typing import Any
import numpy as np
import qutip as qt
from qutip import Qobj
import quantumsimulationlab.photonanalyzer as qp
from quantumsimulationlab.qndanalyzer import QNDAnalyzer
from quantumsimulationlab.tools import create_directory


class System:
    def __init__(
        self,
        initial_state: int,
        epsilon: float,
        A: float,
        coupling_type: str,
        dim_r: int = 30,
        dim_q: int = 2,
        omega_q: float = 1.0,
        omega_r: float = 501.0,
        kappa: float = 0.5,
        t_max: float = 25.0,
        nt: int = 1500,
        ntraj: int = 100,
        theta: float = (1.0 / 128.0) * np.pi,
        xvec: np.ndarray = np.linspace(-6, 6, 150),
        father_directory: str = "cache",
        show=True,
    ) -> None:
        self.initial_state: int = initial_state
        self.coupling_type: str = coupling_type
        self.dim_r: int = dim_r
        self.dim_q: int = dim_q
        self.omega_q: float = omega_q
        self.omega_r: float = omega_r
        self.w_r = self.omega_r
        self.kappa: float = kappa
        self.epsilon: float = epsilon
        self.t_max: float = t_max
        self.nt: int = nt
        self.ntraj: int = ntraj
        self.A: float = A
        self.show: bool = show
        self.theta: float = theta
        self.delta: float = abs(omega_q - omega_r)
        self.tlist: np.ndarray = np.linspace(0, t_max, nt)
        self.n_op: Qobj = qt.tensor(qt.qeye(dim_q), qt.num(dim_r))
        self.psi_q: Qobj = qt.basis(dim_q, initial_state)
        self.psi_r: Qobj = qt.basis(dim_r, 0)
        self.psi: Qobj = qt.tensor(self.psi_q, self.psi_r)
        self.a: Qobj = qt.tensor(qt.qeye(dim_q), qt.destroy(dim_r))
        self.adag: Qobj = self.a.dag()
        self.sigma_z: Qobj = qt.tensor(qt.sigmaz(), qt.qeye(dim_r))
        self.sigma_plus: Qobj = qt.tensor(qt.sigmap(), qt.qeye(dim_r))
        self.sigma_minus: Qobj = qt.tensor(qt.sigmam(), qt.qeye(dim_r))
        self.X_op: Qobj = self.a + self.adag
        self.Y_op: Qobj = 1j * (self.adag - self.a)
        self.e_ops: list[Qobj] = [self.n_op, self.X_op, self.Y_op]
        self.xvec: np.ndarray = xvec
        self.father_directory: str = father_directory
        self.directory_name: str = (
            coupling_type
            + "_kappa_"
            + str(kappa)
            + "_epsilon_"
            + str(epsilon)
            + "_A_"
            + str(A)
        )
        self.data_dir: str = father_directory + "/" + self.directory_name + "/data/"
        self.figures_dir: str = (
            father_directory + "/" + self.directory_name + "/figures/"
        )
        match coupling_type:
            case "transverse":
                self.args = {
                    "w_r": self.w_r,
                    "w_q": self.omega_q,
                    "gamma_func_transverse": self.gamma_func_transverse,
                }
                self.H = self.H_transverse()
            case "longitudinal":
                self.args = {
                    "w_r": self.w_r,
                    "gamma_func_longitudinal": self.gamma_func_longitudinal,
                }
                self.H = self.H_longitudinal()
            case "hybrid":
                self.args = {
                    "w_r": self.w_r,
                    "gamma_func_transverse": self.gamma_func_transverse,
                    "gamma_func_longitudinal": self.gamma_func_longitudinal
                }
                self.H = self.H_hybrid()
            case _:
                raise ValueError(
                    f"Invalid coupling type: {coupling_type}. Choose from {['transverse', 'longitudinal', 'hybrid']}."
                )

    def print_parameters(self) -> None:
        print(f"Initial state: {self.initial_state}")
        print(f"Coupling type: {self.coupling_type}")
        print(f"Dimension of the resonator: {self.dim_r}")
        print(f"Dimension of the qubit: {self.dim_q}")
        print(f"Qubit frequency: {self.omega_q}")
        print(f"Resonator frequency: {self.omega_r}")
        print(f"Kappa: {self.kappa}")
        print(f"Epsilon: {self.epsilon}")
        print(f"A: {self.A}")
        print(f"Theta: {self.theta}")
        print(f"Delta: {self.delta}")
        print(f"Time max: {self.t_max}")
        print(f"Number of time steps: {self.nt}")
        print(f"Number of trajectories: {self.ntraj}")
        print(self.args)

    def create_directories(self) -> None:
        create_directory(self.father_directory, self.directory_name)
        create_directory(self.father_directory + "/" + self.directory_name, "data")
        create_directory(self.father_directory + "/" + self.directory_name, "figures")

    def H_transverse(self) -> list[Any]:
        H0: Qobj = (self.delta / 2.0) * self.sigma_z + self.epsilon * (
            self.a + self.adag
        )
        H1: Qobj = self.sigma_minus * self.adag + self.sigma_plus * self.a  # type: ignore

        def gamma_coeff(t, args):
            return args["gamma_func_transverse"](t)

        H = [H0, [H1, gamma_coeff]]
        return H

    def H_longitudinal(self) -> list[Any]:
        H0: Qobj = self.epsilon * (self.a + self.adag)
        H1: Qobj = self.sigma_z * self.adag  # type: ignore
        H2: Qobj = self.sigma_z * self.a  # type: ignore

        def coeff1(t, args):
            return args["gamma_func_longitudinal"](t) * np.exp(1j * args["w_r"] * t)

        def coeff2(t, args):
            return args["gamma_func_longitudinal"](t) * np.exp(-1j * args["w_r"] * t)

        H = [H0, [H1, coeff1], [H2, coeff2]]

        return H

    def H_hybrid(self) -> list[Any]:
        H0 = (self.delta / 2.0) * self.sigma_z + self.epsilon * (self.a + self.adag)
        H1 = self.sigma_z * self.adag # type: ignore
        H2 = self.sigma_z * self.a # type: ignore
        H3 = self.sigma_minus * self.adag + self.sigma_plus * self.a # type: ignore

        def coeff1(t, args):
            return args['gamma_func_longitudinal'](t) * np.exp(1j * args['w_r'] * t)
        
        def coeff2(t, args):
            return args['gamma_func_longitudinal'](t) * np.exp(-1j * args['w_r'] * t)

        def gamma_coeff(t, args):
            return args['gamma_func_transverse'](t)

        H = [
            H0,
            [H1, coeff1],
            [H2, coeff2],
            [H3, gamma_coeff]
        ]

        return H

    def gamma_func_transverse(self, t: float) -> float:
        return self.A * np.cos(self.theta)

    def gamma_func_longitudinal(self, t: float) -> float:
        return self.A * (np.sign(np.sin(-self.omega_r * t))) * np.sin(self.theta)

    def collapse_ops(self) -> list[list[Any]]:
        def collapse_coeff(t, args) -> complex:
            return np.exp(-1j * args["w_r"] * t)

        c_ops = [[np.sqrt(self.kappa) * self.a, collapse_coeff]]
        return c_ops

    def run_simulation(self) -> tuple:
        result: qt.McResult = qt.mcsolve(
            self.H,
            self.psi,
            self.tlist,
            self.collapse_ops(),
            e_ops=self.e_ops,
            args=self.args,
            ntraj=self.ntraj,
            options=qt.Options(nsteps=5000, store_states=True, progress_bar=True),
        )
        rho_t = result.average_states
        if rho_t is None:
            raise ValueError("Simulation did not return any states. Check parameters.")
        photon_list = qp.get_photon_data_list(self.tlist, result.expect[0])
        qp.save_photon_data(
            photon_list,
            self.data_dir + "photon_data_" + str(self.initial_state) + ".csv",
        )
        return rho_t, photon_list

    def qnd_analysis(self, rho_t: list[Qobj]):
        qnd_analyzer = QNDAnalyzer(
            self.dim_q, self.dim_r, self.psi_q, self.initial_state
        )
        rho_t_reduced, tlist_reduced = qnd_analyzer.select_data(
            len(self.tlist) // 6, self.tlist, rho_t
        )
        qnd_results_reduced = qnd_analyzer.time_evolution_analysis(
            rho_t_reduced, tlist_reduced
        )
        qnd_analyzer.save_results_compressed(
            qnd_results_reduced,
            self.data_dir + "qnd_results_" + str(self.initial_state) + "_reduced",
        )
        qnd_analyzer.plot_qnd_fidelities(
            qnd_results_reduced,
            coupling_type=self.coupling_type,
            epsilon=self.epsilon,
            A=self.A,
            save_path=self.figures_dir
            + "qnd_fidelities_"
            + str(self.initial_state)
            + "_reduced.png",
            show=self.show
        )
        qnd_analyzer.plot_sigmaz_expectations(
            qnd_results_reduced,
            type=self.coupling_type,
            epsilon=self.epsilon,
            A=self.A,
            save_path=self.figures_dir
            + f"qnd_sigmaz_expectations_{self.initial_state}_reduced.png",
            ylim_down=0.99 if self.initial_state == 0 else -1.01,
            ylim_up=1.01 if self.initial_state == 0 else -0.99,
            show=self.show
        )
