import numpy as np
import qutip as qt
from qutip import Qobj
import quantumsimulationlab.photonanalyzer as qp
from quantumsimulationlab.wigneranalyzer import WignerAnalyzer
from quantumsimulationlab.tools import create_directory
from quantumsimulationlab.system import System
import matplotlib.pyplot as plt


class Twins:
    def __init__(
        self,
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
        show: bool = True
    ) -> None:
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
        self.show: bool = show
        self.A: float = A
        self.theta: float = theta
        self.delta: float = abs(omega_q - omega_r)
        self.tlist: np.ndarray = np.linspace(0, t_max, nt)
        self.n_op: Qobj = qt.tensor(qt.qeye(dim_q), qt.num(dim_r))
        self.psi_q_0: Qobj = qt.basis(dim_q, 0)
        self.psi_q_1: Qobj = qt.basis(dim_q, 1)
        self.psi_r: Qobj = qt.basis(dim_r, 0)
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
                pass
            case "longitudinal":
                pass
            case "hybrid":
                pass
            case _:
                raise ValueError(
                    f"Invalid coupling type: {coupling_type}. Choose from {['transverse', 'longitudinal', 'hybrid']}."
                )
        self.system = [
            System(
                initial_state=0,
                coupling_type=coupling_type,
                dim_r=dim_r,
                dim_q=dim_q,
                omega_q=omega_q,
                omega_r=omega_r,
                kappa=kappa,
                epsilon=epsilon,
                t_max=t_max,
                nt=nt,
                ntraj=ntraj,
                A=A,
                theta=theta,
                xvec=xvec,
                father_directory=father_directory,
                show=self.show
            ),
            System(
                initial_state=1,
                coupling_type=coupling_type,
                dim_r=dim_r,
                dim_q=dim_q,
                omega_q=omega_q,
                omega_r=omega_r,
                kappa=kappa,
                epsilon=epsilon,
                t_max=t_max,
                nt=nt,
                ntraj=ntraj,
                A=A,
                theta=theta,
                xvec=xvec,
                father_directory=father_directory,
                show=self.show
            ),
        ]

    def print_parameters(self) -> None:
        print(f"Coupling type: {self.coupling_type}")
        print(f"Dimension of resonator: {self.dim_r}")
        print(f"Dimension of qubit: {self.dim_q}")
        print(f"Qubit frequency: {self.omega_q}")
        print(f"Resonator frequency: {self.omega_r}")
        print(f"Kappa: {self.kappa}")
        print(f"Epsilon: {self.epsilon}")
        print(f"Maximum time: {self.t_max}")
        print(f"Number of time steps: {self.nt}")
        print(f"Number of trajectories: {self.ntraj}")
        print(f"A: {self.A}")
        print(f"Theta: {self.theta}")

        print("----------------------------------------")
        print("State 0")
        self.system[0].print_parameters()
        print("----------------------------------------")
        print("State 1")
        self.system[1].print_parameters()

    def create_directories(self) -> None:
        create_directory(self.father_directory, self.directory_name)
        create_directory(self.father_directory + "/" + self.directory_name, "data")
        create_directory(self.father_directory + "/" + self.directory_name, "figures")
    
    def configure_plot_style(self, fontsize=14):
        plt.rcParams.update({
            'font.size': fontsize,
            'axes.titlesize': fontsize + 2,
            'axes.labelsize': fontsize,
            'xtick.labelsize': fontsize - 2,
            'ytick.labelsize': fontsize - 2,
            'legend.fontsize': fontsize,
            'figure.titlesize': fontsize + 4,
            'lines.linewidth': 2,
            'lines.markersize': 8
        })

    def run_simulation(self):
        rho_ts = []
        photon_lists = []
        for i in range(0, 2):
            rho_t, photon_list = self.system[i].run_simulation()
            rho_ts.append(rho_t)
            photon_lists.append(photon_list)
        return rho_ts, photon_lists

    def qnd_analysis(self, rho_ts):
        for i in range(0, 2):
            self.system[i].qnd_analysis(rho_ts[i])

    def photon_analysis(self, photon_lists):
        for i in range(0, 2):
            qp.save_photon_data(photon_lists[i], self.data_dir + f"photon_data_{i}.csv")

        match self.coupling_type:
            case "transverse":

                def photon_average_transverse(t):
                    chi = (self.A * np.cos(self.theta)) ** 2 / (self.delta)
                    phiqb = 2 * np.arctan(2 * chi / self.kappa)
                    n = (
                        (2 * np.abs(self.epsilon) / self.kappa) ** 2
                        * (np.cos(0.5 * phiqb)) ** 2
                        * (
                            1
                            - 2 * np.cos(chi * t) * np.exp(-0.5 * self.kappa * t)
                            + np.exp(-self.kappa * t)
                        )
                    )
                    return n

                photon_theoretical = qp.get_photon_data_list(
                    self.tlist, [photon_average_transverse(t) for t in self.tlist]
                )

                qp.plot_photon_data_list(
                    [photon_lists[0], photon_lists[1], photon_theoretical],
                    linestyles={2: "-."},
                    labels={
                        0: "|0>",
                        1: "|1>",
                        2: "theory",
                    },
                    type=self.coupling_type,
                    epsilon=self.epsilon,
                    A=self.A,
                    save_path=self.figures_dir + "photon_data.png",
                    show=self.show
                )

            case "longitudinal":
                if self.epsilon < 0.001:

                    def photon_average_longitudinal(t):
                        n = (
                            (np.abs(self.A * np.sin(self.theta))) ** 2 / self.kappa**2
                        ) * (1 - np.exp(-0.5 * self.kappa * t)) ** 2
                        return n

                    def photon_average_longitudinal_fixed(t):
                        n = (
                            (4 / np.pi) ** 2
                            * (
                                (np.abs(self.A * np.sin(self.theta))) ** 2
                                / self.kappa**2
                            )
                            * (1 - np.exp(-0.5 * self.kappa * t)) ** 2
                        )
                        return n

                    photon_theoretical = qp.get_photon_data_list(
                        self.tlist, [photon_average_longitudinal(t) for t in self.tlist]
                    )

                    photon_theoretical_fixed = qp.get_photon_data_list(
                        self.tlist,
                        [photon_average_longitudinal_fixed(t) for t in self.tlist],
                    )

                    qp.plot_photon_data_list(
                        [
                            photon_lists[0],
                            photon_lists[1],
                            photon_theoretical,
                            photon_theoretical_fixed,
                        ],
                        linestyles={2: "-.", 3: "-."},
                        labels={
                            0: "|0>",
                            1: "|1>",
                            2: "theory",
                            3: "theory fixed",
                        },
                        type=self.coupling_type,
                        epsilon=self.epsilon,
                        A=self.A,
                        save_path=self.figures_dir + "photon_data.png",
                        show=self.show
                    )

                else:
                    qp.plot_photon_data_list(
                        [photon_lists[0], photon_lists[1]],
                        labels={
                            0: "|0>",
                            1: "|1>",
                        },
                        type=self.coupling_type,
                        epsilon=self.epsilon,
                        A=self.A,
                        save_path=self.figures_dir + "photon_data.png",
                        show=self.show
                    )

            case "hybrid":

                def photon_average_hybrid(t):
                    g = self.A * np.sin(self.theta)
                    a = 4.0 / np.pi
                    chi = (self.A * np.cos(self.theta)) ** 2 / (self.delta)
                    phiqb = 2 * np.arctan(2 * chi / self.kappa)
                    term1 = (
                        a**2
                        * (np.abs(g)) ** 2
                        / self.kappa**2
                        * (np.cos(0.5 * phiqb)) ** 2
                        * (1 - np.exp(-0.5 * self.kappa * t)) ** 2
                    )
                    term2 = (
                        (2 * np.abs(self.epsilon) / self.kappa) ** 2
                        * (np.cos(0.5 * phiqb)) ** 2
                        * (
                            1
                            - 2 * np.cos(chi * t) * np.exp(-0.5 * self.kappa * t)
                            + np.exp(-self.kappa * t)
                        )
                    )
                    return term1 + term2

                photon_theoretical = qp.get_photon_data_list(
                    self.tlist, [photon_average_hybrid(t) for t in self.tlist]
                )

                qp.plot_photon_data_list(
                    [photon_lists[0], photon_lists[1], photon_theoretical],
                    linestyles={2: "-."},
                    labels={
                        0: "|0>",
                        1: "|1>",
                        2: "theory",
                    },
                    type=self.coupling_type,
                    epsilon=self.epsilon,
                    A=self.A,
                    save_path=self.figures_dir + "photon_data.png",
                    show=self.show
                )

            case _:
                raise ValueError(
                    f"Invalid coupling type: {self.coupling_type}. Choose from {['transverse', 'longitudinal', 'hybrid']}."
                )

    def wigner_analysis(self, rho_ts):
        wigner_analyzer = WignerAnalyzer(self.xvec, point_number=30)
        wigner_lists = []
        for i in range(0, 2):
            wigner_lists.append(wigner_analyzer.get_wigner_list(rho_ts[i]))

        wigner_results = wigner_analyzer.time_evolution_analysis(
            wigner_lists[0], wigner_lists[1], self.tlist
        )
        wigner_analyzer.save_results(wigner_results, self.data_dir + "wigner_results")

        wigner_times = wigner_results["times"]
        wigner_overlaps = wigner_results["overlaps"]
        wigner_separabilities = wigner_results["separabilities"]
        wigner_peak_traces = [
            wigner_results["peak_traces_0"],
            wigner_results["peak_traces_1"],
        ]
        wigner_last_0, wigner_last_1, time_last = wigner_analyzer.get_wigner_last(
            wigner_lists[0], wigner_lists[1], wigner_times
        )
        wigner_analyzer.plot_wigner_together_with_peak_trace(
            wigner_last_0,
            wigner_last_1,
            time_last,
            [wigner_peak_traces[0], wigner_peak_traces[1]],
            type=self.coupling_type,
            epsilon=self.epsilon,
            A=self.A,
            wlim=1.00,
            save_path=self.figures_dir + "wigner_trace_last.png",
            show=self.show
        )
        wigner_analyzer.plot_overlaps(
            wigner_times,
            wigner_overlaps,
            type=self.coupling_type,
            epsilon=self.epsilon,
            A=self.A,
            save_path=self.figures_dir + "wigner_overlaps.png",
            show=self.show
        )
        wigner_analyzer.plot_separabilities(
            wigner_times,
            wigner_separabilities,
            type=self.coupling_type,
            epsilon=self.epsilon,
            A=self.A,
            save_path=self.figures_dir + "wigner_separabilities.png",
            show=self.show
        )

    def replot(self, save_path, fontsize=16):
        create_directory(save_path, self.directory_name)
        create_directory(save_path + "/" + self.directory_name, "figures")
        save_path = save_path + "/" + self.directory_name + "/figures"
        # Replotting wigner results
        self.configure_plot_style(fontsize=fontsize)
        wigner_analyzer = WignerAnalyzer(self.xvec, point_number=30)
        wigner_results = wigner_analyzer.load_results(self.data_dir + "wigner_results")
        wigner_times = wigner_results["times"]
        wigner_list_0 = wigner_results["wigner_list_0"]
        wigner_list_1 = wigner_results["wigner_list_1"]
        wigner_overlaps = wigner_results["overlaps"]
        wigner_separabilities = wigner_results["separabilities"]
        wigner_peak_traces = [
            wigner_results["peak_traces_0"],
            wigner_results["peak_traces_1"],
        ]
        wigner_last_0, wigner_last_1, time_last = wigner_analyzer.get_wigner_last(
            wigner_list_0, wigner_list_1, wigner_times
        )
        wigner_analyzer.plot_wigner_together_with_peak_trace(
            wigner_last_0,
            wigner_last_1,
            time_last,
            [wigner_peak_traces[0], wigner_peak_traces[1]],
            type=self.coupling_type,
            epsilon=self.epsilon,
            A=self.A,
            wlim=1.00,
            save_path=save_path + "/wigner_trace_last.png",
            show=self.show
        )
        wigner_analyzer.plot_overlaps(
            wigner_times,
            wigner_overlaps,
            type=self.coupling_type,
            epsilon=self.epsilon,
            A=self.A,
            save_path=save_path + "/wigner_overlaps.png",
            show=self.show
        )
        wigner_analyzer.plot_separabilities(
            wigner_times,
            wigner_separabilities,
            type=self.coupling_type,
            epsilon=self.epsilon,
            A=self.A,
            save_path=save_path + "/wigner_separabilities.png",
            show=self.show
        )
        self.system[0].replot(save_path=save_path)
        self.system[1].replot(save_path=save_path)
        self.photon_analysis_replot(save_path=save_path)

    def photon_analysis_replot(self, save_path):
        photon_lists = []
        for i in range(0, 2):
            photon_lists.append(qp.load_photon_data(self.data_dir + f"photon_data_{i}.csv"))

        match self.coupling_type:
            case "transverse":

                def photon_average_transverse(t):
                    chi = (self.A * np.cos(self.theta)) ** 2 / (self.delta)
                    phiqb = 2 * np.arctan(2 * chi / self.kappa)
                    n = (
                        (2 * np.abs(self.epsilon) / self.kappa) ** 2
                        * (np.cos(0.5 * phiqb)) ** 2
                        * (
                            1
                            - 2 * np.cos(chi * t) * np.exp(-0.5 * self.kappa * t)
                            + np.exp(-self.kappa * t)
                        )
                    )
                    return n

                photon_theoretical = qp.get_photon_data_list(
                    self.tlist, [photon_average_transverse(t) for t in self.tlist]
                )

                qp.plot_photon_data_list(
                    [photon_lists[0], photon_lists[1], photon_theoretical],
                    linestyles={2: "-."},
                    labels={
                        0: "|0>",
                        1: "|1>",
                        2: "theory",
                    },
                    type=self.coupling_type,
                    epsilon=self.epsilon,
                    A=self.A,
                    save_path=save_path + "/photon_data.png",
                    show=self.show
                )

            case "longitudinal":
                if self.epsilon < 0.001:

                    def photon_average_longitudinal(t):
                        n = (
                            (np.abs(self.A * np.sin(self.theta))) ** 2 / self.kappa**2
                        ) * (1 - np.exp(-0.5 * self.kappa * t)) ** 2
                        return n

                    def photon_average_longitudinal_fixed(t):
                        n = (
                            (4 / np.pi) ** 2
                            * (
                                (np.abs(self.A * np.sin(self.theta))) ** 2
                                / self.kappa**2
                            )
                            * (1 - np.exp(-0.5 * self.kappa * t)) ** 2
                        )
                        return n

                    photon_theoretical = qp.get_photon_data_list(
                        self.tlist, [photon_average_longitudinal(t) for t in self.tlist]
                    )

                    photon_theoretical_fixed = qp.get_photon_data_list(
                        self.tlist,
                        [photon_average_longitudinal_fixed(t) for t in self.tlist],
                    )

                    qp.plot_photon_data_list(
                        [
                            photon_lists[0],
                            photon_lists[1],
                            photon_theoretical,
                            photon_theoretical_fixed,
                        ],
                        linestyles={2: "-.", 3: "-."},
                        labels={
                            0: "|0>",
                            1: "|1>",
                            2: "theory",
                            3: "theory fixed",
                        },
                        type=self.coupling_type,
                        epsilon=self.epsilon,
                        A=self.A,
                        save_path=save_path + "/photon_data.png",
                        show=self.show
                    )

                else:
                    qp.plot_photon_data_list(
                        [photon_lists[0], photon_lists[1]],
                        labels={
                            0: "|0>",
                            1: "|1>",
                        },
                        type=self.coupling_type,
                        epsilon=self.epsilon,
                        A=self.A,
                        save_path=save_path + "/photon_data.png",
                        show=self.show
                    )

            case "hybrid":

                def photon_average_hybrid(t):
                    g = self.A * np.sin(self.theta)
                    a = 4.0 / np.pi
                    chi = (self.A * np.cos(self.theta)) ** 2 / (self.delta)
                    phiqb = 2 * np.arctan(2 * chi / self.kappa)
                    term1 = (
                        a**2
                        * (np.abs(g)) ** 2
                        / self.kappa**2
                        * (np.cos(0.5 * phiqb)) ** 2
                        * (1 - np.exp(-0.5 * self.kappa * t)) ** 2
                    )
                    term2 = (
                        (2 * np.abs(self.epsilon) / self.kappa) ** 2
                        * (np.cos(0.5 * phiqb)) ** 2
                        * (
                            1
                            - 2 * np.cos(chi * t) * np.exp(-0.5 * self.kappa * t)
                            + np.exp(-self.kappa * t)
                        )
                    )
                    return term1 + term2

                photon_theoretical = qp.get_photon_data_list(
                    self.tlist, [photon_average_hybrid(t) for t in self.tlist]
                )

                qp.plot_photon_data_list(
                    [photon_lists[0], photon_lists[1], photon_theoretical],
                    linestyles={2: "-."},
                    labels={
                        0: "|0>",
                        1: "|1>",
                        2: "theory",
                    },
                    type=self.coupling_type,
                    epsilon=self.epsilon,
                    A=self.A,
                    save_path=save_path + "/photon_data.png",
                    show=self.show
                )

            case _:
                raise ValueError(
                    f"Invalid coupling type: {self.coupling_type}. Choose from {['transverse', 'longitudinal', 'hybrid']}."
                )
