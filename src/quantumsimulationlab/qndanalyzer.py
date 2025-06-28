import numpy as np
import qutip as qt
import json
import matplotlib.pyplot as plt


class QNDAnalyzer:
    def __init__(self, dim_q, dim_r, initial_state_q, initial_state: int):
        self.dim_q = dim_q
        self.dim_r = dim_r
        self.initial_state_q = initial_state_q
        self.initial_state_q_dm = self.initial_state_q * self.initial_state_q.dag()
        self.initial_sigmaz = qt.expect(
            qt.tensor(qt.sigmaz(), qt.qeye(self.dim_r)),
            qt.tensor(self.initial_state_q, qt.basis(self.dim_r, 0)),
        )
        self.sigmaz = qt.tensor(qt.sigmaz(), qt.qeye(self.dim_r))
        if initial_state not in [0, 1]:
            raise ValueError("Initial state must be 0 or 1.")
        self.initial_state = initial_state

    def select_data(self, points_number: int, tlist, rho_t):
        indices = np.linspace(0, len(tlist) - 1, points_number, dtype=int)
        tlist_reduced = tlist[indices]
        rho_t_reduced = [rho_t[i] for i in indices]
        return rho_t_reduced, tlist_reduced

    def calculate_and_fidelity(self, rho):
        rho_qubit = rho.ptrace(0)
        fidelity = qt.fidelity(self.initial_state_q_dm, rho_qubit)
        return fidelity

    def calculate_sigmaz_stability(self, rho):
        sigmaz_exp = qt.expect(self.sigmaz, rho)
        stability_metric = 1.0 - np.abs(sigmaz_exp - self.initial_sigmaz) / np.abs(
            self.initial_sigmaz
        )

        return sigmaz_exp, stability_metric

    def time_evolution_analysis(self, rho_t, tlist):
        results = {
            "times": tlist,
            "rho_t": rho_t,
            "fidelities": [],
            "sigmaz_expectations": [],
            "sigmaz_stabilities": [],
            "initial_sigmaz": self.initial_sigmaz,
        }
        for rho in rho_t:
            fidelity = self.calculate_and_fidelity(rho)
            sigmaz_exp, stability_metric = self.calculate_sigmaz_stability(rho)

            results["fidelities"].append(fidelity)
            results["sigmaz_expectations"].append(sigmaz_exp)
            results["sigmaz_stabilities"].append(stability_metric)

        return results

    def save_results(self, results, filename, format="npz"):
        """
        Save time evolution analysis results to file.

        Parameters:
        results: dict - Results from time_evolution_analysis
        filename: str - Output filename (without extension)
        format: str - 'npz' (default) or 'json'
        """
        if format == "npz":
            np.savez_compressed(
                f"{filename}.npz",
                times=np.array(results["times"]),
                rho_t=np.array(results["rho_t"]),
                fidelities=np.array(results["fidelities"]),
                sigmaz_expectations=np.array(results["sigmaz_expectations"]),
                sigmaz_stabilities=np.array(results["sigmaz_stabilities"]),
                initial_sigmaz=results["initial_sigmaz"],
            )
        elif format == "json":
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in results.items()
            }
            with open(f"{filename}.json", "w") as f:
                json.dump(json_results, f, indent=2)
        else:
            raise ValueError("Unsupported format. Use 'npz' or 'json'.")

    def save_results_compressed(self, results, filename):
        np.savez_compressed(
            f"{filename}.npz",
            times=np.array(results["times"]),
            rho_t=np.array(results["rho_t"]),
        )

    def load_results_compressed(self, filename):
        data = np.load(f"{filename}.npz", allow_pickle=True)
        results = self.time_evolution_analysis(data["rho_t"], data["times"])
        return results

    def load_results(self, filename, format="npz"):
        """
        Load previously saved results.

        Parameters:
        filename: str - Input filename (without extension)
        format: str - 'npz' (default) or 'json'

        Returns:
        dict - Loaded results
        """
        if format == "npz":
            data = np.load(f"{filename}.npz")
            return {
                "times": data["times"],
                "rho_t": data["rho_t"],
                "fidelities": data["fidelities"],
                "sigmaz_expectations": data["sigmaz_expectations"],
                "sigmaz_stabilities": data["sigmaz_stabilities"],
                "initial_sigmaz": float(data["initial_sigmaz"]),
            }
        elif format == "json":
            with open(f"{filename}.json", "r") as f:
                return json.load(f)
        else:
            raise ValueError("Unsupported format. Use 'npz' or 'json'.")

    def plot_qnd_fidelities(
        self,
        results,
        save_path=None,
        figsize=(10, 8),
        coupling_type=None,
        epsilon=None,
        A=None,
        show=True
    ):
        title = "QND Fidelity" + " |" + str(self.initial_state) + ">"
        if coupling_type is not None:
            title += f", {coupling_type}"
        if epsilon is not None:
            title += f", ε={epsilon} MHz"
        if A is not None:
            title += f", A={A} MHz"
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.plot(results["times"], results["fidelities"], label="Fidelity")
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Fidelity")
        ax.set_ylim(0.99, 1.01)
        ax.legend()
        if save_path:
            plt.savefig(save_path)
        if show:
            plt.show()

    def plot_sigmaz_stability(
        self,
        results,
        save_path=None,
        figsize=(10, 8),
        type=None,
        epsilon=None,
        A=None,
        ylim_down=None,
        ylim_up=None,
        show=True
    ):
        title = "QND Stability" + " |" + str(self.initial_state) + ">"
        if type is not None:
            title += f", {type}"
        if epsilon is not None:
            title += f", ε={epsilon} MHz"
        if A is not None:
            title += f", A={A} MHz"
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.plot(
            results["times"], results["sigmaz_stabilities"], label="Stability Metric"
        )
        ax.axhline(0, color="red", linestyle="--", label="Initial Sigmaz")
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Stability Metric")

        if ylim_down is not None and ylim_up is not None:
            ax.set_ylim(ylim_down, ylim_up)

        ax.legend()
        if save_path:
            plt.savefig(save_path)

        if show:
            plt.show()

    def plot_sigmaz_expectations(
        self,
        results,
        save_path=None,
        figsize=(10, 8),
        type=None,
        epsilon=None,
        A=None,
        ylim_down=None,
        ylim_up=None,
        show=True
    ):
        title = (
            "QND Sigmaz Expectation" + " |" + str(self.initial_state) + ">"
        )
        if type is not None:
            title += f", {type}"
        if epsilon is not None:
            title += f", ε={epsilon} MHz"
        if A is not None:
            title += f", A={A} MHz"
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.plot(
            results["times"], results["sigmaz_expectations"], label="Sigmaz Expectation"
        )
        ax.axhline(
            self.initial_sigmaz, color="red", linestyle="--", label="Initial Sigmaz"
        )
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Sigmaz Expectation")

        if ylim_down is not None and ylim_up is not None:
            ax.set_ylim(ylim_down, ylim_up)

        ax.legend()
        if save_path:
            plt.savefig(save_path)

        if show:
            plt.show()

    def calculate_qnd_fidelity_list(self, rho_t):
        fidelity_t = []

        for rho in rho_t:
            rho_qubit = rho.ptrace(0)
            fidelity = qt.fidelity(self.initial_state_q_dm, rho_qubit)
            fidelity_t.append(fidelity)

        return fidelity_t

    def calculate_sigmaz_stability_list(self, rho_t):
        sigmaz_expectation = []
        for rho in rho_t:
            sigmaz_exp = qt.expect(self.sigmaz, rho)
            sigmaz_expectation.append(sigmaz_exp)

        stability_metric = 1.0 - np.abs(
            np.array(sigmaz_expectation) - self.initial_sigmaz
        ) / np.abs(self.initial_sigmaz)

        return sigmaz_expectation, stability_metric, self.initial_sigmaz
