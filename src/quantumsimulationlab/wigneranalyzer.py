import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import json
import qutip as qt


class WignerAnalyzer:
    def __init__(self, xvec, yvec=None, point_number=30):
        if yvec is None:
            yvec = xvec
        self.xvec = xvec
        self.yvec = yvec
        self.dx = xvec[1] - xvec[0]
        self.dy = yvec[1] - yvec[0]
        self.point_number = point_number

    def wigner_block(self, rho_block):
        return qt.wigner(qt.Qobj(rho_block), self.xvec, self.yvec)

    def wigner_assembly(self, rho):
        sz = rho.shape[0]
        if sz % 2 != 0:
            raise ValueError("The density matrix size must be even.")

        N = sz // 2

        rho_uu = rho[:N, :N]
        rho_ud = rho[:N, N : 2 * N]
        rho_du = rho[N : 2 * N, :N]
        rho_dd = rho[N : 2 * N, N : 2 * N]

        wigner_uu = self.wigner_block(rho_uu)
        wigner_ud = self.wigner_block(rho_ud)
        wigner_du = self.wigner_block(rho_du)
        wigner_dd = self.wigner_block(rho_dd)

        wigner_sum = wigner_uu + wigner_dd + np.real(wigner_ud) + np.real(wigner_du)  # type: ignore
        return wigner_sum

    def get_wigner_list(self, rho_list):
        n = len(rho_list)
        if self.point_number > n:
            raise ValueError("Frame number exceeds the length of rho_list.")
        indices = np.round(np.linspace(0, n - 1, self.point_number)).astype(int)

        if len(indices) > 0:
            first_result = self.wigner_assembly(rho_list[indices[0]])
            wigner_list = [None] * len(indices)
            wigner_list[0] = first_result

            for idx, i in enumerate(indices):
                if idx > 0:
                    wigner_list[idx] = self.wigner_assembly(rho_list[i])
        else:
            wigner_list = []
        return wigner_list

    def calculate_overlap(self, wigner1, wigner2):
        w1_pos = np.abs(wigner1)
        w2_pos = np.abs(wigner2)

        overlap_region = np.minimum(w1_pos, w2_pos)
        union_region = np.maximum(w1_pos, w2_pos)

        overlap_integral = np.sum(overlap_region) * self.dx * self.dy
        union_integral = np.sum(union_region) * self.dx * self.dy

        if union_integral == 0:
            return 0.0

        return overlap_integral / union_integral

    def calculate_fidelity(self, wigner1, wigner2):
        """
        Wigner fidelty
        F = (∫∫ √(W1(x,p) * W2(x,p)) dx dp)²
        """
        w1_pos = np.abs(wigner1)
        w2_pos = np.abs(wigner2)

        # 计算几何平均
        geometric_mean = np.sqrt(w1_pos * w2_pos)
        fidelity_integral = np.sum(geometric_mean) * self.dx * self.dy

        return fidelity_integral**2

    def find_peak_positions(self, wigner):
        max_idx = np.unravel_index(np.argmax(wigner), wigner.shape)
        x_peak = self.xvec[max_idx[1]]
        y_peak = self.yvec[max_idx[0]]

        return x_peak, y_peak

    def get_peak_trace(self, wigner_list):
        n = len(wigner_list)
        peak_trace_list = []

        for i in range(n):
            wigner = wigner_list[i]
            x_peak, y_peak = self.find_peak_positions(wigner)
            peak_trace_list.append((x_peak, y_peak))
        return peak_trace_list

    def save_peak_trace(self, peak_trace_list, save_path):
        with open(save_path, "w") as f:
            for x, y in peak_trace_list:
                f.write(f"{x},{y}\n")

    def get_save_peak_trace(self, wigner_list, save_path):
        peak_trace_list = self.get_peak_trace(wigner_list)
        self.save_peak_trace(peak_trace_list, save_path)
        return peak_trace_list

    def load_peak_trace(self, load_path):
        peak_trace_list = []
        with open(load_path, "r") as f:
            for line in f:
                coords = line.strip().split(",")
                if len(coords) == 2:
                    peak_trace_list.append((float(coords[0]), float(coords[1])))

        return peak_trace_list

    def calculate_separation_distance(self, wigner1, wigner2):
        x1, y1 = self.find_peak_positions(wigner1)
        x2, y2 = self.find_peak_positions(wigner2)

        return euclidean([x1, y1], [x2, y2])

    def calculate_separability_metric(self, wigner1, wigner2):
        """
        S = D / sqrt(sigma1^2 + sigma2^2)
        where D is the separation distance and sigma1, sigma2 are the standard deviations of the Wigner functions.
        """

        distance = self.calculate_separation_distance(wigner1, wigner2)

        def effective_width(wigner):
            w_pos = np.abs(wigner)
            w_norm = w_pos / np.sum(w_pos)  # Normalize the Wigner function

            x_center = np.sum(np.sum(w_norm, axis=0) * self.xvec)
            y_center = np.sum(np.sum(w_norm, axis=1) * self.yvec)

            x_grid, y_grid = np.meshgrid(self.xvec, self.yvec)
            sigma_x2 = np.sum(w_norm * (x_grid - x_center) ** 2)
            sigma_y2 = np.sum(w_norm * (y_grid - y_center) ** 2)

            return np.sqrt(sigma_x2 + sigma_y2)

        width1 = effective_width(wigner1)
        width2 = effective_width(wigner2)

        return distance / np.sqrt(width1**2 + width2**2)

    def data_times(self, time_points):
        if len(time_points) <= self.point_number:
            selected_indices = range(len(time_points))
        else:
            selected_indices = np.linspace(
                0, len(time_points) - 1, self.point_number, dtype=int
            )
        return time_points[selected_indices]

    def time_evolution_analysis(self, wigner_list_0, wigner_list_1, time_points):
        results = {
            "times": self.data_times(time_points),
            "wigner_list_0": wigner_list_0,
            "wigner_list_1": wigner_list_1,
            "overlaps": [],
            "fidelities": [],
            "distances": [],
            "separabilities": [],
            "peak_traces_0": [],
            "peak_traces_1": [],
        }

        for i, (w0, w1) in enumerate(zip(wigner_list_0, wigner_list_1)):
            overlap = self.calculate_overlap(w0, w1)
            fidelity = self.calculate_fidelity(w0, w1)
            distance = self.calculate_separation_distance(w0, w1)
            separability = self.calculate_separability_metric(w0, w1)
            peak_0 = self.find_peak_positions(w0)
            peak_1 = self.find_peak_positions(w1)

            results["overlaps"].append(overlap)
            results["fidelities"].append(fidelity)
            results["distances"].append(distance)
            results["separabilities"].append(separability)
            results["peak_traces_0"].append(peak_0)
            results["peak_traces_1"].append(peak_1)

        return results

    def save_results(self, results, filename, format="npz"):
        if format == "npz":
            np.savez_compressed(
                f"{filename}.npz",
                times=np.array(results["times"]),
                wigner_list_0=np.array(results["wigner_list_0"]),
                wigner_list_1=np.array(results["wigner_list_1"]),
                overlaps=np.array(results["overlaps"]),
                fidelities=np.array(results["fidelities"]),
                distances=np.array(results["distances"]),
                separabilities=np.array(results["separabilities"]),
                peak_traces_0=np.array(results["peak_traces_0"]),
                peak_traces_1=np.array(results["peak_traces_1"]),
            )
        elif format == "json":
            json_results = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in results.items()
            }
            with open(f"{filename}.json", "w") as f:
                json.dump(json_results, f, indent=2)

    def load_results(self, filename, format="npz"):
        if format == "npz":
            data = np.load(f"{filename}.npz")
            return {
                "times": data["times"],
                "wigner_list_0": data["wigner_list_0"],
                "wigner_list_1": data["wigner_list_1"],
                "overlaps": data["overlaps"],
                "fidelities": data["fidelities"],
                "distances": data["distances"],
                "separabilities": data["separabilities"],
                "peak_traces_0": data["peak_traces_0"],
                "peak_traces_1": data["peak_traces_1"],
            }
        elif format == "json":
            with open(f"{filename}.json", "r") as f:
                return json.load(f)
        else:
            raise ValueError("Unsupported format. Use 'npz' or 'json'.")

    def plot_overlaps(
        self,
        times,
        overlaps,
        save_path=None,
        figsize=(10, 8),
        type=None,
        epsilon=None,
        A=None,
        show=True,
    ):
        title = "Overlap vs Time"
        if type is not None:
            title += f", {type}"
        if epsilon is not None:
            title += f", ε = {epsilon:.2f} MHz"
        if A is not None:
            title += f", A = {A:.2f} MHz"
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(times, overlaps, marker="o", linestyle="-", color="blue")
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Overlap")
        ax.set_title(title)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

    def plot_fidelities(
        self,
        times,
        fidelities,
        save_path=None,
        figsize=(10, 8),
        type=None,
        epsilon=None,
        A=None,
        show=True,
    ):
        title = "Fidelity vs Time"
        if type is not None:
            title += f", {type}"
        if epsilon is not None:
            title += f", ε = {epsilon:.2f} MHz"
        if A is not None:
            title += f", A = {A:.2f} MHz"
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(times, fidelities, marker="o", linestyle="-", color="yellow")
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Fidelity")
        ax.set_title(title)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        if show:
            plt.show()

    def plot_separabilities(
        self,
        times,
        separabilities,
        save_path=None,
        figsize=(10, 8),
        type=None,
        epsilon=None,
        A=None,
        show=True,
    ):
        title = "Separability vs Time"
        if type is not None:
            title += f", {type}"
        if epsilon is not None:
            title += f", ε = {epsilon:.2f} MHz"
        if A is not None:
            title += f", A = {A:.2f} MHz"
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(times, separabilities, marker="o", linestyle="-", color="green")
        ax.set_xlabel("Time (μs)")
        ax.set_ylabel("Separability Metric")
        ax.set_title(title)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        if show:
            plt.show()

    def get_wigner_last(self, wigner_list_0, wigner_list_1, time_points):
        return wigner_list_0[-1], wigner_list_1[-1], self.data_times(time_points)[-1]

    def plot_wigner(self, wigner, time, save_path=None, wlim=0.68, figsize=(10, 8), show=True):
        fig, ax = plt.subplots(figsize=figsize)
        x, y = np.meshgrid(self.xvec, self.yvec)
        contour = ax.contourf(
            x,
            y,
            wigner,
            levels=np.linspace(-wlim, wlim, 100),
            cmap="seismic",
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(rf"Wigner Function at t = {time:.2f} μs")
        plt.colorbar(contour, ax=ax)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        
        if show:
            plt.show()

    def plot_wigner_together(
        self, wigner_0, wigner_1, time, save_path=None, wlim=0.68, figsize=(10, 8), show=True
    ):
        fig, ax = plt.subplots(figsize=figsize)
        x, y = np.meshgrid(self.xvec, self.yvec)
        contour = ax.contourf(
            x,
            y,
            wigner_0 + wigner_1,
            levels=np.linspace(-wlim, wlim, 100),
            cmap="seismic",
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(rf"Wigner Functions at t = {time:.2f} μs")
        # ax.legend()
        plt.colorbar(contour, ax=ax)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

    def plot_peak_trace(
        self,
        peak_trace_lists,
        save_path=None,
        figsize=(10, 8),
        range=4.5,
        type=None,
        epsilon=None,
        A=None,
        show=True,
    ):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        title = "Peak Trace Plot"

        if type is not None:
            title = title + ", " + type

        if epsilon is not None:
            title = title + rf", $\epsilon$ = {epsilon:.2f} MHz"

        if A is not None:
            title = title + rf", $A$ = {A:.2f} MHz"

        ax.set_title(title)
        ax.set_xlim(-range, range)
        ax.set_ylim(-range, range)

        colors = ["red", "blue"]

        for i, peak_trace_list in enumerate(peak_trace_lists):
            color = colors[i % len(colors)]

            x_coords = [point[0] for point in peak_trace_list]
            y_coords = [point[1] for point in peak_trace_list]

            ax.plot(x_coords, y_coords, color=color, label=rf"Trace $|{i}\rangle$")
            ax.scatter(x_coords, y_coords, color=color)

        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

    def plot_wigner_together_with_peak_trace(
        self,
        wigner_0,
        wigner_1,
        time,
        peak_trace_lists,
        save_path=None,
        wlim=0.68,
        figsize=(10, 8),
        range=6,
        type=None,
        epsilon=None,
        A=None,
        show=True,
    ):
        fig, ax = plt.subplots(figsize=figsize)

        ax.set_xlabel("x")
        ax.set_ylabel("y")

        title = "Wigner Functions with Peak Trace"

        if type is not None:
            title = title + ", " + type

        if epsilon is not None:
            title = title + rf", $\epsilon$ = {epsilon:.2f} MHz"

        if A is not None:
            title = title + rf", $A$ = {A:.2f} MHz"

        title = title + rf" at {time:.2f} $\mu s$"

        ax.set_title(title)
        ax.set_xlim(-range, range)
        ax.set_ylim(-range, range)

        colors = ["purple", "orange"]

        x, y = np.meshgrid(self.xvec, self.yvec)
        contour = ax.contourf(
            x,
            y,
            wigner_0 + wigner_1,
            levels=np.linspace(-wlim, wlim, 100),
            cmap="seismic",
        )

        plt.colorbar(contour, ax=ax)

        for i, peak_trace_list in enumerate(peak_trace_lists):
            color = colors[i % len(colors)]

            x_coords = [point[0] for point in peak_trace_list]
            y_coords = [point[1] for point in peak_trace_list]

            ax.plot(x_coords, y_coords, color=color, label=rf"Trace $|{i}\rangle$")
            ax.scatter(x_coords, y_coords, color=color)

        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()
