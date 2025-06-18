import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_photon_data_list(t_list, n_list):
    photon_list = np.column_stack((t_list, n_list))
    return photon_list


def find_equilibrium_time(photon_list, threshold=0.001):
    if len(photon_list[:, 1]) < 10:
        return None

    final_portion = int(0.1 * len(photon_list[:, 1]))
    final_value = np.mean(photon_list[:, 1][-final_portion:])

    for i in range(len(photon_list[:, 1])):
        if abs(photon_list[:, 1][i] - final_value) / (final_value + 1e-10) < threshold:
            if i + 50 < len(photon_list[:, 1]):
                stable = True
                for j in range(i, min(i + 50, len(photon_list[:, 1]))):
                    if (
                        abs(photon_list[:, 1][j] - final_value) / (final_value + 1e-10)
                        > threshold
                    ):
                        stable = False
                        break
                if stable:
                    return photon_list[:, 0][i]

    return None


def save_photon_data(photon_list, save_path):
    df = pd.DataFrame(photon_list, columns=["time", "photon_count"])
    df.to_csv(save_path, index=False, header=False)
    return save_path


def load_photon_data(load_path):
    df = pd.read_csv(load_path, header=None, names=["time", "photon_count"])
    t_list = df["time"].values
    n_list = df["photon_count"].values
    return get_photon_data_list(t_list, n_list)


def plot_photon_data_list(
    photon_lists,
    save_path=None,
    figsize=(10, 8),
    show_equilibrium=True,
    linestyles=None,
    labels=None,
    type=None,
    epsilon = None,
    A = None,
    show=True
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(r"Time ($\mu s$)")
    ax.set_ylabel("Average Photon Number")

    title = ""

    if show_equilibrium:
        title = "Average Photon Number with Equilibrium Time"
    else:
        title = "Average Photon Number vs Time"

    if type is not None:
        title += ", "
        title += type

    if epsilon is not None:
        title += ", "
        title += rf"$\epsilon$ = {epsilon} MHz"

    if A is not None:
        title += ", "
        title += rf"A = {A} MHz"

    ax.set_title(title)

    colors = ["red", "blue", "green", "orange", "purple", "brown", "pink", "cyan"]

    for i, photon_list in enumerate(photon_lists):
        color = colors[i % len(colors)]

        if linestyles and i in linestyles:
            linestyle = linestyles[i]
        else:
            linestyle = "-"

        if labels and i in labels:
            label = labels[i]
        else:
            label = f"Photon List {i + 1}"

        x_coords = [point[0] for point in photon_list]
        y_coords = [point[1] for point in photon_list]

        ax.plot(x_coords, y_coords, color=color, linestyle=linestyle, label=label)

    if show_equilibrium:
        for i, photon_list in enumerate(photon_lists):
            eq_time = find_equilibrium_time(photon_list)

            if labels and i in labels:
                label = labels[i] + rf" eq: {eq_time:.1f}$\mu s$"
            else:
                label = rf"eq: {eq_time:.1f}$\mu s$"

            # equilibrium_time_list.append(eq_time)
            color = colors[i % len(colors)]
            if eq_time is not None:
                ax.axvline(eq_time, color=color, linestyle="--", alpha=0.7, label=label)

    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()


def get_equilibrium_time_list(photon_lists):
    equilibrium_time_list = []
    for i, photon_list in enumerate(photon_lists):
        eq_time = find_equilibrium_time(photon_list)
        equilibrium_time_list.append(eq_time)
    return equilibrium_time_list


def plot_equilibrium_times_epsilon(
    equilibrium_time_list, epsilon_list, save_path=None, figsize=(10, 8), show=True
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(r"Epsilon ($\epsilon$ MHz)")
    ax.set_ylabel(r"Equilibrium Time ($\mu s$)")
    ax.set_title("Equilibrium Time vs Epsilon")

    ax.plot(
        epsilon_list, equilibrium_time_list, marker="o", linestyle="-", color="blue"
    )

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()


def plot_equilibrium_times_vs_A(
    equilibrium_time_list, A_list, save_path=None, figsize=(10, 8), show=True
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(r"Amp ($A$ MHz)")
    ax.set_ylabel(r"Equilibrium Time ($\mu s$)")
    ax.set_title("Equilibrium Time vs Amp")

    ax.plot(A_list, equilibrium_time_list, marker="o", linestyle="-", color="green")

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
