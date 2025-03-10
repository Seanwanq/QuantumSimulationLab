from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import gc

from quantumsimulationlab.tools.wigner import wigner_assembly


def wigner_plotting(
    density_matrix_history, xvec, pvec, frame_index, dt, filename="wigner"
):
    if frame_index >= len(density_matrix_history):
        raise ValueError("Frame index exceeds the number of density matrices.")
    time = frame_index * dt
    fig, ax = plt.subplots(figsize=(8, 5))
    wigner = wigner_assembly(density_matrix_history[frame_index], xvec, pvec)
    x, y = np.meshgrid(xvec, pvec)
    wlim = max(abs(wigner.min()), abs(wigner.max()))
    c = ax.contourf(x, y, wigner, levels=np.linspace(-wlim, wlim, 100), cmap="seismic")
    plt.colorbar(c)
    ax.set_xlabel(r"$\mathrm{Re}\left( \alpha \right)$")
    ax.set_ylabel(r"$\mathrm{Im}\left( \alpha \right)$")
    ax.set_title(f"Wigner function of time {time: 2f}")
    filename = filename + "_time_{:.2f}.png".format(time)
    fig.savefig(filename, dpi=1000, bbox_inches="tight")


def wigner_plotting_multi(
    density_matrix_history, xvec, pvec, frame_number, dt, filename="wigner"
):
    filename_backup = filename
    if frame_number > len(density_matrix_history):
        raise ValueError("Frame number exceeds the number of density matrices.")
    indices = np.linspace(0, len(density_matrix_history) - 1, frame_number, dtype=int)

    selected_density_matrices = [density_matrix_history[int(i)] for i in indices]
    times = [i * dt for i in indices]
    for i in tqdm(range(0, len(selected_density_matrices))):
        plt.clf()
        plt.close("all")
        gc.collect()

        fig, ax = plt.subplots(figsize=(10, 8))
        wigner = wigner_assembly(selected_density_matrices[i], xvec, pvec)
        x, y = np.meshgrid(xvec, pvec)
        # wlim = max(abs(wigner.min()), abs(wigner.max()))
        wlim = 0.68
        levels = np.linspace(-wlim, wlim, 100)
        c = ax.contourf(x, y, wigner, levels=levels, cmap="seismic")
        plt.colorbar(c)
        ax.set_xlabel(r"$\mathrm{Re}\left( \alpha \right)$")
        ax.set_ylabel(r"$\mathrm{Im}\left( \alpha \right)$")
        ax.set_title(f"Wigner function of time {times[i]: 2f}")
        filename = filename + "_time_{:.2f}.png".format(times[i])
        fig.savefig(filename, dpi=500)
        plt.close(fig)
        filename = filename_backup
