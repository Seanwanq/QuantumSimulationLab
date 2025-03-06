import matplotlib.pyplot as plt
import numpy as np

from quantumsimulationlab.wigner import wigner_assembly

def wigner_plotting(
    density_matrix_history, xvec, pvec, frame_index, dt, filename="wigner"
):
    if frame_index >= len(density_matrix_history):
        raise ValueError("Frame index exceeds the number of density matrices.")
    time = frame_index * dt
    fig, ax = plt.subplots(figsize=(8, 5))
    wigner = wigner_assembly(
        density_matrix_history[frame_index], xvec, pvec
    )
    x, y = np.meshgrid(xvec, pvec)
    wlim = max(abs(wigner.min()), abs(wigner.max()))
    c = ax.contourf(
        x, y, wigner, levels=np.linspace(-wlim, wlim, 100), cmap="seismic"
    )
    plt.colorbar(c)
    ax.set_xlabel(r"$\mathrm{Re}\left( \alpha \right)$")
    ax.set_ylabel(r"$\mathrm{Im}\left( \alpha \right)$")
    ax.set_title(f"Wigner function of time {time: 2f}")
    filename = filename + "_time_{:.2f}.png".format(time)
    fig.savefig(filename, dpi=1000, bbox_inches="tight")
    return fig
