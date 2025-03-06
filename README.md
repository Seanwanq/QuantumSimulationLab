# Quantum Simulation Lab

Quantum Simulation Lab is a Python package designed for simulating quantum systems. It provides tools for state evolution, plotting, and various quantum operations.

## Features

- **State Evolution**: Simulate time-dependent state evolution using different Hamiltonian.
- **Quantum Gates**: Apply quantum gates like Hadamard, Pauli-X, Pauli-Y, and Pauli-Z.
- **Plotting**: Visualize Wigner functions and other quantum states.
- **Enums**: Use enumerations for coherent orientations, gate types, and prepared Hamiltonian.

## Installation

To install the package, use the following command:

```sh
uv sync
```

## Dependencies

The project requires the following dependencies:

- numpy>=2.2.3
- pandas>=2.2.3
- scipy>=1.15.2
- moviepy>=2.1.2
- qutip>=5.1.1
- tqdm>=4.67.1
- matplotlib>=3.10.1

For development, the following additional dependencies are required:

- ipykernel>=6.29.5
- ipympl>=0.9.6
- ipynb>=0.5.1

## Usage

### State Evolution

To perform a time-dependent state evolution, use the `TimeDependentStateEvolution` class:

```python
from quantumsimulationlab.evolution import TimeDependentStateEvolution
from quantumsimulationlab.enums import PreparedHamiltonianEnum

evolution = TimeDependentStateEvolution(
    time_total=10,
    time_steps=100,
    omega_b=1.0,
    omega_r=1.0,
    gamma_t=lambda t: 0.1,
    prepared_hamiltonian_index=PreparedHamiltonianEnum.SIGMAX,
)
density_matrix_history, times = evolution.simulation_evolution()
```

### Plotting

To plot the Wigner function, use the `wigner_plotting` function:

```python
from quantumsimulationlab.plotting import wigner_plotting

fig = wigner_plotting(
    density_matrix_history, 
    xvec, 
    pvec, 
    frame_index=0,
    dt=0.1
)
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.