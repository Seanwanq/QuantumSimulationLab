from enum import Enum

class PauliEnum(Enum):
    IDENTITY = "identity"
    SIGMAX = "sigma_x"
    SIGMAY = "sigma_y"
    SIGMAZ = "sigma_z"
    HADAMARD = "hadamard"