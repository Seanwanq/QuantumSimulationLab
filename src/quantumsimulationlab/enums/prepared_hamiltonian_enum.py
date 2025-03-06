from enum import Enum

class PreparedHamiltonianEnum(Enum):
    RSIGMAZ = "rotating_sigma_z"
    RSIGMAX = "rotating_sigma_x"
    SIGMAZ = "sigma_z"
    SIGMAX = "sigma_x"
    NONE = "none"