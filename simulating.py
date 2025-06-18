from quantumsimulationlab.twins import Twins

coupling_types: list[str] = ["transverse", "longitudinal", "hybrid"]

epsilon_list: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

A_list: list[float] = [1.0, 2.0, 3.0, 4.0, 5.0]


def main() -> None:
    for coupling_type in coupling_types:
        for epsilon in epsilon_list:
            for A in A_list:
                twins = Twins(
                    epsilon=epsilon,
                    A=A,
                    coupling_type=coupling_type,
                    show=False
                )
                twins.create_directories()
                rho_ts, photon_lists = twins.run_simulation()
                twins.qnd_analysis(rho_ts=rho_ts)
                twins.photon_analysis(photon_lists=photon_lists)
                twins.wigner_analysis(rho_ts=rho_ts)





if __name__ == "__main__":
    main()
