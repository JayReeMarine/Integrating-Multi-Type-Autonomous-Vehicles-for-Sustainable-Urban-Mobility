from visualization.plot_greedy_pv_av_sweep import main as plot_greedy_pv_av_main
from visualization.plot_greedy_length_sweep import main as plot_greedy_length_main
from visualization.plot_greedy_capacity_sweep import main as plot_greedy_capacity_main

from visualization.plot_hungarian_pv_av_sweep import main as plot_hungarian_pv_av_main
from visualization.plot_hungarian_length_sweep import main as plot_hungarian_length_main
from visualization.plot_hungarian_capacity_sweep import main as plot_hungarian_capacity_main


def main() -> None:
    print("\n==============================")
    print(" Plot All: Greedy & Hungarian ")
    print("==============================\n")

    print("[1/6] Plotting Greedy PV/AV sweep...")
    plot_greedy_pv_av_main()

    print("\n[2/6] Plotting Greedy length sweep...")
    plot_greedy_length_main()

    print("\n[3/6] Plotting Greedy capacity sweep...")
    plot_greedy_capacity_main()

    print("\n[4/6] Plotting Hungarian PV/AV sweep...")
    plot_hungarian_pv_av_main()

    print("\n[5/6] Plotting Hungarian length sweep...")
    plot_hungarian_length_main()

    print("\n[6/6] Plotting Hungarian capacity sweep...")
    plot_hungarian_capacity_main()

    print("\n✅ All plots generated (Greedy + Hungarian).")


if __name__ == "__main__":
    main()
