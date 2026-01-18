from visualization.plot_greedy_pv_av_sweep import main as plot_greedy_pv_av_main
from visualization.plot_greedy_length_sweep import main as plot_greedy_length_main
from visualization.plot_greedy_capacity_sweep import main as plot_greedy_capacity_main

from visualization.plot_hungarian_pv_av_sweep import main as plot_hungarian_pv_av_main
from visualization.plot_hungarian_length_sweep import main as plot_hungarian_length_main
from visualization.plot_hungarian_capacity_sweep import main as plot_hungarian_capacity_main

from visualization.compare_algorithms import main as compare_algorithms_main


def main() -> None:
    print("\n==============================")
    print(" Plot All: Greedy & Hungarian ")
    print("==============================\n")

    print("[1/7] Plotting Greedy PV/AV sweep...")
    plot_greedy_pv_av_main()

    print("\n[2/7] Plotting Greedy length sweep...")
    plot_greedy_length_main()

    print("\n[3/7] Plotting Greedy capacity sweep...")
    plot_greedy_capacity_main()

    print("\n[4/7] Plotting Hungarian PV/AV sweep...")
    plot_hungarian_pv_av_main()

    print("\n[5/7] Plotting Hungarian length sweep...")
    plot_hungarian_length_main()

    print("\n[6/7] Plotting Hungarian capacity sweep...")
    plot_hungarian_capacity_main()

    print("\n[7/7] Generating comparison analysis...")
    compare_algorithms_main()

    print("\n All plots generated (Greedy + Hungarian + Comparison).")


if __name__ == "__main__":
    main()
