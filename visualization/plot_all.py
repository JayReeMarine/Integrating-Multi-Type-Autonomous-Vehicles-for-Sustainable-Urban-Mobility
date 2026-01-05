from visualization.plot_greedy_pv_av_sweep import main as plot_pv_av_main
from visualization.plot_greedy_length_sweep import main as plot_length_main


def main() -> None:
    print("\n==============================")
    print(" Plot All: Greedy Experiments ")
    print("==============================\n")

    print("[1/2] Plotting PV/AV sweep...")
    plot_pv_av_main()

    print("\n[2/2] Plotting length sweep...")
    plot_length_main()

    print("\n✅ All plots generated.")


if __name__ == "__main__":
    main()
