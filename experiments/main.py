from experiments.run_greedy_pv_av_sweep import run_pv_av_sweep
from experiments.run_greedy_length_sweep import run_length_sweep
from experiments.run_greedy_capacity_sweep import run_capacity_sweep


def main() -> None:
    run_pv_av_sweep(output_csv="data/results/greedy/pv_av_sweep.csv")
    run_length_sweep(output_csv="data/results/greedy/length_sweep.csv")
    run_capacity_sweep(output_csv="data/results/greedy/capacity_sweep.csv")


if __name__ == "__main__":
    main()
