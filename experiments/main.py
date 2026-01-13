from experiments.run_greedy_pv_av_sweep import run_pv_av_sweep as run_greedy_pv_av
from experiments.run_greedy_length_sweep import run_length_sweep as run_greedy_length
from experiments.run_greedy_capacity_sweep import run_capacity_sweep as run_greedy_capacity

from experiments.run_hungarian_pv_av_sweep import run_pv_av_sweep as run_hungarian_pv_av
from experiments.run_hungarian_length_sweep import run_length_sweep as run_hungarian_length
from experiments.run_hungarian_capacity_sweep import run_capacity_sweep as run_hungarian_capacity


def main() -> None:
    run_greedy_pv_av(output_csv="data/results/greedy/pv_av_sweep.csv")
    run_greedy_length(output_csv="data/results/greedy/length_sweep.csv")
    run_greedy_capacity(output_csv="data/results/greedy/capacity_sweep.csv")

    run_hungarian_pv_av(output_csv="data/results/hungarian/pv_av_sweep.csv")
    run_hungarian_length(output_csv="data/results/hungarian/length_sweep.csv")
    run_hungarian_capacity(output_csv="data/results/hungarian/capacity_sweep.csv")


if __name__ == "__main__":
    main()
