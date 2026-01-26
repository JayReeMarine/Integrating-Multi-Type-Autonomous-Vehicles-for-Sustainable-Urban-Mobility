from experiments.run_greedy_pv_av_sweep import run_pv_av_sweep as run_greedy_pv_av
from experiments.run_greedy_length_sweep import run_length_sweep as run_greedy_length
from experiments.run_greedy_capacity_sweep import run_capacity_sweep as run_greedy_capacity

from experiments.run_hungarian_pv_av_sweep import run_pv_av_sweep as run_hungarian_pv_av
from experiments.run_hungarian_length_sweep import run_length_sweep as run_hungarian_length
from experiments.run_hungarian_capacity_sweep import run_capacity_sweep as run_hungarian_capacity


def main() -> None:
    time_params = {
        "enable_time_constraints": True,
        "time_tolerance": 5.0,
        "time_window": 100.0,
        "av_speed_range": (0.8, 1.2),
        "pv_speed_range": (0.8, 1.2),
    }
    
    # Greedy
    run_greedy_pv_av(
        output_csv="data/results/greedy/pv_av_sweep.csv",
        **time_params  
    )
    run_greedy_length(
        output_csv="data/results/greedy/length_sweep.csv",
        **time_params
    )
    run_greedy_capacity(
        output_csv="data/results/greedy/capacity_sweep.csv",
        **time_params
    )
    
    # Hungarian
    run_hungarian_pv_av(
        output_csv="data/results/hungarian/pv_av_sweep.csv",
        **time_params
    )
    run_hungarian_length(
        output_csv="data/results/hungarian/length_sweep.csv",
        **time_params
    )
    run_hungarian_capacity(
        output_csv="data/results/hungarian/capacity_sweep.csv",
        **time_params
    )

if __name__ == "__main__":
    main()
