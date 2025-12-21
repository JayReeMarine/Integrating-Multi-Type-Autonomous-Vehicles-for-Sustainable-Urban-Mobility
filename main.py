from data import generate_mock_data
from greedy import greedy_platoon_matching
from metrics import compute_saving_stats, baseline_total_powered_distance


def main():
    # -----------------------
    # Experiment parameters
    # -----------------------
    NUM_AV = 5
    NUM_PV = 20
    HIGHWAY_LENGTH = 100
    AV_CAPACITY_RANGE = (1, 3)
    MIN_TRIP_LENGTH = 10
    SEED = 42

    # Generate experiment data
    avs, pvs, l_min = generate_mock_data(
        num_av=NUM_AV,
        num_pv=NUM_PV,
        highway_length=HIGHWAY_LENGTH,
        av_capacity_range=AV_CAPACITY_RANGE,
        min_trip_length=MIN_TRIP_LENGTH,
        seed=SEED,
    )

    # Baseline stats
    baseline_total = baseline_total_powered_distance(avs, pvs)

    # Greedy
    assignments, greedy_saving = greedy_platoon_matching(avs, pvs, l_min)

    # Comparison stats (percent saving)
    baseline_total, greedy_total, saving_percent = compute_saving_stats(avs, pvs, assignments)
    saved_distance = baseline_total - greedy_total  # == sum(dp-cp)

    print("=== Baseline (No Platooning) ===")
    print(f"Total powered distance: {baseline_total}")
    print("Saved distance: 0")
    print("Saving percent: 0.00%")

    print("\n=== Greedy Platooning ===")
    print(f"Total powered distance: {greedy_total}")
    print(f"Saved distance (sum(dp-cp)): {saved_distance}")
    print(f"Saving percent: {saving_percent:.2f}%")

    print("\nAssignments:")
    for a in assignments:
        print(f"{a.pv.id} -> {a.av.id} (cp={a.cp}, dp={a.dp}), saved={a.saved_distance}")

    # (Optional) Also show the total_saving calculated from greedy.py
    print(f"\nGreedy total_saving (from algorithm): {greedy_saving}")


if __name__ == "__main__":
    main()
