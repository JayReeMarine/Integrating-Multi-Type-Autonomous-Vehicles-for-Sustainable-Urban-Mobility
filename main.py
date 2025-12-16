from data import load_mock_data
from greedy import greedy_platoon_matching

avs, pvs, l_min = load_mock_data()
assignments, total_saving = greedy_platoon_matching(avs, pvs, l_min)

print("Greedy Platoon Matching Results:")
for a in assignments:
    print(a.pv.id, "->", a.av.id, f"({a.cp} to {a.dp})", "saved:", a.saved_distance)
print("Total Energy Saving:", total_saving)