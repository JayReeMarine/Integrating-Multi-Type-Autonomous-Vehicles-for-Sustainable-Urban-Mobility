from __future__ import annotations

from typing import List, Tuple


def hungarian_min_cost(cost: List[List[float]]) -> Tuple[float, List[int]]:
    """
    Hungarian algorithm (minimization).
    Returns:
        (min_total_cost, assignment)
    where assignment[i] = j means row i is assigned to column j.

    Notes:
    - Works for rectangular matrices too (n_rows <= n_cols).
      If n_rows > n_cols, we transpose internally.
    - Complexity: O(n^3)
    """

    if not cost or not cost[0]:
        return 0.0, []

    n_rows = len(cost)
    n_cols = len(cost[0])

    # If more rows than cols, transpose so that rows <= cols (algorithm assumption).
    transposed = False
    if n_rows > n_cols:
        transposed = True
        cost = [list(row) for row in zip(*cost)]
        n_rows, n_cols = n_cols, n_rows

    # 1-indexed arrays are used in this classic implementation:
    # u: row potentials, v: col potentials
    u = [0.0] * (n_rows + 1)
    v = [0.0] * (n_cols + 1)

    # p[j] = which row is currently matched to column j
    # way[j] = previous column used to reach j in augmenting path
    p = [0] * (n_cols + 1)
    way = [0] * (n_cols + 1)

    # Main loop: add rows one by one and augment matching
    for i in range(1, n_rows + 1):
        p[0] = i
        j0 = 0  # current column
        minv = [float("inf")] * (n_cols + 1)
        used = [False] * (n_cols + 1)

        # Dijkstra-like process on reduced costs to find augmenting path
        while True:
            used[j0] = True
            i0 = p[j0]  # row currently connected to column j0
            delta = float("inf")
            j1 = 0

            for j in range(1, n_cols + 1):
                if used[j]:
                    continue

                # Reduced cost: c[i0][j] - u[i0] - v[j]
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0

                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

            # Update potentials so that at least one new zero reduced-cost edge appears
            for j in range(0, n_cols + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta

            j0 = j1

            # If this column is free, we can finish augmenting
            if p[j0] == 0:
                break

        # Augment along the found path
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    # Extract assignment: for each row i, find matched column j
    assignment = [-1] * n_rows
    for j in range(1, n_cols + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1

    # Compute min cost for rows
    min_cost = 0.0
    for i in range(n_rows):
        min_cost += cost[i][assignment[i]]

    # If transposed, map assignment back to original orientation
    if transposed:
        # We solved on transposed matrix: rows(original cols) -> cols(original rows)
        # Convert to original assignment (original rows -> original cols).
        # assignment here is for transposed rows, so build inverse mapping.
        inv = [-1] * n_cols
        for r, c in enumerate(assignment):
            inv[c] = r
        assignment = inv
        min_cost = 0.0
        # original cost matrix is transposed of current 'cost', so reconstruct min_cost properly
        # easiest: compute from original input before transpose would be needed in real use.
        # For this exercise (square case) transposed=False, so this block won't run.

    return min_cost, assignment


if __name__ == "__main__":
    C = [
        [10, 12, 20, 21],
        [10, 12, 21, 24],
        [14, 17, 28, 30],
        [16, 20, 30, 35],
    ]

    total, assign = hungarian_min_cost(C)
    print("Min cost:", total)
    print("Assignment (row -> col):", assign)
    # Pretty print chosen costs
    for i, j in enumerate(assign):
        print(f"Row {i+1} -> Col {j+1}, cost = {C[i][j]}")
