"""
evaluate.py
-----------
Metrics computed strictly from the acc_matrix produced by
training/engine_baseline.py's run_full_sequence().

acc_matrix[i][j] = accuracy (fraction in [0,1], NOT percent) on task i's
val set after training through task j. acc_matrix[i][j] is None for j < i
(task i hasn't been trained yet).

Forgetting metric (Eq. 20):
    F = 1/(n-1) * sum_{i=1}^{n-1} max_j( acc[i,j] - acc[i,n] )

IMPORTANT: this file asserts that all accuracies are stored as fractions
in [0, 1]. This directly guards against the scale bug from the last run
(reported F = 10.633, which is impossible if accuracies are fractions --
max possible F is 1.0). If this assertion fires, you have a scale bug
upstream (percentages mixed with fractions) -- fix it there, don't patch
it here.
"""

from typing import List, Optional


def _assert_valid_fraction(value: float, name: str):
    assert 0.0 <= value <= 1.0, (
        f"{name} = {value} is outside [0,1]. Accuracies must be stored as "
        f"fractions, not percentages. Fix the scale upstream in engine_baseline.py "
        f"before trusting any metric computed here."
    )


def average_accuracy(acc_matrix: List[List[Optional[float]]]) -> float:
    """Mean accuracy across all tasks, evaluated at the END of training
    (i.e. acc_matrix[i][n-1] for every task i)."""
    n = len(acc_matrix)
    final_col = n - 1
    accs = []
    for i in range(n):
        v = acc_matrix[i][final_col]
        assert v is not None, f"Task {i} was never evaluated after the final task."
        _assert_valid_fraction(v, f"acc_matrix[{i}][{final_col}]")
        accs.append(v)
    return sum(accs) / len(accs)


def forgetting_metric(acc_matrix: List[List[Optional[float]]]) -> float:
    """Eq. 20: F = 1/(n-1) * sum_i max_j(acc[i,j] - acc[i,n])."""
    n = len(acc_matrix)
    final_col = n - 1
    total = 0.0
    count = 0
    for i in range(n - 1):  # i = 1..n-1 in the paper's 1-indexed notation -> 0..n-2 here
        acc_i_n = acc_matrix[i][final_col]
        _assert_valid_fraction(acc_i_n, f"acc_matrix[{i}][{final_col}]")
        seen = [acc_matrix[i][j] for j in range(i, n) if acc_matrix[i][j] is not None]
        for v in seen:
            _assert_valid_fraction(v, f"acc_matrix[{i}][j]")
        max_drop = max(v - acc_i_n for v in seen)
        total += max_drop
        count += 1
    F = total / count
    _assert_valid_fraction(F, "forgetting_metric F")
    return F


def backward_transfer(acc_matrix: List[List[Optional[float]]]) -> float:
    """BWT = 1/(n-1) * sum_{i=1}^{n-1} (acc[n,i] - acc[i,i])
    (final-task-column accuracy on task i, minus task i's accuracy right
    after it was first trained)."""
    n = len(acc_matrix)
    final_col = n - 1
    total = 0.0
    count = 0
    for i in range(n - 1):
        acc_n_i = acc_matrix[i][final_col]
        acc_i_i = acc_matrix[i][i]
        _assert_valid_fraction(acc_n_i, f"acc_matrix[{i}][{final_col}]")
        _assert_valid_fraction(acc_i_i, f"acc_matrix[{i}][{i}]")
        total += (acc_n_i - acc_i_i)
        count += 1
    return total / count


def summarize(acc_matrix: List[List[Optional[float]]]) -> dict:
    return {
        "average_accuracy_pct": average_accuracy(acc_matrix) * 100,
        "forgetting_metric": forgetting_metric(acc_matrix),
        "backward_transfer_pct": backward_transfer(acc_matrix) * 100,
    }


if __name__ == "__main__":
    import json
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=str, default=None,
                        help="Path to accuracy matrix JSON (defaults to phase1_acc_matrix.json if exists, else phase0_acc_matrix.json)")
    args = parser.parse_args()

    if args.matrix:
        matrix_file = args.matrix
    elif os.path.exists("phase1_acc_matrix.json"):
        matrix_file = "phase1_acc_matrix.json"
    elif os.path.exists("phase0_acc_matrix.json"):
        matrix_file = "phase0_acc_matrix.json"
    else:
        raise FileNotFoundError("No accuracy matrix JSON found (neither phase1_acc_matrix.json nor phase0_acc_matrix.json).")

    with open(matrix_file) as f:
        acc_matrix = json.load(f)
    results = summarize(acc_matrix)
    print(f"=== Results ({matrix_file}) ===")
    for k, v in results.items():
        print(f"{k}: {v:.3f}")
    print("\nCompare against paper's claim: 72.6% accuracy, 0.183 forgetting metric.")
