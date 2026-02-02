"""
Initial program for the desired task.
"""

import numpy as np
import scipy.linalg as spl


# EVOLVE-BLOCK-START
def construct_solution(n: int=10, seed:int=42) -> np.ndarray:
    np.random.seed(seed)
    solution = np.random.randn(n)
    return solution
# EVOLVE-BLOCK-END


def auxiliary_function(sol: np.ndarray) -> np.ndarray:
    return sol / spl.norm(sol,2)


def run_experiment(seed: int = 42, **kwargs) -> dict:
    """
    Main experiment function called by the evaluator.
    """
    n = kwargs.get('n', 10)  # Number/dimension of matrices

    # Construct the solution
    sol = construct_solution(n=n, seed=seed)

    # Normalize the solution
    sol = auxiliary_function(sol)

    return {
        "valid": True,
        "n": n,
        "top_entry": max(sol),  # This is what we want to maximize
    }


if __name__ == "__main__":
    result = run_experiment(seed=42)
    print(f"Results: {result}")
    if result["valid"]:
        print(f"  Best objective: {result['top_entry']:.4f}")
