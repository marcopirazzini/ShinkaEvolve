"""
Evaluation script for the sample program of maximizing the largest entry in a unit-norm vector.
"""

import argparse
from shinka.core import run_shinka_eval


def get_experiment_kwargs(run_idx: int) -> dict:
    """Generate kwargs for each evaluation run."""
    return {
        "seed": run_idx,
        "n": 10, 
    }


def validate_fn(result) -> tuple[bool, str | None]:
    """Validate that the result is well-formed. Returns (is_valid, error_msg)."""
    if not isinstance(result, dict):
        return False, "Result is not a dictionary"
    if "valid" not in result:
        return False, "Result missing 'valid' key"
    return True, None


def aggregate_metrics_fn(results: list) -> dict:
    """
    Aggregate results from multiple runs.

    For this problem, we want to find the maximum top_entry across all seeds.
    """
    valid_results = [r for r in results if r.get("valid", False)]

    INVALID_PENALTY = -1000.0

    if not valid_results:
        return {
            "combined_score": INVALID_PENALTY,
            "public": {
                "status": "invalid",
                "num_valid": 0,
                "num_runs": len(results),
            },
            "private": {},
        }

    # Extract metrics from all valid results
    top_entries = [r["top_entry"] for r in valid_results]
    n = valid_results[0]["n"]

    max_entry = max(top_entries)
    avg_entry = sum(top_entries) / len(top_entries)

    # The score is the maximum entry, the average is used as additional metric
    combined_score = max_entry

    return {
        "combined_score": combined_score,
        "public": {
            "status": "valid",
            "max_top_entry": max_entry,
            "avg_top_entry": avg_entry,
            "n": n,
            "num_valid": len(valid_results),
            "num_runs": len(results),
        },
        "private": {
            "all_top_entries": top_entries,
        },
    }


def main(program_path: str, results_dir: str):
    """Main evaluation entry point."""
    metrics, correct, err = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_experiment",
        num_runs=5,  # Test with 5 different seeds
        get_experiment_kwargs=get_experiment_kwargs,
        aggregate_metrics_fn=aggregate_metrics_fn,
        validate_fn=validate_fn,
    )
    return metrics, correct, err


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()

    metrics, correct, err = main(args.program_path, args.results_dir)
    print(f"Metrics: {metrics}")
    print(f"Correct: {correct}")
    if err:
        print(f"Error: {err}")
