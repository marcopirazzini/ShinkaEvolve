"""
Run script for evolving the sample program to maximize the top entry in a unit norm vector.

Usage:
    cd /path/to/ShinkaEvolve
    source .venv/bin/activate
    python /path/to/sample_problem/run_evolution.py
"""

from pathlib import Path

from shinka.core import EvolutionRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
INIT_PROGRAM = SCRIPT_DIR / "initial_program.py"
EVAL_PROGRAM = SCRIPT_DIR / "evaluate.py"
RESULTS_DIR = SCRIPT_DIR / "results"

# Task description for the LLM - Make sure to add expert knowledge here
TASK_DESCRIPTION = """
You have to solve a very easy task, let's see if you can figure it out.
"""
# TASK_DESCRIPTION = """
# You want to find an algorithm to construct a vector with maximum largest entry relative to its euclidean norm.
# """


def main():
    # Job configuration - local execution
    job_config = LocalJobConfig(
        eval_program_path=str(EVAL_PROGRAM),
    )

    # Database configuration
    db_config = DatabaseConfig(
        num_islands=4,  # 2 parallel evolution islands
        archive_size=10,
        num_archive_inspirations=3,
        num_top_k_inspirations=2,
    )

    # Evolution configuration
    evo_config = EvolutionConfig(
        init_program_path=str(INIT_PROGRAM),
        results_dir=str(RESULTS_DIR),
        task_sys_msg=TASK_DESCRIPTION,
        num_generations=50,
        max_parallel_jobs=2,
        llm_models=["openai/gpt-4o-mini"],
        # llm_models=["google/gemini-3-flash-preview"],
        embedding_model="openai/text-embedding-3-small",
        patch_types=["diff", "full"],
        patch_type_probs=[0.8, 0.2],
    )

    print(f"Starting sample program evolution...")
    print(f"  Initial program: {INIT_PROGRAM}")
    print(f"  Evaluation script: {EVAL_PROGRAM}")
    print(f"  Results directory: {RESULTS_DIR}")
    print(f"  Generations: {evo_config.num_generations}")
    print(f"  LLM models: {evo_config.llm_models}")
    print()

    runner = EvolutionRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
    )
    runner.run()


if __name__ == "__main__":
    main()
