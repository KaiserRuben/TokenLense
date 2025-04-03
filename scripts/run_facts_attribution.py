#!/usr/bin/env python
"""
Script to run token attribution analysis on the FACTS dataset.

This script runs all permutations of model, attribution method, and prompts 
from the FACTS dataset, saving and organizing the results. It also collects
detailed system information to contextualize the timing measurements, including
CPU/GPU specifications and PyTorch configuration.

Results are saved in the following formats:
- timing_results.csv: CSV with timing data for each model and attribution method
- summary.txt: Human-readable summary of results with system information
- system_info.json: Detailed system specifications in JSON format
"""

import os
import sys
import logging
from pathlib import Path

# Configure MPS for Apple Silicon if needed
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import nltk
from returns.result import Success, Failure

# Initialize NLTK dependencies
nltk.download('averaged_perceptron_tagger_eng')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("attribution_benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import benchmark components
from src.benchmark.config import MODELS, ATTRIBUTION_METHODS, MAX_PROMPTS
from src.benchmark.system import collect_system_info, format_system_info
from src.benchmark.dataset import load_dataset
from src.benchmark.runner import run_all_permutations
from src.benchmark.reporting import (
    create_output_directory,
    create_method_timing_dataframe,
    create_per_prompt_dataframe,
    save_timing_results,
    generate_summary_report
)


def main():
    """Main function to run attribution analysis on FACTS dataset."""
    logger.info("Starting FACTS dataset attribution analysis")

    # Collect and log system information
    logger.info("Collecting system information...")
    system_info = collect_system_info()
    logger.info(format_system_info(system_info))

    # Create output directory
    output_dir, timestamp = create_output_directory()

    # Load dataset
    logger.info("Loading FACTS dataset")
    dataset_result = load_dataset(limit=MAX_PROMPTS)

    match dataset_result:
        case Success(prompts):
            if not prompts:
                logger.error("Dataset loaded but contains no prompts, exiting")
                return

            logger.info(f"Processing {len(prompts)} prompts")

            # Run analysis for all permutations
            logger.info("Starting attribution analysis for all model/method combinations")
            results = run_all_permutations(
                prompts=prompts,
                output_dir=output_dir,
                models=MODELS,
                methods=ATTRIBUTION_METHODS,
                max_workers=1  # Can increase if machine has enough RAM
            )

            if not results:
                logger.error("No results generated, exiting")
                return

            # Create timing dataframes
            method_timing_df = create_method_timing_dataframe(results)
            prompt_timing_df = create_per_prompt_dataframe(results)

            # Save timing results
            save_timing_results(method_timing_df, prompt_timing_df, output_dir)

            # Generate summary report
            generate_summary_report(results, prompts, timestamp, output_dir)

            logger.info("Attribution analysis completed successfully")

        case Failure(error):
            logger.error(f"Failed to load dataset: {error}")
            logger.error("Attribution analysis aborted")


if __name__ == "__main__":
    main()