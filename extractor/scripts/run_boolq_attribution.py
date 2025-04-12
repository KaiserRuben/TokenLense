#!/usr/bin/env python
"""
Script to run token attribution analysis on the BoolQ dataset.

This script runs all permutations of model, attribution method, and prompts 
from the BoolQ dataset, saving and organizing the results. It uses the first 50
questions from the dataset which are shorter and more suitable for models like GPT-2.

Results are saved in the following formats:
- timing_results.csv: CSV with timing data for each model and attribution method
- summary.txt: Human-readable summary of results with system information
- system_info.json: Detailed system specifications in JSON format
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Configure MPS for Apple Silicon if needed
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import nltk
from returns.result import Success, Failure, Result

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
from src.benchmark.config import MODELS, ATTRIBUTION_METHODS
from src.benchmark.system import collect_system_info, format_system_info
from src.benchmark.runner import run_all_permutations
from src.benchmark.reporting import (
    create_output_directory,
    create_method_timing_dataframe,
    create_per_prompt_dataframe,
    save_timing_results,
    generate_summary_report
)

# Maximum number of BoolQ questions to use
MAX_QUESTIONS = 50


def load_boolq_dataset() -> Result[list, Exception]:
    """
    Load BoolQ dataset and extract the first 50 questions.
    
    Returns:
        Result containing a list of question prompts or an error
    """
    try:
        logger.info("Loading BoolQ dataset")
        
        # Define dataset paths
        splits = {
            'train': 'data/train-00000-of-00001.parquet', 
            'validation': 'data/validation-00000-of-00001.parquet'
        }
        
        # Load dataset
        df = pd.read_parquet("hf://datasets/google/boolq/" + splits["train"])
        
        # Extract questions and limit to MAX_QUESTIONS
        questions = df['question'].tolist()[:MAX_QUESTIONS]
        
        if not questions:
            return Failure(Exception("No questions found in BoolQ dataset"))
        
        # Format questions for LLM prompt
        prompts = []
        for q in questions:
            # Create a prompt from the question
            prompt = f"Answer this yes/no question: {q}"
            prompts.append(prompt)
        
        logger.info(f"Loaded {len(prompts)} BoolQ questions")
        return Success(prompts)
    
    except Exception as e:
        logger.error(f"Failed to load BoolQ dataset: {e}")
        return Failure(e)


def main():
    """Main function to run attribution analysis on BoolQ dataset."""
    logger.info("Starting BoolQ dataset attribution analysis")

    # Collect and log system information
    logger.info("Collecting system information...")
    system_info = collect_system_info()
    logger.info(format_system_info(system_info))

    # Create output directory
    output_dir, timestamp = create_output_directory(name="boolq_attribution")

    # Load dataset
    logger.info("Loading BoolQ dataset")
    dataset_result = load_boolq_dataset()

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
                max_workers=1 # Only one is working
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