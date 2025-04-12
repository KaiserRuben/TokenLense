#!/usr/bin/env python
"""
Script to run token attribution analysis on the SimpleQuestions dataset.

This script runs all permutations of model, attribution method, and prompts
from the SimpleQuestions dataset, saving and organizing the results. It uses the first 50
questions from the dataset which are more suitable for models like GPT-2.

Results are saved in the following formats:
- timing_results.csv: CSV with timing data for each model and attribution method
- summary.txt: Human-readable summary of results with system information
- system_info.json: Detailed system specifications in JSON format
"""

import os
import sys
import logging
import requests
import tarfile
import tempfile
import shutil
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
        logging.FileHandler("simple_questions_attribution.log"),
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

# Maximum number of SimpleQuestions to use
MAX_QUESTIONS = 50


def load_simple_questions_dataset() -> Result[list, Exception]:
    """
    Load SimpleQuestions dataset and extract the first 50 questions.

    Returns:
        Result containing a list of question prompts or an error
    """
    try:
        logger.info("Loading SimpleQuestions dataset")

        # Define the Dropbox URL as shown in the original SimpleQuestions_v2 code
        url = "https://www.dropbox.com/s/tohrsllcfy7rch4/SimpleQuestions_v2.tgz?dl=1"

        # Create a temporary directory for downloading and extraction
        temp_dir = tempfile.mkdtemp()
        download_path = os.path.join(temp_dir, "SimpleQuestions_v2.tgz")
        extract_dir = os.path.join(temp_dir, "extracted")

        try:
            # Download the dataset
            logger.info(f"Downloading SimpleQuestions dataset from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(download_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Extract the tar file
            logger.info(f"Extracting dataset to {extract_dir}")
            os.makedirs(extract_dir, exist_ok=True)

            with tarfile.open(download_path, "r:gz") as tar:
                tar.extractall(path=extract_dir)

            # Path to the annotated training data
            data_path = os.path.join(extract_dir, "SimpleQuestions_v2", "annotated_fb_data_train.txt")

            if not os.path.exists(data_path):
                return Failure(Exception(f"Could not find extracted file at {data_path}"))

            # Parse the file according to the original SimpleQuestions_v2 code
            questions = []
            with open(data_path, encoding="utf-8") as f:
                for i, row in enumerate(f):
                    if i >= MAX_QUESTIONS:
                        break

                    row = row.split("\t")
                    if len(row) >= 4:  # Ensure we have enough columns
                        questions.append(row[3].strip())  # Question is in the 4th column

            logger.info(f"Successfully loaded {len(questions)} questions from SimpleQuestions dataset")

        except Exception as e:
            logger.error(f"Failed to download or process SimpleQuestions dataset: {e}")
            return Failure(e)
        finally:
            # Clean up temporary files
            logger.info("Cleaning up temporary files")
            shutil.rmtree(temp_dir, ignore_errors=True)

        if not questions:
            return Failure(Exception("No questions found in SimpleQuestions dataset"))

        if not questions:
            return Failure(Exception("No questions found in SimpleQuestions dataset"))

        # Format questions for LLM prompt
        prompts = []
        for q in questions:
            prompts.append(q)

        logger.info(f"Loaded {len(prompts)} SimpleQuestions")
        return Success(prompts)

    except Exception as e:
        logger.error(f"Failed to load SimpleQuestions dataset: {e}")
        return Failure(e)


def main():
    """Main function to run attribution analysis on SimpleQuestions dataset."""
    logger.info("Starting SimpleQuestions dataset attribution analysis")

    # Collect and log system information
    logger.info("Collecting system information...")
    system_info = collect_system_info()
    logger.info(format_system_info(system_info))

    # Create output directory
    output_dir, timestamp = create_output_directory(name="simple_questions_attribution")

    # Load dataset
    logger.info("Loading SimpleQuestions dataset")
    dataset_result = load_simple_questions_dataset()

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
                max_workers=1  # Only one worker at a time
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