"""
Reporting utilities for benchmarking.

This module provides functions to process, visualize, and save benchmark results.
"""

import logging
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd

from src.benchmark.schema import (
    SystemInfo, AnalysisResultData, PromptResult
)
from src.persistence.storage import TokenAnalysisStorage

logger = logging.getLogger(__name__)


def create_output_directory(name="attribution") -> Tuple[str, str]:
    """
    Create output directory for results.

    Returns:
        Tuple of (output_dir, timestamp)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", name, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved to {output_dir}")

    return output_dir, timestamp


def create_output_structure(
        model_name: str,
        method: str,
        base_path: str
) -> Tuple[Path, TokenAnalysisStorage]:
    """
    Create output directory structure and storage handler.

    Args:
        model_name: Name of the model
        method: Attribution method
        base_path: Base output directory

    Returns:
        Tuple of (output_path, storage_handler)
    """
    # Create specific output path for this model and method
    safe_model_name = model_name.replace("/", "_")
    output_path = Path(base_path) / safe_model_name / f"method_{method}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize storage
    storage = TokenAnalysisStorage(base_path=str(output_path))

    return output_path, storage


def create_per_prompt_dataframe(results: Dict[str, AnalysisResultData]) -> pd.DataFrame:
    """
    Create a DataFrame with per-prompt timing results.

    Args:
        results: Dictionary of analysis results

    Returns:
        DataFrame with per-prompt attribution timing data
    """
    all_prompt_rows = []

    for model_method, result in results.items():
        model_name, method = model_method.split('/')

        for prompt_result in result.prompt_results:
            # Get the basic prompt row
            row = prompt_result.to_row()

            # Add model and method information
            row["model"] = model_name
            row["attribution_method"] = method

            # Add system info
            system_info = result.system_info
            row["device"] = "CUDA" if system_info.torch_cuda_available else (
                "MPS" if system_info.torch_mps_available else "CPU")
            row["gpu_info"] = system_info.gpu_info

            all_prompt_rows.append(row)

    # Create DataFrame
    if not all_prompt_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_prompt_rows)

    # Reorder columns for better readability
    if not df.empty:
        first_columns = ["model", "attribution_method", "prompt_id", "token_count", "output_token_count"]
        timing_columns = ["attribution_time", "tokens_per_second"]

        # Get remaining columns
        other_columns = [col for col in df.columns if col not in first_columns + timing_columns]

        # Reorder columns
        df = df[first_columns + timing_columns + other_columns]

    return df


def create_method_timing_dataframe(results: Dict[str, AnalysisResultData]) -> pd.DataFrame:
    """
    Create a DataFrame with method-level aggregated timing results.

    Args:
        results: Dictionary of analysis results

    Returns:
        DataFrame with aggregated timing data per method
    """
    # Extract timing data from results
    timing_data = [result.to_timing_row() for result in results.values()]

    # Create DataFrame
    df = pd.DataFrame(timing_data)

    # Reorder columns for better readability
    if not df.empty:
        first_columns = ["model", "attribution_method", "successful_prompts", "total_prompts", "success_rate"]
        timing_columns = ["model_loading_time", "attribution_time", "average_prompt_time", "total_time"]

        # Get system info columns
        system_columns = ["platform", "cpu_model", "cpu_cores", "memory_gb", "gpu_info",
                          "cuda_available", "mps_available", "torch_version"]

        # Get remaining columns
        other_columns = [col for col in df.columns
                         if col not in first_columns + timing_columns + system_columns]

        # Reorder columns
        df = df[first_columns + timing_columns + system_columns + other_columns]

    return df


def save_timing_results(method_df: pd.DataFrame, prompt_df: pd.DataFrame, output_dir: str) -> Tuple[str, str]:
    """
    Save timing results to CSV files and log summary statistics.

    Args:
        method_df: DataFrame with method-level timing data
        prompt_df: DataFrame with per-prompt timing data
        output_dir: Output directory

    Returns:
        Tuple of paths to saved CSV files (method_csv_path, prompt_csv_path)
    """
    # Check for empty dataframes
    if method_df.empty and prompt_df.empty:
        logger.warning("No timing data to save")
        return "", ""

    method_csv_path = ""
    prompt_csv_path = ""

    # Save method-level timing data
    if not method_df.empty:
        method_csv_path = os.path.join(output_dir, "method_timing_results.csv")
        method_df.to_csv(method_csv_path, index=False)
        logger.info(f"Method-level timing results saved to {method_csv_path}")

        # Print summary statistics
        logger.info("\nMethod Timing Summary:")
        try:
            fastest_model = method_df.loc[method_df['model_loading_time'].idxmin()]
            logger.info(f"Fastest model loading: {fastest_model['model']} ({fastest_model['model_loading_time']:.2f}s)")

            fastest_attribution = method_df.loc[method_df['attribution_time'].idxmin()]
            logger.info(
                f"Fastest attribution method: {fastest_attribution['model']}/{fastest_attribution['attribution_method']} ({fastest_attribution['attribution_time']:.2f}s)")

            fastest_prompt = method_df.loc[method_df['average_prompt_time'].idxmin()]
            logger.info(
                f"Fastest average prompt time: {fastest_prompt['model']}/{fastest_prompt['attribution_method']} ({fastest_prompt['average_prompt_time']:.2f}s)")
        except Exception as e:
            logger.error(f"Error generating method timing summary: {e}")

    # Save per-prompt timing data
    if not prompt_df.empty:
        prompt_csv_path = os.path.join(output_dir, "prompt_timing_results.csv")
        prompt_df.to_csv(prompt_csv_path, index=False)
        logger.info(f"Per-prompt timing results saved to {prompt_csv_path}")

        # Print summary statistics for per-prompt data
        logger.info("\nPer-Prompt Timing Summary:")
        try:
            # Find prompt with fastest attribution time
            fastest_prompt = prompt_df.loc[prompt_df['attribution_time'].idxmin()]
            logger.info(
                f"Fastest prompt: {fastest_prompt['prompt_id']} for {fastest_prompt['model']}/{fastest_prompt['attribution_method']} ({fastest_prompt['attribution_time']:.2f}s)")

            # Prompt with highest tokens per second
            if 'tokens_per_second' in prompt_df.columns:
                fastest_throughput = prompt_df.loc[prompt_df['tokens_per_second'].idxmax()]
                logger.info(
                    f"Highest token throughput: {fastest_throughput['prompt_id']} for {fastest_throughput['model']}/{fastest_throughput['attribution_method']} ({fastest_throughput['tokens_per_second']:.2f} tokens/s)")

            # Group by model and method to get average timing stats
            if len(prompt_df) > 1:
                grouped = prompt_df.groupby(['model', 'attribution_method']).agg({
                    'attribution_time': 'mean',
                    'tokens_per_second': 'mean' if 'tokens_per_second' in prompt_df.columns else 'sum',
                    'token_count': ['sum', 'mean', 'std'],
                    'output_token_count': ['sum', 'mean', 'std']
                }).reset_index()

                # Flatten multi-level columns
                grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]

                # Save aggregated stats
                agg_csv_path = os.path.join(output_dir, "prompt_aggregated_results.csv")
                grouped.to_csv(agg_csv_path, index=False)
                logger.info(f"Prompt aggregated results saved to {agg_csv_path}")

        except Exception as e:
            logger.error(f"Error generating prompt timing summary: {e}")

    return method_csv_path, prompt_csv_path


def generate_summary_report(
        results: Dict[str, AnalysisResultData],
        prompts: List[str],
        timestamp: str,
        output_dir: str
) -> str:
    """
    Generate and save a detailed summary report.

    Args:
        results: Dictionary of analysis results
        prompts: List of prompts processed
        timestamp: Analysis timestamp
        output_dir: Output directory

    Returns:
        Path to saved summary report
    """
    summary_path = os.path.join(output_dir, "summary.txt")

    with open(summary_path, "w") as f:
        f.write(f"FACTS Attribution Analysis - {timestamp}\n")
        f.write(f"Total prompts: {len(prompts)}\n\n")

        # Add system information if we have results
        if results:
            # Use system info from the first result (should be the same for all)
            first_result = next(iter(results.values()))
            f.write(format_system_info(first_result.system_info))
            f.write("\n\n")

        f.write("Results by model/method combination:\n")

        for combo_key, result in results.items():
            f.write(f"\n{combo_key}:\n")
            f.write(
                f"  Success rate: {result.success_rate:.1f}% ({result.successful_prompts}/{result.total_prompts})\n")

            # Add timing information
            timing = result.timing
            f.write(f"  Timing:\n")
            f.write(f"    Model loading time: {timing.model_loading_time:.2f} seconds\n")
            f.write(f"    Attribution time: {timing.attribution_time:.2f} seconds\n")
            f.write(f"    Average time per prompt: {timing.average_prompt_time:.2f} seconds\n")
            f.write(f"    Total execution time: {timing.total_time:.2f} seconds\n")

            # Write error examples
            if result.errors:
                f.write(f"  Errors: {len(result.errors)}\n")
                for i, error in enumerate(result.errors[:5]):  # Show first 5 errors
                    prompt_id = error.prompt_id or "unknown"
                    f.write(f"    - {prompt_id}: {error.error}\n")
                if len(result.errors) > 5:
                    f.write(f"    - ...and {len(result.errors) - 5} more errors\n")

    # Also save system info separately in JSON format for easy parsing
    if results:
        first_result = next(iter(results.values()))
        system_info_path = os.path.join(output_dir, "system_info.json")
        with open(system_info_path, "w") as f:
            json.dump(first_result.system_info.to_dict(), f, indent=2)
        logger.info(f"System information saved to {system_info_path}")

    logger.info(f"Summary report saved to {summary_path}")
    return summary_path


def format_system_info(system_info: SystemInfo) -> str:
    """
    Format system information for display in summary report.

    Args:
        system_info: SystemInfo object

    Returns:
        Formatted string with system information
    """
    lines = [
        f"System Information:",
        f"  Platform: {system_info.platform} ({system_info.platform_version})",
        f"  CPU: {system_info.cpu_model or system_info.processor}",
        f"  CPU Cores: {system_info.cpu_cores}",
        f"  Memory: {system_info.memory_total_gb:.2f} GB",
    ]

    # GPU information
    if system_info.torch_cuda_available:
        lines.append(f"  GPU: {system_info.gpu_info} (CUDA {system_info.cuda_version})")
    elif system_info.torch_mps_available:
        lines.append(f"  GPU: {system_info.gpu_info} (Apple MPS)")
    else:
        lines.append(f"  GPU: None")

    # PyTorch version
    lines.append(f"  PyTorch: {system_info.torch_version}")

    # Device availability
    devices = []
    if system_info.torch_cuda_available:
        devices.append("CUDA")
    if system_info.torch_mps_available:
        devices.append("MPS")
    if not devices:
        devices.append("CPU only")

    lines.append(f"  Available Devices: {', '.join(devices)}")

    return "\n".join(lines)