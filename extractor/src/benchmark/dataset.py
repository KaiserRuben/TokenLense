"""
Dataset loading and processing utilities for benchmarking.

This module provides functions to load and process datasets for attribution analysis.
"""

import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import traceback

from returns.result import Success, Failure, Result

from src.benchmark.config import DATASET_CONFIG

logger = logging.getLogger(__name__)


def load_dataset(
        source: Optional[str] = None,
        fallback_path: Optional[str] = None,
        limit: Optional[int] = None
) -> Result[List[str], Exception]:
    """
    Load prompts from a dataset.

    Args:
        source: Source path for the dataset (Hugging Face or local path)
        fallback_path: Fallback local path if source fails
        limit: Optional limit on number of prompts to load

    Returns:
        Result containing either a list of prompts or an exception
    """
    # Use default configuration if not specified
    if source is None:
        source = DATASET_CONFIG.get('primary_source', "")
    if fallback_path is None:
        fallback_path = DATASET_CONFIG.get('fallback_path', "")

    try:
        # Try to load from primary source
        try:
            logger.info(f"Loading dataset from {source}")
            df = pd.read_csv(source)
        except Exception as e:
            logger.warning(f"Failed to load from {source}: {e}")
            logger.info(f"Attempting to load from fallback path: {fallback_path}")
            # Fallback to local CSV if available
            df = pd.read_csv(fallback_path)

        # Log dataset info
        logger.info(f"Dataset loaded with {len(df)} entries")
        logger.info(f"Columns: {df.columns.tolist()}")

        # Extract prompts from dataframe
        prompts = extract_prompts_from_dataframe(df)

        # Limit number of prompts if specified
        if limit and limit > 0 and limit < len(prompts):
            logger.info(f"Limiting to {limit} prompts")
            prompts = prompts[:limit]

        # Log sample prompts
        for i, prompt in enumerate(prompts[:3]):
            logger.info(f"Sample prompt {i + 1}: {prompt[:100]}...")

        return Success(prompts)

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error(traceback.format_exc())
        return Failure(e)


def extract_prompts_from_dataframe(df: pd.DataFrame) -> List[str]:
    """
    Extract prompts from a dataframe based on column structure.

    Args:
        df: Pandas DataFrame containing prompt data

    Returns:
        List of extracted prompts
    """
    # Use full_prompt column if available
    if "full_prompt" in df.columns:
        return df["full_prompt"].tolist()

    # Otherwise combine system_instruction and user_request
    if "system_instruction" in df.columns and "user_request" in df.columns:
        return [
            f"{row['system_instruction']}\n\n{row['user_request']}"
            for _, row in df.iterrows()
        ]

    # If we don't have the expected columns, just use whatever we have
    logger.warning("Expected columns not found, using first text column")
    text_cols = [col for col in df.columns if df[col].dtype == 'object']
    if text_cols:
        return df[text_cols[0]].astype(str).tolist()

    # No usable text columns found
    logger.warning("No usable text columns found in dataset")
    return []