from fastapi import APIRouter, HTTPException
import json
import pandas as pd
from pathlib import Path

from models import (
    SystemInfo,
    MethodTimingResult,
    PromptTimingResult,
    TimingResults
)

# Configuration
DATA_DIR = Path("data")  # Path to data directory


# Create router
router = APIRouter(
    prefix="/performance",
    tags=["performance"],
    responses={404: {"description": "Not found"}},
)


@router.get("/system", response_model=SystemInfo)
def get_system_info() -> SystemInfo:
    """Get system information"""
    system_info_path = DATA_DIR / "system_info.json"
    if not system_info_path.exists():
        raise HTTPException(status_code=404, detail="System information not found")

    with open(system_info_path, 'r') as f:
        system_info = json.load(f)

    return SystemInfo(**system_info)


@router.get("/timing", response_model=TimingResults)
def get_timing_results() -> TimingResults:
    """Get timing results from CSV files"""
    method_timing_path = DATA_DIR / "method_timing_results.csv"
    prompt_timing_path = DATA_DIR / "prompt_timing_results.csv"

    if not (method_timing_path.exists() and prompt_timing_path.exists()):
        raise HTTPException(status_code=404, detail="Timing results not found")

    # Load CSVs using pandas
    method_df = pd.read_csv(method_timing_path)
    prompt_df = pd.read_csv(prompt_timing_path)

    # Convert to dictionaries for JSON response
    method_timing = method_df.to_dict(orient="records")
    prompt_timing = prompt_df.to_dict(orient="records")

    # Create Pydantic models
    method_timing_models = [MethodTimingResult(**item) for item in method_timing]
    prompt_timing_models = [PromptTimingResult(**item) for item in prompt_timing]

    return TimingResults(
        method_timing=method_timing_models,
        prompt_timing=prompt_timing_models
    )