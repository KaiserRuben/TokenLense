from fastapi import APIRouter, HTTPException, Query
import json
import sys
from pathlib import Path

from data_loader import get_available_models_and_methods
from models import ModelInfo, ModelMethods, ModelMethodFile

# Configuration
DATA_DIR = Path("data")  # Path to data directory


# Create router
router = APIRouter(
    prefix="/models",
    tags=["models"],
    responses={404: {"description": "Not found"}},
)


@router.get("/", response_model=ModelInfo)
def get_models() -> ModelInfo:
    """Get list of available models"""
    models_methods = get_available_models_and_methods(DATA_DIR)
    return ModelInfo(models=list(models_methods.keys()))


@router.get("/{model}/methods", response_model=ModelMethods)
def get_methods(model: str) -> ModelMethods:
    """Get attribution methods available for a model"""
    models_methods = get_available_models_and_methods(DATA_DIR)
    if model not in models_methods:
        raise HTTPException(status_code=404, detail=f"Model {model} not found")
    return ModelMethods(model=model, methods=models_methods[model])


@router.get("/{model}/methods/{method}/files", response_model=ModelMethodFile)
def get_files(model: str, method: str, include_details: bool = False) -> ModelMethodFile:
    """Get attribution files for a model and method"""
    # First check if model exists
    model_dir = DATA_DIR / model
    if not model_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model {model} not found"
        )
        
    # Then check if method exists
    method_dir = model_dir / f"method_{method}" / "data"
    if not method_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No data directory found for model {model} and method {method}"
        )
        
    # Get files - directory is guaranteed to exist at this point
    files = [f.name for f in method_dir.glob("*.json")]
    files = [f for f in files if "_inseq" in f]
    
    file_details = None
    if include_details:
        # Extract basic information from each file without loading the full content
        file_details = []
        for file_name in files:
            file_path = method_dir / file_name
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                # Extract minimal info
                info = data["attributes"]["info"]
                file_details.append({
                    "prompt": info["input_texts"][0] if "input_texts" in info and info["input_texts"] else "",
                    "generation": info["generated_texts"][0] if "generated_texts" in info and info["generated_texts"] else "",
                    "exec_time": info.get("exec_time", None),
                    "timestamp": file_name.split("_")[0]
                })
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                file_details.append({
                    "error": f"Could not parse file: {str(e)}",
                    "timestamp": file_name.split("_")[0] if "_" in file_name else ""
                })
    
    return ModelMethodFile(
        model=model, 
        method=method, 
        files=files, 
        file_details=file_details
    )