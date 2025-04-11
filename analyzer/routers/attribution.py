from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List
import json
import numpy as np
from pathlib import Path

from data_loader import (
    load_attribution_file,
    extract_tokens_and_attributions,
    aggregate_attribution
)

from inseq_processor import (
    AggregationMethod as InseqAggregationMethod,
    process_attribution_file,
    compare_attributions,
    compare_token_importance
)

from models import (
    AggregationMethod,
    TokenWithId,
    AttributionMatrixInfo,
    AttributionResponse,
    AttributionDetailedResponse,
    AggregationOptions
)

# Configuration
DATA_DIR = Path("data")  # Path to data directory


# Map API enum to inseq enum
def map_aggregation_method(method: AggregationMethod) -> InseqAggregationMethod:
    """Map API aggregation method to inseq aggregation method"""
    mapping = {
        AggregationMethod.SUM: InseqAggregationMethod.SUM,
        AggregationMethod.MEAN: InseqAggregationMethod.MEAN,
        AggregationMethod.L2_NORM: InseqAggregationMethod.L2_NORM,
        AggregationMethod.ABS_SUM: InseqAggregationMethod.SUM,  # Fallback, not exact match
        AggregationMethod.MAX: InseqAggregationMethod.MAX,
    }
    return mapping.get(method, InseqAggregationMethod.SUM)


# Create router
router = APIRouter(
    prefix="/attribution",
    tags=["attribution"],
    responses={404: {"description": "Not found"}},
)


@router.get("/aggregation_methods", response_model=AggregationOptions)
def get_aggregation_methods() -> AggregationOptions:
    """Get available aggregation methods"""
    methods = [method for method in AggregationMethod]
    return AggregationOptions(methods=methods, default=AggregationMethod.SUM)


@router.get("/{model}/{method}/{file_id}", response_model=AttributionResponse)
async def get_attribution(
    model: str,
    method: str,
    file_id: int,
    aggregation: AggregationMethod = Query(AggregationMethod.SUM, description="Aggregation method for attribution tensor")
) -> AttributionResponse:
    """Get attribution data for a model, method, and file"""
    # Import here to avoid circular imports
    from routers.models import get_files
    
    # Get the file list for this model and method
    files_response = get_files(model, method)
    
    if file_id < 0 or file_id >= len(files_response.files):
        raise HTTPException(
            status_code=404,
            detail=f"File ID {file_id} out of range for model {model} and method {method}"
        )
    
    file_name = files_response.files[file_id]
    file_path = DATA_DIR / model / f"method_{method}" / "data" / file_name
    
    try:
        # Using the new inseq processor
        inseq_aggregation = map_aggregation_method(aggregation)
        result = process_attribution_file(file_path, inseq_aggregation)
        
        # Extract data from the processed result
        metadata = result["metadata"]
        source_tokens = [token["token"] for token in result["source_tokens"]]
        target_tokens = [token["token"] for token in result["target_tokens"]]
        attribution_matrix = result["attribution_matrix"]
        
        return AttributionResponse(
            model=model,
            method=method,
            file_id=file_id,
            prompt=metadata["prompt"],
            generation=metadata["generation"],
            source_tokens=source_tokens,
            target_tokens=target_tokens,
            attribution_matrix=attribution_matrix,
            aggregation=aggregation
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing attribution file: {str(e)}"
        )


@router.get("/{model}/{method}/{file_id}/detailed", response_model=AttributionDetailedResponse)
async def get_detailed_attribution(
    model: str,
    method: str,
    file_id: int,
    aggregation: AggregationMethod = Query(AggregationMethod.SUM, description="Aggregation method for attribution tensor")
) -> AttributionDetailedResponse:
    """Get detailed attribution data including token IDs and tensor information"""
    # Import here to avoid circular imports
    from routers.models import get_files
    
    # Get the file list for this model and method
    files_response = get_files(model, method)
    
    if file_id < 0 or file_id >= len(files_response.files):
        raise HTTPException(
            status_code=404,
            detail=f"File ID {file_id} out of range for model {model} and method {method}"
        )
    
    file_name = files_response.files[file_id]
    file_path = DATA_DIR / model / f"method_{method}" / "data" / file_name
    
    try:
        # Using the new inseq processor
        inseq_aggregation = map_aggregation_method(aggregation)
        result = process_attribution_file(file_path, inseq_aggregation)
        
        # Extract data from the processed result
        metadata = result["metadata"]
        matrix_info = result["matrix_info"]
        
        # Convert source and target tokens to TokenWithId objects
        source_tokens = [
            TokenWithId(id=token["id"], token=token["token"])
            for token in result["source_tokens"]
        ]
        
        target_tokens = [
            TokenWithId(id=token["id"], token=token["token"])
            for token in result["target_tokens"]
        ]
        
        # Create the matrix info object
        attr_matrix_info = AttributionMatrixInfo(
            shape=matrix_info["shape"],
            dtype=matrix_info["dtype"],
            is_attention=matrix_info["is_attention"],
            tensor_type=matrix_info["tensor_type"]
        )
        
        return AttributionDetailedResponse(
            model=model,
            method=method,
            file_id=file_id,
            prompt=metadata["prompt"],
            generation=metadata["generation"],
            source_tokens=source_tokens,
            target_tokens=target_tokens,
            attribution_matrix=result["attribution_matrix"],
            matrix_info=attr_matrix_info,
            aggregation=aggregation,
            exec_time=metadata["exec_time"],
            original_attribution_shape=matrix_info["shape"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing attribution file: {str(e)}"
        )


@router.get("/{model}/{method}/{file_id}/raw")
async def get_raw_attribution(
    model: str,
    method: str,
    file_id: int
) -> Dict[str, Any]:
    """Get raw attribution data directly from the file (for debugging)"""
    # Import here to avoid circular imports
    from routers.models import get_files
    
    # Get the file list for this model and method
    files_response = get_files(model, method)
    
    if file_id < 0 or file_id >= len(files_response.files):
        raise HTTPException(
            status_code=404,
            detail=f"File ID {file_id} out of range for model {model} and method {method}"
        )
    
    file_name = files_response.files[file_id]
    file_path = DATA_DIR / model / f"method_{method}" / "data" / file_name
    
    try:
        # Load the file but don't decode tensors to save memory
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Remove large tensor data to avoid overwhelming the API response
        if "attributes" in data and "sequence_attributions" in data["attributes"]:
            for seq_attr in data["attributes"]["sequence_attributions"]:
                if "attributes" in seq_attr and "target_attributions" in seq_attr["attributes"]:
                    if "__ndarray__" in seq_attr["attributes"]["target_attributions"]:
                        seq_attr["attributes"]["target_attributions"]["__ndarray__"] = "[tensor data omitted]"
        
        return data
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading attribution file: {str(e)}"
        )


# New endpoints for comparing attributions across methods and models

@router.get("/compare", response_model=Dict[str, Any])
async def compare_attribution_files(
    files: List[str] = Query(..., description="List of file paths to compare"),
    aggregation: AggregationMethod = Query(AggregationMethod.SUM, description="Aggregation method")
) -> Dict[str, Any]:
    """
    Compare attribution data across multiple files
    
    Files should be specified as model/method/file_id format, e.g., "BART/attention/0"
    """
    file_paths = []
    
    for file_spec in files:
        parts = file_spec.split("/")
        if len(parts) != 3:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file specification '{file_spec}'. Format should be 'model/method/file_id'"
            )
        
        model, method, file_id_str = parts
        
        try:
            file_id = int(file_id_str)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file ID '{file_id_str}' in '{file_spec}'. File ID must be an integer."
            )
        
        # Import here to avoid circular imports
        from routers.models import get_files
        
        try:
            files_response = get_files(model, method)
            
            if file_id < 0 or file_id >= len(files_response.files):
                raise HTTPException(
                    status_code=404,
                    detail=f"File ID {file_id} out of range for model {model} and method {method}"
                )
            
            file_name = files_response.files[file_id]
            file_path = DATA_DIR / model / f"method_{method}" / "data" / file_name
            file_paths.append(file_path)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error accessing file {file_spec}: {str(e)}"
            )
    
    try:
        inseq_aggregation = map_aggregation_method(aggregation)
        result = compare_attributions(file_paths, inseq_aggregation)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error comparing attribution files: {str(e)}"
        )


@router.get("/token_importance", response_model=Dict[str, Any])
async def get_token_importance(
    files: List[str] = Query(..., description="List of file paths to compare"),
    token_index: int = Query(..., description="Index of the token to compare"),
    is_target: bool = Query(True, description="Whether the token is in the target (True) or source (False) sequence"),
    aggregation: AggregationMethod = Query(AggregationMethod.SUM, description="Aggregation method")
) -> Dict[str, Any]:
    """
    Compare importance of a specific token across multiple attribution files
    
    Files should be specified as model/method/file_id format, e.g., "BART/attention/0"
    """
    file_paths = []
    
    for file_spec in files:
        parts = file_spec.split("/")
        if len(parts) != 3:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file specification '{file_spec}'. Format should be 'model/method/file_id'"
            )
        
        model, method, file_id_str = parts
        
        try:
            file_id = int(file_id_str)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file ID '{file_id_str}' in '{file_spec}'. File ID must be an integer."
            )
        
        # Import here to avoid circular imports
        from routers.models import get_files
        
        try:
            files_response = get_files(model, method)
            
            if file_id < 0 or file_id >= len(files_response.files):
                raise HTTPException(
                    status_code=404,
                    detail=f"File ID {file_id} out of range for model {model} and method {method}"
                )
            
            file_name = files_response.files[file_id]
            file_path = DATA_DIR / model / f"method_{method}" / "data" / file_name
            file_paths.append(file_path)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error accessing file {file_spec}: {str(e)}"
            )
    
    try:
        inseq_aggregation = map_aggregation_method(aggregation)
        result = compare_token_importance(file_paths, token_index, is_target, inseq_aggregation)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error comparing token importance: {str(e)}"
        )