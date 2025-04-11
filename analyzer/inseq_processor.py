"""
Inseq processing utilities for attribution data
"""
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import inseq
from pathlib import Path
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AggregationMethod(str, Enum):
    """
    Aggregation methods for attribution data
    Maps to inseq's aggregation functions
    """
    SUM = "sum"
    MEAN = "mean"
    L2_NORM = "vnorm"  # Vector norm in inseq corresponds to L2 norm
    MAX = "max"
    MIN = "min"
    ABS_MAX = "absmax"
    PROD = "prod"


def load_attribution(file_path: Union[str, Path]) -> inseq.FeatureAttributionOutput:
    """
    Load an inseq attribution file
    
    Args:
        file_path: Path to the attribution file
        
    Returns:
        Loaded attribution data
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"Attribution file not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
        
    try:
        # First try using the standard inseq loader
        return inseq.FeatureAttributionOutput.load(str(file_path))
    except Exception as e:
        logger.error(f"Error loading attribution file with inseq loader: {e}")
        # Fall back to direct JSON loading with our data_loader
        try:
            from data_loader import load_attribution_file
            data = load_attribution_file(str(file_path))
            
            # We need to create an object that mimics FeatureAttributionOutput
            # Pass an empty list to the constructor to avoid errors
            attribution_output = inseq.FeatureAttributionOutput([])
            
            # Add attributes directly to the object
            attribution_output.info = data["attributes"]["info"]
            
            # Process sequence_attributions
            if "sequence_attributions" in data["attributes"]:
                # Create a class to mimic the sequence attribution output structure
                class TokenWithId:
                    def __init__(self, token_data):
                        if "__instance_type__" in token_data:
                            self.__instance_type__ = token_data["__instance_type__"]
                        self.attributes = token_data["attributes"]
                        # For direct access (needed for some methods)
                        self.id = token_data["attributes"]["id"]
                        self.token = token_data["attributes"]["token"]
                        
                class SeqAttr:
                    def __init__(self, seq_attr):
                        # Copy all attributes from the JSON
                        attrs = seq_attr["attributes"]
                        self.__dict__["raw_attributes"] = attrs  # Keep original for debugging
                        
                        # Store special attributes first (need careful handling)
                        self.target_attributions = attrs.get("target_attributions", {})
                        self.source_attributions = attrs.get("source_attributions", {})
                        self.sequence_scores = attrs.get("sequence_scores", {})
                        self.step_scores = attrs.get("step_scores", {})
                        self.attr_pos_start = attrs.get("attr_pos_start", 0)
                        self.attr_pos_end = attrs.get("attr_pos_end", None)
                        
                        # Process source and target tokens specially
                        if "source" in attrs:
                            self.source = [TokenWithId(token) for token in attrs["source"]]
                            
                        if "target" in attrs:
                            self.target = [TokenWithId(token) for token in attrs["target"]]
                            
                        self.__instance_type__ = seq_attr.get("__instance_type__")
                
                # Create sequence attribution objects
                attribution_output.sequence_attributions = [
                    SeqAttr(seq_attr) 
                    for seq_attr in data["attributes"]["sequence_attributions"]
                ]
            
            return attribution_output
        except Exception as fallback_error:
            logger.error(f"Fallback loading also failed for {file_path}: {fallback_error}")
            raise


def get_attribution_metadata(attribution: inseq.FeatureAttributionOutput) -> Dict[str, Any]:
    """
    Extract metadata from an attribution object
    
    Args:
        attribution: Inseq attribution object
        
    Returns:
        Dictionary of metadata
    """
    info = attribution.info
    metadata = {
        "model_name": info.get("model_name", "unknown"),
        "attribution_method": info.get("attribution_method", "unknown"),
        "exec_time": info.get("exec_time", 0.0),
        "prompt": info.get("input_texts", [""])[0] if info.get("input_texts") else "",
        "generation": info.get("generated_texts", [""])[0] if info.get("generated_texts") else "",
        "tokenizer_class": info.get("tokenizer_class", "unknown"),
        "model_class": info.get("model_class", "unknown"),
    }
    return metadata


def get_tokens_and_ids(attribution: inseq.FeatureAttributionOutput) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Extract tokens and their IDs from an attribution object
    
    Args:
        attribution: Inseq attribution object
        
    Returns:
        Tuple of (source_tokens, target_tokens) with their IDs
    """
    if not attribution.sequence_attributions:
        return [], []
    
    seq_attr = attribution.sequence_attributions[0]
    
    # TokenWithId objects can have different structures depending on the source
    source_tokens = []
    for token in seq_attr.source:
        if hasattr(token, 'attributes') and isinstance(token.attributes, dict):
            # JSON-loaded tokens have an attributes dict
            source_tokens.append({"id": token.attributes["id"], "token": token.attributes["token"]})
        else:
            # TokenWithId objects from inseq have direct properties
            source_tokens.append({"id": token.id, "token": token.token})
    
    target_tokens = []
    for token in seq_attr.target:
        if hasattr(token, 'attributes') and isinstance(token.attributes, dict):
            target_tokens.append({"id": token.attributes["id"], "token": token.attributes["token"]})
        else:
            target_tokens.append({"id": token.id, "token": token.token})
    
    return source_tokens, target_tokens


def get_attribution_matrix(
    attribution: inseq.FeatureAttributionOutput,
    aggregation_method: AggregationMethod = AggregationMethod.SUM
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Extract and aggregate attribution matrix from an attribution object
    
    Args:
        attribution: Inseq attribution object
        aggregation_method: Method to aggregate attribution values
        
    Returns:
        Tuple of (attribution_matrix, matrix_info)
    """
    if not attribution.sequence_attributions:
        return np.array([[]]), {}
    
    seq_attr = attribution.sequence_attributions[0]
    target_attributions = seq_attr.target_attributions
    
    # Handle different types of target_attributions
    if target_attributions is None:
        logger.error("target_attributions is None")
        return np.array([[]]), {}
    
    # Check if target_attributions is already a tensor (for some attribution methods)
    import torch
    if isinstance(target_attributions, torch.Tensor):
        # Convert PyTorch tensor to numpy
        try:
            tensor = target_attributions.detach().cpu().numpy()
            logger.info(f"Converted PyTorch tensor of shape {tensor.shape} to numpy")
        except Exception as e:
            logger.error(f"Error converting PyTorch tensor to numpy: {e}")
            return np.array([[]]), {}
    elif isinstance(target_attributions, np.ndarray):
        # Already a numpy array
        tensor = target_attributions
        logger.info(f"Using numpy array of shape {tensor.shape}")
    elif isinstance(target_attributions, dict):
        # JSON-loaded data structure
        if "tensor" in target_attributions:
            # Already decoded tensor
            tensor = target_attributions["tensor"]
        elif "__ndarray__" in target_attributions and "dtype" in target_attributions and "shape" in target_attributions:
            # Decode from base64 gzipped ndarray
            try:
                from data_loader import decode_ndarray
                tensor = decode_ndarray(
                    target_attributions["__ndarray__"],
                    target_attributions["dtype"],
                    target_attributions["shape"]
                )
                # Cache for future use
                target_attributions["tensor"] = tensor
            except Exception as e:
                logger.error(f"Error decoding tensor: {e}")
                return np.array([[]]), {}
        else:
            logger.error(f"Dict target_attributions lacks required fields: {list(target_attributions.keys())}")
            return np.array([[]]), {}
    else:
        # Unknown type
        logger.error(f"Unsupported target_attributions type: {type(target_attributions)}")
        return np.array([[]]), {}
        
    if tensor is None:
        logger.error("Failed to obtain a valid tensor")
        return np.array([[]]), {}
    
    # Now we have the tensor, determine its properties
    original_shape = list(tensor.shape)
    is_attention = tensor.ndim == 4
    
    # Determine the tensor type based on method name from metadata
    method = attribution.info.get("attribution_method", "unknown")
    tensor_type_map = {
        "attention": "attention",
        "input_x_gradient": "input_x_gradient",
        "lime": "lime", 
        "integrated_gradients": "integrated_gradients",
        "saliency": "saliency",
        "layer_gradient_x_activation": "layer_gradient_x_activation"
    }
    tensor_type = tensor_type_map.get(method, "unknown")
    
    # Aggregate the tensor
    try:
        # Using inseq's aggregation would be ideal, but it requires proper FeatureAttributionSequenceOutput objects
        # which are not fully retained when loading from JSON
        # Instead, we use our custom aggregation logic for now
        from data_loader import aggregate_attribution
        
        # First, handle any NaN values in the tensor - these cause JSON serialization to fail
        has_nans = np.isnan(tensor).any()
        if has_nans:
            logger.warning(f"Tensor contains {np.isnan(tensor).sum()} NaN values, replacing with zeros")
            tensor = np.nan_to_num(tensor, nan=0.0)
            
        # Map our enum to the string expected by our function
        aggregation_str = aggregation_method.value
        attribution_matrix = aggregate_attribution(tensor, aggregation_str, is_attention)
        
        # Handle NaNs in the final attribution matrix as well
        if np.isnan(attribution_matrix).any():
            logger.warning(f"Attribution matrix contains {np.isnan(attribution_matrix).sum()} NaN values, replacing with zeros")
            attribution_matrix = np.nan_to_num(attribution_matrix, nan=0.0)
        
        matrix_info = {
            "shape": original_shape,
            "dtype": str(tensor.dtype),
            "is_attention": is_attention,
            "tensor_type": tensor_type,
            "aggregation_method": aggregation_method.value,
            "had_nan_values": has_nans
        }
        
        return attribution_matrix, matrix_info
    
    except Exception as e:
        logger.error(f"Error aggregating attribution matrix: {e}")
        return np.array([[]]), {}


def process_attribution_file(
    file_path: Union[str, Path],
    aggregation_method: AggregationMethod = AggregationMethod.SUM
) -> Dict[str, Any]:
    """
    Process an attribution file into a standardized format for the API
    
    Args:
        file_path: Path to the attribution file
        aggregation_method: Method to aggregate attribution values
        
    Returns:
        Dictionary with processed attribution data
    """
    file_path = Path(file_path)
    attribution = load_attribution(file_path)
    
    metadata = get_attribution_metadata(attribution)
    source_tokens, target_tokens = get_tokens_and_ids(attribution)
    attribution_matrix, matrix_info = get_attribution_matrix(attribution, aggregation_method)
    
    # Extract filename components for additional info
    filename = file_path.name
    file_parts = filename.split("_")
    timestamp = file_parts[0] if len(file_parts) > 0 else ""
    
    result = {
        "metadata": metadata,
        "source_tokens": source_tokens,
        "target_tokens": target_tokens,
        "attribution_matrix": attribution_matrix.tolist(),
        "matrix_info": matrix_info,
        "filename": filename,
        "timestamp": timestamp,
        "aggregation": aggregation_method.value
    }
    
    # Convert any numpy types to Python native types for serialization
    result = convert_numpy_types(result)
    
    return result


def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types to ensure serializability
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.number):
        return obj.item()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def compare_attributions(files: List[Union[str, Path]], aggregation_method: AggregationMethod = AggregationMethod.SUM) -> Dict[str, Any]:
    """
    Compare multiple attribution files
    
    Args:
        files: List of attribution file paths
        aggregation_method: Method to aggregate attribution values
        
    Returns:
        Dictionary with compared attribution data
    """
    results = []
    
    for file_path in files:
        try:
            result = process_attribution_file(file_path, aggregation_method)
            # Convert any numpy types to Python native types for serialization
            result = convert_numpy_types(result)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
    
    # Extract common tokens and normalize matrices for comparison
    # TODO: Implement token alignment and matrix normalization
    
    return {
        "results": results,
        "aggregation": aggregation_method.value,
        "count": len(results)
    }


# Example token comparison function (e.g., for comparing the same token across methods/models)
def compare_token_importance(
    files: List[Union[str, Path]],
    token_index: int,
    is_target: bool = True,
    aggregation_method: AggregationMethod = AggregationMethod.SUM
) -> Dict[str, Any]:
    """
    Compare importance of a specific token across multiple attribution files
    
    Args:
        files: List of attribution file paths
        token_index: Index of the token to compare
        is_target: Whether the token is in the target (True) or source (False) sequence
        aggregation_method: Method to aggregate attribution values
        
    Returns:
        Dictionary with token importance comparison
    """
    results = []
    
    for file_path in files:
        try:
            attribution = load_attribution(file_path)
            matrix, info = get_attribution_matrix(attribution, aggregation_method)
            
            if matrix.size == 0:
                continue
                
            metadata = get_attribution_metadata(attribution)
            source_tokens, target_tokens = get_tokens_and_ids(attribution)
            
            if is_target and token_index < len(target_tokens):
                token = target_tokens[token_index]
                # For target tokens, get the column of influences from all source tokens
                if token_index < matrix.shape[0]:
                    influences = matrix[token_index, :].tolist()
                    result = {
                        "model": metadata["model_name"],
                        "method": metadata["attribution_method"],
                        "token": token,
                        "position": token_index,
                        "influences": influences,
                        "sum_influence": sum(influences),
                        "max_influence": max(influences),
                        "filename": Path(file_path).name
                    }
                    results.append(result)
            elif not is_target and token_index < len(source_tokens):
                token = source_tokens[token_index]
                # For source tokens, get the row of influences to all target tokens
                if token_index < matrix.shape[1]:
                    influences = matrix[:, token_index].tolist()
                    result = {
                        "model": metadata["model_name"],
                        "method": metadata["attribution_method"],
                        "token": token,
                        "position": token_index,
                        "influences": influences,
                        "sum_influence": sum(influences),
                        "max_influence": max(influences),
                        "filename": Path(file_path).name
                    }
                    results.append(result)
        except Exception as e:
            logger.error(f"Error processing file {file_path} for token {token_index}: {e}")
    
    # Convert any numpy values to Python native types for serialization
    results = convert_numpy_types(results)
    
    return {
        "token_index": token_index,
        "is_target": convert_numpy_types(is_target),  # Convert bool_ if needed
        "results": results,
        "count": len(results),
        "aggregation": aggregation_method.value
    }