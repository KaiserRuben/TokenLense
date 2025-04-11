import os
import json
import base64
import gzip
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


def decode_ndarray(encoded_data: str, dtype_str: str, shape: List[int]) -> np.ndarray:
    """Decode a base64 gzipped ndarray from Inseq format"""
    # Remove the "b64.gz:" prefix if present
    if encoded_data.startswith("b64.gz:"):
        encoded_data = encoded_data[7:]

    # Decode base64 and decompress gzip
    decoded_binary = base64.b64decode(encoded_data)
    decompressed = gzip.decompress(decoded_binary)

    # Convert to numpy array
    dtype = np.dtype(dtype_str)
    array = np.frombuffer(decompressed, dtype=dtype)

    # Reshape according to the shape field
    tensor = array.reshape(shape)

    return tensor


def load_attribution_file(file_path: str) -> Dict[str, Any]:
    """Load and decode an Inseq attribution file"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Process the tensors in the data
    sequence_attributions = data["attributes"]["sequence_attributions"]
    for seq_attr in sequence_attributions:
        target_attributions = seq_attr["attributes"]["target_attributions"]
        if target_attributions and "__ndarray__" in target_attributions:
            tensor = decode_ndarray(
                target_attributions["__ndarray__"],
                target_attributions["dtype"],
                target_attributions["shape"]
            )

            # Store decoded tensor
            target_attributions["tensor"] = tensor

    return data


def aggregate_attribution(tensor: np.ndarray, method: str = "sum", is_attention: bool = False) -> np.ndarray:
    """Aggregate attribution tensor for visualization"""
    # Handle any non-finite values before aggregation
    if not np.isfinite(tensor).all():
        # Replace NaN with 0, inf with large value, and -inf with small value
        tensor = np.nan_to_num(tensor, nan=0.0, posinf=1e30, neginf=-1e30)
    
    if is_attention:  # 4D tensor: [target_len, source_len, num_heads, head_dim]
        # First aggregate over head dimension (axis=3)
        if method == "sum":
            head_agg = np.sum(tensor, axis=3)
        elif method == "mean":
            head_agg = np.mean(tensor, axis=3)
        elif method == "l2_norm":
            head_agg = np.sqrt(np.sum(tensor ** 2, axis=3))
        elif method == "abs_sum":
            head_agg = np.sum(np.abs(tensor), axis=3)
        elif method == "max":
            head_agg = np.max(tensor, axis=3)
        else:
            # Default to mean if method is not recognized
            head_agg = np.mean(tensor, axis=3)

        # Then aggregate across heads (axis=2)
        if method == "sum":
            result = np.sum(head_agg, axis=2)
        elif method == "mean":
            result = np.mean(head_agg, axis=2)
        elif method == "l2_norm":
            result = np.sqrt(np.sum(head_agg ** 2, axis=2))
        elif method == "abs_sum":
            result = np.sum(np.abs(head_agg), axis=2)
        elif method == "max":
            result = np.max(head_agg, axis=2)
        else:
            # Default to mean if method is not recognized
            result = np.mean(head_agg, axis=2)
    else:  # 3D tensor: [target_len, source_len, hidden_size]
        if method == "sum":
            result = np.sum(tensor, axis=2)
        elif method == "mean":
            result = np.mean(tensor, axis=2)
        elif method == "l2_norm":
            result = np.sqrt(np.sum(tensor ** 2, axis=2))
        elif method == "abs_sum":
            result = np.sum(np.abs(tensor), axis=2)
        elif method == "max":
            result = np.max(tensor, axis=2)
        else:
            # Default to mean if method is not recognized
            result = np.mean(tensor, axis=2)
    
    # Final check for any remaining non-finite values
    if not np.isfinite(result).all():
        result = np.nan_to_num(result, nan=0.0, posinf=1e30, neginf=-1e30)
    
    return result


def get_available_models_and_methods(data_dir: str) -> Dict[str, List[str]]:
    """Get all available models and their attribution methods"""
    result = {}
    for model_dir in os.listdir(data_dir):
        model_path = os.path.join(data_dir, model_dir)
        if os.path.isdir(model_path) and not model_dir.startswith('.'):
            methods = []
            for method_dir in os.listdir(model_path):
                method_path = os.path.join(model_path, method_dir)
                if method_dir.startswith("method_") and os.path.isdir(method_path):
                    method_name = method_dir.replace("method_", "")
                    methods.append(method_name)
            if methods:
                result[model_dir] = methods
    return result


def extract_tokens_and_attributions(data: Dict[str, Any],
                                    aggregation_method: str = "sum") -> Tuple[List[str], List[str], np.ndarray]:
    """Extract tokens and attribution matrix from data"""
    if not data["attributes"]["sequence_attributions"]:
        return [], [], None

    seq_attr = data["attributes"]["sequence_attributions"][0]
    source_tokens = [token["attributes"]["token"] for token in seq_attr["attributes"]["source"]]
    target_tokens = [token["attributes"]["token"] for token in seq_attr["attributes"]["target"]]

    target_attributions = seq_attr["attributes"]["target_attributions"]
    if not target_attributions or "tensor" not in target_attributions:
        return source_tokens, target_tokens, None

    tensor = target_attributions["tensor"]

    # Determine if this is an attention tensor (4D) or other (3D)
    is_attention = tensor.ndim == 4
    attribution_matrix = aggregate_attribution(tensor, aggregation_method, is_attention)

    return source_tokens, target_tokens, attribution_matrix