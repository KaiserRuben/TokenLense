"""
Pydantic models for Inseq attribution data structures.

This module defines Pydantic models that mirror the Inseq attribution output format,
allowing for easy serialization/deserialization of Inseq results.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
from enum import Enum

from .schema import TokenData, AnalysisResult, AnalysisMetadata, AssociationData


# The supported attribution methods should now be checked using 
# inseq.list_feature_attribution_methods() instead of an enum
# This was causing compatibility issues with Inseq 0.6.0+


class InseqTokenWithId(BaseModel):
    """Model representing an Inseq token with its ID"""
    token: str
    token_id: int
    
    model_config = ConfigDict(
        protected_namespaces=()
    )


class InseqFeatureAttributionSequence(BaseModel):
    """Model representing a single sequence attribution from Inseq"""
    source: List[str] = Field(..., description="Source/input tokens")
    target: List[str] = Field(..., description="Target/output tokens")
    source_attributions: Any = Field(..., description="Attribution scores for source tokens")
    target_attributions: Optional[Any] = Field(None, description="Attribution scores for target tokens")
    step_scores: Dict[str, Any] = Field(default_factory=dict, description="Step-specific scores")
    sequence_scores: Optional[Dict[str, Any]] = Field(None, description="Sequence-level scores")
    attr_pos_start: int = Field(0, description="Start position for attribution")
    attr_pos_end: Optional[int] = Field(None, description="End position for attribution")
    
    model_config = ConfigDict(
        protected_namespaces=(),
        arbitrary_types_allowed=True
    )
    
    def to_token_analyzer_format(self, model_id: str, prompt: str, attribution_method: str) -> AnalysisResult:
        """
        Convert Inseq attribution sequence to the standard TokenAnalyzer format.
        
        Args:
            model_id: The model identifier
            prompt: Original input prompt
            attribution_method: Name of the attribution method used
            
        Returns:
            AnalysisResult compatible with existing visualization
        """
        # Convert source tokens to TokenData
        input_tokens = [
            TokenData(
                token=str(token),
                token_id=0,  # Will be populated later
                clean_token=str(token).lstrip('Ġ').strip()
            )
            for token in self.source
        ]
        
        # Convert target tokens to TokenData
        output_tokens = [
            TokenData(
                token=str(token),
                token_id=0,  # Will be populated later
                clean_token=str(token).lstrip('Ġ').strip()
            )
            for token in self.target
        ]
        
        # Process attribution matrix - first check if we have source attributions
        attribution_matrix = self.source_attributions
        attribution_source = "source"
        
        import logging
        logger = logging.getLogger(__name__)
        
        # If source attributions are None, try using target attributions (common for some methods)
        if attribution_matrix is None and hasattr(self, 'target_attributions') and self.target_attributions is not None:
            logger.info("Using target_attributions since source_attributions is None")
            attribution_matrix = self.target_attributions
            attribution_source = "target"
        
        # Check if attribution_matrix is None or empty after trying target_attributions
        if attribution_matrix is None:
            logger.warning("No attribution data available, creating zero matrix")
            # Create default matrix matching input/output dimensions
            association_matrix_list = [[0.0 for _ in range(len(self.source))] for _ in range(len(self.target))]
            normalized_matrix_list = [[0.0 for _ in range(len(self.source))] for _ in range(len(self.target))]
        else:
            try:
                logger.info(f"Processing {attribution_source} attribution matrix of type {type(attribution_matrix)}")
                
                # Convert to numpy if it's a tensor
                if hasattr(attribution_matrix, 'numpy'):
                    attribution_matrix = attribution_matrix.numpy()
                    logger.info(f"Converted tensor to numpy array with shape {attribution_matrix.shape}")
                    
                # Ensure we're working with float64 to avoid precision issues
                if hasattr(attribution_matrix, 'dtype'):
                    attribution_matrix = attribution_matrix.astype(np.float64)
                    logger.info(f"Converted to float64 dtype, min: {attribution_matrix.min()}, max: {attribution_matrix.max()}")
                
                # Convert to numpy array if possible
                # Handle case when attribution_matrix might be a list instead of tensor
                if isinstance(attribution_matrix, list):
                    # Convert nested lists to numpy array
                    attribution_matrix = np.array(attribution_matrix, dtype=np.float64)
                    logger.info(f"Converted list to numpy array with shape {attribution_matrix.shape}")
                else:
                    # Make sure we have a numpy array
                    attribution_matrix = np.array(attribution_matrix, dtype=np.float64)
                
                # Handle multi-dimensional attributions - aggregate if needed
                if len(attribution_matrix.shape) > 2:
                    logger.info(f"Processing multi-dimensional attribution with shape {attribution_matrix.shape}")
                    
                    # For 3D matrices (like gradient attributions)
                    if len(attribution_matrix.shape) == 3:
                        # Handle NaN values - replace with zeros
                        if np.isnan(attribution_matrix).any():
                            logger.warning(f"NaN values detected in attribution matrix, replacing with zeros")
                            attribution_matrix = np.nan_to_num(attribution_matrix, nan=0.0)
                            
                        # Take absolute values to ensure no negative numbers
                        attribution_matrix = np.abs(attribution_matrix)
                        
                        # Take L2 norm across the last dimension
                        attribution_matrix = np.linalg.norm(attribution_matrix, axis=-1)
                        logger.info(f"Took L2 norm over dim 2, new shape: {attribution_matrix.shape}")
                    
                    # For 4D matrices (like attention attributions)
                    elif len(attribution_matrix.shape) == 4:
                        # Handle NaN values - replace with zeros
                        if np.isnan(attribution_matrix).any():
                            logger.warning(f"NaN values detected in attribution matrix, replacing with zeros")
                            attribution_matrix = np.nan_to_num(attribution_matrix, nan=0.0)
                        
                        # First average across attention heads (dim 3)
                        attribution_matrix = attribution_matrix.mean(axis=3)
                        logger.info(f"Averaged across attention heads, new shape: {attribution_matrix.shape}")
                        
                        # Then average across layers (dim 2)
                        attribution_matrix = attribution_matrix.mean(axis=2)
                        logger.info(f"Averaged across layers, new shape: {attribution_matrix.shape}")
                        
                    # Final check for NaN values after all transformations
                    if np.isnan(attribution_matrix).any():
                        logger.warning(f"NaN values still present after transformation, replacing with random values")
                        nan_mask = np.isnan(attribution_matrix)
                        attribution_matrix[nan_mask] = np.random.uniform(0.1, 0.2, size=np.count_nonzero(nan_mask))
                
                # For attention matrices, ensure correct dimension ordering (target x source)
                # Often attention has dimensions (source_len, target_len) but we need (target_len, source_len)
                if attribution_method == "attention" and attribution_matrix.shape[0] == len(self.source) and attribution_matrix.shape[1] == len(self.target):
                    logger.info(f"Transposing attention matrix to match target x source dimensions")
                    attribution_matrix = attribution_matrix.T
                    logger.info(f"After transpose, shape: {attribution_matrix.shape}, min: {attribution_matrix.min()}, max: {attribution_matrix.max()}")
                
                # Convert to native Python lists for Pydantic compatibility
                association_matrix_list = attribution_matrix.tolist()
                
                # Normalize for visualization
                normalized_matrix = self._normalize_attribution_matrix(attribution_matrix)
                normalized_matrix_list = normalized_matrix.tolist()
                
                logger.info(f"Final attribution matrix shape: {attribution_matrix.shape}")
            except Exception as e:
                logger.error(f"Error processing attribution matrix: {e}", exc_info=True)
                association_matrix_list = [[0.0 for _ in range(len(self.source))] for _ in range(len(self.target))]
                normalized_matrix_list = [[0.0 for _ in range(len(self.source))] for _ in range(len(self.target))]
        
        # Determine attribution source
        from ..persistence.schema import AttributionSource
        
        attribution_source = AttributionSource.UNKNOWN
        if attribution_matrix is self.source_attributions and self.source_attributions is not None:
            attribution_source = AttributionSource.SOURCE
        elif attribution_matrix is self.target_attributions and self.target_attributions is not None:
            attribution_source = AttributionSource.TARGET
        
        # Store raw dimensions information if available
        raw_dimensions = None
        if hasattr(attribution_matrix, 'shape'):
            raw_dimensions = list(attribution_matrix.shape)
            logger.info(f"Raw attribution dimensions: {raw_dimensions}")
        
        # Create the standard analysis result with enhanced metadata
        return AnalysisResult(
            metadata=AnalysisMetadata(
                llm_id=model_id,
                llm_version=model_id.split('/')[-1],
                prompt=prompt,
                attribution_method=attribution_method,
                generation_params={
                    "max_new_tokens": len(output_tokens)
                }
            ),
            data=AssociationData(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                association_matrix=association_matrix_list,
                normalized_association=normalized_matrix_list,
                attribution_source=attribution_source,
                raw_dimensions=raw_dimensions
            )
        )
    
    @staticmethod
    def _normalize_attribution_matrix(matrix: np.ndarray) -> np.ndarray:
        """
        Normalize attribution matrix values to [0,1] range.
        
        Args:
            matrix: Raw attribution matrix
            
        Returns:
            Normalized attribution matrix
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # Convert input to numpy array if it's not already
        if not isinstance(matrix, np.ndarray):
            try:
                matrix = np.array(matrix, dtype=np.float64)
            except Exception as e:
                logger.error(f"Error converting to numpy array: {e}")
                return np.array([[0.0]])
        
        # Ensure we have a 2D matrix - check dimensionality first
        matrix_shape = getattr(matrix, "shape", None)
        if matrix_shape is None or len(matrix_shape) == 0:
            # If not an array or empty, return an empty array
            logger.warning("Matrix has no shape, returning empty array")
            return np.array([[0.0]])
            
        # Handle 1D array - convert to 2D (1 x N)
        if len(matrix_shape) == 1:
            matrix = matrix.reshape(1, -1)
            logger.info(f"Reshaped 1D array to {matrix.shape}")
            
        # Handle higher dimensions (3D+)
        elif len(matrix_shape) > 2:
            # Take L2 norm to reduce to 2D
            matrix = np.linalg.norm(matrix, axis=-1)
            logger.info(f"Reduced higher dimensions to {matrix.shape}")
        
        # Ensure we have some data
        if matrix.size == 0 or len(matrix.shape) < 2:
            logger.warning("Matrix has no data, returning empty array")
            return np.array([[0.0]])
        
        # Check for NaN or Inf values and replace them
        if np.isnan(matrix).any() or np.isinf(matrix).any():
            logger.warning(f"Matrix contains NaN or Inf values, replacing with zeros")
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=0.0)
        
        # If matrix is all zeros, keep it as zeros
        if np.all(matrix == 0):
            logger.warning("Matrix contains only zeros")
            normalized = np.zeros(matrix.shape)
            return normalized
        
        # Global normalization approach - normalize the entire matrix at once
        matrix_min = matrix.min()
        matrix_max = matrix.max()
        
        # Only proceed with normalization if we have meaningful values
        if abs(matrix_max - matrix_min) < 1e-10:
            logger.warning(f"Matrix has uniform values (min={matrix_min}, max={matrix_max})")
            # If matrix has all the same value, use the value itself (0-1 range)
            # or if outside 0-1 range, use 0.5
            if 0 <= matrix_min <= 1:
                normalized = np.full(matrix.shape, matrix_min)
            else:
                normalized = np.full(matrix.shape, 0.5)
            return normalized
            
        # Normalize the entire matrix
        normalized = (matrix - matrix_min) / (matrix_max - matrix_min)
        
        # Final sanity check - replace any remaining NaN values with zeros
        if np.isnan(normalized).any():
            logger.warning("Normalized matrix still contains NaN values, replacing with zeros")
            nan_mask = np.isnan(normalized)
            normalized[nan_mask] = 0.0
            
        logger.info(f"Normalized matrix range: {normalized.min()} to {normalized.max()}")
        return normalized


class InseqFeatureAttributionStep(BaseModel):
    """Model representing a single attribution step from Inseq"""
    # Step-level attribution data
    step_idx: int
    token: str
    token_id: int
    attributions: Any
    scores: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        protected_namespaces=(),
        arbitrary_types_allowed=True
    )


class InseqFeatureAttributionOutput(BaseModel):
    """Model representing the complete Inseq attribution output"""
    sequence_attributions: List[InseqFeatureAttributionSequence]
    step_attributions: Optional[List[InseqFeatureAttributionStep]] = None
    info: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(
        protected_namespaces=(),
        arbitrary_types_allowed=True
    )
    
    def to_token_analyzer_format(self, model_id: str, prompt: str, attribution_method: str) -> AnalysisResult:
        """
        Convert the first sequence attribution to TokenAnalyzer format.
        
        Args:
            model_id: The model identifier
            prompt: Original input prompt
            attribution_method: Name of the attribution method used
            
        Returns:
            AnalysisResult compatible with existing visualization
        """
        if not self.sequence_attributions:
            raise ValueError("No sequence attributions available")
        
        # Convert the first sequence attribution
        return self.sequence_attributions[0].to_token_analyzer_format(
            model_id=model_id,
            prompt=prompt,
            attribution_method=attribution_method
        )