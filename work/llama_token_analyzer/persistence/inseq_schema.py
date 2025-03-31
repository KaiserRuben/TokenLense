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
        
        # Process attribution matrix
        attribution_matrix = self.source_attributions
        
        # Check if attribution_matrix is None or empty
        if attribution_matrix is None:
            # Create default matrix matching input/output dimensions
            association_matrix_list = [[0.0 for _ in range(len(self.source))] for _ in range(len(self.target))]
            normalized_matrix_list = [[0.0 for _ in range(len(self.source))] for _ in range(len(self.target))]
        else:
            try:
                # Convert to numpy if it's not already
                if hasattr(attribution_matrix, 'numpy'):
                    attribution_matrix = attribution_matrix.numpy()
                    
                # Ensure we're working with float64 to avoid precision issues
                if hasattr(attribution_matrix, 'dtype'):
                    attribution_matrix = attribution_matrix.astype(np.float64)
                
                # Convert to numpy array
                attribution_matrix = np.array(attribution_matrix)
                
                # Handle multi-dimensional attributions - aggregate if needed
                if len(attribution_matrix.shape) > 2:
                    # For 3D matrices (like gradient attributions)
                    if len(attribution_matrix.shape) == 3:
                        # Take L2 norm across the last dimension
                        attribution_matrix = np.linalg.norm(attribution_matrix, axis=-1)
                    # For 4D matrices (like attention attributions)
                    elif len(attribution_matrix.shape) == 4:
                        # Flatten the last two dimensions and take L2 norm
                        orig_shape = attribution_matrix.shape
                        flattened = attribution_matrix.reshape(orig_shape[0], orig_shape[1], -1)
                        attribution_matrix = np.linalg.norm(flattened, axis=-1)
                
                # Convert to native Python lists for Pydantic compatibility
                association_matrix_list = attribution_matrix.tolist()
                
                # Normalize for visualization
                normalized_matrix = self._normalize_attribution_matrix(attribution_matrix)
                normalized_matrix_list = normalized_matrix.tolist()
            except Exception as e:
                # If there's any error in processing, create a default matrix
                print(f"Error processing attribution matrix: {e}")
                association_matrix_list = [[0.0 for _ in range(len(self.source))] for _ in range(len(self.target))]
                normalized_matrix_list = [[0.0 for _ in range(len(self.source))] for _ in range(len(self.target))]
        
        # Create the standard analysis result
        return AnalysisResult(
            metadata=AnalysisMetadata(
                llm_id=model_id,
                llm_version=model_id.split('/')[-1],
                prompt=prompt,
                generation_params={
                    "attribution_method": attribution_method,
                    "max_new_tokens": len(output_tokens)
                }
            ),
            data=AssociationData(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                association_matrix=association_matrix_list,
                normalized_association=normalized_matrix_list
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
        # Ensure we have a 2D matrix - check dimensionality first
        matrix_shape = getattr(matrix, "shape", None)
        if matrix_shape is None or len(matrix_shape) == 0:
            # If not an array or empty, return an empty array
            return np.array([[0.0]])
            
        # Handle 1D array - convert to 2D (1 x N)
        if len(matrix_shape) == 1:
            matrix = matrix.reshape(1, -1)
            
        # Handle higher dimensions (3D+)
        elif len(matrix_shape) > 2:
            # Take L2 norm to reduce to 2D
            matrix = np.linalg.norm(matrix, axis=-1)
        
        # Ensure we have some data
        if matrix.size == 0 or len(matrix.shape) < 2:
            return np.array([[0.0]])
            
        # Create normalized array
        normalized = np.zeros_like(matrix)
        
        # Normalize each column if we have columns
        if matrix.shape[1] > 0:
            for i in range(matrix.shape[1]):
                # Get non-zero values only
                col_values = matrix[:, i]
                valid_values = col_values[col_values != 0]
                
                if len(valid_values) > 0:
                    column_min = valid_values.min()
                    column_max = valid_values.max()
                    if column_max > column_min:
                        normalized[:, i] = (
                                (matrix[:, i] - column_min)
                                / (column_max - column_min)
                        )
                    else:
                        # If all values are the same, set to 0.5
                        normalized[:, i] = 0.5
                else:
                    # If all values are 0, leave as 0
                    normalized[:, i] = 0.0
                
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