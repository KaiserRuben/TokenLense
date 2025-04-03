"""
Inseq-based token attribution analysis for LLMs.

This module provides an implementation of token analysis using the Inseq library
for attribution methods, replacing the custom gradient-based approach.
"""

from typing import List, Tuple, Dict, Any, Union, Optional
import torch
import numpy as np
from returns.result import Result, Success, Failure
import logging
from tqdm import tqdm
import inseq
import numpy.linalg
from datetime import datetime
from .model import ModelManager
from ..persistence.schema import AnalysisResult, AnalysisMetadata, AssociationData, TokenData
from ..persistence.inseq_schema import (
    InseqFeatureAttributionOutput, 
    InseqFeatureAttributionSequence
)
from ..persistence.storage import TokenAnalysisStorage
from ..utils.functional import safe_operation

logger = logging.getLogger(__name__)


class InseqError(Exception):
    """Base class for Inseq-specific errors"""
    pass


class AttributionError(InseqError):
    """Raised when attribution computation fails"""
    pass


class InseqTokenAnalyzer:
    """Analyzes token attributions using Inseq library"""

    def __init__(
            self,
            model_manager: ModelManager,
            attribution_method: str = "saliency"
    ) -> None:
        """
        Initialize InseqTokenAnalyzer with model manager.

        Args:
            model_manager: Initialized ModelManager instance
            attribution_method: Inseq attribution method (saliency, attention, etc.)
        """
        self.model_manager = model_manager
        # Store attribution method as string (Inseq 0.6.0+ uses string identifiers)
        if attribution_method in inseq.list_feature_attribution_methods():
            self.attribution_method = attribution_method
        else:
            logger.warning(f"Unknown attribution method: {attribution_method}, using saliency")
            self.attribution_method = "saliency"
            
        self.inseq_model = self._initialize_inseq_model()

    def _initialize_inseq_model(self):
        """
        Initialize Inseq model with the specified attribution method.
        
        Returns:
            Initialized Inseq model
        """
        try:
            # Use string method name for compatibility with Inseq 0.6.0+
            inseq_model = inseq.load_model(
                self.model_manager.model,
                self.attribution_method,
                tokenizer=self.model_manager.tokenizer
            )
            return inseq_model
        except Exception as e:
            logger.error(f"Failed to initialize Inseq model: {str(e)}")
            raise AttributionError(f"Inseq model initialization failed: {str(e)}")

    def create_analysis_pipeline(self, storage: TokenAnalysisStorage):
        """
        Creates a reusable analysis pipeline function.
        
        Args:
            storage: Storage instance for saving analysis results
            
        Returns:
            Analysis pipeline function
        """
        @safe_operation
        def process_single(prompt: str) -> Result[AnalysisResult, Exception]:
            """Process a single prompt through the attribution pipeline"""
            try:
                # Get attribution from Inseq
                attribution = self.inseq_model.attribute(prompt, pretty_progress=False, show_progress=False)
                
                try:
                    # Save native Inseq format
                    inseq_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_name = self.model_manager.config.llm_id.split('/')[-1]
                    clean_prompt = "".join(c for c in prompt.split()[:5] if c.isalnum() or c in "._- ").replace(" ", "_")[:50]
                    inseq_filename = f"{inseq_timestamp}_{model_name}_{clean_prompt}_inseq.json"
                    inseq_path = storage.data_path / inseq_filename
                    
                    # Save in native Inseq format
                    attribution.save(str(inseq_path))
                    logger.info(f"Saved Inseq native format to {inseq_path}")
                    
                    # Convert Inseq attribution to compatible format
                    analysis_result = self._convert_to_analysis_result(attribution, prompt)
                    
                    # Save to our custom storage format
                    storage.save(analysis_result)
                    
                    return Success(analysis_result)
                except Exception as convert_error:
                    logger.error(f"Error converting attribution: {convert_error}")
                    # Return a failure while preserving the error type
                    return Failure(convert_error)
            except Exception as e:
                logger.error(f"Analysis failed for prompt: {prompt}, {e}")
                return Failure(e)

        def process_batch(prompts: List[str]) -> Result[List[AnalysisResult], Exception]:
            """Process a batch of prompts"""
            results = []
            for prompt in tqdm(prompts, desc="Processing prompts", unit="prompt"):
                result = process_single(prompt)
                match result:
                    case Success(r):
                        results.append(r)
                    case Failure(error):
                        logger.error(f"Analysis failed for prompt: {prompt}, {error}")

            return Success(results) if results else Failure(Exception("No successful results"))

        def pipeline(input_data: Union[str, List[str]], logger=None) -> Result:
            """Main attribution pipeline"""
            result = (
                process_single(input_data) if isinstance(input_data, str)
                else process_batch(input_data)
            )

            if logger:
                match result:
                    case Success(_):
                        logger.info("Inseq analysis pipeline completed successfully")
                    case Failure(error):
                        logger.error(f"Inseq analysis pipeline failed: {error}")

            return result

        return pipeline

    def _convert_to_analysis_result(self, attribution, prompt: str) -> AnalysisResult:
        """
        Convert Inseq attribution to AnalysisResult format.
        
        Args:
            attribution: Inseq attribution result (FeatureAttributionOutput)
            prompt: Original input prompt
            
        Returns:
            AnalysisResult object compatible with existing visualization
        """
        try:
            logger.info(f"Processing Inseq attribution with structure: {type(attribution)}")
            
            # Debugging to understand structure
            if hasattr(attribution, 'sequence_attributions'):
                seq_count = len(attribution.sequence_attributions)
                logger.info(f"Found {seq_count} sequence attributions")
                
                if seq_count > 0:
                    first_seq = attribution.sequence_attributions[0]
                    logger.info(f"First sequence has source len: {len(first_seq.source)}, target len: {len(first_seq.target)}")
                    
                    # Log all attribution information
                    if hasattr(first_seq, 'source_attributions') and first_seq.source_attributions is not None:
                        logger.info(f"Source attributions present with shape: {getattr(first_seq.source_attributions, 'shape', 'unknown')}")
                        # Additional handling for list-type attributions
                        if isinstance(first_seq.source_attributions, list):
                            logger.info(f"Source attributions is a list of type: {type(first_seq.source_attributions[0])}")
                            # If it's a list of integers, we need special handling
                            if isinstance(first_seq.source_attributions[0], (int, list)):
                                # Convert to numpy array when passing to model
                                first_seq.source_attributions = np.array(first_seq.source_attributions, dtype=np.float64)
                    else:
                        logger.warning("Source attributions are None or not present")
                    
                    # Check for target attributions too
                    if hasattr(first_seq, 'target_attributions') and first_seq.target_attributions is not None:
                        logger.info(f"Target attributions present with shape: {getattr(first_seq.target_attributions, 'shape', 'unknown')}")
                        if isinstance(first_seq.target_attributions, list):
                            logger.info(f"Target attributions is a list of type: {type(first_seq.target_attributions[0])}")
                            if isinstance(first_seq.target_attributions[0], (int, list)):
                                first_seq.target_attributions = np.array(first_seq.target_attributions, dtype=np.float64)
                    else:
                        logger.warning("Target attributions are None or not present")
            
            # Convert raw Inseq attribution to our Pydantic model
            inseq_output = self._to_pydantic_model(attribution)
            
            # Process token IDs for the input and output tokens
            # since the Inseq attribution might not include them
            if inseq_output.sequence_attributions:
                seq_attr = inseq_output.sequence_attributions[0]  # Use the first sequence
                # Update token IDs in the Pydantic model data
                result = self._enrich_with_token_ids(
                    seq_attr.to_token_analyzer_format(
                        model_id=self.model_manager.config.llm_id,
                        prompt=prompt,
                        attribution_method=self.attribution_method
                    )
                )
                return result
            else:
                raise AttributionError("No sequence attributions found in Inseq output")
                
        except Exception as e:
            logger.error(f"Failed to convert Inseq attribution: {str(e)}", exc_info=True)
            raise AttributionError(f"Failed to convert Inseq attribution: {str(e)}")
            
    def _to_pydantic_model(self, attribution) -> InseqFeatureAttributionOutput:
        """
        Convert raw Inseq attribution to Pydantic model for easier processing.
        
        Args:
            attribution: Raw Inseq attribution output
            
        Returns:
            InseqFeatureAttributionOutput Pydantic model
        """
        try:
            # Extract sequence attributions
            if not hasattr(attribution, 'sequence_attributions'):
                raise AttributionError("Attribution object does not have sequence_attributions")
                
            sequences = []
            for seq_attr in attribution.sequence_attributions:
                # Convert each sequence attribution
                # For TokenWithId objects, extract the token string
                source_tokens = []
                for token in seq_attr.source:
                    if hasattr(token, 'token'):
                        source_tokens.append(token.token)
                    else:
                        source_tokens.append(str(token))
                
                target_tokens = []
                for token in seq_attr.target:
                    if hasattr(token, 'token'):
                        target_tokens.append(token.token)
                    else:
                        target_tokens.append(str(token))
                
                # Handle source_attributions
                source_attributions = None
                if hasattr(seq_attr, 'source_attributions') and seq_attr.source_attributions is not None:
                    source_attributions = seq_attr.source_attributions
                    if isinstance(source_attributions, list):
                        try:
                            # Convert list to numpy array for uniform handling
                            source_attributions = np.array(source_attributions, dtype=np.float64)
                            logger.info(f"Converted source attributions list to numpy array with shape {source_attributions.shape}")
                        except Exception as e:
                            logger.warning(f"Could not convert source attributions list to numpy: {e}")
                else:
                    logger.warning("Source attributions not available in sequence")
                
                # Handle target_attributions
                target_attributions = None
                if hasattr(seq_attr, 'target_attributions') and seq_attr.target_attributions is not None:
                    target_attributions = seq_attr.target_attributions
                    if isinstance(target_attributions, list):
                        try:
                            # Convert list to numpy array for uniform handling
                            target_attributions = np.array(target_attributions, dtype=np.float64)
                            logger.info(f"Converted target attributions list to numpy array with shape {target_attributions.shape}")
                        except Exception as e:
                            logger.warning(f"Could not convert target attributions list to numpy: {e}")
                else:
                    logger.warning("Target attributions not available in sequence")
                
                # Create the sequence with all available data
                sequence = InseqFeatureAttributionSequence(
                    source=source_tokens,
                    target=target_tokens,
                    source_attributions=source_attributions,
                    target_attributions=target_attributions,
                    step_scores=seq_attr.step_scores if hasattr(seq_attr, 'step_scores') else {},
                    sequence_scores=seq_attr.sequence_scores if hasattr(seq_attr, 'sequence_scores') else None,
                    attr_pos_start=seq_attr.attr_pos_start if hasattr(seq_attr, 'attr_pos_start') else 0,
                    attr_pos_end=seq_attr.attr_pos_end if hasattr(seq_attr, 'attr_pos_end') else None
                )
                
                sequences.append(sequence)
                
            # Create the complete output model
            return InseqFeatureAttributionOutput(
                sequence_attributions=sequences,
                step_attributions=attribution.step_attributions if hasattr(attribution, 'step_attributions') else None,
                info=attribution.info if hasattr(attribution, 'info') else {}
            )
        except Exception as e:
            logger.error(f"Error converting to Pydantic model: {str(e)}", exc_info=True)
            raise AttributionError(f"Error converting to Pydantic model: {str(e)}")
    
    def _enrich_with_token_ids(self, result: AnalysisResult) -> AnalysisResult:
        """
        Add token IDs to the analysis result.
        
        Args:
            result: Analysis result to update
            
        Returns:
            Updated analysis result with token IDs
        """
        try:
            # Update input token IDs
            for i, token_data in enumerate(result.data.input_tokens):
                token_data.token_id = self.model_manager.tokenizer.convert_tokens_to_ids(token_data.token)
                
            # Update output token IDs
            for i, token_data in enumerate(result.data.output_tokens):
                token_data.token_id = self.model_manager.tokenizer.convert_tokens_to_ids(token_data.token)
                
            return result
        except Exception as e:
            logger.error(f"Error enriching token IDs: {str(e)}")
            return result  # Return original result if we can't enrich

    @staticmethod
    def load_native_inseq(filepath: str) -> Result:
        """
        Load native Inseq attribution from file.
        
        Args:
            filepath: Path to saved Inseq attribution file
            
        Returns:
            Result containing loaded Inseq attribution
        """
        try:
            attribution = inseq.FeatureAttributionOutput.load(filepath)
            return Success(attribution)
        except Exception as e:
            logger.error(f"Failed to load Inseq attribution from {filepath}: {str(e)}")
            return Failure(e)
            
    def convert_native_to_analysis_result(self, filepath: str, prompt: str) -> Result[AnalysisResult, Exception]:
        """
        Convert a native Inseq attribution file to our AnalysisResult format.
        
        Args:
            filepath: Path to saved Inseq attribution file
            prompt: Original prompt used for attribution
            
        Returns:
            Result containing converted AnalysisResult
        """
        try:
            # Load native Inseq format
            result = self.load_native_inseq(filepath)
            match result:
                case Success(attribution):
                    # Convert to our format
                    analysis_result = self._convert_to_analysis_result(attribution, prompt)
                    return Success(analysis_result)
                case Failure(error):
                    return Failure(error)
        except Exception as e:
            logger.error(f"Failed to convert Inseq attribution: {str(e)}")
            return Failure(e)
    
    @staticmethod
    def _normalize_attribution_matrix(matrix: np.ndarray) -> np.ndarray:
        """
        Normalize attribution matrix values to [0,1] range.
        
        Args:
            matrix: Raw attribution matrix
            
        Returns:
            Normalized attribution matrix
        """
        try:
            normalized = np.zeros_like(matrix)
            for i in range(matrix.shape[1]):
                # Get non-zero values only
                valid_values = matrix[:, i][matrix[:, i] != 0]
                if len(valid_values) > 0:
                    column_min = valid_values.min()
                    column_max = valid_values.max()
                    if column_max > column_min:
                        normalized[:, i] = (
                                (matrix[:, i] - column_min)
                                / (column_max - column_min)
                        )
                    else:
                        normalized[:, i] = 0.5
            return normalized
        except Exception as e:
            logger.error(f"Failed to normalize attribution matrix: {str(e)}")
            raise AttributionError(f"Failed to normalize attribution matrix: {str(e)}")