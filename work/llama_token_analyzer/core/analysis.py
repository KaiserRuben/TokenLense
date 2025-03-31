from typing import List, Tuple, Dict, Any, Union
import torch
import numpy as np
from returns.result import Result, Success, Failure
import logging
from tqdm import tqdm

from .model import ModelManager
from ..persistence.schema import AnalysisResult, AnalysisMetadata, AssociationData, TokenData
from ..persistence.storage import TokenAnalysisStorage, pad_matrix
from ..utils.functional import safe_operation
from ..visualization.plots import visualize_summed_token_association, visualize_token_influence

logger = logging.getLogger(__name__)


class AnalysisError(Exception):
    """Base class for analysis-specific errors"""
    pass


class TokenizationError(AnalysisError):
    """Raised when token processing fails"""
    pass


class GenerationError(AnalysisError):
    """Raised when text generation fails"""
    pass


class AssociationError(AnalysisError):
    """Raised when association computation fails"""
    pass


def prepare_prompt(text: str) -> str:
    """
    Format prompt for model input.
    Detects if the input already contains conversation formatting.
    Always ensures it ends with assistant header for model response.

    Args:
        text: Raw input text or formatted conversation

    Returns:
        Formatted prompt
    """
    if not text.strip():  # Handle empty or whitespace-only input
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    if "<|begin_of_text|>" in text:
        # Remove any trailing whitespace
        text = text.rstrip()
        # For pre-formatted conversations, just add the assistant header if needed
        if not text.endswith("<|start_header_id|>assistant<|end_header_id|>"):
            # If it ends with eot_id, just append the assistant header
            if text.endswith("<|eot_id|>"):
                return text + "<|start_header_id|>assistant<|end_header_id|>"
            # If it already has content after the last eot_id, add both eot_id and assistant header
            return text + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        return text

    # For single-turn prompts, format as a conversation
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""


class TokenAnalyzer:
    """Analyzes token importance and associations in model outputs"""

    def __init__(self, model_manager: ModelManager) -> None:
        """
        Initialize TokenAnalyzer with model manager.

        Args:
            model_manager: Initialized ModelManager instance
        """
        self.model_manager = model_manager

    def create_analysis_pipeline(self, storage):
        """Creates a reusable analysis pipeline function."""

        @safe_operation
        def process_single(prompt: str) -> Result[str, Exception]:

            result =  (Success(prompt)
                    .map(prepare_prompt)
                    .map(self.generate_text)
                    .map(self.compute_token_association))

            match result:
                case Success(r):
                    self.process_and_save(r, storage)
                    return r
                case Failure(error):
                    raise error

        def process_batch(prompts: List[str]) -> Result[List[str], Exception]:
            results = []
            for prompt in tqdm(prompts, desc="Processing prompts", unit="prompt"):
                result = process_single(prompt)
                match result:
                    case Success(r):
                        results.append(r)
                    case Failure(error):
                        logger.error(f"Analysis failed for prompt: {prompt}, {error}")

            return Success(results) if len(results) > 0 else Failure(Exception("No successful results"))

        def pipeline(input_data: Union[str, List[str]], logger=None) -> Result:
            result = (
                process_single(input_data) if isinstance(input_data, str)
                else process_batch(input_data)
            )

            if logger:
                match result:
                    case Success(_):
                        logger.info("Analysis pipeline completed successfully")
                    case Failure(error):
                        logger.error(f"Analysis pipeline failed: {error}")

            return result

        return pipeline

    def generate_text(
            self,
            prompt: str,
            max_new_tokens: int = 1000,
            **generation_kwargs: Any
    ) -> Result[Tuple[torch.Tensor, torch.Tensor, str], Exception]:
        """
        Generate text from prompt and return relevant tensors.

        Args:
            prompt: Formatted prompt
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation parameters

        Returns:
            Result containing tuple of (input_ids, generated_ids, generated_text)
            or Failure with error
        """
        text_result = self.model_manager.generate_text(
            prompt,
            max_new_tokens,
            **generation_kwargs
        )
        match text_result:
            case Success((input_ids, generated_ids)):
                return input_ids, generated_ids, self.model_manager.tokenizer.decode(generated_ids)
            case Failure(error):
                return error

    @safe_operation
    def process_and_save(self, result: AnalysisResult, storage: TokenAnalysisStorage):
        """Process and save analysis results"""
        try:
            storage.save(result)
            # Convert the list of lists to a numpy array before padding
            matrix_np = np.array(result.data.association_matrix)

            # Then pad the matrix
            result.data.association_matrix = pad_matrix(matrix_np,
                                                        len(result.data.input_tokens),
                                                        len(result.data.output_tokens))
        except Exception as e:
            logger.error(f"Analysis failed: {e}")

    def _try_process_tokens(
            self,
            input_ids: torch.Tensor,
            generated_ids: torch.Tensor
    ) -> Result[Tuple[List[TokenData], List[TokenData]], TokenizationError]:
        """
        Safely process input and output tokens.
        """
        try:
            input_tokens = self._process_tokens(input_ids[0])
            output_tokens = self._process_tokens(generated_ids)
            return Success((input_tokens, output_tokens))
        except Exception as e:
            return Failure(TokenizationError(f"Failed to process tokens: {str(e)}"))

    def _try_calculate_associations(
            self,
            input_ids: torch.Tensor,
            generated_ids: torch.Tensor
    ) -> Result[np.ndarray, AssociationError]:
        """
        Safely calculate token associations.
        """
        try:
            matrix = self._calculate_association_matrix(input_ids, generated_ids)
            return Success(matrix)
        except Exception as e:
            return Failure(AssociationError(f"Failed to calculate associations: {str(e)}"))

    def compute_token_association(
            self,
            generation_tuple: Tuple[torch.Tensor, torch.Tensor, str]
    ) -> Result[AnalysisResult, Exception]:
        """
        Compute token associations for generated text.

        Args:
            generation_tuple: Tuple from generate_text containing
                            (input_ids, generated_ids, generated_text)

        Returns:
            Result containing analysis results or Failure with error
        """

        input_ids, generated_ids, generated_text = generation_tuple

        # First, process the tokens
        tokens_result = self._try_process_tokens(input_ids, generated_ids)

        # Then, calculate associations
        matrix_result = self._try_calculate_associations(input_ids, generated_ids)

        # Combine both results to create the final analysis
        result = tokens_result.bind(
            lambda tokens: matrix_result.map(
                lambda matrix: AnalysisResult(
                    metadata=AnalysisMetadata(
                        llm_id=self.model_manager.config.llm_id,
                        llm_version=self.model_manager.config.llm_id.split('/')[-1],
                        prompt=generated_text,
                        generation_params={"max_new_tokens": len(generated_ids)}
                    ),
                    data=AssociationData(
                        input_tokens=tokens[0],
                        output_tokens=tokens[1],
                        association_matrix=matrix.tolist(),
                        normalized_association=self._normalize_association(matrix).tolist()
                    )
                )
            )
        )
        match result:
            case Success(r):
                return r
            case Failure(e):
                raise e

    def _calculate_association_matrix(
            self,
            input_ids: torch.Tensor,
            generated_ids: torch.Tensor
    ) -> np.ndarray:
        try:
            association_list = []
            generated_ids = generated_ids.unsqueeze(0)
            max_length = input_ids.size(1) + generated_ids.size(1)  # Total sequence length

            for i in tqdm(range(generated_ids.size(1)), desc="Calculating token association", unit="token"):
                self.model_manager.model.zero_grad()

                # Prepare input sequence
                current_input_ids = torch.cat([input_ids, generated_ids[:, :i]], dim=-1)

                with torch.set_grad_enabled(True):
                    embed = self.model_manager.model.get_input_embeddings()(current_input_ids)
                    embed.requires_grad_(True)
                    embed.retain_grad()

                    outputs = self.model_manager.model(inputs_embeds=embed)

                    target_id = (
                        generated_ids[0, i]
                        if i < generated_ids.size(1)
                        else self.model_manager.tokenizer.eos_token_id
                    )

                    loss = -torch.log_softmax(outputs.logits, dim=-1)[0, -1, target_id]
                    loss.backward()

                    # Get gradients and pad to max length
                    token_association = torch.norm(
                        embed.grad[0, :current_input_ids.shape[1]],
                        dim=-1
                    ).detach().cpu().numpy()

                    # Pad the array to max_length
                    padded_association = np.zeros(max_length)
                    padded_association[:len(token_association)] = token_association
                    association_list.append(padded_association)

            # Now all arrays in the list have the same length
            return np.array(association_list)

        except Exception as e:
            logger.error(f"Failed in association matrix calculation: {str(e)}", exc_info=True)
            raise AssociationError(f"Failed to calculate association matrix: {str(e)}")

    def _process_tokens(self, ids: torch.Tensor) -> List[TokenData]:
        """
        Convert token IDs to TokenData objects.

        Args:
            ids: Token IDs

        Returns:
            List of TokenData objects

        Raises:
            TokenizationError: If token processing fails
        """
        try:
            tokens = []
            for token_id in ids:
                token = self.model_manager.tokenizer.convert_ids_to_tokens(
                    token_id.item()
                )
                tokens.append(TokenData(
                    token=token,
                    token_id=token_id.item(),
                    clean_token=token.lstrip('Ġ').strip()
                ))
            return tokens
        except Exception as e:
            raise TokenizationError(f"Failed to process tokens: {str(e)}")

    @staticmethod
    def _normalize_association(matrix: np.ndarray) -> np.ndarray:
        """
        Normalize association matrix values to [0,1] range.

        Args:
            matrix: Raw association matrix

        Returns:
            Normalized association matrix

        Raises:
            AssociationError: If normalization fails
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
            raise AssociationError(f"Failed to normalize association matrix: {str(e)}")
