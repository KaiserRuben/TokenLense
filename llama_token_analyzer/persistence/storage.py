from pathlib import Path
from typing import Dict, List, Optional, Any
import json
from datetime import datetime, timezone
import logging
import numpy as np
from returns.result import Result, Success, Failure
from returns.maybe import Maybe, Some, Nothing

from .schema import (
    AnalysisResult,
    AnalysisMetadata,
    AssociationData,
    DataVersion,
    TokenData
)

logger = logging.getLogger(__name__)


def is_system_token(token):
    """Helper function to identify system tokens."""
    system_tokens = {
        "<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>",
        "<|eot_id|>", "system", "user", "assistant"
    }
    return token in system_tokens or token.startswith('\\u')


def pad_matrix(matrix: np.ndarray, n_input_tokens: int, n_output_tokens: int) -> np.ndarray:
    """
    Pad a matrix to match token dimensions, handling the fact that the matrix
    is structured as [output_tokens x input_tokens].

    Args:
        matrix: Input matrix with shape [n_output, n_input]
        n_input_tokens: Number of input tokens
        n_output_tokens: Number of output tokens

    Returns:
        Padded matrix with shape [n_output_tokens, n_input_tokens]
    """
    current_output, current_input = matrix.shape

    # First pad to match the number of input tokens (columns)
    if current_input < n_input_tokens:
        padding_width = ((0, 0), (0, n_input_tokens - current_input))
        matrix = np.pad(matrix, padding_width, mode='constant', constant_values=0)
    elif current_input > n_input_tokens:
        # Trim if necessary
        matrix = matrix[:, :n_input_tokens]

    # Then pad to match the number of output tokens (rows)
    if current_output < n_output_tokens:
        padding_height = ((0, n_output_tokens - current_output), (0, 0))
        matrix = np.pad(matrix, padding_height, mode='constant', constant_values=0)
    elif current_output > n_output_tokens:
        # Trim if necessary
        matrix = matrix[:n_output_tokens, :]

    return matrix


class TokenAnalysisStorage:
    def __init__(
            self,
            base_path: str | Path,
            version: DataVersion = DataVersion.V1_0_0
    ) -> None:
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "data"
        self.graph_path = self.base_path / "graphs"
        self.version = version
        self._ensure_storage_directory()

    def load(self, filepath: str | Path) -> Result[AnalysisResult, Exception]:
        """
        Load analysis results from storage with matrix padding.
        """
        try:
            filepath = Path(filepath)
            with filepath.open('r') as f:
                data = json.load(f)

            # Before creating the AnalysisResult, pad the association matrix
            if 'data' in data and 'association_matrix' in data['data']:
                matrix = np.array(data['data']['association_matrix'])
                input_tokens = data['data']['input_tokens']
                output_tokens = data['data']['output_tokens']

                # Log original dimensions
                logger.debug(f"Original matrix shape: {matrix.shape}")
                logger.debug(f"Input tokens length: {len(input_tokens)}")
                logger.debug(f"Output tokens length: {len(output_tokens)}")

                # Pad matrix to match token counts, keeping orientation [output x input]
                padded_matrix = pad_matrix(
                    matrix,
                    n_input_tokens=len(input_tokens),
                    n_output_tokens=len(output_tokens)
                )

                # Update the data with padded matrix
                data['data']['association_matrix'] = padded_matrix.tolist()
                logger.info(f"Padded matrix to shape: {padded_matrix.shape}")

            return Success(AnalysisResult.model_validate(data))
        except Exception as e:
            logger.error(f"Failed to load analysis results: {str(e)}")
            return Failure(e)

    def save(self, analysis_result: AnalysisResult) -> Result[Path, Exception]:
        """
        Save analysis results to storage, ensuring matrix dimensions match token counts.
        """
        try:
            # Validate and adjust matrix dimensions before saving
            data = analysis_result.model_dump()
            input_tokens = data['data']['input_tokens']
            output_tokens = data['data']['output_tokens']
            matrix = np.array(data['data']['association_matrix'])

            # Pad matrix to match token counts, keeping orientation [output x input]
            padded_matrix = pad_matrix(
                matrix,
                n_input_tokens=len(input_tokens),
                n_output_tokens=len(output_tokens)
            )

            # Update the data with padded matrix
            data['data']['association_matrix'] = padded_matrix.tolist()
            logger.info(f"Padded matrix before saving to shape: {padded_matrix.shape}")

            filepath = self._generate_filename(analysis_result.metadata)
            with filepath.open('w') as f:
                json.dump(
                    data,
                    f,
                    indent=2,
                    default=self._json_serializer
                )
            logger.info(f"Saved analysis results to {filepath}")
            return Success(filepath)

        except Exception as e:
            logger.error(f"Failed to save analysis results: {str(e)}")
            return Failure(e)

    def save_graph(self, plt, filename):
        plt.savefig(
            self.graph_path / f"{filename}",
            bbox_inches='tight',
            dpi=500
        )
        logger.info(f"Saved visualization to {self.graph_path /filename}")

    def save_multiple(self, analysis_results: List[AnalysisResult]) -> List[Result[Path, Exception]]:
        """
        Save multiple analysis results to storage.
        """
        return [self.save(result) for result in analysis_results]

    def load_multiple(
            self,
            filter_criteria: Optional[Dict[str, Any]] = None
    ) -> List[Result[AnalysisResult, Exception]]:
        """
        Load multiple analysis results matching filter criteria.
        """
        results: List[Result[AnalysisResult, Exception]] = []

        for filepath in self.data_path.glob("*.json"):
            result = self.load(filepath)
            match result:
                case Success(analysis_result):
                    if self._matches_criteria(analysis_result, filter_criteria):
                        results.append(result)
                case Failure(error):
                    logger.warning(f"Failed to load {filepath}: {str(error)}")
                    results.append(result)

        return results

    def _generate_filename(self, metadata: AnalysisMetadata) -> Path:
        """Generate a filename for storing analysis results."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        clean_prompt = "".join(
            c for c in metadata.prompt.split()[:5]
            if c.isalnum() or c in "._- "
        ).replace(" ", "_")[:50]
        filename = f"{timestamp}_{metadata.llm_version}_{clean_prompt}.json"
        return self.data_path / filename

    def _ensure_storage_directory(self) -> Result[None, Exception]:
        """Ensures storage directory exists."""
        try:
            self.data_path.mkdir(parents=True, exist_ok=True)
            self.graph_path.mkdir(parents=True, exist_ok=True)
            return Success(None)
        except Exception as e:
            logger.error(f"Failed to create storage directory: {str(e)}")
            return Failure(e)

    def _matches_criteria(self, result: AnalysisResult, criteria: Optional[Dict[str, Any]] = None) -> bool:
        """Check if result matches filter criteria."""
        if not criteria:
            return True

        for key, value in criteria.items():
            if key == "prompt_contains":
                if value.lower() not in result.metadata.prompt.lower():
                    return False
            elif key == "date_range":
                start_dt = datetime.fromisoformat(value[0])
                end_dt = datetime.fromisoformat(value[1])
                timestamp = result.metadata.timestamp
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=timezone.utc)
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=timezone.utc)
                if not start_dt <= timestamp <= end_dt:
                    return False
            elif key == "model_version":
                if value != result.metadata.llm_version:
                    return False
        return True

    @staticmethod
    def _json_serializer(obj: Any) -> Any:
        """Serialize special types to JSON."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, DataVersion):
            return obj.value
        raise TypeError(f"Type {type(obj)} not serializable")
