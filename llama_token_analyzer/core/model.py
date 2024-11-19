from typing import Dict, Optional, Any, Tuple
import torch
from returns.pointfree import bind
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from returns.result import Result, Success, Failure
from returns.pipeline import flow
import logging
from pydantic import BaseModel, Field, ConfigDict
from ..utils.functional import safe_operation, retry_operation

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Configuration for model initialization"""
    llm_id: str = Field(..., description="Model identifier")
    device: str = Field(default="auto")
    torch_dtype: str = Field(default="float16")
    tokenizer_padding_side: str = Field(default="left")
    max_memory: Optional[Dict[int, str]] = None
    load_in_8bit: bool = Field(default=False)
    trust_remote_code: bool = Field(default=False)

    model_config = ConfigDict(
        protected_namespaces=()
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for model initialization"""
        config_dict = self.model_dump()
        # Convert torch dtype string to actual dtype
        config_dict["torch_dtype"] = getattr(torch, config_dict["torch_dtype"])
        return config_dict


class ModelManager:
    """Manages model and tokenizer loading and operations"""

    def __init__(
            self,
            model: PreTrainedModel,
            tokenizer: PreTrainedTokenizer,
            config: ModelConfig
    ) -> None:
        """
        Initialize ModelManager with loaded model and tokenizer.

        Args:
            model: Loaded PyTorch model
            tokenizer: Loaded tokenizer
            config: Model configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self._setup_tokenizer()

    @classmethod
    def initialize(cls, config: Dict[str, Any]) -> Result[
        'ModelManager',
        Exception
    ]:
        """
        Safely initialize ModelManager from configuration.

        Args:
            config: Dictionary containing model configuration

        Returns:
            Result containing ModelManager instance or error
        """

        def create_config(cfg: Dict[str, Any]) -> Result[ModelConfig, Exception]:
            try:
                return Success(ModelConfig(**cfg))
            except Exception as e:
                logger.error(f"Failed to create model config: {str(e)}")
                return Failure(e)

        def load_model_and_tokenizer(
                config: ModelConfig
        ) -> Result[ModelManager, Exception]:
            """
            Load model and tokenizer with configuration.
            Returns a single Result instead of nested Results.
            """
            logger.info(f"Loading model {config.llm_id}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    config.llm_id,
                    trust_remote_code=config.trust_remote_code
                )

                device_map = cls._determine_device_map(config.device)

                model = AutoModelForCausalLM.from_pretrained(
                    config.llm_id,
                    device_map=device_map,
                    torch_dtype=getattr(torch, config.torch_dtype),
                    max_memory=config.max_memory,
                    load_in_8bit=config.load_in_8bit,
                    trust_remote_code=config.trust_remote_code
                )

                instance = cls(model, tokenizer, config)
                return Success(instance)
            except Exception as e:
                logger.error(f"Failed to load model and tokenizer: {str(e)}")
                return Failure(e)

        # Use flow to compose the operations
        return flow(
            config,
            create_config,
            bind(load_model_and_tokenizer)
        )

    @classmethod
    @retry_operation(max_attempts=2)
    def _load_model_and_tokenizer(
            cls,
            config: ModelConfig
    ) -> 'ModelManager':
        """
        Load model and tokenizer with configuration.

        Args:
            config: Model configuration

        Returns:
            ModelManager instance

        Raises:
            RuntimeError: If model or tokenizer loading fails
        """
        logger.info(f"Loading model {config.llm_id}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                config.llm_id,
                trust_remote_code=config.trust_remote_code
            )

            device_map = cls._determine_device_map(config.device)

            model = AutoModelForCausalLM.from_pretrained(
                config.llm_id,
                device_map=device_map,
                torch_dtype=getattr(torch, config.torch_dtype),
                max_memory=config.max_memory,
                load_in_8bit=config.load_in_8bit,
                trust_remote_code=config.trust_remote_code
            )

            instance = cls(model, tokenizer, config)
            return Success(instance)
        except Exception as e:
            logger.error(f"Failed to load model and tokenizer: {str(e)}")
            return Failure(e)

    @staticmethod
    def _determine_device_map(device: str) -> str:
        """
        Determine appropriate device mapping for model.

        Args:
            device: Requested device string

        Returns:
            Device mapping string
        """
        if device != "auto":
            return device

        # Check available devices
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _setup_tokenizer(self) -> None:
        """Configure tokenizer settings"""
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = self.config.tokenizer_padding_side

    @safe_operation
    def generate_text(
            self,
            prompt: str,
            max_new_tokens: int = 1000,
            **generation_kwargs: Any
    ) -> Result[Tuple[torch.Tensor, torch.Tensor], Exception]:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            **generation_kwargs: Additional generation parameters

        Returns:
            Result containing tuple of (input_ids, generated_ids)
            or Failure with error
        """
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True
            ).to(self.model.device)

            input_length = inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **generation_kwargs
                )

            return inputs.input_ids, outputs[0][input_length:]
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise e

    def get_token_ids_and_text(
            self,
            ids: torch.Tensor
    ) -> Tuple[torch.Tensor, str]:
        """
        Convert token IDs to text while preserving IDs.

        Args:
            ids: Tensor of token IDs

        Returns:
            Tuple of (token_ids, decoded_text)
        """
        return ids, self.tokenizer.decode(ids)

    def tokenize_text(
            self,
            text: str
    ) -> torch.Tensor:
        """
        Tokenize text to token IDs.

        Args:
            text: Input text

        Returns:
            Tensor of token IDs
        """
        return self.tokenizer(
            text,
            return_tensors="pt"
        ).input_ids[0]

    @property
    def device(self) -> torch.device:
        """Get current model device"""
        return self.model.device

    def __enter__(self) -> 'ModelManager':
        """Context manager enter"""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with cleanup"""
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()