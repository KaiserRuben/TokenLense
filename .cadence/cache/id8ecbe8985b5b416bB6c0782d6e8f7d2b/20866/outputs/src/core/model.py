import os
os.environ['HF_HOME'] = './cache/model/'

from typing import Dict, Any, Tuple, Union

import psutil
import torch
from returns.pointfree import bind
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from returns.result import Result, Success, Failure
from returns.pipeline import flow
import logging
from pydantic import BaseModel, Field, ConfigDict
from ..utils.functional import safe_operation

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Configuration for model initialization"""
    llm_id: str = Field(..., description="Model identifier")
    device: str = Field(default="auto")
    torch_dtype: str = Field(default="float16")
    tokenizer_padding_side: str = Field(default="left")
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
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = self.config.tokenizer_padding_side

    @classmethod
    def initialize(cls, config: Dict[str, Any]) -> Result['ModelManager', Exception]:
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

        def load_model_and_tokenizer(config: ModelConfig) -> Result[ModelManager, Exception]:
            logger.info(f"Loading model {config.llm_id}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    config.llm_id,
                    trust_remote_code=config.trust_remote_code
                )

                device_map = cls._determine_device_map(device=config.device)

                # Print memory allocation info
                logger.info(f"Device map: {device_map}")

                model = AutoModelForCausalLM.from_pretrained(
                    config.llm_id,
                    device_map=device_map,
                    torch_dtype=config.torch_dtype,
                    load_in_8bit=config.load_in_8bit,
                    trust_remote_code=config.trust_remote_code
                )

                instance = cls(model, tokenizer, config)
                return Success(instance)
            except Exception as e:
                logger.error(f"Failed to load model and tokenizer: {str(e)}")
                return Failure(e)

        return flow(
            config,
            create_config,
            bind(load_model_and_tokenizer)
        )

    @classmethod
    def _setup_device_map(cls) -> Dict[str, str]:
        """
        Create an optimized device map for multiple GPUs and CPU.
        Returns a dictionary mapping device identifiers to memory allocations.

        Returns:
            Dict[str, str]: Device map with format {"device": "memoryGiB"}
                           e.g., {"cpu": "24GiB", "0": "18GiB", "1": "18GiB"}
        """
        # Get available CPU memory
        cpu_memory_bytes = psutil.virtual_memory().available
        cpu_memory_gib = int(cpu_memory_bytes * 0.7 / (1024 ** 3))  # Use 70% of available memory
        device_map = {"cpu": f"{cpu_memory_gib}GiB"}

        if not torch.cuda.is_available():
            logger.info("No CUDA devices available, using CPU only")
            return device_map

        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            logger.info("No GPU devices found, using CPU only")
            return device_map

        # Get memory info for each GPU
        for i in range(n_gpus):
            try:
                gpu_properties = torch.cuda.get_device_properties(i)
                total_memory = gpu_properties.total_memory
                # Convert to GiB and leave buffer (use 80% of available memory)
                usable_memory = int(total_memory * 0.8 / (1024 ** 3))
                device_map[str(i)] = f"{usable_memory}GiB"
                logger.info(f"GPU {i}: {gpu_properties.name}, "
                            f"Total Memory: {total_memory / (1024 ** 3):.1f} GiB, "
                            f"Allocated: {usable_memory} GiB")
            except Exception as e:
                logger.warning(f"Failed to get properties for GPU {i}: {str(e)}")
                continue

        return device_map
    @classmethod
    def _determine_device_map(cls,
                              device: str) -> Union[str, Dict[str, str]]:
        """
        Determine appropriate device mapping for model.

        Args:
            device: Requested device string
            model_id: Model identifier for memory requirement checking

        Returns:
            Union[str, Dict[str, str]]: Device mapping configuration
        """
        if device != "auto":
            return device

        if not torch.cuda.is_available():
            if torch.mps.is_available():
                return "mps"
            return "cpu"

        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            return "cpu"
        elif n_gpus == 1:
            return "cuda"
        else:
            return cls._setup_device_map()

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