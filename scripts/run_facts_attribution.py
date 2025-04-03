#!/usr/bin/env python
"""
Script to run token attribution analysis on the FACTS dataset.

This script runs all permutations of model, attribution method, and prompts 
from the FACTS dataset, saving and organizing the results. It also collects
detailed system information to contextualize the timing measurements, including
CPU/GPU specifications and PyTorch configuration.

Results are saved in the following formats:
- timing_results.csv: CSV with timing data for each model and attribution method
- summary.txt: Human-readable summary of results with system information
- system_info.json: Detailed system specifications in JSON format
"""

import os
import sys
import logging
import time
import platform
import socket
import subprocess
import json
from dataclasses import dataclass, field, asdict
from functools import partial

import nltk
import pandas as pd
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple, NamedTuple
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Configure MPS for Apple Silicon if needed
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import inseq
from returns.result import Success, Failure, Result

from src import (
    ModelManager,
    InseqTokenAnalyzer,
    TokenAnalysisStorage
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("attribution_benchmark.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

nltk.download('averaged_perceptron_tagger_eng')


@dataclass
class SystemInfo:
    """Data class for storing system information"""
    hostname: str = ""
    platform: str = ""
    platform_version: str = ""
    processor: str = ""
    cpu_model: str = ""
    cpu_cores: int = 0
    memory_total_gb: float = 0.0
    gpu_info: str = ""
    cuda_version: str = ""
    torch_version: str = ""
    torch_cuda_available: bool = False
    torch_mps_available: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def collect(cls) -> 'SystemInfo':
        """Collect system information"""
        info = cls()
        
        # Basic system info
        info.hostname = socket.gethostname()
        info.platform = platform.system()
        info.platform_version = platform.version()
        info.processor = platform.processor()
        
        # CPU info
        try:
            if info.platform == "Darwin":  # macOS
                # Get CPU model name
                cpu_model = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
                info.cpu_model = cpu_model
                
                # Get CPU cores
                cpu_cores = int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).decode().strip())
                info.cpu_cores = cpu_cores
                
                # Get RAM
                mem_total = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip())
                info.memory_total_gb = mem_total / (1024**3)  # Convert to GB
                
            elif info.platform == "Linux":
                # CPU model from /proc/cpuinfo
                cpu_info = subprocess.check_output(["cat", "/proc/cpuinfo"]).decode()
                model_name_line = [line for line in cpu_info.split('\n') if "model name" in line]
                if model_name_line:
                    info.cpu_model = model_name_line[0].split(':')[1].strip()
                
                # CPU cores
                info.cpu_cores = os.cpu_count() or 0
                
                # Get RAM
                mem_info = subprocess.check_output(["cat", "/proc/meminfo"]).decode()
                mem_line = [line for line in mem_info.split('\n') if "MemTotal" in line]
                if mem_line:
                    mem_kb = int(mem_line[0].split(':')[1].strip().split()[0])
                    info.memory_total_gb = mem_kb / (1024**2)  # Convert KB to GB
                
            elif info.platform == "Windows":
                # On Windows, use wmic for CPU info
                cpu_model = subprocess.check_output(["wmic", "cpu", "get", "name"]).decode()
                cpu_model_lines = cpu_model.strip().split('\n')
                if len(cpu_model_lines) > 1:
                    info.cpu_model = cpu_model_lines[1].strip()
                
                # CPU cores
                info.cpu_cores = os.cpu_count() or 0
                
                # Get RAM
                mem_info = subprocess.check_output(["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"]).decode()
                mem_lines = mem_info.strip().split('\n')
                if len(mem_lines) > 1:
                    mem_bytes = int(mem_lines[1].strip())
                    info.memory_total_gb = mem_bytes / (1024**3)  # Convert bytes to GB
        except Exception as e:
            logger.warning(f"Failed to collect detailed CPU information: {e}")
            info.cpu_model = platform.processor()
            info.cpu_cores = os.cpu_count() or 0
        
        # GPU information
        try:
            # CUDA
            info.torch_cuda_available = torch.cuda.is_available()
            if info.torch_cuda_available:
                info.cuda_version = torch.version.cuda or "Unknown"
                gpu_count = torch.cuda.device_count()
                gpu_models = []
                for i in range(gpu_count):
                    gpu_models.append(torch.cuda.get_device_name(i))
                info.gpu_info = ", ".join(gpu_models)
            
            # MPS (Apple Silicon)
            info.torch_mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            if info.torch_mps_available and not info.gpu_info:
                if info.platform == "Darwin":
                    # Try to get Apple Silicon model
                    try:
                        chip_info = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
                        if "Apple" in chip_info:
                            info.gpu_info = f"Apple MPS ({chip_info})"
                        else:
                            info.gpu_info = "Apple MPS"
                    except:
                        info.gpu_info = "Apple MPS"
        except Exception as e:
            logger.warning(f"Failed to collect GPU information: {e}")
        
        # PyTorch version
        info.torch_version = torch.__version__
        
        return info


@dataclass
class TimingData:
    """Data class for storing timing information"""
    model_loading_time: float = 0.0
    attribution_time: float = 0.0
    total_time: float = 0.0
    average_prompt_time: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class PromptTimingData:
    """Data class for storing timing information for a single prompt"""
    attribution_time: float = 0.0  # Time taken by inseq attribution
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class PromptResult:
    """Data class for storing individual prompt results"""
    prompt_id: str
    prompt_text: str
    success: bool
    error_message: Optional[str] = None
    timing: PromptTimingData = field(default_factory=PromptTimingData)
    token_count: int = 0  # Number of tokens in the prompt
    output_token_count: int = 0  # Number of tokens in the output
    
    @property
    def attribution_time(self) -> float:
        """Get attribution time"""
        return self.timing.attribution_time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "success": self.success,
            "error_message": self.error_message,
            "attribution_time": self.timing.attribution_time,
            "token_count": self.token_count,
            "output_token_count": self.output_token_count
        }
    
    def to_row(self) -> Dict[str, Any]:
        """Convert to a row for the per-prompt DataFrame"""
        return {
            "prompt_id": self.prompt_id,
            "prompt_text": self.prompt_text,
            "success": self.success,
            "attribution_time": self.timing.attribution_time,
            "token_count": self.token_count,
            "output_token_count": self.output_token_count,
            "tokens_per_second": self.token_count / self.timing.attribution_time if self.timing.attribution_time > 0 else 0
        }


@dataclass
class AnalysisError:
    """Data class for storing error information"""
    stage: str
    error: str
    prompt_id: Optional[str] = None


@dataclass
class AnalysisResultData:
    """Data class for storing analysis results"""
    model_name: str
    attribution_method: str
    total_prompts: int
    successful_prompts: int = 0
    failed_prompts: int = 0
    prompt_results: List[PromptResult] = field(default_factory=list)
    errors: List[AnalysisError] = field(default_factory=list)
    timing: TimingData = field(default_factory=TimingData)
    system_info: SystemInfo = field(default_factory=SystemInfo.collect)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_prompts == 0:
            return 0.0
        return (self.successful_prompts / self.total_prompts) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "model_name": self.model_name,
            "attribution_method": self.attribution_method,
            "total_prompts": self.total_prompts,
            "successful_prompts": self.successful_prompts,
            "failed_prompts": self.failed_prompts,
            "success_rate": self.success_rate,
            "errors": [asdict(error) for error in self.errors],
            "timing": asdict(self.timing),
            "system_info": asdict(self.system_info)
        }
    
    def to_timing_row(self) -> Dict[str, Any]:
        """Convert to a row for the timing DataFrame"""
        # Get system info details to include
        system = self.system_info
        
        # Create basic timing info
        row = {
            "model": self.model_name,
            "attribution_method": self.attribution_method,
            "model_loading_time": self.timing.model_loading_time,
            "attribution_time": self.timing.attribution_time,
            "total_time": self.timing.total_time,
            "average_prompt_time": self.timing.average_prompt_time,
            "successful_prompts": self.successful_prompts,
            "total_prompts": self.total_prompts,
            "success_rate": self.success_rate,
        }
        
        # Add system information
        system_data = {
            "platform": system.platform,
            "cpu_model": system.cpu_model,
            "cpu_cores": system.cpu_cores,
            "memory_gb": round(system.memory_total_gb, 2),
            "gpu_info": system.gpu_info or "None",
            "cuda_available": system.torch_cuda_available,
            "mps_available": system.torch_mps_available,
            "torch_version": system.torch_version,
        }
        
        # Add system info to row
        row.update(system_data)
        
        return row

# Define models to test - starting with smaller subset for initial tests
MODELS = [
    {
        "name": "BART",
        "llm_id": "facebook/bart-large",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    {
        "name": "GPT-2",
        "llm_id": "gpt2",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
 # ----------
 #    {
 #        "name": "Llama-3-2-Instruct-3B",
 #        "llm_id": "meta-llama/Llama-3.2-3B-Instruct",
 #        "device": "auto",
 #        "torch_dtype": "float16",
 #        "type": "causal"
 #    },
 #    {
 #        "llm_id": "meta-llama/Llama-3.2-1B",
 #        "name": "Llama-3-2-1B",
 #        "device": "auto",
 #        "torch_dtype": "float16",
 #        "type": "causal"
 #    },
 #    {
 #        "name": "Llama-3-1-8B",
 #        "llm_id": "meta-llama/Llama-3.1-8B",
 #        "device": "auto",
 #        "torch_dtype": "float16",
 #        "type": "causal"
 #    },
 #    # ----------
 #    {
 #        "name": "DeepSeek-R1-Distill-Llama-8B",
 #        "llm_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
 #        "device": "auto",
 #        "torch_dtype": "float16",
 #        "type": "causal"
 #    },
 #    {
 #        "llm_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
 #        "name": "DeepSeek-R1-Distill-Qwen-1.5B",
 #        "device": "auto",
 #        "torch_dtype": "float16",
 #        "type": "causal"
 #    },
 #    {
 #        "name": "DeepSeek-R1-Distill-Qwen-7B",
 #        "llm_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
 #        "device": "auto",
 #        "torch_dtype": "float16",
 #        "type": "causal"
 #    },
 #    {
 #        "llm_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
 #        "name": "DeepSeek-R1-Distill-Qwen-14B",
 #        "device": "auto",
 #        "torch_dtype": "float16",
 #        "type": "causal"
 #    },
 #    # ----------
 #    {
 #        "llm_id": "Qwen/Qwen2.5-Omni-7B",
 #        "name": "Qwen2.5-Omni-7B",
 #        "device": "auto",
 #        "torch_dtype": "float16",
 #        "type": "causal"
 #    },
 #    {
 #        "llm_id": "Qwen/Qwen2.5-VL-3B-Instruct",
 #        "name": "Qwen2.5-VL-3B-Instruct",
 #        "device": "auto",
 #        "torch_dtype": "float16",
 #        "type": "causal"
 #    },
 #    {
 #        "llm_id": "Qwen/Qwen2.5-VL-7B-Instruct",
 #        "name": "Qwen2.5-VL-7B-Instruct",
 #        "device": "auto",
 #        "torch_dtype": "float16",
 #        "type": "causal"
 #    },
 #    {
 #        "llm_id": "Qwen/QwQ-32B",
 #        "name": "QwQ-32B",
 #        "device": "auto",
 #        "torch_dtype": "float16",
 #        "type": "causal"
 #    },
 #    # ----------
 #    {
 #        "llm_id": "mistralai/Mistral-Small-3.1-24B-Base-2503",
 #        "name": "Mistral-Small-3.1-24B-Base-2503",
 #        "device": "auto",
 #        "torch_dtype": "float16",
 #        "type": "causal"
 #    },
 #    {
 #        "llm_id": "mistralai/Mistral-Nemo-Base-2407",
 #        "name": "Mistral-Nemo-Base-2407",
 #        "device": "auto",
 #        "torch_dtype": "float16",
 #        "type": "causal"
 #    },
 #    # ----------
 #    {
 #        "llm_id": "google/gemma-3-27b-it",
 #        "name": "Gemma-3-27b-it",
 #        "device": "auto",
 #        "torch_dtype": "float16",
 #        "type": "causal"
 #    },
 #    {
 #        "llm_id": "google/gemma-3-27b-pt",
 #        "name": "Gemma-3-27b-pt",
 #        "device": "auto",
 #        "torch_dtype": "float16",
 #        "type": "causal"
 #    },
 #    {
 #        "llm_id": "google/gemma-3-12b-it",
 #        "name": "Gemma-3-12b-it",
 #        "device": "auto",
 #        "torch_dtype": "float16",
 #        "type": "causal"
 #    },
 #    {
 #        "llm_id": "google/gemma-3-12b-pt",
 #        "name": "Gemma-3-12b-pt",
 #        "device": "auto",
 #        "torch_dtype": "float16",
 #        "type": "causal"
 #    },
 #    {
 #        "llm_id": "google/gemma-3-4b-it",
 #        "name": "Gemma-3-4b-it",
 #        "device": "auto",
 #        "torch_dtype": "float16",
 #        "type": "causal"
 #    },
 #    {
 #        "llm_id": "google/gemma-3-4b-pt",
 #        "name": "Gemma-3-4b-pt",
 #        "device": "auto",
 #        "torch_dtype": "float16",
 #        "type": "causal"
 #    },
 #    {
 #        "llm_id": "google/gemma-3-1b-it",
 #        "name": "Gemma-3-1b-it",
 #        "device": "auto",
 #        "torch_dtype": "float16",
 #        "type": "causal"
 #    },
 #    {
 #        "llm_id": "google/gemma-3-1b-pt",
 #        "name": "Gemma-3-1b-pt",
 #        "device": "auto",
 #        "torch_dtype": "float16",
 #        "type": "causal"
 #    }
]

# Define attribution methods to test
ATTRIBUTION_METHODS = [
    'input_x_gradient',

    # 'layer_deeplift',

    'discretized_integrated_gradients',
    # 'layer_integrated_gradients',
    'value_zeroing',
    # 'deeplift',
    'saliency',
    'lime',
    # 'reagent',
    'integrated_gradients',
    'layer_gradient_x_activation',
    # 'occlusion',
    'attention',
    'gradient_shap',
    # 'sequential_integrated_gradients'
]

# Number of prompts to process from dataset (set to None for all)
MAX_PROMPTS = 20  # Adjust based on available compute and time constraints


def load_dataset(
    source: str = "hf://datasets/google/FACTS-grounding-public/examples.csv",
    fallback_path: str = "data/facts_examples.csv",
    limit: Optional[int] = None
) -> Result[List[str], Exception]:
    """
    Load prompts from a dataset.
    
    Args:
        source: Source path for the dataset (Hugging Face or local path)
        fallback_path: Fallback local path if source fails
        limit: Optional limit on number of prompts to load
        
    Returns:
        Result containing either a list of prompts or an exception
    """
    try:
        # Try to load from primary source
        try:
            logger.info(f"Loading dataset from {source}")
            df = pd.read_csv(source)
        except Exception as e:
            logger.warning(f"Failed to load from {source}: {e}")
            logger.info(f"Attempting to load from fallback path: {fallback_path}")
            # Fallback to local CSV if available
            df = pd.read_csv(fallback_path)
        
        # Log dataset info
        logger.info(f"Dataset loaded with {len(df)} entries")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Extract prompts from dataframe
        prompts = extract_prompts_from_dataframe(df)
        
        # Limit number of prompts if specified
        if limit and limit > 0 and limit < len(prompts):
            logger.info(f"Limiting to {limit} prompts")
            prompts = prompts[:limit]
        
        # Log sample prompts
        for i, prompt in enumerate(prompts[:3]):
            logger.info(f"Sample prompt {i+1}: {prompt[:100]}...")
        
        return Success(prompts)
    
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error(traceback.format_exc())
        return Failure(e)


def extract_prompts_from_dataframe(df: pd.DataFrame) -> List[str]:
    """
    Extract prompts from a dataframe based on column structure.
    
    Args:
        df: Pandas DataFrame containing prompt data
        
    Returns:
        List of extracted prompts
    """
    # Use full_prompt column if available
    if "full_prompt" in df.columns:
        return df["full_prompt"].tolist()
    
    # Otherwise combine system_instruction and user_request
    if "system_instruction" in df.columns and "user_request" in df.columns:
        return [
            f"{row['system_instruction']}\n\n{row['user_request']}"
            for _, row in df.iterrows()
        ]
    
    # If we don't have the expected columns, just use whatever we have
    logger.warning("Expected columns not found, using first text column")
    text_cols = [col for col in df.columns if df[col].dtype == 'object']
    if text_cols:
        return df[text_cols[0]].astype(str).tolist()
    
    # No usable text columns found
    logger.warning("No usable text columns found in dataset")
    return []


def create_output_structure(
    model_name: str, 
    method: str, 
    base_path: str
) -> Tuple[Path, TokenAnalysisStorage]:
    """
    Create output directory structure and storage handler.
    
    Args:
        model_name: Name of the model
        method: Attribution method
        base_path: Base output directory
        
    Returns:
        Tuple of (output_path, storage_handler)
    """
    # Create specific output path for this model and method
    safe_model_name = model_name.replace("/", "_")
    output_path = Path(base_path) / safe_model_name / f"method_{method}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize storage
    storage = TokenAnalysisStorage(base_path=str(output_path))
    
    return output_path, storage


def load_model(model_config: Dict[str, Any]) -> Result[Tuple[ModelManager, float], Exception]:
    """
    Load and initialize model with timing measurement.
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        Result containing either (model_manager, loading_time) or an exception
    """
    model_name = model_config["name"]
    logger.info(f"Loading model: {model_name}")
    
    # Start timing for model loading
    start_time = time.time()
    
    # Initialize model
    model_data = {
        "llm_id": model_config["llm_id"],
        "device": model_config["device"],
        "torch_dtype": model_config["torch_dtype"]
    }
    
    model_result = ModelManager.initialize(model_data)
    
    # Calculate loading time
    loading_time = time.time() - start_time
    
    # Add timing information to result
    match model_result:
        case Success(manager):
            logger.info(f"Model {model_name} loaded successfully in {loading_time:.2f} seconds")
            return Success((manager, loading_time))
        case Failure(error):
            logger.error(f"❌ Failed to load model {model_name}: {error}")
            return Failure(error)


def initialize_analyzer(
    manager: ModelManager, 
    method: str
) -> Result[Tuple[Callable, float], Exception]:
    """
    Initialize attribution analyzer with timing measurement.
    
    Args:
        manager: Initialized model manager
        method: Attribution method to use
        
    Returns:
        Result containing either (analysis_function, init_time) or an exception
    """
    try:
        # Start timing analyzer initialization
        start_time = time.time()
        
        # Initialize analyzer
        analyzer = InseqTokenAnalyzer(manager, attribution_method=method)
        
        # Calculate initialization time
        init_time = time.time() - start_time
        logger.info(f"Analyzer initialized with method '{method}' in {init_time:.2f} seconds")
        
        return Success((analyzer, init_time))
    except Exception as e:
        logger.error(f"❌ Failed to initialize analyzer with method {method}: {e}")
        logger.error(traceback.format_exc())
        return Failure(e)


def process_prompt(
    analyze_fn: Callable,
    model_manager: ModelManager,
    prompt: str,
    prompt_idx: int,
    total_prompts: int
) -> PromptResult:
    """
    Process a single prompt with attribution timing measurement.
    
    Args:
        analyze_fn: Analysis pipeline function
        model_manager: ModelManager instance (used only for error case token counting)
        prompt: Prompt text to analyze
        prompt_idx: Index of the prompt (0-based)
        total_prompts: Total number of prompts
        
    Returns:
        PromptResult with success status and attribution timing
    """
    prompt_id = f"prompt_{prompt_idx+1}"
    logger.info(f"Processing prompt {prompt_idx+1}/{total_prompts}")
    
    # Initialize timing data
    timing = PromptTimingData()
    
    # Initialize token counts
    token_count = 0
    output_token_count = 0
    
    try:
        # Start timing the attribution process
        start_time = time.time()
        
        # Run attribution analysis - this is the operation we want to measure
        result = analyze_fn(prompt)
        
        # Record attribution time
        timing.attribution_time = time.time() - start_time
        
        logger.info(f"Prompt {prompt_idx+1} processed in {timing.attribution_time:.2f} seconds")
        
        # Process result based on success/failure
        match result:
            case Success(analysis_result):
                # Success already logged in the outer logging
                
                # Extract token counts from the analysis result
                try:
                    token_count = len(analysis_result.data.input_tokens)
                    output_token_count = len(analysis_result.data.output_tokens)
                except (AttributeError, TypeError):
                    # If we can't get token counts from the result, estimate from the tokenizer
                    # This is a fallback and should rarely be needed if attribution is successful
                    token_count = len(model_manager.tokenizer(prompt, return_tensors="pt")["input_ids"][0])
                    output_token_count = 0
                
                return PromptResult(
                    prompt_id=prompt_id,
                    prompt_text=prompt,
                    success=True,
                    timing=timing,
                    token_count=token_count,
                    output_token_count=output_token_count
                )
            case Failure(error):
                logger.error(f"❌ Failed to process prompt {prompt_idx+1}: {error}")
                
                # In case of failure, get token count for analysis
                try:
                    token_count = len(model_manager.tokenizer(prompt, return_tensors="pt")["input_ids"][0])
                except Exception:
                    token_count = 0
                
                return PromptResult(
                    prompt_id=prompt_id,
                    prompt_text=prompt,
                    success=False,
                    error_message=str(error),
                    timing=timing,
                    token_count=token_count
                )
    except Exception as e:
        # Handle unexpected exceptions
        attribution_time = time.time() - start_time if 'start_time' in locals() else 0
        timing.attribution_time = attribution_time
        
        logger.error(f"❌ Exception processing prompt {prompt_idx+1}: {e}")
        logger.error(traceback.format_exc())
        
        # Try to get token count for the error case
        try:
            token_count = len(model_manager.tokenizer(prompt, return_tensors="pt")["input_ids"][0])
        except Exception:
            token_count = 0
        
        return PromptResult(
            prompt_id=prompt_id,
            prompt_text=prompt,
            success=False,
            error_message=str(e),
            timing=timing,
            token_count=token_count
        )


def run_attribution_analysis(
    model_config: Dict[str, Any],
    method: str,
    prompts: List[str],
    output_base_path: str
) -> AnalysisResultData:
    """
    Run attribution analysis for a specific model, method, and list of prompts.
    
    Args:
        model_config: Model configuration
        method: Attribution method to use
        prompts: List of prompts to analyze
        output_base_path: Base output directory
        
    Returns:
        AnalysisResultData with complete results including execution time
    """
    model_name = model_config["name"]
    logger.info(f"Testing model: {model_name} with method: {method}")
    
    # Initialize result object
    result = AnalysisResultData(
        model_name=model_name,
        attribution_method=method,
        total_prompts=len(prompts)
    )
    
    # Start timing for total execution
    start_time_total = time.time()
    
    # Create output structure
    output_path, storage = create_output_structure(model_name, method, output_base_path)
    
    # Load model
    model_load_result = load_model(model_config)
    
    match model_load_result:
        case Success((manager, model_loading_time)):
            # Record model loading time
            result.timing.model_loading_time = model_loading_time
            
            # Initialize analyzer
            analyzer_result = initialize_analyzer(manager, method)
            
            match analyzer_result:
                case Success((analyzer, _)):
                    # Create analysis pipeline
                    analyze = analyzer.create_analysis_pipeline(storage)
                    
                    # Start timing attribution process
                    attribution_start_time = time.time()
                    
                    # Process each prompt
                    for i, prompt in enumerate(tqdm(prompts, desc=f"{model_name}/{method} processing")):
                        prompt_result = process_prompt(analyze, manager, prompt, i, len(prompts))
                        result.prompt_results.append(prompt_result)
                        
                        # Update success/failure counts
                        if prompt_result.success:
                            result.successful_prompts += 1
                        else:
                            result.failed_prompts += 1
                            # Add to errors list
                            result.errors.append(AnalysisError(
                                stage="prompt_processing",
                                error=prompt_result.error_message or "Unknown error",
                                prompt_id=prompt_result.prompt_id
                            ))
                    
                    # Calculate attribution time
                    attribution_time = time.time() - attribution_start_time
                    result.timing.attribution_time = attribution_time
                    
                    # Calculate average attribution time and token throughput
                    if result.prompt_results:
                        # Calculate average attribution time
                        avg_attribution_time = sum(p.attribution_time for p in result.prompt_results) / len(result.prompt_results)
                        
                        # Set in result
                        result.timing.average_prompt_time = avg_attribution_time
                        
                        # Log timing information
                        logger.info(f"Average attribution time: {avg_attribution_time:.2f} seconds")
                        
                        # Calculate and log token processing throughput
                        total_tokens = sum(p.token_count for p in result.prompt_results)
                        successful_prompts = sum(1 for p in result.prompt_results if p.success)
                        
                        if total_tokens > 0 and attribution_time > 0 and successful_prompts > 0:
                            tokens_per_second = total_tokens / attribution_time
                            avg_tokens_per_prompt = total_tokens / successful_prompts
                            logger.info(f"Token processing throughput: {tokens_per_second:.2f} tokens/second")
                            logger.info(f"Average tokens per prompt: {avg_tokens_per_prompt:.1f} tokens")
                
                case Failure(error):
                    # Handle analyzer initialization failure (error already logged in initialize_analyzer)
                    result.failed_prompts = len(prompts)
                    result.errors.append(AnalysisError(
                        stage="analyzer_initialization",
                        error=str(error)
                    ))
        
        case Failure(error):
            # Handle model loading failure (error already logged in load_model)
            result.failed_prompts = len(prompts)
            result.errors.append(AnalysisError(
                stage="model_loading",
                error=str(error)
            ))
    
    # Calculate total execution time
    total_time = time.time() - start_time_total
    result.timing.total_time = total_time
    logger.info(f"Total execution time for {model_name}/{method}: {total_time:.2f} seconds")
    
    return result


def filter_available_methods(methods_to_check: List[str]) -> List[str]:
    """
    Filter attribution methods to only those available in Inseq.
    
    Args:
        methods_to_check: List of methods to check for availability
        
    Returns:
        List of available methods
    """
    # Get all available methods from Inseq
    available_methods = inseq.list_feature_attribution_methods()
    logger.info(f"Available attribution methods: {available_methods}")
    
    # Filter to only those in our list that are available
    filtered_methods = [m for m in methods_to_check if m in available_methods]
    logger.info(f"Using attribution methods: {filtered_methods}")
    
    return filtered_methods


def create_task_list(
    models: List[Dict[str, Any]],
    methods: List[str],
    prompts: List[str],
    output_dir: str
) -> List[Tuple[str, Callable[[], AnalysisResultData]]]:
    """
    Create a list of tasks for parallel execution.
    
    Args:
        models: List of model configurations
        methods: List of attribution methods
        prompts: List of prompts to process
        output_dir: Base output directory
        
    Returns:
        List of (task_id, task_fn) tuples
    """
    tasks = []
    
    for model_config in models:
        model_name = model_config["name"]
        
        for method in methods:
            # Create a unique key for this combination
            combo_key = f"{model_name}/{method}"
            logger.info(f"Creating task for {combo_key}")
            
            # Create a task function that captures the arguments
            task_fn = lambda mc=model_config, m=method: run_attribution_analysis(
                mc, m, prompts, output_dir
            )
            
            tasks.append((combo_key, task_fn))
    
    return tasks


def run_tasks_in_parallel(
    tasks: List[Tuple[str, Callable]],
    max_workers: int = 1
) -> Dict[str, AnalysisResultData]:
    """
    Run tasks in parallel using a thread pool.
    
    Args:
        tasks: List of (task_id, task_fn) tuples
        max_workers: Maximum number of parallel workers
        
    Returns:
        Dictionary mapping task_id to result
    """
    all_results = {}
    futures = []
    
    # Create a worker pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the pool
        for task_id, task_fn in tasks:
            logger.info(f"Scheduling task for {task_id}")
            future = executor.submit(task_fn)
            futures.append((task_id, future))
        
        # Process results as they complete
        for task_id, future in tqdm(futures, desc="Processing model/method combinations"):
            try:
                result = future.result()
                all_results[task_id] = result
                logger.info(f"✅ Completed {task_id}: {result.successful_prompts}/{result.total_prompts} successful")
            except Exception as e:
                logger.error(f"❌ Task {task_id} failed with error: {e}")
                logger.error(traceback.format_exc())
                # Create a failure result
                error_result = AnalysisResultData(
                    model_name=task_id.split('/')[0],
                    attribution_method=task_id.split('/')[1],
                    total_prompts=0,  # We don't know how many prompts at this level
                    successful_prompts=0,
                    failed_prompts=0
                )
                error_result.errors.append(AnalysisError(
                    stage="task_execution",
                    error=str(e)
                ))
                all_results[task_id] = error_result
    
    return all_results


def run_all_permutations(prompts: List[str], output_dir: str) -> Dict[str, AnalysisResultData]:
    """
    Run all permutations of models, methods, and prompts in parallel.
    
    Args:
        prompts: List of prompts to process
        output_dir: Base output directory
        
    Returns:
        Dictionary with results for all permutations
    """
    # Filter attribution methods to only those available in Inseq
    filtered_methods = filter_available_methods(ATTRIBUTION_METHODS)
    
    if not filtered_methods:
        logger.error("No valid attribution methods available")
        return {}
    
    # Create task list
    tasks = create_task_list(MODELS, filtered_methods, prompts, output_dir)
    
    if not tasks:
        logger.warning("No tasks created")
        return {}
    
    # Run tasks in parallel
    return run_tasks_in_parallel(tasks, max_workers=1)  # Can increase if machine has enough RAM


def create_per_prompt_dataframe(results: Dict[str, AnalysisResultData]) -> pd.DataFrame:
    """
    Create a DataFrame with per-prompt timing results.
    
    Args:
        results: Dictionary of analysis results
        
    Returns:
        DataFrame with per-prompt attribution timing data
    """
    all_prompt_rows = []
    
    for model_method, result in results.items():
        model_name, method = model_method.split('/')
        
        for prompt_result in result.prompt_results:
            # Get the basic prompt row
            row = prompt_result.to_row()
            
            # Add model and method information
            row["model"] = model_name
            row["attribution_method"] = method
            
            # Add system info
            system_info = result.system_info
            row["device"] = "CUDA" if system_info.torch_cuda_available else ("MPS" if system_info.torch_mps_available else "CPU")
            row["gpu_info"] = system_info.gpu_info
            
            all_prompt_rows.append(row)
    
    # Create DataFrame
    if not all_prompt_rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_prompt_rows)
    
    # Reorder columns for better readability
    if not df.empty:
        first_columns = ["model", "attribution_method", "prompt_id", "token_count", "output_token_count"]
        timing_columns = ["attribution_time", "tokens_per_second"]
        
        # Get remaining columns
        other_columns = [col for col in df.columns if col not in first_columns + timing_columns]
        
        # Reorder columns
        df = df[first_columns + timing_columns + other_columns]
    
    return df


def create_method_timing_dataframe(results: Dict[str, AnalysisResultData]) -> pd.DataFrame:
    """
    Create a DataFrame with method-level aggregated timing results.
    
    Args:
        results: Dictionary of analysis results
        
    Returns:
        DataFrame with aggregated timing data per method
    """
    # Extract timing data from results
    timing_data = [result.to_timing_row() for result in results.values()]
    
    # Create DataFrame
    df = pd.DataFrame(timing_data)
    
    # Reorder columns for better readability
    if not df.empty:
        first_columns = ["model", "attribution_method", "successful_prompts", "total_prompts", "success_rate"]
        timing_columns = ["model_loading_time", "attribution_time", "average_prompt_time", "total_time"]
        
        # Get system info columns
        system_columns = ["platform", "cpu_model", "cpu_cores", "memory_gb", "gpu_info", 
                          "cuda_available", "mps_available", "torch_version"]
        
        # Get remaining columns
        other_columns = [col for col in df.columns 
                        if col not in first_columns + timing_columns + system_columns]
        
        # Reorder columns
        df = df[first_columns + timing_columns + system_columns + other_columns]
    
    return df


def save_timing_results(method_df: pd.DataFrame, prompt_df: pd.DataFrame, output_dir: str) -> Tuple[str, str]:
    """
    Save timing results to CSV files and log summary statistics.
    
    Args:
        method_df: DataFrame with method-level timing data
        prompt_df: DataFrame with per-prompt timing data
        output_dir: Output directory
        
    Returns:
        Tuple of paths to saved CSV files (method_csv_path, prompt_csv_path)
    """
    # Check for empty dataframes
    if method_df.empty and prompt_df.empty:
        logger.warning("No timing data to save")
        return "", ""
    
    method_csv_path = ""
    prompt_csv_path = ""
    
    # Save method-level timing data
    if not method_df.empty:
        method_csv_path = os.path.join(output_dir, "method_timing_results.csv")
        method_df.to_csv(method_csv_path, index=False)
        logger.info(f"Method-level timing results saved to {method_csv_path}")
        
        # Print summary statistics
        logger.info("\nMethod Timing Summary:")
        try:
            fastest_model = method_df.loc[method_df['model_loading_time'].idxmin()]
            logger.info(f"Fastest model loading: {fastest_model['model']} ({fastest_model['model_loading_time']:.2f}s)")
            
            fastest_attribution = method_df.loc[method_df['attribution_time'].idxmin()]
            logger.info(f"Fastest attribution method: {fastest_attribution['model']}/{fastest_attribution['attribution_method']} ({fastest_attribution['attribution_time']:.2f}s)")
            
            fastest_prompt = method_df.loc[method_df['average_prompt_time'].idxmin()]
            logger.info(f"Fastest average prompt time: {fastest_prompt['model']}/{fastest_prompt['attribution_method']} ({fastest_prompt['average_prompt_time']:.2f}s)")
        except Exception as e:
            logger.error(f"Error generating method timing summary: {e}")
    
    # Save per-prompt timing data
    if not prompt_df.empty:
        prompt_csv_path = os.path.join(output_dir, "prompt_timing_results.csv")
        prompt_df.to_csv(prompt_csv_path, index=False)
        logger.info(f"Per-prompt timing results saved to {prompt_csv_path}")
        
        # Print summary statistics for per-prompt data
        logger.info("\nPer-Prompt Timing Summary:")
        try:
            # Find prompt with fastest attribution time
            fastest_prompt = prompt_df.loc[prompt_df['attribution_time'].idxmin()]
            logger.info(f"Fastest prompt: {fastest_prompt['prompt_id']} for {fastest_prompt['model']}/{fastest_prompt['attribution_method']} ({fastest_prompt['attribution_time']:.2f}s)")
            
            # Prompt with highest tokens per second
            if 'tokens_per_second' in prompt_df.columns:
                fastest_throughput = prompt_df.loc[prompt_df['tokens_per_second'].idxmax()]
                logger.info(f"Highest token throughput: {fastest_throughput['prompt_id']} for {fastest_throughput['model']}/{fastest_throughput['attribution_method']} ({fastest_throughput['tokens_per_second']:.2f} tokens/s)")
            
            # Group by model and method to get average timing stats
            if len(prompt_df) > 1:
                grouped = prompt_df.groupby(['model', 'attribution_method']).agg({
                    'attribution_time': 'mean',
                    'tokens_per_second': 'mean' if 'tokens_per_second' in prompt_df.columns else 'sum',
                    'token_count': ['sum', 'mean', 'std'],
                    'output_token_count': ['sum', 'mean', 'std']
                }).reset_index()
                
                # Flatten multi-level columns
                grouped.columns = ['_'.join(col).strip('_') for col in grouped.columns.values]
                
                # Save aggregated stats
                agg_csv_path = os.path.join(output_dir, "prompt_aggregated_results.csv")
                grouped.to_csv(agg_csv_path, index=False)
                logger.info(f"Prompt aggregated results saved to {agg_csv_path}")
                
        except Exception as e:
            logger.error(f"Error generating prompt timing summary: {e}")
    
    return method_csv_path, prompt_csv_path


def format_system_info(system_info: SystemInfo) -> str:
    """
    Format system information for display in summary report.
    
    Args:
        system_info: SystemInfo object
        
    Returns:
        Formatted string with system information
    """
    lines = [
        f"System Information:",
        f"  Platform: {system_info.platform} ({system_info.platform_version})",
        f"  CPU: {system_info.cpu_model or system_info.processor}",
        f"  CPU Cores: {system_info.cpu_cores}",
        f"  Memory: {system_info.memory_total_gb:.2f} GB",
    ]
    
    # GPU information
    if system_info.torch_cuda_available:
        lines.append(f"  GPU: {system_info.gpu_info} (CUDA {system_info.cuda_version})")
    elif system_info.torch_mps_available:
        lines.append(f"  GPU: {system_info.gpu_info} (Apple MPS)")
    else:
        lines.append(f"  GPU: None")
    
    # PyTorch version
    lines.append(f"  PyTorch: {system_info.torch_version}")
    
    # Device availability
    devices = []
    if system_info.torch_cuda_available:
        devices.append("CUDA")
    if system_info.torch_mps_available:
        devices.append("MPS")
    if not devices:
        devices.append("CPU only")
    
    lines.append(f"  Available Devices: {', '.join(devices)}")
    
    return "\n".join(lines)


def generate_summary_report(
    results: Dict[str, AnalysisResultData], 
    prompts: List[str], 
    timestamp: str, 
    output_dir: str
) -> str:
    """
    Generate and save a detailed summary report.
    
    Args:
        results: Dictionary of analysis results
        prompts: List of prompts processed
        timestamp: Analysis timestamp
        output_dir: Output directory
        
    Returns:
        Path to saved summary report
    """
    summary_path = os.path.join(output_dir, "summary.txt")
    
    with open(summary_path, "w") as f:
        f.write(f"FACTS Attribution Analysis - {timestamp}\n")
        f.write(f"Total prompts: {len(prompts)}\n\n")
        
        # Add system information if we have results
        if results:
            # Use system info from the first result (should be the same for all)
            first_result = next(iter(results.values()))
            f.write(format_system_info(first_result.system_info))
            f.write("\n\n")
        
        f.write("Results by model/method combination:\n")
        
        for combo_key, result in results.items():
            f.write(f"\n{combo_key}:\n")
            f.write(f"  Success rate: {result.success_rate:.1f}% ({result.successful_prompts}/{result.total_prompts})\n")
            
            # Add timing information
            timing = result.timing
            f.write(f"  Timing:\n")
            f.write(f"    Model loading time: {timing.model_loading_time:.2f} seconds\n")
            f.write(f"    Attribution time: {timing.attribution_time:.2f} seconds\n")
            f.write(f"    Average time per prompt: {timing.average_prompt_time:.2f} seconds\n")
            f.write(f"    Total execution time: {timing.total_time:.2f} seconds\n")
            
            # Write error examples
            if result.errors:
                f.write(f"  Errors: {len(result.errors)}\n")
                for i, error in enumerate(result.errors[:5]):  # Show first 5 errors
                    prompt_id = error.prompt_id or "unknown"
                    f.write(f"    - {prompt_id}: {error.error}\n")
                if len(result.errors) > 5:
                    f.write(f"    - ...and {len(result.errors) - 5} more errors\n")
    
    # Also save system info separately in JSON format for easy parsing
    if results:
        first_result = next(iter(results.values()))
        system_info_path = os.path.join(output_dir, "system_info.json")
        with open(system_info_path, "w") as f:
            json.dump(first_result.system_info.to_dict(), f, indent=2)
        logger.info(f"System information saved to {system_info_path}")
    
    logger.info(f"Summary report saved to {summary_path}")
    return summary_path


def create_output_directory() -> Tuple[str, str]:
    """
    Create output directory for results.
    
    Returns:
        Tuple of (output_dir, timestamp)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", "facts_attribution", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved to {output_dir}")
    
    return output_dir, timestamp


def main():
    """Main function to run attribution analysis on FACTS dataset."""
    logger.info("Starting FACTS dataset attribution analysis")
    
    # Collect and log system information
    logger.info("Collecting system information...")
    system_info = SystemInfo.collect()
    logger.info(format_system_info(system_info))
    
    # Create output directory
    output_dir, timestamp = create_output_directory()
    
    # Load dataset
    logger.info("Loading FACTS dataset")
    dataset_result = load_dataset(limit=MAX_PROMPTS)
    
    match dataset_result:
        case Success(prompts):
            if not prompts:
                logger.error("Dataset loaded but contains no prompts, exiting")
                return
            
            logger.info(f"Processing {len(prompts)} prompts")
            
            # Run analysis for all permutations
            logger.info("Starting attribution analysis for all model/method combinations")
            results = run_all_permutations(prompts, output_dir)
            
            if not results:
                logger.error("No results generated, exiting")
                return
            
            # Create timing dataframes
            method_timing_df = create_method_timing_dataframe(results)
            prompt_timing_df = create_per_prompt_dataframe(results)
            
            # Save timing results
            save_timing_results(method_timing_df, prompt_timing_df, output_dir)
            
            # Generate summary report
            generate_summary_report(results, prompts, timestamp, output_dir)
            
            logger.info("Attribution analysis completed successfully")
        
        case Failure(error):
            logger.error(f"Failed to load dataset: {error}")
            logger.error("Attribution analysis aborted")


if __name__ == "__main__":
    main()