"""
Schema definitions for benchmarking data models.

This module defines data structures used to represent and track benchmarking results
for model attribution analysis.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime


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
    system_info: SystemInfo = field(default_factory=SystemInfo)

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