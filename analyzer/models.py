from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Union
from enum import Enum


# Enum for aggregation methods
class AggregationMethod(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    L2_NORM = "l2_norm"
    ABS_SUM = "abs_sum"
    MAX = "max"


# Base models
class TokenWithId(BaseModel):
    """A token with its ID"""
    id: int
    token: str


class AttributionMatrixInfo(BaseModel):
    """Information about the attribution matrix dimensions and structure"""
    shape: List[int]
    dtype: str
    is_attention: bool
    tensor_type: str = Field(..., description="One of: attention, input_x_gradient, lime, integrated_gradients, saliency, layer_gradient_x_activation")


# Response models
class ModelInfo(BaseModel):
    """Model for model listing response"""
    models: List[str]


class ModelMethods(BaseModel):
    """Model for method listing response"""
    model: str
    methods: List[str]


class ModelMethodFile(BaseModel):
    """Model for file listing response"""
    model: str
    method: str
    files: List[str]
    file_details: Optional[List[Dict[str, Any]]] = Field(None, description="Optional details about each file")


class AttributionResponse(BaseModel):
    """Response model for attribution data"""
    model: str
    method: str
    file_id: int
    prompt: str
    generation: str
    source_tokens: List[str]
    target_tokens: List[str]
    attribution_matrix: List[List[float]]
    aggregation: str


class AttributionDetailedResponse(BaseModel):
    """Detailed response model for attribution data including token IDs"""
    model: str
    method: str
    file_id: int
    prompt: str
    generation: str
    source_tokens: List[TokenWithId]
    target_tokens: List[TokenWithId]
    attribution_matrix: List[List[float]]
    matrix_info: AttributionMatrixInfo
    aggregation: str
    exec_time: float = Field(..., description="Execution time of the attribution in seconds")
    original_attribution_shape: List[int] = Field(..., description="Original shape of the attribution tensor before aggregation")


class AggregationOptions(BaseModel):
    """Available aggregation methods for a given file"""
    methods: List[AggregationMethod]
    default: AggregationMethod


class SystemHardwareInfo(BaseModel):
    """Hardware component of system information"""
    name: str
    details: Dict[str, Any]


class SystemInfo(BaseModel):
    """System information model"""
    hostname: Optional[str] = None
    platform: Optional[str] = None
    platform_version: Optional[str] = None
    processor: Optional[str] = None
    cpu_model: Optional[str] = None
    cpu_cores: Optional[int] = None
    memory_total_gb: Optional[float] = None
    gpu_info: Optional[str] = None
    cuda_version: Optional[str] = None
    torch_version: Optional[str] = None
    torch_cuda_available: Optional[bool] = None
    torch_mps_available: Optional[bool] = None
    
    model_config = {
        "populate_by_name": True
    }


class MethodTimingResult(BaseModel):
    """Model for method timing result"""
    model: str
    method: str  # This is 'attribution_method' in CSV
    attribution_method: Optional[str] = None  # For compatibility with CSV
    successful_prompts: Optional[int] = None
    total_prompts: Optional[int] = None
    success_rate: float
    model_loading_time: Optional[float] = None
    attribution_time: Optional[float] = None
    average_time: Optional[float] = None  # CSV field for 'average_prompt_time'
    average_prompt_time: Optional[float] = None  # Field for CSV
    total_time: Optional[float] = None
    min_time: Optional[float] = None
    max_time: Optional[float] = None
    tokens_per_second: Optional[float] = None
    platform: Optional[str] = None
    cpu_model: Optional[str] = None
    cpu_cores: Optional[int] = None
    memory_gb: Optional[float] = None
    gpu_info: Optional[str] = None
    cuda_available: Optional[bool] = None
    mps_available: Optional[bool] = None
    torch_cuda_available: Optional[bool] = None  # CSV field
    torch_mps_available: Optional[bool] = None  # CSV field
    torch_version: Optional[str] = None
    
    model_config = {
        "populate_by_name": True  # Allow both alias and original names to be used
    }


class PromptTimingResult(BaseModel):
    """Model for prompt timing result"""
    model: str
    method: str  # This is 'attribution_method' in CSV
    attribution_method: Optional[str] = None  # For compatibility
    prompt: str  # This is 'prompt_text' in CSV
    prompt_text: Optional[str] = None  # For compatibility
    prompt_id: Optional[str] = None
    tokens: Optional[int] = None  # This is 'token_count' in CSV
    token_count: Optional[int] = None  # For compatibility
    output_token_count: Optional[int] = None
    time: Optional[float] = None  # This is 'attribution_time' in CSV
    attribution_time: Optional[float] = None  # For compatibility
    tokens_per_second: Optional[float] = None
    success: Optional[bool] = None
    device: Optional[str] = None
    gpu_info: Optional[str] = None
    
    model_config = {
        "populate_by_name": True  # Allow both alias and original names to be used
    }


class TimingResults(BaseModel):
    """Model for timing results response"""
    method_timing: List[MethodTimingResult]
    prompt_timing: List[PromptTimingResult]