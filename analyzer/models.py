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
    cpu: Dict[str, Any] = Field(..., description="CPU information")
    memory: Dict[str, Any] = Field(..., description="Memory information")
    gpu: Optional[List[Dict[str, Any]]] = Field(None, description="GPU information if available")
    python_version: str = Field(..., description="Python version")
    os: Dict[str, Any] = Field(..., description="Operating system information")
    timestamp: str = Field(..., description="When the information was collected")


class MethodTimingResult(BaseModel):
    """Model for method timing result"""
    model: str
    method: str
    average_time: float
    min_time: float
    max_time: float
    success_rate: float
    tokens_per_second: float


class PromptTimingResult(BaseModel):
    """Model for prompt timing result"""
    prompt: str
    model: str
    method: str
    time: float
    success: bool
    tokens: int
    tokens_per_second: float


class TimingResults(BaseModel):
    """Model for timing results response"""
    method_timing: List[MethodTimingResult]
    prompt_timing: List[PromptTimingResult]