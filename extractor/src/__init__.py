"""
LLaMA Token Analyzer - A framework for analyzing token importance in LLaMA models
"""

from .core.model import ModelManager, ModelConfig
from .core.analysis import TokenAnalyzer
from .core.inseq_analysis import InseqTokenAnalyzer
from .persistence.storage import TokenAnalysisStorage
from .persistence.schema import AnalysisResult, AnalysisMetadata
from .persistence.inseq_schema import (
    InseqFeatureAttributionOutput, 
    InseqFeatureAttributionSequence
)

__version__ = "0.1.0"
__all__ = [
    "ModelManager",
    "ModelConfig",
    "TokenAnalyzer",
    "InseqTokenAnalyzer",
    "TokenAnalysisStorage",
    "AnalysisResult",
    "AnalysisMetadata",
    "InseqFeatureAttributionOutput",
    "InseqFeatureAttributionSequence"
]