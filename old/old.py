#%% md
# # LLaMA Token Analyzer - Complete Example
#
# This notebook demonstrates the complete functionality of the LLaMA Token Analyzer framework.
#
import os

from src.visualization.main import visualize

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
from returns.result import Success, Failure
from src import (
    ModelManager,
    TokenAnalyzer,
    TokenAnalysisStorage
)

import logging

logging.basicConfig(level=logging.INFO)

model_config = {
    "llm_id": "meta-llama/Llama-3.2-3B-Instruct",
    "device": "auto",
    "torch_dtype": "float16"
}

model_result = ModelManager.initialize(model_config)

# Initialize other components
storage = TokenAnalysisStorage(base_path="../work/notebooks/output")

match model_result:
    case Success(manager):
        analyzer = TokenAnalyzer(manager)
        analyze = analyzer.create_analysis_pipeline(storage)
    case Failure(error):
        raise RuntimeError(f"Failed to load model: {error}")

prompt = "Hello, "

analysis_result = analyze(prompt)

match analysis_result:
    case Success(r):
        visualize(r, storage=storage)
    case Failure(error):
        raise RuntimeError(f"Analysis pipeline failed: {error}")

