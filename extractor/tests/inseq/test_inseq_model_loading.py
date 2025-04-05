#!/usr/bin/env python
"""
Test script for verifying model loading with Inseq.

This script tests whether various models can be successfully loaded with Inseq
to ensure compatibility before running more extensive attribution tests.
"""

import os
import sys
import logging
from pathlib import Path

# Configure MPS for Apple Silicon if needed
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import inseq
from returns.result import Success, Failure

# Add project root to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import (
    ModelManager,
    InseqTokenAnalyzer
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define models to test
TEST_MODELS = [
    {
        "name": "GPT-2",
        "llm_id": "gpt2",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
 # ----------
    {
        "name": "Llama-3-2-Instruct-3B",
        "llm_id": "meta-llama/Llama-3.2-3B-Instruct",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    {
        "llm_id": "meta-llama/Llama-3.2-1B",
        "name": "Llama-3-2-1B",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    {
        "name": "Llama-3-1-8B",
        "llm_id": "meta-llama/Llama-3.1-8B",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    # ----------
    {
        "name": "DeepSeek-R1-Distill-Llama-8B",
        "llm_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    {
        "llm_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "name": "DeepSeek-R1-Distill-Qwen-1.5B",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    {
        "name": "DeepSeek-R1-Distill-Qwen-7B",
        "llm_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    {
        "llm_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "name": "DeepSeek-R1-Distill-Qwen-14B",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    # ----------
    {
        "llm_id": "Qwen/Qwen2.5-Omni-7B",
        "name": "Qwen2.5-Omni-7B",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    {
        "llm_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "name": "Qwen2.5-VL-3B-Instruct",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    {
        "llm_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "name": "Qwen2.5-VL-7B-Instruct",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    {
        "llm_id": "Qwen/QwQ-32B",
        "name": "QwQ-32B",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    # ----------
    {
        "llm_id": "mistralai/Mistral-Small-3.1-24B-Base-2503",
        "name": "Mistral-Small-3.1-24B-Base-2503",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    {
        "llm_id": "mistralai/Mistral-Nemo-Base-2407",
        "name": "Mistral-Nemo-Base-2407",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    {
        "llm_id": "mistralai/Mistral-Nemo-Base-2407",
        "name": "Mistral-Nemo-Base-2407",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    # ----------
    {
        "llm_id": "google/gemma-3-27b-it",
        "name": "Gemma-3-27b-it",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    {
        "llm_id": "google/gemma-3-27b-pt",
        "name": "Gemma-3-27b-pt",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    {
        "llm_id": "google/gemma-3-12b-it",
        "name": "Gemma-3-12b-it",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    {
        "llm_id": "google/gemma-3-12b-pt",
        "name": "Gemma-3-12b-pt",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    {
        "llm_id": "google/gemma-3-4b-it",
        "name": "Gemma-3-4b-it",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    {
        "llm_id": "google/gemma-3-4b-pt",
        "name": "Gemma-3-4b-pt",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    {
        "llm_id": "google/gemma-3-1b-it",
        "name": "Gemma-3-1b-it",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    {
        "llm_id": "google/gemma-3-1b-pt",
        "name": "Gemma-3-1b-pt",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    }
]

# Attribution methods to try for each model (use a smaller subset for quick testing)
TEST_METHODS = ['saliency', 'attention', 'integrated_gradients']


def test_model_loading(model_config: dict) -> dict:
    """
    Test if a model can be loaded with Inseq.
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        Dictionary with test results
    """
    model_name = model_config["name"]
    llm_id = model_config["llm_id"]
    logger.info(f"Testing model loading: {model_name} ({llm_id})")
    
    results = {
        "model_loading": None,
        "attribution_methods": {}
    }
    
    # Step 1: Test basic model loading with ModelManager
    model_data = {
        "llm_id": model_config["llm_id"],
        "device": model_config["device"],
        "torch_dtype": model_config["torch_dtype"]
    }
    
    model_result = ModelManager.initialize(model_data)
    
    match model_result:
        case Success(manager):
            logger.info(f"✅ Model {model_name} loaded successfully with ModelManager")
            results["model_loading"] = "✅ Success"
            
            # Step 2: Test loading with Inseq for each attribution method
            for method in TEST_METHODS:
                if method in inseq.list_feature_attribution_methods():
                    logger.info(f"Testing method {method} with {model_name}")
                    try:
                        # Initialize Inseq analyzer with this method
                        analyzer = InseqTokenAnalyzer(manager, attribution_method=method)
                        
                        # Check if inseq_model was created successfully
                        if hasattr(analyzer, 'inseq_model') and analyzer.inseq_model is not None:
                            logger.info(f"✅ Successfully loaded {model_name} with {method} method")
                            results["attribution_methods"][method] = "✅ Success"
                        else:
                            logger.error(f"❌ Failed to load {model_name} with {method} method - model is None")
                            results["attribution_methods"][method] = "❌ Failed - model is None"
                    except Exception as e:
                        logger.error(f"❌ Failed to load {model_name} with {method} method: {e}")
                        results["attribution_methods"][method] = f"❌ Failed - {str(e)}"
                else:
                    logger.warning(f"⚠️ Method {method} not available in Inseq")
                    results["attribution_methods"][method] = "⚠️ Not Available"
        
        case Failure(error):
            logger.error(f"❌ Failed to load model {model_name}: {error}")
            results["model_loading"] = f"❌ Failed - {str(error)}"
            # Mark all methods as failed due to model loading error
            results["attribution_methods"] = {method: "❌ Model Loading Failed" for method in TEST_METHODS}
    
    return results


def main():
    """Main function to run model loading tests."""
    logger.info("Starting Inseq model loading tests")
    
    # Print available attribution methods
    available_methods = inseq.list_feature_attribution_methods()
    logger.info(f"Available attribution methods in Inseq: {available_methods}")
    
    # Track overall results
    all_results = {}
    
    # Test each model
    for model_config in TEST_MODELS:
        model_name = model_config["name"]
        logger.info(f"\n===== Testing model: {model_name} =====")
        
        # Test the model loading and collect results
        results = test_model_loading(model_config)
        all_results[model_name] = results
    
    # Print summary
    logger.info("\n===== MODEL LOADING TEST RESULTS =====")
    for model_name, results in all_results.items():
        logger.info(f"\nModel: {model_name}")
        logger.info(f"  Model Loading: {results['model_loading']}")
        
        logger.info("  Attribution Methods:")
        for method, result in results["attribution_methods"].items():
            logger.info(f"    - {method}: {result}")
    
    # Count successful models
    successful_models = sum(1 for results in all_results.values() if results["model_loading"] == "✅ Success")
    logger.info(f"\nSUMMARY: Successfully loaded {successful_models}/{len(TEST_MODELS)} models")
    
    # For each model, count successful attribution methods
    for model_name, results in all_results.items():
        if results["model_loading"] == "✅ Success":
            successful_methods = sum(1 for result in results["attribution_methods"].values() 
                                    if result == "✅ Success")
            total_methods = len(results["attribution_methods"])
            logger.info(f"  {model_name}: {successful_methods}/{total_methods} attribution methods successful")
    
    logger.info("\nTest completed!")


if __name__ == "__main__":
    main()