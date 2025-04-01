#!/usr/bin/env python
"""
Test script for different Inseq attribution methods.

This script tests multiple attribution methods and models with the Inseq
token analyzer implementation to validate compatibility and performance.
"""

import os
import sys
import logging
from pathlib import Path

from src.visualization.main import visualize

# Configure MPS for Apple Silicon if needed
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import inseq
from returns.result import Success, Failure
import numpy as np

# Add project root to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import (
    ModelManager,
    InseqTokenAnalyzer,
    TokenAnalysisStorage
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
    {
        "name": "Llama-3-Instruct",
        "llm_id": "meta-llama/Llama-3.2-3B-Instruct",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    {
        "name": "DeepSeek-R1",
        "llm_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    }
]

# Define attribution methods to test
ATTRIBUTION_METHODS = [
    "saliency",
    "attention",
    "integrated_gradients",
]

# Test prompts for different model types
TEST_PROMPTS = {
    "causal": "Explain the importance of attribution analysis in language models:",
    "seq2seq": "Translate to French: Hello, how are you today?"
}


def test_attribution_method(model_manager, method: str, prompt: str, storage_path: str) -> bool:
    """
    Test a specific attribution method with the given model and prompt.
    
    Args:
        model_manager: The initialized model manager
        method: The attribution method to test
        prompt: The prompt to analyze
        storage_path: Base path for storage
        
    Returns:
        True if test passed, False otherwise
    """
    logger.info(f"Testing attribution method: {method}")
    
    # Create storage with method-specific subfolder
    method_path = os.path.join(storage_path, f"method_{method}")
    os.makedirs(method_path, exist_ok=True)
    storage = TokenAnalysisStorage(base_path=method_path)
    
    try:
        # Initialize analyzer with the method
        analyzer = InseqTokenAnalyzer(model_manager, attribution_method=method)
        analyze = analyzer.create_analysis_pipeline(storage)
        
        # Run analysis
        logger.info(f"Running analysis with prompt: '{prompt}'")
        result = analyze(prompt)
        
        # Check results
        match result:
            case Success(analysis_result):
                logger.info(f"✅ Success with {method}")
                
                # Handle potential double wrapping of Success
                try:
                    # First, check if this is a Success container
                    if hasattr(analysis_result, 'unwrap'):
                        actual_result = analysis_result.unwrap()
                        logger.info(f"Unwrapped Success container for {method}")
                    else:
                        actual_result = analysis_result
                        
                    # Print matrix shape 
                    matrix_shape = np.array(actual_result.data.association_matrix).shape
                    logger.info(f"Association matrix shape: {matrix_shape}")
                    
                    # Try visualization
                    visualize(actual_result, storage=storage)
                    logger.info(f"Visualization successful for {method}")
                    return True
                except Exception as viz_error:
                    logger.error(f"Visualization error for {method}: {viz_error}")
                    logger.error(f"Result type: {type(analysis_result)}")
                    if hasattr(analysis_result, '_inner_value'):
                        logger.info(f"Inner value type: {type(analysis_result._inner_value)}")
                    return False
                
            case Failure(error):
                logger.error(f"❌ Failed with {method}: {error}")
                return False
            
    except Exception as e:
        logger.error(f"❌ Error testing {method}: {e}")
        return False


def test_model(model_config: dict, storage_path: str) -> dict:
    """
    Test all attribution methods with a specific model.
    
    Args:
        model_config: Model configuration dictionary
        storage_path: Base path for storage
        
    Returns:
        Dictionary of method -> success/failure
    """
    model_name = model_config["name"]
    logger.info(f"Testing model: {model_name}")
    
    # Create model-specific storage path
    model_path = os.path.join(storage_path, model_name.replace("/", "_"))
    os.makedirs(model_path, exist_ok=True)
    
    # Initialize model
    model_data = {
        "llm_id": model_config["llm_id"],
        "device": model_config["device"],
        "torch_dtype": model_config["torch_dtype"]
    }
    
    model_result = ModelManager.initialize(model_data)
    
    results = {}
    
    match model_result:
        case Success(manager):
            logger.info(f"Model {model_name} loaded successfully")
            
            # Select appropriate prompt for model type
            prompt_type = model_config.get("type", "causal")
            prompt = TEST_PROMPTS.get(prompt_type, TEST_PROMPTS["causal"])
            
            # Test each attribution method
            for method in ATTRIBUTION_METHODS:
                # Check if method is available in Inseq
                if method in inseq.list_feature_attribution_methods():
                    logger.info(f"Testing method {method} on {model_name}")
                    success = test_attribution_method(manager, method, prompt, model_path)
                    results[method] = "✅ Success" if success else "❌ Failed"
                else:
                    logger.warning(f"Method {method} not available in Inseq")
                    results[method] = "⚠️ Not Available"
                    
        case Failure(error):
            logger.error(f"Failed to load model {model_name}: {error}")
            # Mark all methods as failed due to model loading error
            results = {method: f"❌ Model Loading Failed: {error}" for method in ATTRIBUTION_METHODS}
    
    return results


def main():
    """Main function to run tests on all models and methods."""
    # Create output directory
    output_path = os.path.join("../../work/notebooks/work", "notebooks", "output", "inseq_methods_test")
    os.makedirs(output_path, exist_ok=True)
    
    # Print available attribution methods
    logger.info(f"Available attribution methods: {inseq.list_feature_attribution_methods()}")
    
    # Track overall results
    all_results = {}
    
    # Test each model
    for model_config in TEST_MODELS:
        model_name = model_config["name"]
        logger.info(f"🔍 Testing model: {model_name}")
        
        try:
            results = test_model(model_config, output_path)
            all_results[model_name] = results
        except Exception as e:
            logger.error(f"Error testing model {model_name}: {e}")
            all_results[model_name] = {method: f"❌ Error: {e}" for method in ATTRIBUTION_METHODS}
    
    # Print summary of results
    logger.info("\n===== TEST RESULTS SUMMARY =====")
    for model_name, results in all_results.items():
        logger.info(f"\nModel: {model_name}")
        for method, result in results.items():
            logger.info(f"  - {method}: {result}")
    
    logger.info("\nTest completed!")


if __name__ == "__main__":
    main()