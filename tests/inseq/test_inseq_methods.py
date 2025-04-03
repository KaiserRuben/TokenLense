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
from datetime import datetime

import nltk

from src.visualization.main import visualize

# Configure MPS for Apple Silicon if needed
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
nltk.download('averaged_perceptron_tagger_eng')

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
 # # ----------
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

    'layer_deeplift',

    'discretized_integrated_gradients',
    'layer_integrated_gradients',
    'value_zeroing',
    'deeplift',
    'saliency',
    'lime',
    'reagent',
    'integrated_gradients',
    'layer_gradient_x_activation',
    'occlusion',
    'attention',
    'gradient_shap',
    'sequential_integrated_gradients'
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


def test_batch_processing(model_config: dict, storage_path: str, batch_methods: list) -> dict:
    """
    Test batch processing with multiple diverse prompts on a specific model.
    
    Args:
        model_config: Model configuration dictionary
        storage_path: Base path for storage
        batch_methods: List of attribution methods to test in batch mode
        
    Returns:
        Dictionary of method -> success/failure
    """
    model_name = model_config["name"]
    logger.info(f"Testing batch processing on model: {model_name}")

    # Create model-specific storage path for batch tests
    model_path = os.path.join(storage_path, model_name.replace("/", "_"), "batch_test")
    os.makedirs(model_path, exist_ok=True)

    # Initialize model
    model_data = {
        "llm_id": model_config["llm_id"],
        "device": model_config["device"],
        "torch_dtype": model_config["torch_dtype"]
    }

    model_result = ModelManager.initialize(model_data)
    results = {}

    # Create a diverse batch of test prompts with entirely different content
    batch_prompts = [
        "Explain the importance of attribution analysis in language models",
        "Describe three key benefits of using large language models in education",
        "What are the ethical considerations when implementing AI systems?",
        "Summarize the history of neural networks in three sentences"
    ]

    match model_result:
        case Success(manager):
            logger.info(f"Model {model_name} loaded successfully for batch processing")

            # Test each selected attribution method in batch mode
            for method in batch_methods:
                # Check if method is available in Inseq
                if method in inseq.list_feature_attribution_methods():
                    logger.info(f"Testing batch processing with method {method} on {model_name}")
                    
                    # Create storage with method-specific subfolder
                    method_path = os.path.join(model_path, f"method_{method}")
                    os.makedirs(method_path, exist_ok=True)
                    storage = TokenAnalysisStorage(base_path=method_path)
                    
                    try:
                        # Initialize analyzer with the method
                        analyzer = InseqTokenAnalyzer(manager, attribution_method=method)
                        analyze = analyzer.create_analysis_pipeline(storage)
                        
                        # Run batch analysis
                        logger.info(f"Running batch analysis with {len(batch_prompts)} different prompts")
                        result = analyze(batch_prompts)
                        
                        # Check results
                        match result:
                            case Success(analysis_results):
                                if not isinstance(analysis_results, list):
                                    logger.error(f"❌ Batch processing for {method} returned non-list result: {type(analysis_results)}")
                                    results[method] = "❌ Failed - Invalid result type"
                                    continue
                                    
                                logger.info(f"✅ Batch processing successful with {method}, received {len(analysis_results)} results")
                                
                                # Verify each result in the batch
                                success_count = 0
                                prompt_match_count = 0
                                file_save_count = 0
                                
                                for i, analysis_result in enumerate(analysis_results):
                                    try:
                                        # Check if result needs unwrapping
                                        if hasattr(analysis_result, 'unwrap'):
                                            actual_result = analysis_result.unwrap()
                                        else:
                                            actual_result = analysis_result
                                        
                                        # Print matrix shape for each result
                                        matrix_shape = np.array(actual_result.data.association_matrix).shape
                                        logger.info(f"Result {i+1} for prompt '{batch_prompts[i][:30]}...': Association matrix shape: {matrix_shape}")
                                        
                                        # Check original prompt is correctly stored
                                        stored_prompt = actual_result.metadata.prompt
                                        if stored_prompt and batch_prompts[i] in stored_prompt:
                                            logger.info(f"✓ Result {i+1} correctly stores original prompt")
                                            prompt_match_count += 1
                                        else:
                                            logger.warning(f"✗ Result {i+1} has incorrect prompt: '{stored_prompt[:50]}...'")
                                        
                                        # Check files were saved
                                        timestamp = datetime.now().strftime("%Y%m%d")
                                        model_short = model_config["llm_id"].split('/')[-1]
                                        
                                        # Get clean prompt for filename matching
                                        clean_prompt = "".join(c for c in batch_prompts[i].split()[:5] 
                                                             if c.isalnum() or c in "._- ").replace(" ", "_")[:20]
                                        
                                        # Find matching files
                                        inseq_files = list(storage.data_path.glob(f"*{timestamp}*{model_short}*{clean_prompt}*_inseq.json"))
                                        custom_files = [f for f in storage.data_path.glob(f"*{timestamp}*{model_short}*{clean_prompt}*.json") 
                                                       if "_inseq.json" not in f.name]
                                        
                                        if inseq_files and custom_files:
                                            logger.info(f"✓ Found both file formats for result {i+1}")
                                            file_save_count += 1
                                        else:
                                            logger.warning(f"✗ Missing files for result {i+1}: Inseq files: {len(inseq_files)}, Custom files: {len(custom_files)}")
                                        
                                        # Try visualization for each result
                                        visualize(actual_result, storage=storage)
                                        success_count += 1
                                    except Exception as viz_error:
                                        logger.error(f"Visualization error for batch item {i+1} with {method}: {viz_error}")
                                
                                # Determine overall success
                                if success_count == len(analysis_results) and prompt_match_count == len(analysis_results) and file_save_count == len(analysis_results):
                                    results[method] = f"✅ Complete success - All prompts processed and verified"
                                elif success_count > 0:
                                    results[method] = f"⚠️ Partial success - {success_count}/{len(batch_prompts)} results, {prompt_match_count} prompt matches, {file_save_count} file saves"
                                else:
                                    results[method] = "❌ Failed - No successful results"
                                
                            case Failure(error):
                                logger.error(f"❌ Batch processing failed with {method}: {error}")
                                results[method] = f"❌ Failed - {str(error)}"
                        
                    except Exception as e:
                        logger.error(f"❌ Error testing batch processing with {method}: {e}")
                        results[method] = f"❌ Failed - {str(e)}"
                else:
                    logger.warning(f"Method {method} not available in Inseq")
                    results[method] = "⚠️ Not Available"

        case Failure(error):
            logger.error(f"Failed to load model {model_name} for batch processing: {error}")
            # Mark all methods as failed due to model loading error
            results = {method: f"❌ Model Loading Failed: {error}" for method in batch_methods}

    return results


def test_file_formats(model_config: dict, storage_path: str) -> dict:
    """
    Test if batch processing correctly saves data in both Inseq and custom formats.
    
    Args:
        model_config: Model configuration dictionary
        storage_path: Base path for storage
        
    Returns:
        Dictionary with test results
    """
    model_name = model_config["name"]
    logger.info(f"Testing file format saving for model: {model_name}")

    # Create model-specific storage path
    model_path = os.path.join(storage_path, model_name.replace("/", "_"), "format_test")
    os.makedirs(model_path, exist_ok=True)

    # Initialize model
    model_data = {
        "llm_id": model_config["llm_id"],
        "device": model_config["device"],
        "torch_dtype": model_config["torch_dtype"]
    }

    model_result = ModelManager.initialize(model_data)
    
    if not isinstance(model_result, Success):
        logger.error(f"Failed to load model {model_name} for format test: {model_result.failure()}")
        return {"format_test": f"❌ Model Loading Failed: {model_result.failure()}"}
    
    manager = model_result.unwrap()
    storage = TokenAnalysisStorage(base_path=model_path)
    
    # Use a simple method for format testing
    method = "saliency"
    test_prompt = "Test prompt for file format verification"
    
    try:
        # Initialize analyzer with the method
        analyzer = InseqTokenAnalyzer(manager, attribution_method=method)
        analyze = analyzer.create_analysis_pipeline(storage)
        
        # Run analysis to generate files
        logger.info(f"Running analysis to test file formats")
        result = analyze(test_prompt)
        
        # Check results
        if isinstance(result, Success):
            # Check if files were saved
            inseq_files = list(storage.data_path.glob("*_inseq.json"))
            custom_files = [f for f in storage.data_path.glob("*.json") if "_inseq.json" not in f.name]
            
            logger.info(f"Found {len(inseq_files)} Inseq format files and {len(custom_files)} custom format files")
            
            # Verify content of files
            format_status = {}
            
            # Check Inseq format
            if inseq_files:
                inseq_file = inseq_files[0]
                try:
                    # Try to load file with Inseq's native loader
                    inseq_result = InseqTokenAnalyzer.load_native_inseq(str(inseq_file))
                    if isinstance(inseq_result, Success):
                        format_status["inseq_format"] = "✅ Successfully saved and loaded"
                    else:
                        format_status["inseq_format"] = f"❌ File exists but failed to load: {inseq_result.failure()}"
                except Exception as e:
                    format_status["inseq_format"] = f"❌ Failed to verify: {str(e)}"
            else:
                format_status["inseq_format"] = "❌ No Inseq format files found"
            
            # Check custom format
            if custom_files:
                custom_file = custom_files[0]
                try:
                    # Try to load file using storage
                    custom_result = storage.load_by_path(custom_file)
                    if custom_result:
                        format_status["custom_format"] = "✅ Successfully saved and loaded"
                    else:
                        format_status["custom_format"] = "❌ File exists but failed to load"
                except Exception as e:
                    format_status["custom_format"] = f"❌ Failed to verify: {str(e)}"
            else:
                format_status["custom_format"] = "❌ No custom format files found"
            
            return format_status
        else:
            error = result.failure()
            logger.error(f"❌ Format test failed: {error}")
            return {"format_test": f"❌ Failed - {str(error)}"}
            
    except Exception as e:
        logger.error(f"❌ Error testing file formats: {e}")
        return {"format_test": f"❌ Failed - {str(e)}"}

BATCH_ACTIVE = True  # Set to True to enable batch processing tests

def main():
    """Main function to run tests on all models and methods."""    # Create output directory
    output_path = os.path.join("./output", "inseq_methods_test")
    os.makedirs(output_path, exist_ok=True)

    # Print available attribution methods
    logger.info(f"Available attribution methods: {inseq.list_feature_attribution_methods()}")

    # Track overall results
    all_results = {}
    batch_results = {}
    format_results = {}

    # Test each model
    for model_config in TEST_MODELS:
        model_name = model_config["name"]
        logger.info(f"Testing model: {model_name}")

        # Test the model with single prompts and collect results
        results = test_model(model_config, output_path)
        all_results[model_name] = results
        
        # Test batch processing with a subset of methods
        if BATCH_ACTIVE:
            logger.info(f"Testing batch processing for model: {model_name}")
            # Choose a few efficient methods for batch testing
            batch_methods = ['saliency', 'input_x_gradient', 'attention']
            batch_test_results = test_batch_processing(model_config, output_path, batch_methods)
            batch_results[model_name] = batch_test_results
        
        # Test file format saving
        format_test_results = test_file_formats(model_config, output_path)
        format_results[model_name] = format_test_results

    # Print summary of standard results
    logger.info("\n===== TEST RESULTS SUMMARY =====")
    for model_name, results in all_results.items():
        logger.info(f"\nModel: {model_name}")
        for method, result in results.items():
            logger.info(f"  - {method}: {result}")
    
    # Print batch processing results
    if BATCH_ACTIVE:
        logger.info("\n===== BATCH PROCESSING RESULTS =====")
        for model_name, results in batch_results.items():
            logger.info(f"\nModel: {model_name} (Batch Processing)")
            for method, result in results.items():
                logger.info(f"  - {method}: {result}")
            
    # Print file format results
    logger.info("\n===== FILE FORMAT TEST RESULTS =====")
    for model_name, results in format_results.items():
        logger.info(f"\nModel: {model_name} (File Format Test)")
        for format_type, result in results.items():
            logger.info(f"  - {format_type}: {result}")

    logger.info("\nTest completed!")


if __name__ == "__main__":
    main()
