#!/usr/bin/env python
"""
Script to run token attribution analysis on the FACTS dataset.

This script runs all permutations of model, attribution method, and prompts 
from the FACTS dataset, saving and organizing the results.
"""

import os
import sys
import logging

import nltk
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Configure MPS for Apple Silicon if needed
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import inseq
from returns.result import Success, Failure

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

# Define models to test - starting with smaller subset for initial tests
MODELS = [
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


def load_facts_dataset(limit: Optional[int] = None) -> List[str]:
    """
    Load prompts from the FACTS dataset.
    
    Args:
        limit: Optional limit on number of prompts to load
        
    Returns:
        List of prompts from the dataset
    """
    try:
        # Try to load from Hugging Face datasets
        try:
            logger.info("Loading FACTS dataset from Hugging Face")
            df = pd.read_csv("hf://datasets/google/FACTS-grounding-public/examples.csv")
        except Exception as e:
            logger.warning(f"Failed to load from Hugging Face: {e}")
            logger.info("Attempting to load from local CSV")
            # Fallback to local CSV if available
            df = pd.read_csv("data/facts_examples.csv")
        
        # Log dataset info
        logger.info(f"Dataset loaded with {len(df)} entries")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Use full_prompt column if available
        if "full_prompt" in df.columns:
            prompts = df["full_prompt"].tolist()
        else:
            # Otherwise combine system_instruction and user_request
            prompts = []
            for _, row in df.iterrows():
                if "system_instruction" in df.columns and "user_request" in df.columns:
                    full_prompt = f"{row['system_instruction']}\n\n{row['user_request']}"
                    prompts.append(full_prompt)
                else:
                    # If we don't have the expected columns, just use whatever we have
                    logger.warning("Expected columns not found, using first text column")
                    text_cols = [col for col in df.columns if df[col].dtype == 'object']
                    if text_cols:
                        prompts.append(str(row[text_cols[0]]))
        
        # Limit number of prompts if specified
        if limit and limit > 0 and limit < len(prompts):
            logger.info(f"Limiting to {limit} prompts")
            prompts = prompts[:limit]
        
        # Log sample prompts
        for i, prompt in enumerate(prompts[:3]):
            logger.info(f"Sample prompt {i+1}: {prompt[:100]}...")
        
        return prompts
    
    except Exception as e:
        logger.error(f"Failed to load FACTS dataset: {e}")
        logger.error(traceback.format_exc())
        # Return empty list if dataset loading fails
        return []


def run_attribution_analysis(
    model_config: Dict[str, Any],
    method: str,
    prompts: List[str],
    output_base_path: str
) -> Dict[str, Any]:
    """
    Run attribution analysis for a specific model, method, and list of prompts.
    
    Args:
        model_config: Model configuration
        method: Attribution method to use
        prompts: List of prompts to analyze
        output_base_path: Base output directory
        
    Returns:
        Dictionary with test results
    """
    model_name = model_config["name"]
    logger.info(f"Testing model: {model_name} with method: {method}")
    
    # Create specific output path for this model and method
    output_path = os.path.join(output_base_path, model_name.replace("/", "_"), f"method_{method}")
    os.makedirs(output_path, exist_ok=True)
    storage = TokenAnalysisStorage(base_path=output_path)
    
    results = {
        "model_name": model_name,
        "attribution_method": method,
        "total_prompts": len(prompts),
        "successful_prompts": 0,
        "failed_prompts": 0,
        "errors": []
    }
    
    # Initialize model
    model_data = {
        "llm_id": model_config["llm_id"],
        "device": model_config["device"],
        "torch_dtype": model_config["torch_dtype"]
    }
    
    model_result = ModelManager.initialize(model_data)
    
    match model_result:
        case Success(manager):
            logger.info(f"Model {model_name} loaded successfully")
            
            try:
                # Initialize analyzer with method
                analyzer = InseqTokenAnalyzer(manager, attribution_method=method)
                analyze = analyzer.create_analysis_pipeline(storage)
                
                # Process each prompt
                for i, prompt in enumerate(tqdm(prompts, desc=f"{model_name}/{method} processing")):
                    prompt_id = f"prompt_{i+1}"
                    
                    try:
                        # Run analysis on this prompt
                        logger.info(f"Processing prompt {i+1}/{len(prompts)}")
                        result = analyze(prompt)
                        
                        match result:
                            case Success(_):
                                logger.info(f"✅ Successfully processed prompt {i+1}")
                                results["successful_prompts"] += 1
                            case Failure(error):
                                logger.error(f"❌ Failed to process prompt {i+1}: {error}")
                                results["failed_prompts"] += 1
                                results["errors"].append({
                                    "prompt_id": prompt_id,
                                    "error": str(error)
                                })
                    except Exception as e:
                        logger.error(f"❌ Exception processing prompt {i+1}: {e}")
                        logger.error(traceback.format_exc())
                        results["failed_prompts"] += 1
                        results["errors"].append({
                            "prompt_id": prompt_id,
                            "error": str(e)
                        })
            
            except Exception as e:
                logger.error(f"❌ Failed to initialize analyzer with method {method}: {e}")
                logger.error(traceback.format_exc())
                results["failed_prompts"] = len(prompts)
                results["errors"].append({
                    "stage": "analyzer_initialization",
                    "error": str(e)
                })
        
        case Failure(error):
            logger.error(f"❌ Failed to load model {model_name}: {error}")
            logger.error(traceback.format_exc())
            results["failed_prompts"] = len(prompts)
            results["errors"].append({
                "stage": "model_loading",
                "error": str(error)
            })
    
    # Return detailed results
    return results


def run_all_permutations(prompts: List[str], output_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Run all permutations of models, methods, and prompts in parallel.
    
    Args:
        prompts: List of prompts to process
        output_dir: Base output directory
        
    Returns:
        Dictionary with results for all permutations
    """
    all_results = {}
    tasks = []
    
    # List available attribution methods
    available_methods = inseq.list_feature_attribution_methods()
    logger.info(f"Available attribution methods: {available_methods}")
    
    # Filter attribution methods to only those available in Inseq
    filtered_methods = [m for m in ATTRIBUTION_METHODS if m in available_methods]
    logger.info(f"Using attribution methods: {filtered_methods}")
    
    # Create a worker pool
    with ThreadPoolExecutor(max_workers=1) as executor:  # Can increase if machine has enough RAM
        # Submit all model+method combinations to the pool
        for model_config in MODELS:
            model_name = model_config["name"]
            
            for method in filtered_methods:
                # Create a unique key for this combination
                combo_key = f"{model_name}/{method}"
                logger.info(f"Scheduling task for {combo_key}")
                
                # Submit task to the pool
                future = executor.submit(
                    run_attribution_analysis,
                    model_config,
                    method,
                    prompts,
                    output_dir
                )
                tasks.append((combo_key, future))
        
        # Process results as they complete
        for combo_key, future in tqdm(tasks, desc="Processing model/method combinations"):
            try:
                result = future.result()
                all_results[combo_key] = result
                logger.info(f"✅ Completed {combo_key}: {result['successful_prompts']}/{result['total_prompts']} successful")
            except Exception as e:
                logger.error(f"❌ Task {combo_key} failed with error: {e}")
                logger.error(traceback.format_exc())
                all_results[combo_key] = {
                    "error": str(e),
                    "successful_prompts": 0,
                    "failed_prompts": len(prompts)
                }
    
    return all_results


def main():
    """Main function to run attribution analysis on FACTS dataset."""
    logger.info("Starting FACTS dataset attribution analysis")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("output", "facts_attribution", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Results will be saved to {output_dir}")
    
    # Load dataset
    logger.info("Loading FACTS dataset")
    prompts = load_facts_dataset(limit=MAX_PROMPTS)
    
    if not prompts:
        logger.error("No prompts loaded from dataset, exiting")
        return
    
    logger.info(f"Processing {len(prompts)} prompts")
    
    # Run all permutations
    logger.info("Starting attribution analysis for all model/method combinations")
    results = run_all_permutations(prompts, output_dir)
    
    # Save summary report
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"FACTS Attribution Analysis - {timestamp}\n")
        f.write(f"Total prompts: {len(prompts)}\n\n")
        
        f.write("Results by model/method combination:\n")
        for combo_key, result in results.items():
            success_rate = result.get("successful_prompts", 0) / result.get("total_prompts", 1) * 100
            f.write(f"\n{combo_key}:\n")
            f.write(f"  Success rate: {success_rate:.1f}% ({result.get('successful_prompts', 0)}/{result.get('total_prompts', 0)})\n")
            
            # Write error examples if any
            errors = result.get("errors", [])
            if errors:
                f.write(f"  Errors: {len(errors)}\n")
                for i, error in enumerate(errors[:5]):  # Show the first 5 errors
                    f.write(f"    - {error.get('prompt_id', 'unknown')}: {error.get('error', 'unknown error')}\n")
                if len(errors) > 5:
                    f.write(f"    - ...and {len(errors) - 5} more errors\n")
    
    logger.info(f"Summary report saved to {summary_path}")
    logger.info("Attribution analysis completed")


if __name__ == "__main__":
    main()