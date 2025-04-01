#!/usr/bin/env python
"""
Attribution benchmark script for comparing different attribution methods.

This script runs benchmarks on multiple models and attribution methods,
measuring performance metrics and storing results.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any
import argparse

# Configure MPS for Apple Silicon if needed
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import inseq
from returns.result import Success, Failure
import numpy as np
from matplotlib import pyplot as plt
import torch

# Add project root to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import (
    ModelManager,
    InseqTokenAnalyzer,
    TokenAnalysisStorage
)

# Setup logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default test prompts
DEFAULT_PROMPTS = [
    "Explain the importance of attribution analysis in language models:",
    "Describe how token attribution works in transformer models:",
    "What makes a good visualization of model attributions?",
    "How do different attribution methods compare in terms of performance?",
    "Explain the relationship between attention and feature importance:"
]

class AttributionBenchmark:
    """Run attribution benchmarks on various models and methods"""

    def __init__(self, output_dir: str = "output/benchmark"):
        """Initialize the benchmark system"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Directory for data output
        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        # Directory for visualization output
        self.plots_dir = self.output_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Cache for loaded models
        self.model_cache = {}
        
        # Metrics storage
        self.metrics = {
            "attribution_times": {},
            "memory_usage": {},
            "success_rates": {}
        }

    def load_prompts(self, prompt_path: str = None) -> List[str]:
        """Load test prompts from file or use defaults"""
        if prompt_path and os.path.exists(prompt_path):
            try:
                with open(prompt_path, 'r') as f:
                    prompts = json.load(f)
                    logger.info(f"Loaded {len(prompts)} prompts from {prompt_path}")
                    return prompts
            except Exception as e:
                logger.warning(f"Error loading prompts from {prompt_path}: {e}")
                logger.warning("Using default prompts instead")
        
        logger.info(f"Using {len(DEFAULT_PROMPTS)} default prompts")
        return DEFAULT_PROMPTS

    def get_model_manager(self, model_name: str) -> ModelManager:
        """Get or create a model manager instance"""
        if model_name in self.model_cache:
            logger.info(f"Using cached model: {model_name}")
            return self.model_cache[model_name]
        
        logger.info(f"Loading model: {model_name}")
        model_data = {
            "llm_id": model_name,
            "device": "auto",
            "torch_dtype": "float16"
        }
        
        result = ModelManager.initialize(model_data)
        match result:
            case Success(manager):
                logger.info(f"Model {model_name} loaded successfully")
                self.model_cache[model_name] = manager
                return manager
            case Failure(error):
                logger.error(f"Failed to load model {model_name}: {error}")
                raise Exception(f"Model loading failed: {error}")

    def run_benchmark(
        self, 
        models: List[str], 
        methods: List[str], 
        prompts: List[str],
        storage_path: str = None
    ) -> Dict[str, Any]:
        """
        Run benchmark tests across models, methods, and prompts.
        
        Args:
            models: List of model names
            methods: List of attribution methods
            prompts: List of test prompts
            storage_path: Optional path for storage
            
        Returns:
            Dictionary of benchmark results
        """
        results = {}
        
        for model_name in models:
            logger.info(f"Benchmarking model: {model_name}")
            results[model_name] = {}
            
            try:
                # Get model manager (loaded or from cache)
                manager = self.get_model_manager(model_name)
                model_short_name = model_name.split("/")[-1]
                
                # Create model-specific storage
                model_path = storage_path or str(self.data_dir / model_short_name)
                os.makedirs(model_path, exist_ok=True)
                
                # Test each method
                for method in methods:
                    if method not in inseq.list_feature_attribution_methods():
                        logger.warning(f"Method {method} not available in Inseq, skipping")
                        continue
                        
                    logger.info(f"Testing method: {method}")
                    results[model_name][method] = {
                        "time_taken": [],
                        "success": [],
                        "matrix_shape": [],
                        "error_types": []
                    }
                    
                    # Create method-specific storage
                    method_path = os.path.join(model_path, f"method_{method}")
                    os.makedirs(method_path, exist_ok=True)
                    storage = TokenAnalysisStorage(base_path=method_path)
                    
                    # Init analyzer with this method
                    analyzer = InseqTokenAnalyzer(manager, attribution_method=method)
                    analyze = analyzer.create_analysis_pipeline(storage)
                    
                    # Test each prompt
                    for i, prompt in enumerate(prompts):
                        logger.info(f"Testing prompt {i+1}/{len(prompts)}: {prompt[:30]}...")
                        
                        # Measure timing
                        start_time = time.time()
                        
                        try:
                            # Log memory before
                            mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                            
                            # Run analysis
                            result = analyze(prompt)
                            
                            # Log memory after
                            mem_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
                            memory_used = mem_after - mem_before
                            
                            # Record time
                            elapsed = time.time() - start_time
                            results[model_name][method]["time_taken"].append(elapsed)
                            
                            # Process result
                            match result:
                                case Success(analysis_result):
                                    results[model_name][method]["success"].append(True)
                                    
                                    # Get matrix shape
                                    matrix = np.array(analysis_result.data.association_matrix)
                                    results[model_name][method]["matrix_shape"].append(matrix.shape)
                                    
                                    # Log memory usage
                                    results[model_name][method]["memory_used"] = memory_used
                                    
                                case Failure(error):
                                    results[model_name][method]["success"].append(False)
                                    results[model_name][method]["error_types"].append(type(error).__name__)
                                    logger.error(f"Analysis failed: {error}")
                        
                        except Exception as e:
                            elapsed = time.time() - start_time
                            results[model_name][method]["time_taken"].append(elapsed)
                            results[model_name][method]["success"].append(False)
                            results[model_name][method]["error_types"].append(type(e).__name__)
                            logger.error(f"Exception during analysis: {e}")
                    
                    # Calculate success rate
                    success_rate = sum(results[model_name][method]["success"]) / len(prompts)
                    results[model_name][method]["success_rate"] = success_rate
                    logger.info(f"Method {method} success rate: {success_rate:.1%}")
                    
                    # Calculate average time
                    avg_time = sum(results[model_name][method]["time_taken"]) / len(results[model_name][method]["time_taken"])
                    results[model_name][method]["avg_time"] = avg_time
                    logger.info(f"Method {method} average time: {avg_time:.2f}s")
            
            except Exception as e:
                logger.error(f"Error benchmarking model {model_name}: {e}")
                results[model_name]["error"] = str(e)
        
        # Save results
        self.save_results(results)
        
        # Generate plots
        self.generate_plots(results)
        
        return results

    def save_results(self, results: Dict[str, Any]) -> None:
        """Save benchmark results to JSON"""
        result_path = self.output_dir / "benchmark_results.json"
        try:
            # Convert non-serializable objects like numpy shapes
            serializable_results = self._make_serializable(results)
            
            with open(result_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
                
            logger.info(f"Saved benchmark results to {result_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def _make_serializable(self, obj):
        """Convert non-serializable objects to serializable formats"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item) for item in obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'shape'):  # Handle numpy shapes
            return str(obj)
        else:
            return obj

    def generate_plots(self, results: Dict[str, Any]) -> None:
        """Generate comparative plots of benchmark results"""
        try:
            # 1. Attribution time comparison
            self._plot_attribution_times(results)
            
            # 2. Success rate comparison
            self._plot_success_rates(results)
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")

    def _plot_attribution_times(self, results: Dict[str, Any]) -> None:
        """Plot attribution times by model and method"""
        plt.figure(figsize=(12, 8))
        
        models = list(results.keys())
        methods = set()
        for model in models:
            for method in results[model]:
                if isinstance(results[model][method], dict) and "avg_time" in results[model][method]:
                    methods.add(method)
        
        methods = sorted(methods)
        x = np.arange(len(models))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            times = []
            for model in models:
                if method in results[model] and "avg_time" in results[model][method]:
                    times.append(results[model][method]["avg_time"])
                else:
                    times.append(0)
            
            plt.bar(x + i * width - 0.4 + width/2, times, width, label=method)
        
        plt.xlabel('Models')
        plt.ylabel('Attribution Time (seconds)')
        plt.title('Attribution Time by Model and Method')
        plt.xticks(x, [m.split('/')[-1] for m in models], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / "attribution_times.png"
        plt.savefig(plot_path)
        logger.info(f"Saved attribution time plot to {plot_path}")
        plt.close()

    def _plot_success_rates(self, results: Dict[str, Any]) -> None:
        """Plot success rates by model and method"""
        plt.figure(figsize=(12, 8))
        
        models = list(results.keys())
        methods = set()
        for model in models:
            for method in results[model]:
                if isinstance(results[model][method], dict) and "success_rate" in results[model][method]:
                    methods.add(method)
        
        methods = sorted(methods)
        x = np.arange(len(models))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            rates = []
            for model in models:
                if method in results[model] and "success_rate" in results[model][method]:
                    rates.append(results[model][method]["success_rate"] * 100)
                else:
                    rates.append(0)
            
            plt.bar(x + i * width - 0.4 + width/2, rates, width, label=method)
        
        plt.xlabel('Models')
        plt.ylabel('Success Rate (%)')
        plt.title('Attribution Success Rate by Model and Method')
        plt.xticks(x, [m.split('/')[-1] for m in models], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        
        # Save plot
        plot_path = self.plots_dir / "success_rates.png"
        plt.savefig(plot_path)
        logger.info(f"Saved success rate plot to {plot_path}")
        plt.close()


def main():
    """Main function to run attribution benchmarks"""
    parser = argparse.ArgumentParser(description="Run attribution method benchmarks")
    parser.add_argument("--models", type=str, default=os.environ.get("MODELS", "gpt2"),
                      help="Comma-separated list of models to test")
    parser.add_argument("--methods", type=str, default=os.environ.get("ATTRIBUTION_METHODS", "saliency,attention"),
                      help="Comma-separated list of attribution methods to test")
    parser.add_argument("--prompts", type=str, default=os.environ.get("TEST_PROMPTS", ""),
                      help="Path to JSON file containing test prompts")
    parser.add_argument("--output", type=str, default="output/benchmark",
                      help="Output directory for benchmark results")
    
    args = parser.parse_args()
    
    # Parse arguments
    models = [m.strip() for m in args.models.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]
    output_dir = args.output
    
    # Initialize benchmark
    benchmark = AttributionBenchmark(output_dir=output_dir)
    
    # Load prompts
    prompts = benchmark.load_prompts(args.prompts if args.prompts else None)
    
    # Save prompts for reference
    with open(os.path.join(output_dir, "prompts.json"), "w") as f:
        json.dump(prompts, f, indent=2)
    
    # Print configuration
    logger.info("=== Attribution Benchmark Configuration ===")
    logger.info(f"Models: {models}")
    logger.info(f"Methods: {methods}")
    logger.info(f"Number of prompts: {len(prompts)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("========================================")
    
    # Run benchmark
    results = benchmark.run_benchmark(models, methods, prompts)
    
    # Print summary
    logger.info("\n===== BENCHMARK RESULTS SUMMARY =====")
    for model_name in results:
        logger.info(f"\nModel: {model_name}")
        for method, data in results[model_name].items():
            if isinstance(data, dict) and "success_rate" in data:
                logger.info(f"  - {method}: {data['success_rate']:.1%} success, {data['avg_time']:.2f}s avg time")
    
    logger.info("\nBenchmark completed!")


if __name__ == "__main__":
    main()