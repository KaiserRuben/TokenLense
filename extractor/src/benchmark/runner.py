"""
Core execution logic for benchmark experiments.

This module provides functions to execute attribution benchmarks across
different models, methods, and prompts.
"""

import logging
import time
import traceback
from typing import Dict, List, Tuple, Any, Callable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import inseq
import torch
from returns.result import Result, Success, Failure

from src import ModelManager, InseqTokenAnalyzer
from src.benchmark.schema import (
    SystemInfo, TimingData, PromptTimingData, PromptResult,
    AnalysisError, AnalysisResultData
)
from src.benchmark.system import collect_system_info
from src.benchmark.reporting import create_output_structure

logger = logging.getLogger(__name__)


def load_model(model_config: Dict[str, Any]) -> Result[Tuple[ModelManager, float], Exception]:
    """
    Load and initialize model with timing measurement.

    Args:
        model_config: Model configuration dictionary

    Returns:
        Result containing either (model_manager, loading_time) or an exception
    """
    model_name = model_config["name"]
    logger.info(f"Loading model: {model_name}")

    # Start timing for model loading
    start_time = time.time()

    # Initialize model
    model_data = {
        "llm_id": model_config["llm_id"],
        "device": model_config["device"],
        "torch_dtype": model_config["torch_dtype"]
    }

    model_result = ModelManager.initialize(model_data)

    # Calculate loading time
    loading_time = time.time() - start_time

    # Add timing information to result
    match model_result:
        case Success(manager):
            logger.info(f"Model {model_name} loaded successfully in {loading_time:.2f} seconds")
            return Success((manager, loading_time))
        case Failure(error):
            logger.error(f"❌ Failed to load model {model_name}: {error}")
            return Failure(error)


def initialize_analyzer(
        manager: ModelManager,
        method: str
) -> Result[Tuple[InseqTokenAnalyzer, float], Exception]:
    """
    Initialize attribution analyzer with timing measurement.

    Args:
        manager: Initialized model manager
        method: Attribution method to use

    Returns:
        Result containing either (analyzer, init_time) or an exception
    """
    try:
        # Start timing analyzer initialization
        start_time = time.time()

        # Initialize analyzer
        analyzer = InseqTokenAnalyzer(manager, attribution_method=method)

        # Calculate initialization time
        init_time = time.time() - start_time
        logger.info(f"Analyzer initialized with method '{method}' in {init_time:.2f} seconds")

        return Success((analyzer, init_time))
    except Exception as e:
        logger.error(f"❌ Failed to initialize analyzer with method {method}: {e}")
        logger.error(traceback.format_exc())
        return Failure(e)


def process_prompt(
        analyze_fn: Callable,
        model_manager: ModelManager,
        prompt: str,
        prompt_idx: int,
        total_prompts: int
) -> PromptResult:
    """
    Process a single prompt with attribution timing measurement.

    Args:
        analyze_fn: Analysis pipeline function
        model_manager: ModelManager instance (used only for error case token counting)
        prompt: Prompt text to analyze
        prompt_idx: Index of the prompt (0-based)
        total_prompts: Total number of prompts

    Returns:
        PromptResult with success status and attribution timing
    """
    prompt_id = f"prompt_{prompt_idx + 1}"
    logger.info(f"Processing prompt {prompt_idx + 1}/{total_prompts}")

    # Initialize timing data
    timing = PromptTimingData()

    # Initialize token counts
    token_count = 0
    output_token_count = 0

    try:
        # Start timing the attribution process
        start_time = time.time()

        # Run attribution analysis - this is the operation we want to measure
        result = analyze_fn(prompt)

        # Record attribution time
        timing.attribution_time = time.time() - start_time

        logger.info(f"Prompt {prompt_idx + 1} processed in {timing.attribution_time:.2f} seconds")

        # Process result based on success/failure
        match result:
            case Success(analysis_result):
                # Success already logged in the outer logging

                # Extract token counts from the analysis result
                try:
                    token_count = len(analysis_result.data.input_tokens)
                    output_token_count = len(analysis_result.data.output_tokens)
                except (AttributeError, TypeError):
                    # If we can't get token counts from the result, estimate from the tokenizer
                    # This is a fallback and should rarely be needed if attribution is successful
                    token_count = len(model_manager.tokenizer(prompt, return_tensors="pt")["input_ids"][0])
                    output_token_count = 0

                return PromptResult(
                    prompt_id=prompt_id,
                    prompt_text=prompt,
                    success=True,
                    timing=timing,
                    token_count=token_count,
                    output_token_count=output_token_count
                )
            case Failure(error):
                logger.error(f"❌ Failed to process prompt {prompt_idx + 1}: {error}")

                # In case of failure, get token count for analysis
                try:
                    token_count = len(model_manager.tokenizer(prompt, return_tensors="pt")["input_ids"][0])
                except Exception:
                    token_count = 0

                return PromptResult(
                    prompt_id=prompt_id,
                    prompt_text=prompt,
                    success=False,
                    error_message=str(error),
                    timing=timing,
                    token_count=token_count
                )
    except Exception as e:
        # Handle unexpected exceptions
        attribution_time = time.time() - start_time if 'start_time' in locals() else 0
        timing.attribution_time = attribution_time

        logger.error(f"❌ Exception processing prompt {prompt_idx + 1}: {e}")
        logger.error(traceback.format_exc())

        # Try to get token count for the error case
        try:
            token_count = len(model_manager.tokenizer(prompt, return_tensors="pt")["input_ids"][0])
        except Exception:
            token_count = 0

        return PromptResult(
            prompt_id=prompt_id,
            prompt_text=prompt,
            success=False,
            error_message=str(e),
            timing=timing,
            token_count=token_count
        )


def run_attribution_analysis(
        model_config: Dict[str, Any],
        method: str,
        prompts: List[str],
        output_base_path: str
) -> AnalysisResultData:
    """
    Run attribution analysis for a specific model, method, and list of prompts.

    Args:
        model_config: Model configuration
        method: Attribution method to use
        prompts: List of prompts to analyze
        output_base_path: Base output directory

    Returns:
        AnalysisResultData with complete results including execution time
    """
    model_name = model_config["name"]
    logger.info(f"Testing model: {model_name} with method: {method}")

    # Initialize result object
    result = AnalysisResultData(
        model_name=model_name,
        attribution_method=method,
        total_prompts=len(prompts),
        system_info=collect_system_info()
    )

    # Start timing for total execution
    start_time_total = time.time()

    # Create output structure
    output_path, storage = create_output_structure(model_name, method, output_base_path)

    # Load model
    model_load_result = load_model(model_config)

    match model_load_result:
        case Success((manager, model_loading_time)):
            # Record model loading time
            result.timing.model_loading_time = model_loading_time

            # Initialize analyzer
            analyzer_result = initialize_analyzer(manager, method)

            match analyzer_result:
                case Success((analyzer, _)):
                    # Create analysis pipeline
                    analyze = analyzer.create_analysis_pipeline(storage)

                    # Start timing attribution process
                    attribution_start_time = time.time()

                    # Process each prompt
                    for i, prompt in enumerate(tqdm(prompts, desc=f"{model_name}/{method} processing")):
                        prompt_result = process_prompt(analyze, manager, prompt, i, len(prompts))
                        result.prompt_results.append(prompt_result)

                        # Update success/failure counts
                        if prompt_result.success:
                            result.successful_prompts += 1
                        else:
                            result.failed_prompts += 1
                            # Add to errors list
                            result.errors.append(AnalysisError(
                                stage="prompt_processing",
                                error=prompt_result.error_message or "Unknown error",
                                prompt_id=prompt_result.prompt_id
                            ))

                    # Calculate attribution time
                    attribution_time = time.time() - attribution_start_time
                    result.timing.attribution_time = attribution_time

                    # Calculate average attribution time and token throughput
                    if result.prompt_results:
                        # Calculate average attribution time
                        avg_attribution_time = sum(p.attribution_time for p in result.prompt_results) / len(
                            result.prompt_results)

                        # Set in result
                        result.timing.average_prompt_time = avg_attribution_time

                        # Log timing information
                        logger.info(f"Average attribution time: {avg_attribution_time:.2f} seconds")

                        # Calculate and log token processing throughput
                        total_tokens = sum(p.token_count for p in result.prompt_results)
                        successful_prompts = sum(1 for p in result.prompt_results if p.success)

                        if total_tokens > 0 and attribution_time > 0 and successful_prompts > 0:
                            tokens_per_second = total_tokens / attribution_time
                            avg_tokens_per_prompt = total_tokens / successful_prompts
                            logger.info(f"Token processing throughput: {tokens_per_second:.2f} tokens/second")
                            logger.info(f"Average tokens per prompt: {avg_tokens_per_prompt:.1f} tokens")

                case Failure(error):
                    # Handle analyzer initialization failure (error already logged in initialize_analyzer)
                    result.failed_prompts = len(prompts)
                    result.errors.append(AnalysisError(
                        stage="analyzer_initialization",
                        error=str(error)
                    ))

        case Failure(error):
            # Handle model loading failure (error already logged in load_model)
            result.failed_prompts = len(prompts)
            result.errors.append(AnalysisError(
                stage="model_loading",
                error=str(error)
            ))

    # Calculate total execution time
    total_time = time.time() - start_time_total
    result.timing.total_time = total_time
    logger.info(f"Total execution time for {model_name}/{method}: {total_time:.2f} seconds")

    return result


def filter_available_methods(methods_to_check: List[str]) -> List[str]:
    """
    Filter attribution methods to only those available in Inseq.

    Args:
        methods_to_check: List of methods to check for availability

    Returns:
        List of available methods
    """
    # Get all available methods from Inseq
    available_methods = inseq.list_feature_attribution_methods()
    logger.info(f"Available attribution methods: {available_methods}")

    # Filter to only those in our list that are available
    filtered_methods = [m for m in methods_to_check if m in available_methods]
    logger.info(f"Using attribution methods: {filtered_methods}")

    return filtered_methods


def create_task_list(
        models: List[Dict[str, Any]],
        methods: List[str],
        prompts: List[str],
        output_dir: str
) -> List[Tuple[str, Callable[[], AnalysisResultData]]]:
    """
    Create a list of tasks for parallel execution.

    Args:
        models: List of model configurations
        methods: List of attribution methods
        prompts: List of prompts to process
        output_dir: Base output directory

    Returns:
        List of (task_id, task_fn) tuples
    """
    tasks = []

    for model_config in models:
        model_name = model_config["name"]

        for method in methods:
            # Create a unique key for this combination
            combo_key = f"{model_name}/{method}"
            logger.info(f"Creating task for {combo_key}")

            # Create a task function that captures the arguments
            task_fn = lambda mc=model_config, m=method: run_attribution_analysis(
                mc, m, prompts, output_dir
            )

            tasks.append((combo_key, task_fn))

    return tasks


def run_tasks_in_parallel(
        tasks: List[Tuple[str, Callable]],
        max_workers: int = 1
) -> Dict[str, AnalysisResultData]:
    """
    Run tasks in parallel using a thread pool.

    Args:
        tasks: List of (task_id, task_fn) tuples
        max_workers: Maximum number of parallel workers

    Returns:
        Dictionary mapping task_id to result
    """
    all_results = {}
    futures = []

    # Create a worker pool
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the pool
        for task_id, task_fn in tasks:
            logger.info(f"Scheduling task for {task_id}")
            future = executor.submit(task_fn)
            futures.append((task_id, future))

        # Process results as they complete
        for task_id, future in tqdm(futures, desc="Processing model/method combinations"):
            try:
                result = future.result()
                all_results[task_id] = result
                logger.info(f"✅ Completed {task_id}: {result.successful_prompts}/{result.total_prompts} successful")
            except Exception as e:
                logger.error(f"❌ Task {task_id} failed with error: {e}")
                logger.error(traceback.format_exc())
                # Create a failure result
                error_result = AnalysisResultData(
                    model_name=task_id.split('/')[0],
                    attribution_method=task_id.split('/')[1],
                    total_prompts=0,  # We don't know how many prompts at this level
                    successful_prompts=0,
                    failed_prompts=0,
                    system_info=collect_system_info()
                )
                error_result.errors.append(AnalysisError(
                    stage="task_execution",
                    error=str(e)
                ))
                all_results[task_id] = error_result

    return all_results


def run_all_permutations(
        prompts: List[str],
        output_dir: str,
        models: Optional[List[Dict[str, Any]]] = None,
        methods: Optional[List[str]] = None,
        max_workers: int = 1
) -> Dict[str, AnalysisResultData]:
    """
    Run all permutations of models, methods, and prompts in parallel.

    Args:
        prompts: List of prompts to process
        output_dir: Base output directory
        models: List of model configurations (if None, uses config.MODELS)
        methods: List of attribution methods (if None, uses config.ATTRIBUTION_METHODS)
        max_workers: Maximum number of parallel workers

    Returns:
        Dictionary with results for all permutations
    """
    # Import here to avoid circular imports
    if models is None or methods is None:
        from src.benchmark.config import MODELS, ATTRIBUTION_METHODS
        if models is None:
            models = MODELS
        if methods is None:
            methods = ATTRIBUTION_METHODS

    # Filter attribution methods to only those available in Inseq
    filtered_methods = filter_available_methods(methods)

    if not filtered_methods:
        logger.error("No valid attribution methods available")
        return {}

    # Create task list
    tasks = create_task_list(models, filtered_methods, prompts, output_dir)

    if not tasks:
        logger.warning("No tasks created")
        return {}

    # Run tasks in parallel
    return run_tasks_in_parallel(tasks, max_workers=max_workers)