import os
import torch
import psutil
from returns.result import Success, Failure
from work.llama_token_analyzer import ModelManager, TokenAnalyzer, TokenAnalysisStorage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set environment variables before any other imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent tokenizer parallelism warning
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # MPS memory management


def get_optimal_device_map():
    """
    Determines the optimal device mapping based on available hardware.
    Handles MPS (Mac), single GPU, multi-GPU, and CPU configurations.
    """
    # Check for MPS (Mac) first
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logging.info("MPS (Mac) device detected")
        # For MPS, we don't want to return "mps" directly, as we need more control
        return {"device": "mps", "is_mps": True}

    # Check CUDA availability
    if not torch.cuda.is_available():
        logging.info("No GPU detected, using CPU")
        return {"device": "cpu", "is_mps": False}

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        logging.info("No CUDA devices available, using CPU")
        return {"device": "cpu", "is_mps": False}
    elif n_gpus == 1:
        logging.info("Single GPU detected")
        return {"device": "cuda", "is_mps": False}
    else:
        logging.info(f"Multiple GPUs detected: {n_gpus}")
        # Create device map for multiple GPUs
        device_map = {}

        # Add CPU memory
        cpu_memory = psutil.virtual_memory()
        cpu_memory_gib = int(cpu_memory.available * 0.7 / (1024 ** 3))
        device_map["cpu"] = f"{cpu_memory_gib}GiB"

        # Add GPU memory
        for i in range(n_gpus):
            try:
                props = torch.cuda.get_device_properties(i)
                usable_memory = int(props.total_memory * 0.8 / (1024 ** 3))
                device_map[str(i)] = f"{usable_memory}GiB"
                logging.info(f"GPU {i}: {props.name}, Total Memory: {props.total_memory / (1024 ** 3):.1f} GiB, "
                             f"Allocated: {usable_memory} GiB")
            except Exception as e:
                logging.warning(f"Error getting properties for GPU {i}: {e}")
                continue

        return {"device_map": device_map, "is_mps": False}


def initialize_model():
    # Get optimal device mapping
    device_config = get_optimal_device_map()
    logging.info(f"Using device configuration: {device_config}")

    # Basic model configuration
    model_config = {
        "llm_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "torch_dtype": "float16",
        "trust_remote_code": True,
        "load_in_8bit": False
    }

    # Configure device settings based on environment
    if device_config.get("is_mps"):
        model_config.update({
            "device": "mps",
            "low_cpu_mem_usage": True,  # Important for MPS
        })
    elif "device_map" in device_config:
        model_config.update({
            "device": "auto",
            "device_map": device_config["device_map"]
        })
    else:
        model_config.update({
            "device": device_config["device"]
        })

    # Initialize storage
    storage = TokenAnalysisStorage(base_path="output")

    # Try to initialize the model
    try:
        model_result = ModelManager.initialize(model_config)

        match model_result:
            case Success(manager):
                analyzer = TokenAnalyzer(manager)
                return Success((analyzer, storage))
            case Failure(error):
                return Failure(f"Failed to initialize model: {error}")

    except Exception as e:
        return Failure(f"Unexpected error during initialization: {e}")


print("LLaMA Token Analyzer - Robust Device Mapping Example\n")
# Initialize components
init_result = initialize_model()

match init_result:
    case Success((analyzer, storage)):
        # Test with a simple prompt
        prompt = "Answer with ok."
        analyze = analyzer.create_analysis_pipeline(storage)

        try:
            analysis_result = analyze(prompt)
            match analysis_result:
                case Success(r):
                    print("Analysis successful!")
                    from work.llama_token_analyzer.visualization.main import visualize

                    visualize(r, storage=storage)
                case Failure(error):
                    print(f"Analysis failed: {error}")
        except Exception as e:
            print(f"Error during analysis: {e}")

    case Failure(error):
        print(f"Initialization failed: {error}")
