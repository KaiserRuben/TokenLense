# Running TokenLense with Custom Datasets

This guide explains how to run the [TokenLense Extractor](../ReadMe.md) with your own custom datasets to generate token attribution data for visualization and analysis.

🔗 **Related Documentation**: [Main Project](../../README.md) | [Extractor](../ReadMe.md) | [Analyzer API](../../analyzer/README.md) | [Visualizer](../../visualizer/README.md)

## Overview

The extractor component in TokenLense performs attribution analysis on language models using different attribution methods. The framework is designed to be flexible, allowing you to use pre-configured datasets or your own custom data.

## Custom Dataset Options

There are three main approaches to running attribution analysis with your own data:

1. **Create a custom script** - Create a new script based on the existing examples
2. **Modify the config.py** - Change the dataset configuration in the existing scripts
3. **Create a CSV dataset** - Prepare a CSV file that follows the expected format

## Method 1: Creating a Custom Script

This approach gives you the most flexibility and is recommended for complex datasets.

1. Create a new script in the `extractor/scripts/` directory, using the existing scripts as templates:

```python
#!/usr/bin/env python
"""
Script to run token attribution analysis on your custom dataset.
"""

import os
import sys
import logging
from pathlib import Path

# Configure MPS for Apple Silicon if needed
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import nltk
from returns.result import Success, Failure, Result

# Initialize NLTK dependencies
nltk.download('averaged_perceptron_tagger_eng')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("custom_attribution.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import benchmark components
from src.benchmark.config import MODELS, ATTRIBUTION_METHODS
from src.benchmark.system import collect_system_info, format_system_info
from src.benchmark.runner import run_all_permutations
from src.benchmark.reporting import (
    create_output_directory,
    create_method_timing_dataframe,
    create_per_prompt_dataframe,
    save_timing_results,
    generate_summary_report
)

# Maximum number of examples to process
MAX_EXAMPLES = 50


def load_custom_dataset() -> Result[list, Exception]:
    """
    Load your custom dataset.
    
    Returns:
        Result containing a list of prompts or an error
    """
    try:
        # Replace this with your own data loading logic
        # Example - loading from a text file:
        prompts = []
        with open("path/to/your/dataset.txt", "r") as f:
            for line in f:
                prompts.append(line.strip())
                
        # Example - creating prompts programmatically:
        # questions = ["What is machine learning?", "Explain quantum computing"]
        # prompts = [f"Answer this question: {q}" for q in questions]
        
        # Limit the number of prompts if needed
        prompts = prompts[:MAX_EXAMPLES]
        
        if not prompts:
            return Failure(Exception("No prompts found in dataset"))
            
        logger.info(f"Loaded {len(prompts)} prompts")
        return Success(prompts)
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return Failure(e)


def main():
    """Main function to run attribution analysis on custom dataset."""
    logger.info("Starting custom dataset attribution analysis")

    # Collect and log system information
    logger.info("Collecting system information...")
    system_info = collect_system_info()
    logger.info(format_system_info(system_info))

    # Create output directory
    output_dir, timestamp = create_output_directory(name="custom_attribution")

    # Load dataset
    logger.info("Loading custom dataset")
    dataset_result = load_custom_dataset()

    match dataset_result:
        case Success(prompts):
            if not prompts:
                logger.error("Dataset loaded but contains no prompts, exiting")
                return

            logger.info(f"Processing {len(prompts)} prompts")

            # Run analysis for all permutations
            logger.info("Starting attribution analysis for all model/method combinations")
            results = run_all_permutations(
                prompts=prompts,
                output_dir=output_dir,
                models=MODELS,
                methods=ATTRIBUTION_METHODS,
                max_workers=1  # Adjust based on your hardware
            )

            if not results:
                logger.error("No results generated, exiting")
                return

            # Create timing dataframes
            method_timing_df = create_method_timing_dataframe(results)
            prompt_timing_df = create_per_prompt_dataframe(results)

            # Save timing results
            save_timing_results(method_timing_df, prompt_timing_df, output_dir)

            # Generate summary report
            generate_summary_report(results, prompts, timestamp, output_dir)

            logger.info("Attribution analysis completed successfully")

        case Failure(error):
            logger.error(f"Failed to load dataset: {error}")
            logger.error("Attribution analysis aborted")


if __name__ == "__main__":
    main()
```

2. Make your script executable:
   ```bash
   chmod +x scripts/run_custom_attribution.py
   ```

3. Run your script:
   ```bash
   poetry run python scripts/run_custom_attribution.py
   ```

## Method 2: Modifying the Configuration

This approach is simpler but less flexible. It's suitable for basic dataset changes.

1. Edit the `src/benchmark/config.py` file to use your own dataset:

```python
# Dataset configuration
DATASET_CONFIG = {
    'primary_source': "path/to/your/dataset.csv",  # Local CSV file
    'fallback_path': "data/fallback.csv",          # Fallback path
}
```

2. If your dataset is in CSV format, make sure it has one of the following column structures:
   - A `full_prompt` column containing complete prompts
   - Both `system_instruction` and `user_request` columns
   - At least one text column that can be used for prompts

3. Run one of the existing scripts:
   ```bash
   poetry run python scripts/run_facts_attribution.py
   ```

## Method 3: Creating a CSV Dataset

This approach is the simplest if you already have your data in a structured format.

1. Create a CSV file with your prompts. The file should have one of these structures:

   **Option 1 - Single column:**
   ```csv
   full_prompt
   "Write a story about a robot learning to feel emotions."
   "Explain how photosynthesis works in simple terms."
   ```

   **Option 2 - System and user instructions:**
   ```csv
   system_instruction,user_request
   "You are a helpful assistant.","Write a story about a robot."
   "You are a science teacher.","Explain photosynthesis."
   ```

2. Place your CSV file in an accessible location, such as `extractor/data/custom_dataset.csv`

3. Edit the `src/benchmark/config.py` file to use your dataset:
   ```python
   DATASET_CONFIG = {
       'primary_source': "data/custom_dataset.csv",
       'fallback_path': "data/facts_examples.csv",
   }
   ```

4. Run one of the existing scripts:
   ```bash
   poetry run python scripts/run_facts_attribution.py
   ```

## Customizing Models and Attribution Methods

You can also customize which models and attribution methods are used in your analysis:

1. Edit the `src/benchmark/config.py` file to modify the models:

```python
# Use a subset of models to speed up processing
MY_MODELS = [
    {
        "name": "GPT-2",
        "llm_id": "gpt2",
        "device": "auto",
        "torch_dtype": "float16",
        "type": "causal"
    },
    # Add more models as needed
]

# Use specific attribution methods
MY_METHODS = [
    'input_x_gradient',
    'saliency',
    # Add more methods as needed
]
```

2. In your custom script, use your subsets:

```python
from src.benchmark.config import MY_MODELS, MY_METHODS

# ...

results = run_all_permutations(
    prompts=prompts,
    output_dir=output_dir,
    models=MY_MODELS,      # Use your custom models
    methods=MY_METHODS,    # Use your custom methods
    max_workers=1
)
```

## Accessing the Results

The attribution analysis results will be saved in the `extractor/scripts/output/` directory. The specific output directory will include a timestamp and the name you specified in `create_output_directory()`.

The main outputs include:

1. **Attribution Data Files**: JSON files containing token attribution data
2. **Timing Results**: CSV files with performance metrics
3. **Summary Report**: Text file with analysis overview

To view the attribution results in the TokenLense visualizer, you need to copy the generated data to the analyzer component:

```bash
# Create the necessary directories
mkdir -p analyzer/data/MODEL_NAME/method_ATTRIBUTION_METHOD/data

# Copy the attribution data
cp extractor/scripts/output/custom_attribution/data/*.json analyzer/data/MODEL_NAME/method_ATTRIBUTION_METHOD/data/
```

Replace `MODEL_NAME` with the name of the model (e.g., "GPT-2") and `ATTRIBUTION_METHOD` with the attribution method (e.g., "saliency").

## Tips for Efficient Processing

1. **Start Small**: Begin with a small number of prompts (5-10) to verify your setup
2. **Use Smaller Models**: Start with smaller models like GPT-2 before scaling to larger ones
3. **Limit Methods**: Begin with just one or two attribution methods like 'saliency' and 'attention'
4. **Hardware Requirements**: 
   - Attribution analysis is compute-intensive
   - GPU support significantly speeds up processing
   - For large models (>7B parameters), 16GB+ GPU memory is recommended

## Troubleshooting

### Out of Memory Errors
- Reduce the number of prompts
- Use smaller models
- Ensure only one attribution analysis runs at a time
- Increase swap space on your system

### Installation Issues
- Make sure you have the correct PyTorch version for your CUDA setup
- Install Inseq dependencies separately if needed: `pip install "inseq[all]"`
- On macOS, ensure MPS is configured correctly