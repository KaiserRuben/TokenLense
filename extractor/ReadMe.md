# TokenLense Extractor

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/Poetry-Package_Manager-blue)](https://python-poetry.org/)
[![Inseq](https://img.shields.io/badge/Inseq-0.6.0-blue)](https://inseq.org/)

The Extractor component of the [TokenLense](../README.md) framework for analyzing, visualizing, and interpreting token relationships in Large Language Models. This tool generates the attribution data for visualization.

🔗 **Project**: [Main Documentation](../README.md) | [Analyzer API](../analyzer/Documentation/get_attribution_endpoint.md) | [Visualizer](../visualizer/README.md) | [Custom Datasets](docs/custom_dataset_guide.md)

## Features

- **Token Attribution Analysis**: Track how input tokens influence output tokens using gradient-based methods
- **Multiple Attribution Methods**: Support for both custom gradient tracking and Inseq-powered attribution methods
- **Interactive Visualization**: Explore token relationships through the TokenLens UI
- **Flexible Model Support**: Compatible with Hugging Face models including Llama, GPT-2, and more
- **Batch Processing**: Analyze multiple prompts efficiently
- **Data Persistence**: Store analysis results in structured JSON format
- **Comprehensive Benchmarking**: Measure performance across different models and attribution methods

## Installation

### Using Poetry (Recommended)
```bash
poetry install
```

### Using pip
```bash
pip install -r requirements.txt
```

## Authentication

The framework requires a Hugging Face token for model access:
```bash
export HUGGINGFACE_TOKEN="your_token_here"
# OR
huggingface-cli login
```

## Quick Start

### Basic Usage with Custom Attribution

```python
from src import ModelManager, TokenAnalyzer, TokenAnalysisStorage
from src import visualize

# Initialize model
model_config = {
    "llm_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "device": "auto",
    "torch_dtype": "float16"
}
manager = ModelManager.initialize(model_config)

# Setup analysis pipeline
storage = TokenAnalysisStorage(base_path="output")
analyzer = TokenAnalyzer(manager)
analyze = analyzer.create_analysis_pipeline(storage)

# Analyze a prompt
result = analyze("Your prompt here")

# Visualize the result
visualize(result, storage=storage)
```

### Using Inseq Attribution Methods

```python
from src import ModelManager, InseqTokenAnalyzer, TokenAnalysisStorage
from src import visualize

# Initialize model
model_config = {
    "llm_id": "gpt2",
    "device": "auto",
    "torch_dtype": "float16"
}
manager = ModelManager.initialize(model_config)

# Setup Inseq analysis pipeline with chosen attribution method
storage = TokenAnalysisStorage(base_path="output")
analyzer = InseqTokenAnalyzer(manager, attribution_method="saliency")
analyze = analyzer.create_analysis_pipeline(storage)

# Analyze a prompt
result = analyze("Your prompt here")

# Visualize the result
visualize(result, storage=storage)
```

## Benchmarking Attribution Methods

The framework includes a comprehensive benchmarking module to evaluate the performance and accuracy of different attribution methods across various models.

### Running a Benchmark

```bash
# Run the benchmark script with default settings
python scripts/run_facts_attribution.py
```

### Configuring Benchmarks

The benchmark configuration can be customized in `src/benchmark/config.py`:

```python
# Edit benchmark configuration
from src.benchmark.config import MODELS, ATTRIBUTION_METHODS, MAX_PROMPTS

# Add your model to the benchmark
MODELS.append({
    "name": "Your-Model-Name",
    "llm_id": "your-model-id",
    "device": "auto",
    "torch_dtype": "float16",
    "type": "causal"
})

# Choose specific attribution methods
MY_METHODS = ['saliency', 'integrated_gradients', 'attention']

# Run with custom configuration
from src.benchmark.runner import run_all_permutations
from src.benchmark.dataset import load_dataset
from src.benchmark.reporting import create_output_directory

output_dir, _ = create_output_directory()
dataset_result = load_dataset(limit=10)
if dataset_result.is_success():
    prompts = dataset_result.unwrap()
    results = run_all_permutations(
        prompts=prompts,
        output_dir=output_dir,
        models=MY_MODELS,
        methods=MY_METHODS
    )
```

### Benchmark Components

The benchmarking system includes:

- **Dataset Module**: Load and preprocess prompts for evaluation
- **Runner Module**: Execute attribution methods with detailed timing
- **System Module**: Collect hardware and environment information
- **Reporting Module**: Generate visualizations and summary metrics
- **Schema Module**: Define structured data models for benchmark results

### Benchmark Output

Benchmarks generate multiple outputs:

- **Summary Report**: Human-readable overview of results
- **CSV Files**: Detailed timing metrics for analysis
- **System Information**: Hardware configuration details
- **Per-prompt Metrics**: Token processing rates and success rates
- **Error Logs**: Detailed diagnostics for failed attributions

Example summary output:
```
FACTS Attribution Analysis - 20250403_123045
Total prompts: 20

System Information:
  Platform: Linux (5.15.0-1041-azure)
  CPU: AMD EPYC 7763 64-Core Processor
  CPU Cores: 16
  Memory: 64.00 GB
  GPU: NVIDIA A100-SXM4-40GB (CUDA 11.8)
  PyTorch: 2.0.1+cu118
  Available Devices: CUDA

Results by model/method combination:

GPT-2/saliency:
  Success rate: 100.0% (20/20)
  Timing:
    Model loading time: 2.34 seconds
    Attribution time: 15.67 seconds
    Average time per prompt: 0.78 seconds
    Total execution time: 18.01 seconds
```

## Available Attribution Methods

### Custom Gradient-Based Attribution
- Implemented in `TokenAnalyzer` class
- Directly calculates token associations using gradient tracking
- Best for detailed exploration of direct token relationships

### Inseq-Powered Attribution Methods
- Implemented in `InseqTokenAnalyzer` class
- Available methods include:
  - `saliency`: Simple gradient-based saliency maps
  - `attention`: Model's internal attention weights
  - `integrated_gradients`: Path integral of gradients
  - `input_x_gradient`: Input multiplied by gradients
  - `layer_deeplift`: Layer-wise DeepLIFT attribution
  - Many more from the Inseq library

## Visualization Options

### 1. Notebook Visualization
For quick analysis in Jupyter notebooks:
```python
visualize(result, storage=storage)
```

### 2. TokenLens UI (Recommended)
For in-depth, interactive exploration:

1. Generate JSON files using the analyzer
2. Copy the generated files from your output directory to `ui/src/data/`
3. Start the UI: `cd ui && bun run dev`
4. Open http://localhost:5173 in your browser

The UI provides advanced features:
- Interactive token exploration with detailed metrics
- Heat map visualization of token relationships
- Association strength filtering
- Side-by-side comparison of attribution patterns

## Development and Testing

### Commands
- **Run Tests**: `poetry run pytest` or `python -m pytest`
- **Run Single Test**: `poetry run pytest tests/test_file.py::test_function`
- **Code Formatting**: `poetry run black .`
- **Type Checking**: `poetry run mypy .`
- **Run Benchmarks**: `python scripts/run_facts_attribution.py`
- **UI Development**: `cd ui && bun run dev`
- **UI Lint**: `cd ui && bun run lint`
- **UI Build**: `cd ui && bun run build`

### Project Architecture

#### Core Components
- `src/core/model.py`: Model management and initialization
- `src/core/analysis.py`: Custom gradient-based token analysis
- `src/core/inseq_analysis.py`: Inseq-powered attribution methods
- `src/persistence/storage.py`: Analysis result storage and retrieval
- `src/visualization/`: Visualization utilities and plotting functions
- `src/benchmark/`: Performance evaluation and attribution method comparison
- `ui/`: React/TypeScript UI for interactive exploration

## Output Format

Generated JSON files follow the TokenLens UI specification:
```json
{
  "metadata": {
    "timestamp": "ISO-8601",
    "llm_id": "model_id",
    "prompt": "input_text",
    "attribution_method": "method_name",
    "generation_params": {}
  },
  "data": {
    "input_tokens": [],
    "output_tokens": [],
    "association_matrix": [],
    "normalized_association": []
  }
}
```

## Native Inseq Format Support

The `InseqTokenAnalyzer` saves both custom format and Inseq's native format:

```python
# Load a native Inseq file
result = InseqTokenAnalyzer.load_native_inseq("path/to/file_inseq.json")

# Convert a native Inseq file to our format
analyzer = InseqTokenAnalyzer(model_manager)
result = analyzer.convert_native_to_analysis_result("path/to/file_inseq.json", original_prompt)

# Visualize with Inseq's built-in tools
attribution = inseq.FeatureAttributionOutput.load("path/to/file_inseq.json")
attribution.show()  # HTML visualization in Jupyter
```