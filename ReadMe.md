# Token Analyzer

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Poetry](https://img.shields.io/badge/Poetry-Package_Manager-blue)](https://python-poetry.org/)

A framework for analyzing and visualizing token relationships and importance metrics in Large Language Model outputs. 

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

1. Import required components:
```python
from llama_token_analyzer import ModelManager, TokenAnalyzer, TokenAnalysisStorage
from llama_token_analyzer.visualization.main import visualize
```

2. Initialize the model:
```python
model_config = {
    "llm_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "device": "auto",
    "torch_dtype": "float16"
}
manager = ModelManager.initialize(model_config)
```

3. Setup analysis pipeline:
```python
storage = TokenAnalysisStorage(base_path="output")
analyzer = TokenAnalyzer(manager)
analyze = analyzer.create_analysis_pipeline(storage)
```

4. Analyze prompts:
```python
# Single prompt
result = analyze("Your prompt here")

# Batch analysis
results = analyze([
    "Prompt 1",
    "Prompt 2"
])
```

## Visualization Options

### 1. Notebook Visualization
The example notebook (`notebooks/example.ipynb`) demonstrates immediate visualization using:
```python
visualize(result, storage=storage)
```

### 2. TokenLens UI (Recommended)
For a more interactive exploration:

1. Generate JSON files using the analyzer
2. Copy the generated files from your output directory to `ui/src/data/`
3. Follow the [TokenLens UI setup instructions](ui/README.md) to explore your visualizations

The UI provides enhanced features like:
- Interactive token relationship exploration
- Association strength visualization
- Statistical analysis tools

See the [UI documentation](ui/README.md) for deployment options and features.

## Example Notebook

`notebooks/example.ipynb` provides a complete workflow:
- Model initialization
- Single prompt analysis
- Batch processing
- Direct visualization
- JSON file generation for UI exploration

## Configuration

### Model Configuration
```python
model_config = {
    "llm_id": "model_identifier",  # Hugging Face model ID
    "device": "auto",              # "cpu", "cuda", "mps", or "auto"
    "torch_dtype": "float16"       # Model precision
}
```

### Storage Configuration
```python
storage = TokenAnalysisStorage(
    base_path="output",            # Output directory
    format="json"                  # Output format
)
```

## Output Format

Generated JSON files follow TokenLens UI specification:
```json
{
  "metadata": {
    "timestamp": "ISO-8601",
    "llm_id": "model_id",
    "prompt": "input_text",
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
See [sample Data](ui/src/data/sample.json) for a complete example.
