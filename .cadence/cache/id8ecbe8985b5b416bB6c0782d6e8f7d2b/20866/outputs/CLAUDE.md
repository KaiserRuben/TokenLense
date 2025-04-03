# CLAUDE.md - Llama Token Analyzer Repository

## Commands
- **Install**: `poetry install` or `pip install -r requirements.txt`
- **Run Tests**: `poetry run pytest` or `python -m pytest`
- **Run Single Test**: `poetry run pytest tests/test_file.py::test_function`
- **Lint**: `poetry run black .` (code formatting)
- **Type Check**: `poetry run mypy .`
- **UI Development**: `cd ui && bun run dev`
- **UI Lint**: `cd ui && bun run lint`
- **UI Build**: `cd ui && bun run build`

## Style Guidelines
- **Python**: Black formatted (88 chars), MyPy typed (strict)
- **Types**: Use type hints for all functions, disallow untyped defs
- **Imports**: Group standard lib, third-party, local imports
- **Naming**: snake_case for Python, PascalCase for React components
- **Error Handling**: Use `returns` library Result/Maybe monads
- **Documentation**: Docstrings for modules/classes/functions
- **Functional Style**: Prefer pure functions, use toolz library
- **React/TS**: Follow TypeScript strict mode, use TailwindCSS for UI

## Project Architecture

### Current Implementation
- Custom token attribution analysis based on gradient tracking
- Implementation in `work/llama_token_analyzer/core/analysis.py`
- Uses direct gradient calculation with `requires_grad_(True)`
- Matrix calculation in `_calculate_association_matrix`

### Inseq Implementation (Initial work complete)
- Created implementation in `work/llama_token_analyzer/core/inseq_analysis.py`
- Added `InseqTokenAnalyzer` class to replace custom gradient calculation
- Used Inseq's built-in attribution methods (saliency as default)
- Example usage in `work/notebooks/inseq_example.py`
- Remaining work:
  1. Verify visualization compatibility with Inseq attribution format
  2. Enhance attribution result processing for larger inputs
  3. Add support for advanced Inseq features

## Inseq Integration Guide
1. Use `inseq.load_model(model, method, tokenizer)` instead of custom attribution
2. Available attribution methods: "saliency", "attention", "integrated_gradients", "input_x_gradient", "layer_deeplift", and many more
3. Import `InseqTokenAnalyzer` instead of `TokenAnalyzer`
4. Initialize with desired attribution method:
```python
analyzer = InseqTokenAnalyzer(manager, attribution_method="saliency")
```
5. Same API as before for analysis pipeline:
```python
analyze = analyzer.create_analysis_pipeline(storage)
analysis_result = analyze("Your prompt here")
```

## Native Inseq Format Support
The analyzer now saves both our custom format and Inseq's native format:

1. Custom format: Saved to the data directory with the standard naming pattern
2. Inseq native format: Saved in the same directory with "_inseq.json" suffix

To work with native Inseq files:
```python
# Load a native Inseq file
result = InseqTokenAnalyzer.load_native_inseq("path/to/file_inseq.json")

# Convert a native Inseq file to our format
analyzer = InseqTokenAnalyzer(model_manager)
result = analyzer.convert_native_to_analysis_result("path/to/file_inseq.json", original_prompt)
```

You can also visualize native Inseq files using Inseq's built-in visualization:
```python
attribution = inseq.FeatureAttributionOutput.load("path/to/file_inseq.json")
attribution.show()  # HTML visualization in Jupyter
html = attribution.show(display=False, return_html=True)  # Get HTML as string
```

## Migration Path
1. Support both implementations during transition:
   - `TokenAnalyzer` - Original gradient-based implementation
   - `InseqTokenAnalyzer` - New Inseq-based implementation
2. Continue using existing visualization without changes
3. Future enhancements:
   - Add method selection to UI
   - Support advanced Inseq attribution parameters
   - Improve error handling for Inseq-specific issues