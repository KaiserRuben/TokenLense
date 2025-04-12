# TokenLense Analyzer Documentation

The Analyzer component of the [TokenLense](../../README.md) framework provides a FastAPI-based service that processes attribution data and makes it available through REST endpoints.

## Documentation

- [API Overview](README.md) - This document
- [Attribution Endpoint Documentation](Documentation/get_attribution_endpoint.md) - Detailed schema for attribution data

## API Endpoints

The API provides a set of endpoints for accessing attribution data:

- **/models/**: List available models
- **/models/{model}/methods**: Get attribution methods for a specific model
- **/models/{model}/methods/{method}/files**: Get files available for a model and method combination
- **/attribution/{model}/{method}/{fileId}**: Get attribution data for visualization
- **/attribution/{model}/{method}/{fileId}/detailed**: Get detailed attribution including tensor information
- **/attribution/aggregation_methods**: Get available aggregation methods
- **/performance/system**: Get system hardware information
- **/performance/timing**: Get timing data for attribution methods

## Data Structure

The analyzer expects attribution data to be organized in the following directory structure:

```
analyzer/data/
├── MODEL_NAME/
│   ├── method_ATTRIBUTION_METHOD/
│   │   └── data/
│   │       ├── <attribution_data>_inseq.json
│   │       └── ...
```

Where:
- `MODEL_NAME` is the name of the model (e.g., `gpt2`)
- `ATTRIBUTION_METHOD` is the name of the attribution method (e.g., `input_times_gradient`)
- `<attribution_data>_inseq.json` are the attribution data files

## Related Documentation

- [Main Project Documentation](../../README.md)
- [Extractor Component](../../extractor/ReadMe.md)
- [Visualizer Component](../../visualizer/README.md)