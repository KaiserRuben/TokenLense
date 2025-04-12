# API Integration

## Overview
TokenLense's frontend interacts with a backend API to fetch model data, attribution information, and performance metrics. This document describes the API integration approach used throughout the application.

## Core API Module (`lib/api.ts`)

### Base Configuration
- API base URL is defined from environment variable `NEXT_PUBLIC_API_URL` or defaults to `http://localhost:8000`
- Common error handling and request formatting is centralized

### Key API Functions

#### Model Data
- `getModels()`: Retrieves list of available models
- `getModelMethods(model)`: Gets attribution methods for a specific model
- `getModelMethodFiles(model, method)`: Gets files available for a model and method combination

#### Attribution Data
- `getAttribution(model, method, fileId, aggregation)`: Retrieves attribution data for visualization
- `getDetailedAttribution(model, method, fileId, aggregation)`: Gets detailed attribution including tensor info
- `getAggregationMethods()`: Gets available aggregation methods and default
- `getTokenImportanceAcrossFiles(model, method, aggregation)`: Aggregates token importance across files

#### Performance Data
- `getSystemInfo()`: Retrieves system hardware information
- `getTimingResults()`: Gets raw timing data for attribution methods
- `getPerformanceData()`: Processes timing data into visualization-ready formats
- Helper functions for processing different performance metrics:
  - `processMethodPerformance()`
  - `processTokensPerSecond()`
  - `processHardwareComparison()`

### Data Processing
- Transforms API responses into formats expected by visualization components
- Calculates derived metrics (like average times and tokens per second)
- Normalizes data for consistent visualization

## Data Types (`lib/types.ts`)

### API Response Types
- `ModelInfo`: List of available models
- `ModelMethods`: Methods available for a model
- `ModelMethodFile`: Files available for a model/method
- `AttributionResponse`: Core attribution data
- `AttributionDetailedResponse`: Detailed attribution with matrix info
- `SystemInfo`: Hardware/environment information
- `TimingResults`: Raw performance data

### Processed Data Types
- `AnalysisResult`: Transformed attribution data for visualization
- `AnalysisData`: Core data subset of analysis result
- `AnalysisMetadata`: Metadata about the analysis
- `TokenData`: Processed token information
- `MethodPerformanceData`: Performance data formatted for charts
- `TokensPerSecondData`: Throughput data for heatmap
- `HardwareComparisonData`: Cross-hardware performance data

### Configuration Types
- `AggregationMethod`: Type for supported aggregation methods
- `VisualizationSettings`: Interface for visualization configuration

## Error Handling
- API requests include try/catch blocks for error handling
- Errors are logged to console for debugging
- User-friendly error messages are displayed in the UI
- Loading states are managed during API requests

## Performance Optimizations
- Data transformation happens client-side to reduce API complexity
- Reuse of fetched data where possible
- Local caching of processed results

## API Endpoints
TokenLense frontend relies on these key API endpoints:

1. `/models/`
2. `/models/{model}/methods`  
3. `/models/{model}/methods/{method}/files`
4. `/attribution/{model}/{method}/{fileId}`
5. `/attribution/{model}/{method}/{fileId}/detailed`
6. `/attribution/aggregation_methods`
7. `/performance/system`
8. `/performance/timing`

The API integration is designed to be flexible and efficient, with appropriate data transformation happening at the boundary between the frontend and backend systems.