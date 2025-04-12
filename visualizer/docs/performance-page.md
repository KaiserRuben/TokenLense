# Performance Page

## Overview
The Performance page provides comprehensive benchmarks and visualizations for comparing the execution speed and resource usage of different attribution methods across models and hardware configurations.

## Components

### Performance Page (`app/performance/page.tsx`)
- **Purpose**: Visualize and compare performance metrics for attribution methods
- **Data Source**: Fetches data from API `/performance/timing` with both method and prompt timing results
- **Key Features**:
  - Summary statistics (success rate, fastest/slowest methods, best throughput)
  - Multiple chart types for different performance aspects
  - Filtering capabilities by model and method

### Method Performance Chart (`components/charts/MethodPerformanceChart.tsx`)
- **Purpose**: Compare execution times across methods and models
- **Configuration Options**:
  - Group by: Model or Method
  - Sort by: Alphabetical or Performance
- **Presentation**: Bar chart with color-coded bars for easy visual comparison

### Tokens Per Second Heatmap (`components/charts/TokensPerSecondHeatmap.tsx`)
- **Purpose**: Visualize throughput metrics in a compact format
- **Configuration Options**:
  - Scale: Linear or Logarithmic
- **Presentation**: Color-coded heatmap where darker red indicates higher tokens per second

### Hardware Comparison Chart (`components/charts/HardwareComparisonChart.tsx`)
- **Purpose**: Compare method performance across different hardware configurations
- **Presentation**: Shows how hardware affects each method's performance

### Performance Insights Card (`components/performance/PerformanceInsightsCard.tsx`)
- **Purpose**: Provide analytical insights into performance data
- **Presentation**: Detailed findings about method efficiency and resource usage

## Configuration Options
- **Model Filter**: Select which models to include in visualizations
- **Method Filter**: Select which attribution methods to include in visualizations
- **Tab Selection**: Switch between different analysis views:
  - Overview
  - Performance Insights
  - Models (future feature)
  - Methods (future feature)

## Presentation
- Dashboard style with multiple cards and visualizations
- Interactive elements for exploring data
- Color-coded indicators for performance metrics
- Summary statistics cards at the top

## Data Structure
- Performance data is processed from raw timing results into specialized formats for each visualization
- Key metrics include:
  - Average processing time
  - Tokens processed per second
  - Success rate
  - Min/max execution times
  - Hardware-specific performance

## User Flow
1. User views summary stats to understand overall performance landscape
2. User applies filters to focus on models/methods of interest
3. User explores different tabs for varying levels of detail
4. User compares performance across methods, models, and hardware

The Performance page enables users to make data-driven decisions about which attribution methods are most appropriate for their use case based on efficiency and resource requirements.