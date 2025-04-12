# Token Importance Page

## Overview
The Token Importance page provides a global view of token importance across multiple files for a selected model and attribution method. It ranks tokens based on their aggregated attribution scores, revealing which tokens consistently have high influence across different inputs.

## Components

### Token Importance Page (`app/token-importance/page.tsx`)
- **Purpose**: Visualize token importance across all analyzed files
- **Data Source**: 
  - Aggregates data from multiple API endpoints
  - Uses `/models/`, `/models/{model}/methods/`, and `/attribution/` endpoints
- **Key Features**:
  - Token importance ranking visualization
  - Filter controls for model, method, and aggregation
  - Information about file counts and analysis breadth

## Configuration Options
- **Model Selection**: Choose which model to analyze
- **Attribution Method**: Select which attribution method to use
- **Aggregation Method**: How to combine attribution values:
  - Sum: Add all attribution values (default)
  - Mean: Average of attribution values
  - L2 Norm: Square root of sum of squares
  - Abs Sum: Sum of absolute values
  - Max: Maximum attribution value

## Presentation
- Ranked list of tokens with visual importance bars
- Shows token text and frequency information
- Percentage-based importance visualization
- File count indicators for each token

## Data Processing
- Aggregates attribution data across all available files
- Normalizes importance scores for easier comparison
- Tracks which files each token appears in
- Sorts tokens by their overall importance

## User Flow
1. User selects a model from the dropdown
2. User selects an attribution method for that model
3. User selects an aggregation method to determine how importance is calculated
4. The system loads and displays token importance data across all available files
5. User can view tokens ranked by their overall importance

## URL Parameters
The page supports URL parameters for direct access to specific views:
- `model`: Selected model name
- `method`: Selected attribution method
- `aggregation`: Selected aggregation method

## Technical Implementation
- Client-side aggregation of token importance
- Processes attribution matrices to extract token importance
- Normalizes importance values to a 0-1 scale for visualization
- Tracks token occurrence across files

The Token Importance page provides valuable insights into which tokens consistently have high influence across different inputs, helping researchers identify patterns in model behavior that might not be apparent when looking at individual files.