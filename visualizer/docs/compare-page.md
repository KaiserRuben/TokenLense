# Compare Page

## Overview
The Compare page enables side-by-side comparison of attribution results between different models, methods, or both. This provides a powerful way to directly visualize differences in how models or attribution methods interpret the same prompt.

## Components

### Compare Selector Page (`app/compare-selector/page.tsx`)
- **Purpose**: Configure comparison parameters before viewing results
- **Key Features**:
  - Selection of comparison type (models, methods, or both)
  - Model selection
  - Method selection
  - Aggregation method selection

### Comparison Page (`app/compare/page.tsx`)
- **Purpose**: Display side-by-side attribution visualizations
- **Data Source**: Fetches attribution data from API based on selected parameters
- **Key Features**:
  - Split view of two attribution results
  - Synchronized token highlighting across views
  - Shared prompt and generation display
  - Visualization controls

### Comparison Controls (`components/comparison/ComparisonControls.tsx`)
- **Purpose**: Provide settings for comparison visualization
- **Configuration Options**:
  - Max connections
  - Relative strength normalization
  - Background highlighting or connection lines
  - Independent controls for each side

### Split View (`components/comparison/SplitView.tsx`)
- **Purpose**: Display side-by-side token clouds
- **Key Features**:
  - Coordinated token selection (hover/click)
  - Attribution information display
  - Prompt and generation display

### Comparison Token Cloud (`components/comparison/ComparisonTokenCloud.tsx`)
- **Purpose**: Visualize token relationships with coordinated state
- **Key Features**:
  - Shows input/output tokens
  - Highlights relationships based on selected token
  - Shares hover/click state with the other side

## Configuration Options
- **Comparison Type**:
  - Models (same method, different models)
  - Methods (same model, different methods)
  - Both (different models and methods)
- **Visualization Settings** (for each side):
  - Max Connections: Number of connections to display (1-10)
  - Use Relative Strength: Normalize connection strengths
  - Show Background: Use background highlighting for relationships
  - Show Connections: Use connection lines for relationships

## Presentation
- Side-by-side token clouds with synchronized hover/selection
- Shared prompt and generation display
- Color-coded inputs (orange) and outputs (blue)
- Connection lines or background highlight to show relationships

## User Interaction
- **Token Hover**: Highlights relationships on both sides
- **Token Click**: Locks selection to enable deeper comparison
- **Settings Adjustment**: Updates visualization style in real-time

## Data Dependencies
- URL parameters define the comparison:
  - `type`: 'models', 'methods', or 'both'
  - `fileId`: File identifier
  - Model parameters: `leftModel`, `rightModel`, `sharedModel`
  - Method parameters: `leftMethod`, `rightMethod`, `sharedMethod`
  - Aggregation parameters: `leftAggregation`, `rightAggregation`

The Compare page is an essential analytical tool in TokenLense that allows researchers to directly contrast how different models or methods interpret the same prompt, revealing subtle differences in attribution patterns.