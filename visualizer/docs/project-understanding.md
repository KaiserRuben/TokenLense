# TokenLense Project Understanding

## API Structure

The TokenLense API provides access to token attribution data for language models. The API allows users to:

1. Browse available language models (`/models/`)
2. View supported attribution methods for each model (`/models/{model}/methods`)
3. Access attribution files for a given model and method (`/models/{model}/methods/{method}/files`)
4. Retrieve and visualize attribution data (`/attribution/{model}/{method}/{file_id}`)
5. Compare attribution data across different files (`/attribution/compare`)
6. Analyze token importance (`/attribution/token_importance`)
7. View performance metrics (`/performance/timing` and `/performance/system`)

## Data Schema

### Attribution Data

The core attribution data follows a consistent structure:

- **Model**: The language model used (e.g., "GPT-2")
- **Method**: The attribution method used (e.g., "attention", "integrated_gradients")
- **Source Tokens**: Tokenized representation of the input text
- **Target Tokens**: Tokenized representation of the generated output
- **Attribution Matrix**: A 2D matrix of values showing the influence of each token on the generation process

The attribution matrix is key to visualizations. It has dimensions:
- Rows: Number of target (output) tokens
- Columns: Number of source (input) + target tokens

Each cell value represents the attribution score showing how much influence the input token (or previous output token) had on generating the current output token.

### Aggregation Methods

The API supports multiple aggregation methods for converting high-dimensional tensors to 2D matrices:
- `sum`: Sum over feature dimensions
- `mean`: Average over feature dimensions
- `l2_norm`: L2 norm over feature dimensions
- `abs_sum`: Absolute sum over feature dimensions
- `max`: Maximum value across feature dimensions

## Existing Visualization Components

### WordCloud Component

The existing `WordCloud` component:
- Displays source and target tokens in separate sections
- Shows connections (influences) between tokens when hovering or clicking
- Allows toggling connections, background colors, and importance bars
- Displays token importance based on the sum of attribution values
- Provides interactive features like locking token selections

### TokenExplorer Component

The `TokenExplorer` component:
- Wraps the `WordCloud` component
- Provides controls for visualization settings:
  - Maximum connections per token
  - Background color toggle
  - Connection visibility toggle
  - Importance bars toggle

### Dashboard Component

The `Dashboard` view:
- Displays token relationships using the `TokenExplorer`
- Shows an alternative `AssociationMatrix` visualization
- Provides metadata about the analysis
- Offers tabbed navigation between different visualization types

## Data Flow in Existing Implementation

1. Data is loaded and validated from JSON files
2. The `AnalysisContext` provides selected analysis data to components
3. The `TokenExplorer` sets visualization parameters
4. The `WordCloud` component handles interactive behaviors and rendering
5. The attribution matrix is used to calculate token importance and connections

## Challenges to Address

1. **Data Format Alignment**: The existing code expects a different data format than the API provides
2. **Scalability**: Handle potentially large attribution matrices efficiently
3. **Method Selection**: Support toggling between different attribution methods
4. **Model Comparison**: Enable side-by-side comparison of different models
5. **Context Window**: Implement sliding context window for exploring broader token relationships

## Key Implementation Requirements

1. **Matrix Transformation**: Convert API-provided attribution matrices to the format expected by visualization components
2. **TypeScript Interfaces**: Define proper interfaces for API responses
3. **Method Toggling**: Allow users to select and compare multiple attribution methods
4. **Model Selection**: Implement model browsing and selection UI
5. **Context Adjustment**: Add slider control for context window size
6. **Enhanced Visualization**: Improve token relationship visualization with additional features