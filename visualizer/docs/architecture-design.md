# TokenLense Architecture Design

## Application Structure

### Page/View Hierarchy

```
└── Root Layout
    ├── Home Page (/)
    │   └── Model Selection
    ├── Model Page (/models/:modelId)
    │   └── Method Selection
    ├── Analysis Page (/attribution/:model/:method/:fileId)
    │   ├── Token Explorer
    │   ├── Matrix Visualization
    │   └── Performance Stats
    ├── Comparison Page (/compare)
    │   └── Model/Method Comparison View
    └── Performance Page (/performance)
        ├── Method Performance
        └── System Information
```

### Component Hierarchy

```
└── Layout Components
    ├── Header
    └── Footer
└── Model Components
    ├── ModelList
    ├── ModelCard
    └── ModelSelector
└── Method Components
    ├── MethodList
    ├── MethodCard
    └── MethodSelector
└── File Components
    ├── FileList
    ├── FileCard
    └── FileSelector
└── Visualization Components
    ├── TokenCloud
    │   ├── TokenRenderer
    │   ├── ConnectionRenderer
    │   └── ImportanceBar
    ├── AttributionMatrix
    │   ├── HeatmapGrid
    │   └── MatrixControls
    ├── TokenImportance
    │   ├── ImportanceChart
    │   └── TokenDetail
    └── MethodComparison
        ├── ComparisonGrid
        └── DifferenceHighlighter
└── Shared UI Components
    ├── Card
    ├── Tabs
    ├── Button
    ├── Slider
    ├── Switch
    ├── Tooltip
    └── Select
```

## Data Flow Patterns

1. **API Data Flow**:
   ```
   API Client → Data Fetching Hooks → Components
   ```

2. **User Interaction Flow**:
   ```
   User Input → Component State → UI Update → (Optional) API Request
   ```

3. **Context Window Adjustment**:
   ```
   Slider Control → Context Size State → TokenCloud Re-render
   ```

4. **Method Toggling Flow**:
   ```
   Method Selection → Attribution State → Visualization Update
   ```

## State Management Approach

For a Next.js 15 application, we'll use a combination of:

1. **React's useState and useReducer**:
   - For component-local state
   - For UI controls and interactive elements
   - For managing animation states

2. **React Query**:
   - For API data fetching and caching
   - For managing loading/error states
   - For data refetching policies

3. **Custom hooks**:
   - For shared, reusable logic
   - For aggregating related state operations
   - For abstracting complex state transitions

## Folder Structure

```
visualizer/
├── app/
│   ├── api/                     # API route handlers
│   │   └── performance/
│   │       └── route.ts         # Performance data endpoint
│   ├── attribution/             # Attribution pages
│   │   └── [model]/[method]/[fileId]/
│   │       └── page.tsx
│   ├── compare/                 # Comparison pages
│   │   └── page.tsx
│   ├── models/                  # Model pages
│   │   └── [modelId]/
│   │       └── page.tsx
│   ├── performance/             # Performance pages
│   │   └── page.tsx
│   ├── favicon.ico
│   ├── globals.css              # Global CSS
│   ├── layout.tsx               # Root layout
│   └── page.tsx                 # Homepage
├── components/                  # React components
│   ├── attribution/             # Attribution visualization components
│   │   ├── TokenCloud.tsx       # Enhanced WordCloud component
│   │   ├── AttributionMatrix.tsx
│   │   └── TokenExplorer.tsx
│   ├── charts/                  # Chart components
│   │   ├── HeatmapChart.tsx
│   │   └── ImportanceChart.tsx
│   ├── layout/                  # Layout components
│   │   ├── Header.tsx
│   │   └── Footer.tsx
│   ├── model/                   # Model selection components
│   │   ├── ModelCard.tsx
│   │   └── ModelList.tsx
│   ├── method/                  # Method selection components
│   │   ├── MethodCard.tsx
│   │   └── MethodList.tsx
│   └── ui/                      # UI components (shadcn/ui)
│       ├── button.tsx
│       ├── card.tsx
│       └── ...other UI components
├── hooks/                       # Custom React hooks
│   ├── useAttribution.ts        # Attribution data handling
│   ├── useModels.ts             # Model data fetching
│   └── useMethods.ts            # Method data fetching
├── lib/                         # Utility functions and shared code
│   ├── api.ts                   # API client
│   ├── types.ts                 # TypeScript interfaces
│   └── utils.ts                 # Utility functions
├── public/                      # Static assets
│   ├── file.svg
│   ├── globe.svg
│   └── ...other assets
└── docs/                        # Documentation
    ├── project-understanding.md
    ├── architecture-design.md
    ├── implementation-blueprint.md
    └── progress.md
```

## User Stories

### Model Exploration
- As a user, I want to see a list of available models
- As a user, I want to select a model to explore its attribution methods
- As a user, I want to see metadata about each model

### Method Selection
- As a user, I want to see available attribution methods for a selected model
- As a user, I want to compare different attribution methods
- As a user, I want to understand the differences between attribution methods

### Attribution Visualization
- As a user, I want to visualize token-to-token relationships
- As a user, I want to see which tokens most influence the generation
- As a user, I want to interact with tokens to explore their relationships
- As a user, I want to adjust the visualization parameters

### Context Window Adjustment
- As a user, I want to control how many tokens are shown in context
- As a user, I want to focus on specific regions of token relationships
- As a user, I want to slide the context window to explore different regions

### Method Comparison
- As a user, I want to compare attribution scores from different methods
- As a user, I want to toggle multiple methods on/off in the visualization
- As a user, I want to see combined contributions with color differentiation

### Model Comparison
- As a user, I want to compare attribution patterns between different models
- As a user, I want a side-by-side view of different models' attributions
- As a user, I want to identify differences in how models assign attribution

### Performance Analysis
- As a user, I want to see performance metrics for attribution methods
- As a user, I want to understand the tradeoffs between different methods
- As a user, I want to view system information related to the analysis

## API to Component Mapping

| API Endpoint | Component |
|--------------|-----------|
| `/models/` | `ModelList`, `ModelSelector` |
| `/models/{model}/methods` | `MethodList`, `MethodSelector` |
| `/models/{model}/methods/{method}/files` | `FileList`, `FileSelector` |
| `/attribution/{model}/{method}/{file_id}` | `TokenCloud`, `AttributionMatrix` |
| `/attribution/{model}/{method}/{file_id}/detailed` | `TokenDetail` |
| `/attribution/aggregation_methods` | `AggregationSelector` |
| `/attribution/compare` | `ComparisonGrid` |
| `/attribution/token_importance` | `ImportanceChart` |
| `/performance/system` | `SystemInfo` |
| `/performance/timing` | `PerformanceMetrics` |

## Technology Stack

- **Framework**: Next.js 15 with App Router
- **Styling**: Tailwind CSS with shadcn/ui components
- **Data Fetching**: React Query or SWR
- **Data Visualization**: D3.js for custom visualizations
- **TypeScript**: Strong typing throughout the application