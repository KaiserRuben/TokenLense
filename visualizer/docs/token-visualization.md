# Token Visualization Components

## Overview
TokenLense features powerful token visualization components that reveal the relationships and influences between tokens in language model generations. These components form the core of the attribution visualization system.

## Primary Components

### TokenCloud (`components/attribution/TokenCloud.tsx`)
- **Purpose**: Visualize relationships between input and output tokens
- **Key Features**:
  - Interactive token selection via hover/click
  - Visual relationship indicators (background or connections)
  - Importance-based token highlighting
  - Token relationship visualization

### TokenExplorer (`components/attribution/TokenExplorer.tsx`)
- **Purpose**: Wrapper component with controls for TokenCloud
- **Key Features**:
  - Visualization mode selection (background or connections)
  - Maximum connections slider control
  - Normalization toggle for relative strengths
  - Controls for customizing the visualization experience

### ComparisonTokenCloud (`components/comparison/ComparisonTokenCloud.tsx`)
- **Purpose**: Specialized TokenCloud for side-by-side comparisons
- **Key Features**:
  - Synchronized token selection with another instance
  - Compact layout for side-by-side display
  - External state management for coordinated interactions
  - Shared hover/selection state

## Visualization Modes

### Background Highlighting
- Shows token relationships using background color intensity
- When a token is selected:
  - The selected token is highlighted
  - Related tokens are highlighted with intensity proportional to relationship strength
  - Unrelated tokens remain unhighlighted or show only their general importance

### Connection Lines
- Shows token relationships using curved connecting lines
- When a token is selected:
  - Lines connect the selected token to its most influential related tokens
  - Line thickness indicates relationship strength
  - Number of connections limited by max connections setting

## Token Data Representation
- **Input Tokens**: Shown with orange indicators
- **Output Tokens**: Shown with blue indicators
- **Token Text**: Displays the cleaned token text
- **Token ID**: Available via tooltip

## Interaction Model
- **Hover**: Temporarily highlights relationships for the hovered token
- **Click**: Locks the selection to enable sustained examination
- **Click Again**: Unlocks the selection

## Configuration Options
- **Max Connections**: Number of connections to display (1-10)
- **Relative Strength**: Toggle between absolute and relative strength normalization
- **Visualization Mode**: Background highlighting or connection lines
- **Context Size**: Number of surrounding tokens to include in context (where applicable)

## Technical Implementation
- Uses React functional components with hooks
- Calculates token importance from attribution matrices
- Finds top influences for each token based on the attribution data
- Creates visual connections based on DOM positioning
- Uses useRef for DOM element references
- Sets up event handlers for interaction

## Styling
- Color scheme: Orange for input tokens, Blue for output tokens
- Background gradient for relationship strength
- SVG path elements for connection lines
- Responsive design with appropriate spacing

The token visualization components provide an intuitive way to explore how language models attribute importance between tokens, revealing patterns in how the model makes predictions and generates text.