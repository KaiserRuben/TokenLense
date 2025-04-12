# TokenCloud Component Enhancements

## Overview

The TokenCloud component is a central visualization in the TokenLense framework, showing the relationships between tokens in language model generation. It's based on the original WordCloud component from the previous codebase but has been enhanced and integrated with the API data format.

## Current Features

1. **Token Rendering**
   - Display source (input) and target (output) tokens in separate sections
   - Show token text with proper formatting
   - Handle special tokens and whitespace

2. **Token Importance**
   - Calculate token importance based on attribution matrix
   - Display optional importance bars under each token
   - Normalize importance values for consistent visualization

3. **Interactive Connections**
   - Show connections between tokens on hover or click
   - Visualize connection strength with line opacity and thickness
   - Support locking focus on a specific token

4. **Visual Styling**
   - Distinct colors for input vs. output tokens
   - Background highlighting to show token relationships
   - Smooth transitions and animations for interactions

## Planned Enhancements

### 1. Context Window Adjustment

**Description:** Allow adjusting the number of tokens shown in context, both preceding and following the focused token.

**Implementation Approach:**
- Add a context window slider control in TokenExplorer
- Filter visible tokens based on the context window size
- Show only tokens within the specified range of the active token
- Implement smooth transitions when changing context size

**Code Changes:**
- Add context filtering logic in the TokenCloud component
- Implement panning/scrolling to navigate through tokens
- Add visual indicator for tokens outside the context window

### 2. Method Selection with Color Differentiation

**Description:** Enable toggling multiple attribution methods with distinct color coding to compare methodologies.

**Implementation Approach:**
- Allow selecting up to 2 attribution methods simultaneously
- Use distinct colors for each method (e.g., green and red)
- Show combined contributions with a blended color (e.g., yellow)
- Implement a method selector component

**Code Changes:**
- Modify the API client to fetch multiple methods
- Extend TokenCloud to handle multiple attribution matrices
- Implement color blending for overlapping attributions
- Add method selection controls in the UI

### 3. Enhanced Token Exploration

**Description:** Improve the hover and click interactions to provide more detailed information about token relationships.

**Implementation Approach:**
- Add detailed tooltips showing exact attribution values
- Implement filtering options for connection strength
- Show attribution distribution across methods
- Add ability to filter connections by type (input-to-output, output-to-input)

**Code Changes:**
- Enhance the connection rendering logic
- Implement more sophisticated tooltips
- Add filtering controls in the TokenExplorer

### 4. Model Comparison with Split View

**Description:** Add the ability to compare attribution patterns between different models in a split view.

**Implementation Approach:**
- Create a side-by-side view of two TokenCloud components
- Synchronize interactions between the views
- Highlight differences in attribution patterns
- Allow selecting different models for comparison

**Code Changes:**
- Create a ModelComparison component
- Implement synchronized scrolling and focus
- Add visual indicators for attribution differences
- Create model selection controls

## Implementation Priorities

1. Context Window Adjustment - Most fundamental for exploring large token sequences
2. Method Selection - Essential for comparing different attribution approaches
3. Enhanced Token Exploration - Improves the analytical capabilities
4. Model Comparison - Advanced feature for comparative analysis

## Technical Challenges

1. **Performance:** Efficiently rendering potentially hundreds of tokens and connections
2. **Layout:** Managing the visual arrangement of tokens and connections
3. **Data Transformation:** Converting API data to the format needed by visualization
4. **Interaction:** Maintaining responsive and intuitive user interactions

## Progress Tracking

Each enhancement will be documented with:
- Detailed implementation notes
- Before/after screenshots
- Performance considerations
- User feedback and iterations