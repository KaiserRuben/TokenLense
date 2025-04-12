# Models Page

## Overview
The Models page serves as the entry point and navigation hub for exploring different language models available in TokenLense. It allows users to select models and attribution methods for token analysis.

## Components

### Home Page (`app/page.tsx`)
- **Purpose**: Main landing page for selecting a language model
- **Data Source**: Fetches model list from API `/models/`
- **Key Features**:
  - Grid of available models with cards for each
  - Loading states during API calls
  - Error handling
  - Navigation to individual model pages

### Model Page (`app/models/[modelId]/page.tsx`)
- **Purpose**: Display and select attribution methods for a specific model
- **Data Source**: Fetches methods from API `/models/{modelId}/methods`
- **Key Features**:
  - Lists available attribution methods
  - Provides descriptions for common methods:
    - Attention
    - Input × Gradient
    - Integrated Gradients
    - Layer Gradient × Activation
    - LIME
    - Saliency
  - Navigation to specific method file selection
  - Educational content about attribution methods

## Configuration Options
- No explicit configuration options on these pages
- Model and method selection flow

## Presentation
- Model selection cards with model names
- Method selection cards with descriptive text
- Information sections explaining attribution concepts
- Back navigation between levels

## User Flow
1. User lands on Home page and sees available models
2. User selects a model to navigate to the Model page
3. On the Model page, user sees available attribution methods for that model
4. User selects a method to navigate to file selection

## Data Dependencies
- API endpoint `/models/` must return an array of model names
- API endpoint `/models/{modelId}/methods` must return an array of method names

The Models section serves as the starting point for exploration, allowing users to navigate through models and attribution methods before diving into visualizations.