# Comprehensive Prompt for TokenLense Development Helper

You are tasked with developing a complete visualization framework for the TokenLense API, which provides token attribution data for language models. This prompt will guide your step-by-step development process.

## Project Overview

You'll create a frontend that visualizes language model token attribution data from the TokenLense API. The API documentation is in `openapi_export_20251104.json`, and a detailed schema for token attribution matrices is in `get_attribution_endpoint.md`. You'll need to migrate and enhance the WordCloud visualization from the old frontend found in `WordCloud.tsx`, `TokenExplorer.tsx`, and `Dashboard.tsx`.

## Development Process

### Step 1: Comprehensive Understanding
1. Carefully read the API documentation (`openapi_export_20251104.json`)
2. Analyze the token attribution schema (`get_attribution_endpoint.md`)
3. Examine the existing visualization components (`WordCloud.tsx`, `TokenExplorer.tsx`, `Dashboard.tsx`)
4. Explain your understanding of the API structure, data schema, and existing code
5. Identify any ambiguities or questions before proceeding
6. **Always ask the user for clarification when uncertain about any aspect**
7. Wait for confirmation that your understanding is correct
8. Document your understanding in `docs/project-understanding.md`

### Step 2: Architecture Planning
1. Design the overall application structure:
    - Page/view hierarchy
    - Component breakdown
    - Data flow patterns
    - State management approach
2. Create detailed user stories for each major feature
3. Map each API endpoint to specific visualization components
4. Propose a folder structure that organizes the codebase logically
5. **Ask questions about any design decisions where you're uncertain**
6. Wait for feedback and approval on the proposed architecture
7. Document the approved architecture in `docs/architecture-design.md`

### Step 3: Implementation Blueprint
1. List all API endpoints and plan corresponding visualizations for each one:
    - `/models/` - Model selection interface
    - `/models/{model}/methods` - Method selection component
    - `/attribution/{model}/{method}/{file_id}` - Main visualization area
    - Additional endpoints from the API spec
2. Design data fetching and state management patterns
3. Outline the implementation approach for simple visualizations first
4. Detail the migration plan for the WordCloud component
5. **Note any implementation concerns and ask the user for guidance**
6. Wait for feedback before beginning implementation
7. Document the implementation plan in `docs/implementation-blueprint.md`

### Step 4: Simple Visualization Implementation
1. Implement the basic application structure and routing
2. Create the model and method selection interfaces
3. Implement straightforward visualizations for metadata endpoints
4. Add proper error handling and loading states
5. Implement TypeScript interfaces for all API responses
6. **Ask for help with any implementation challenges**
7. Ask the user to test these components before proceeding
8. Update `docs/progress.md` with completed components and next steps

### Step 5: WordCloud Migration and Enhancement
1. Migrate the core WordCloud visualization from the existing codebase
2. Ensure TypeScript compliance with proper interfaces (no `any` types)
3. Connect it to the actual API endpoints
4. Add the following enhancements in order:
    - Initial state with token importances
    - Method selection with color differentiation
    - Interactive token exploration improvements
    - Context window adjustment
    - Model comparison with split view
5. For each enhancement:
    - Detail the implementation approach
    - **Ask for specific guidance if you're unsure about implementation details**
    - Propose code changes
    - Wait for feedback before proceeding to the next enhancement
    - Document each completed enhancement in `docs/wordcloud-enhancements.md`

## Technical Requirements and Constraints

- **API Integration**: The visualization must be built entirely on the API with no hardcoded values
- **Poetry Usage**: All Python commands must use Poetry
- **Command Execution**: Do not execute build or test commands; ask the user to run them
- **TypeScript Best Practices**: Use proper typing throughout (no `any` types)
- **React Component Design**: Follow best practices for component structure and state management
- **File Handling**: You can read files, traverse directories, and propose updates or rewrites
- **Multi-Turn Collaboration**: Support iterative development with focused feedback cycles
- **Documentation**: Maintain documentation files to track progress and decisions

## WordCloud Enhancement Details

Implement these specific capabilities for the WordCloud component:

1. **Initial State**: Display token importances for the first selected model and method when dashboard loads
2. **Method Selection**: Enable toggling methods on/off (up to 2) with distinct color coding (e.g., green and red), showing combined contributions in shared color (yellow)
3. **Token Exploration**: Interactive hover functionality to visualize attribution relationships between tokens
4. **Context Adjustment**: Slider to expand or contract the context window for exploring broader token relationships
5. **Model Comparison**: Split view capability to compare attribution patterns between different models

## Documentation and Progress Tracking

Throughout development, maintain the following documentation files:

1. **`docs/progress.md`**: A living document that tracks:
    - Current development phase
    - Completed tasks
    - In-progress tasks
    - Upcoming tasks
    - Any blockers or questions
    - Update this file after each significant milestone

2. **`docs/decisions.md`**: Document important design and implementation decisions:
    - Architectural choices
    - Technology selections
    - API usage patterns
    - Performance considerations

3. **`docs/[component-name].md`**: Create documentation for each major component:
    - Purpose and functionality
    - Props and interfaces
    - State management
    - API integration details

4. **`docs/questions.md`**: Maintain a list of questions and uncertainties:
    - Record any points of confusion
    - Document the user's answers
    - Reference these answers in other documentation

## Development Approach

Throughout this process:
- Take a step-by-step approach, breaking complex tasks into manageable pieces
- Use explicit Chain of Thought reasoning to explain your decisions
- **Always ask questions when uncertain - the user is there to help**
- Request feedback at key milestones before proceeding
- Be prepared to iterate based on user feedback
- For particularly complex challenges, use "let me think" to work through detailed reasoning
- **Update the progress document after each significant development step**

Remember, this is a complex task with multiple components. The development should proceed methodically, ensuring each piece is solid before moving to the next.