# TokenLense Onboarding User Stories

## Overview
This document outlines user stories for the TokenLense onboarding experience. These stories help guide development by focusing on concrete user needs and interactions.

## First-time User Stories

### General Onboarding

**US-ON-01: First-time Welcome**
```
As a first-time user
I want to see a welcome introduction to TokenLense
So that I understand what the platform offers and how it can help me
```
**Acceptance Criteria:**
- Welcome message shows on first visit to the home page
- Includes a brief explanation of TokenLense purpose
- Provides navigation to next steps or option to skip
- Is only shown once (tracked in localStorage)

**US-ON-02: Feature Overview**
```
As a new user
I want to see an overview of key platform features
So that I know what capabilities are available to me
```
**Acceptance Criteria:**
- Shows main sections/pages available
- Brief explanation of each section's purpose
- Visual indicators for main navigation options
- Option to learn more about specific sections

**US-ON-03: Onboarding Dismissal**
```
As a user
I want to be able to skip or dismiss onboarding
So that I can directly explore the platform if I prefer
```
**Acceptance Criteria:**
- Clear "Skip" or "Close" button visible on all onboarding screens
- Closing onboarding remembers this preference
- Provides option to view onboarding again later

## Page-Specific User Stories

### Models Page

**US-ON-04: Model Selection Guidance**
```
As a new user on the Models page
I want guidance on how to select and explore models
So that I can effectively start using TokenLense
```
**Acceptance Criteria:**
- Explains purpose of model selection
- Points out model cards and how to select them
- Describes what happens after model selection
- Shows only on first visit to models page

### Attribution Methods Page

**US-ON-05: Attribution Method Explanation**
```
As a user exploring attribution methods
I want a clear explanation of what attribution methods are
So that I can choose the appropriate method for my analysis
```
**Acceptance Criteria:**
- Explains what attribution methods do
- Highlights differences between methods
- Provides guidance on method selection
- Shows visual example of method output

### Performance Page

**US-ON-06: Performance Dashboard Orientation**
```
As a user viewing the Performance page for the first time
I want help understanding the charts and metrics
So that I can effectively interpret performance data
```
**Acceptance Criteria:**
- Explains purpose of performance dashboard
- Points out main visualization areas
- Explains key metrics and their significance
- Shows how to use filters and settings

**US-ON-07: Filter Usage Guidance**
```
As a user on the Performance page
I want to understand how to use the filters
So that I can focus on specific models or methods
```
**Acceptance Criteria:**
- Demonstrates how to select/deselect models and methods
- Explains how filters affect visualizations
- Shows examples of filtered results

### Compare Page

**US-ON-08: Comparison Setup Guidance**
```
As a user setting up a comparison
I want guidance on selecting comparison parameters
So that I can create meaningful comparisons
```
**Acceptance Criteria:**
- Explains different comparison types (models/methods/both)
- Guides through selection of comparison parameters
- Shows how to configure visualization settings
- Explains what the comparison will show

**US-ON-09: Comparison Interaction Guidance**
```
As a user viewing a comparison
I want to understand how to interact with the visualization
So that I can gain insights from the comparison
```
**Acceptance Criteria:**
- Explains token selection/hover behavior
- Demonstrates how relationship highlighting works
- Shows how to adjust comparison settings
- Explains how to interpret differences between sides

### Token Importance Page

**US-ON-10: Token Importance Concept**
```
As a user viewing the Token Importance page
I want to understand what token importance means
So that I can interpret the ranked list correctly
```
**Acceptance Criteria:**
- Explains token importance concept
- Describes how tokens are ranked
- Shows how to interpret importance scores
- Explains relationship to file occurrence

**US-ON-11: Aggregation Method Selection**
```
As a user on the Token Importance page
I want to understand different aggregation methods
So that I can choose the most appropriate for my analysis
```
**Acceptance Criteria:**
- Explains available aggregation methods
- Describes when to use each method
- Shows how changing aggregation affects results
- Provides examples of different aggregation outputs

## Advanced User Stories

**US-ON-12: Reset Onboarding**
```
As a returning user
I want the option to view the onboarding again
So that I can refresh my understanding of TokenLense features
```
**Acceptance Criteria:**
- Provides option to reset onboarding preferences
- Accessible from user menu or help section
- Clearly explains what resetting will do
- Allows selective reset for specific pages

**US-ON-13: Contextual Help**
```
As an existing user
I want access to contextual help for specific features
So that I can learn about new or complex functionality as needed
```
**Acceptance Criteria:**
- Help icons or buttons near complex features
- Opens focused explanations without full onboarding
- Provides more detailed information than initial onboarding
- Does not reset onboarding preferences