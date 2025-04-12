# TokenLense Onboarding Strategy

## Overview
This document outlines the approach for implementing user onboarding in the TokenLense visualization platform. The onboarding experience aims to help new users understand the dashboard features and how to use them effectively.

## Goals
1. Provide first-time users with clear guidance on using TokenLense
2. Explain key features and interactions specific to each page
3. Avoid overwhelming users with too much information at once
4. Only show onboarding once per page (using local storage tracking)
5. Support progressive disclosure of complex features

## Component Structure

### `Onboarding` Component
- Reusable component that displays different content based on current page
- Tracks viewed pages in localStorage to prevent repeat displays
- Supports multi-step guides with pagination controls
- Can be closed/dismissed by the user

## Technical Implementation

### Local Storage Tracking
```typescript
// Key for localStorage tracking
const ONBOARDING_KEY = 'tokenlense-onboarding-viewed'

// Check if page has been viewed
const viewedPages = JSON.parse(localStorage.getItem(ONBOARDING_KEY) || '{}')
if (!viewedPages[currentPath]) {
  // Show onboarding
  // Mark as viewed
  const updatedViewedPages = { ...viewedPages, [currentPath]: true }
  localStorage.setItem(ONBOARDING_KEY, JSON.stringify(updatedViewedPages))
}
```

### UI Approach Options

#### Option 1: Modal Overlay (Recommended)
- Full-screen modal with semi-transparent background
- Focuses user attention on onboarding content
- Provides clear next/previous/close controls
- Can highlight specific UI elements with "spotlights"

#### Option 2: Slide-in Panel
- Uses shadcn/ui Sheet component
- Less intrusive than full modal
- Can slide in from bottom/side
- Good for shorter content pieces

#### Option 3: Tooltip Sequence
- Series of tooltips pointing to specific UI elements
- Most contextual but requires precise positioning
- Challenging to implement across all viewport sizes
- Better for feature-specific guidance than initial onboarding

#### Option 4: Overlay Cards
- Cards that appear above specific UI sections
- Balance between contextual and explanatory
- Can be positioned relative to key features
- May require complex positioning logic

## Integration Options

### Option 1: Root Layout Integration (Recommended)
- Add to app/layout.tsx to handle all pages
- Access current route via usePathname() from next/navigation
- Centralized management of onboarding state

```tsx
// In app/layout.tsx
<html lang="en">
  <body>
    <Header />
    <main>
      {children}
    </main>
    <Onboarding path={pathname} />
  </body>
</html>
```

### Option 2: Page-specific Implementation
- Implement onboarding separately on each page
- More targeted but requires duplication
- Better for very different onboarding experiences per page

### Option 3: Layout Group Implementation
- Add to layout files for specific sections
- Balance between centralized and page-specific

## Content Strategy

### Home/Models Page
- Introduction to TokenLense purpose and value
- Guidance on selecting models
- Explanation of attribution methods

### Performance Page
- Explain performance metrics and charts
- Guide on using filters and comparison options
- Help interpreting visualization results

### Compare Page
- Explain comparison functionality
- Guide on selecting comparison parameters
- Help interpreting side-by-side views

### Token Importance Page
- Explain token importance concept
- Guide on interpreting the ranked list
- Help understanding aggregation methods

## Development Approach

1. Create the base Onboarding component
2. Implement localStorage tracking
3. Design UI for the onboarding modal/panel
4. Create content for each page
5. Add to layout with path-based content switching
6. Test across different viewport sizes
7. Test localStorage persistence

## Future Enhancements

1. User preference to reset/replay onboarding
2. More interactive onboarding with "try it" steps
3. Video demonstrations for complex features
4. Expandable help sections for advanced concepts