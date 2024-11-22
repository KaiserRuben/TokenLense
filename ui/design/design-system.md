# Data-Driven Design System

## Core Principles

### 1. Purposeful Color Usage
- Use monochromatic design as the base (black, white, grays)
- Apply color only when it carries meaning
- Primary colors indicate data changes:
  - Positive changes: `text-emerald-400`
  - Negative changes: `text-red-400`
- Avoid decorative color usage

### 2. Visual Hierarchy
```css
/* Title */
text-sm font-medium text-gray-400  // Secondary information

/* Primary Value */
text-4xl font-light text-white     // Main focus

/* Supporting Info */
text-sm text-gray-500              // Background information
```

### 3. Interactive Elements
```css
/* Card Base */
.metric-card {
  background: rgba(20, 20, 25, 0.9);
  border: 1px solid rgba(255, 255, 255, 0.08);
  box-shadow: 
    0 4px 24px -4px rgba(0, 0, 0, 0.3),
    0 0 0 1px rgba(255, 255, 255, 0.1) inset;
}

/* Card Hover State */
.metric-card:hover {
  border-color: rgba(255, 255, 255, 0.15);
  box-shadow: 
    0 8px 32px -4px rgba(0, 0, 0, 0.5),
    0 0 0 1px rgba(255, 255, 255, 0.15) inset;
  transform: translateY(-2px);
}
```

### 4. Data Visualization
- Use progress bars to show magnitude
- Color intensity reinforces data meaning
- Transitions should be smooth but subtle
  ```css
  transition-all duration-300
  ```

## Component Patterns

### Metric Card
```jsx
<div className="metric-card rounded-xl p-6 transition-all duration-300">
  <div className="space-y-2">
    <div className="text-sm font-medium text-gray-400">{title}</div>
    <div className="text-4xl font-light text-white">{value}</div>
    <div className="text-sm flex items-center gap-2 text-gray-500">
      {/* Progress Indicator */}
      <div className="w-full h-1 rounded-full bg-gray-800">
        <div className="h-full rounded-full bg-emerald-500/20" />
      </div>
    </div>
  </div>
</div>
```

### Background Treatment
```jsx
<div className="background-shapes opacity-40">
  <div className="background-shape">
    {/* Subtle texture using radial gradients */}
    background: radial-gradient(
      circle, 
      rgba(255, 255, 255, 0.03) 0%, 
      transparent 70%
    );
  </div>
</div>
```

## Usage Guidelines

1. Information Architecture
   - Group related metrics
   - Use consistent spacing (gap-6)
   - Maintain clear visual hierarchy
   - Keep interface density balanced

2. Interaction Design
   - Subtle hover states
   - Meaningful animations
   - Progressive disclosure where appropriate
   - Clear feedback on interaction

3. Accessibility Considerations
   - Maintain high contrast ratios
   - Use semantic HTML
   - Ensure keyboard navigation
   - Provide alternative text for visual elements

4. Performance
   - Use opacity and transform for animations
   - Implement efficient blur effects
   - Optimize gradient renders
   - Consider reducing effects on lower-end devices

## Implementation Notes

This design system is built with:
- TailwindCSS for utility classes
- React for component architecture
- CSS custom properties for theming
- Modern CSS features (backdrop-filter, etc.)

Remember to:
- Keep components modular
- Document prop interfaces
- Maintain consistent naming
- Follow accessibility best practices

This system prioritizes clear data presentation while maintaining visual interest through subtle depth and interaction rather than color.