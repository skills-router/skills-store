# Session Configuration Schema

Complete schema for video session configuration files.

## Overview

Each session (`session-01/config.json`, etc.) follows this schema:

```typescript
interface SessionConfig {
  title: string;           // Session title
  scripts: ScriptItem[];   // Array of script items
  metadata?: {
    keywords?: string[];   // Keywords
    mood?: string;         // Mood
    pace?: 'slow' | 'medium' | 'fast';  // Pace
  };
}

interface ScriptItem {
  type: CompositionType;           // Composition type
  content: string;                 // Content
  transition?: TransitionType;     // Transition effect
  resources?: string[];            // Resource paths
  narration?: string;              // Voiceover text
  prompts?: string;                // AI prompts for video content generation
  timing?: {
    start?: number;                // Start time
    end?: number;                  // End time
  };
}

type CompositionType =
  | 'title'           // Title screen
  | 'text'            // Text on background
  | 'image'           // Single image
  | 'image-text'      // Image with text overlay
  | 'video'           // Video clip
  | 'code'            // Code snippet
  | 'diagram'         // Diagram/flowchart
  | 'chart'           // Chart/graph
  | 'split'           // Split screen
  | 'list'            // Bullet points
  | 'quote'           // Quote highlight
  | 'transition';     // Transition slide

type TransitionType =
  | 'fade'            // Fade in/out
  | 'slide-left'      // Slide left
  | 'slide-right'     // Slide right
  | 'slide-up'        // Slide up
  | 'slide-down'      // Slide down
  | 'zoom-in'         // Zoom in
  | 'zoom-out'        // Zoom out
  | 'wipe'            // Wipe
  | 'none';           // No transition
```

## Configuration Examples

### Title Screen

```json
{
  "title": "Introduction",
  "scripts": [
    {
      "type": "title",
      "content": "Understanding Machine Learning",
      "transition": "fade",
      "prompts": "Dark gradient background with subtle particle animation, centered title with subtitle below, modern, tech feel, professional"
    }
  ]
}
```

### Image with Voiceover

```json
{
  "title": "Core Concepts",
  "scripts": [
    {
      "type": "image-text",
      "content": "Neural networks are inspired by the human brain",
      "transition": "slide-left",
      "resources": ["assets/neural-network.png"],
      "narration": "Neural networks are computing systems inspired by biological neural networks in the human brain.",
      "prompts": "Neural network diagram with animated connections, image occupies left 60%, text occupies right 40%"
    }
  ]
}
```

### Code Example

```json
{
  "title": "Implementation",
  "scripts": [
    {
      "type": "code",
      "content": "const model = tf.sequential();\nmodel.add(tf.layers.dense({units: 1, inputShape: [1]}));",
      "transition": "fade",
      "narration": "Here's how to create a simple neural network using TensorFlow.js",
      "prompts": "Code editor with syntax highlighting, full-screen code with line numbers, dark theme, large font"
    }
  ]
}
```

### Multi-item Session

```json
{
  "title": "Benefits of Machine Learning",
  "scripts": [
    {
      "type": "list",
      "content": "1. Automation\n2. Prediction\n3. Personalization\n4. Scale",
      "transition": "fade",
      "narration": "Machine learning provides four key benefits for businesses.",
      "prompts": "Animated list items appearing one by one, centered list with icons"
    },
    {
      "type": "image-text",
      "content": "Automation reduces 80% of manual work",
      "duration": 15,
      "transition": "slide-up",
      "resources": ["assets/automation-chart.png"],
      "narration": "Research shows automation can reduce up to 80% of manual tasks.",
      "prompts": "Chart highlighting key data"
    }
  ],
  "metadata": {
    "keywords": ["automation", "efficiency", "scale"],
    "mood": "professional",
    "pace": "medium"
  }
}
```

## Field Guidelines

### title
- Keep short (2-5 words)
- Descriptive for navigation
- Used for Remotion composition naming

### duration
- Sum of all script item durations
- Typical session: 20-60 seconds
- Total video: 3-10 minutes

### scripts[].type
Choose based on content:
- **title**: Opening/closing screens
- **text**: Voiceover-only segments
- **image**: Visual demonstration
- **image-text**: Most common - visual + explanation
- **code**: Technical tutorials
- **diagram**: Process explanations
- **chart**: Data/statistics
- **list**: Point enumeration
- **quote**: Expert quotes, testimonials

### scripts[].transition
- **fade**: Default, works for all
- **slide-***: Directional flow
- **zoom-***: Emphasis changes
- **none**: Instant cut (use sparingly)

### scripts[].narration
- Target 150 words per minute
- Short sentences for clarity
- Natural language, avoid jargon
- Match tone to audience

### scripts[].prompts
Guidance for Remotion implementation:
- **visual**: What to show
- **layout**: How to arrange
- **style**: Aesthetic direction

## Validation

Before generating video, validate:
1. Total duration matches sum of script items
2. All resource paths exist
3. Voiceover length fits duration (150 words/min)
4. Transition effects are compatible
5. No overlapping times

## Best Practices

1. **Strong Opening**: First session grabs attention
2. **Pacing Variation**: Mix fast and slow segments
3. **Visual Diversity**: Don't repeat same type too often
4. **Voiceover Flow**: Read aloud to check naturalness
5. **Timing Buffers**: Add 0.5 seconds between major transitions
