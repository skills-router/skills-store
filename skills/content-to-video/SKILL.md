---
name: content-to-video
description: Convert articles, papers, GitHub repositories, and various content sources into professional videos. Use when users want to transform text content (URLs, documents, code repositories) into video format. Handles research, scriptwriting, scene planning, and Remotion-based video generation. Triggers on requests like "convert this article to video", "create a video from this paper", or "make a video about this GitHub project".
---

# Content to Video

Convert various content sources (articles, papers, GitHub repositories, etc.) into professional videos with structured research, scriptwriting, and Remotion-based generation workflow.

## Workflow

### 1. Gather Requirements

You need to first check the remotion-best-practices skill, which is the core skill for video generation

Ask the user:
- **Content Source**: URL, file path, or topic description
- **Target Audience**: Developers, general public, students, etc.
- **Video Style**: Educational, promotional, documentary, tutorial
- **Visual Style**: Apple keynote style (minimalist tech), Tech startup pitch style (TED / startup launch), Internet product promo style (Figma / Notion style), Documentary style (Netflix style), Fast-paced short video style (TikTok / Reals style)
- **Format**: Landscape (16:9), Portrait (9:16), Square (1:1)
- **Language**: For voiceover and subtitles, whether multiple language versions of the video are needed

Create project directory: `<topic-slug>/`
Write all user requirements to `<topic-slug>/video-style.md`

### 2. Deep Research

Create `<topic-slug>/deep-research/` directory to store:
- Source content (downloaded articles, scraped web pages)
- Resources extracted from source content: images, videos (downloaded, AI-generated, screenshots)
- Additional resource generation and creation
- BGM options (search and download, see free-resource skill for details)
- Key insights and data points
- Finally generate a Readme.md for this directory for later use

**Processing by Source Type:**
- **URLs**: Fetch full content, extract key points, download related images
- **Papers**: Parse PDF, extract abstract/methods/results, generate visualizations
- **GitHub**: Analyze repository structure, README, code highlights, generate diagrams

See [references/research-workflow_CN.md](references/research-workflow_CN.md) for details.

### 3. Plan Video Structure

Create `<topic-slug>/video-sessions/` directory with subdirectories:
- `session-01/`, `session-02/`, etc.

Each session contains:
- `config.json` - Session configuration (see schema below)
- `assets/` - Session-specific images, videos, audio

**Session config.json schema:**
```json
{
  "title": "Session Title",
  "scripts": [
    {
      "type": "composition_type",
      "content": "Text or content description",
      "transition": "fade",
      "resources": ["path/to/image.png", "path/to/video.mp4"],
      "narration": "Voiceover text",
      "prompts": "AI visual layout prompts, style to display, elements and text to generate, detailed prompts for resources used"
    }
  ]
}
```

See [references/session-config-schema_CN.md](references/session-config-schema_CN.md) for full schema.

### 4. User Review

Present complete video structure:
- Total duration and session breakdown
- Session titles and key points
- Sample voiceover text
- Expected visual style

Get user approval before proceeding. Allow iteration on:
- Session order and content
- Voiceover tone and length
- Visual style preferences

### 5. Generate with Remotion

Create `<topic-slug>/remotion-video/` directory:
- Initialize Remotion project
- Create session compositions: `src/compositions/session-01/`, etc.
- Generate audio voiceover (TTS or user-provided), use qwen3-audio skill, generated audio returns duration, note to reserve blank transition time before and after each session switch, must wait for audio to finish before playing next session or animation
- Add assets to `public/assets/`
- Configure transitions and effects

**Session Structure:**
```
session-01/
├── index.tsx          - Main composition
├── components/        - Session-specific components
├── assets/           - Audio, images
└── config.ts         - Session configuration
```

See [references/remotion-patterns_CN.md](references/remotion-patterns_CN.md) for common patterns.



## Directory Structure

Final project structure:
```
<topic-slug>/
├── deep-research/
│   ├── content/
│   ├── images/
│   ├── bgm/
│   └── notes.md
├── video-sessions/
│   ├── session-01/
│   │   ├── config.json
│   │   └── assets/
│   ├── session-02/
│   └── ...
├── remotion-video/
│   ├── src/
│   ├── public/
│   └── package.json
└── output/
    └── final-video.mp4
```

## Key Considerations

- **Token Efficiency**: Research phase may consume many tokens. Use incremental saving and summarization.
- **Parallel Work**: Can prepare Remotion templates while researching.
- **Iteration**: Expect 2-3 rounds of user feedback on scripts before video generation.
- **Quality vs Speed**: Ask user preference upfront - quick prototype or polished production.
