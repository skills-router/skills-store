# Research Workflow

Detailed guidance for the deep research phase of content-to-video conversion.

## Content Source Processing

### URLs and Articles

1. **Fetch Content**:
   - If the user doesn't provide specific URLs and article links, you need to deeply explore various platforms and use WebSearch to gather supporting materials
   - Use WebFetch tool to extract main content
   - Remove ads, navigation, footers
   - Extract author, date, publication

2. **Key Information**:
   - Main arguments/thesis
   - Supporting points (3-5 key points)
   - Quotes and data
   - Images and charts

3. **Save to** `deep-research/content/`:
   - `article.md` - Cleaned markdown
   - `metadata.json` - Title, author, URL, date
   - `highlights.md` - Key excerpts

### Academic Papers

1. **Parse PDF**:
   - Extract text and structure
   - Identify sections: Abstract, Introduction, Methods, Results, Conclusion

2. **Key Information**:
   - Research question
   - Methods summary
   - Main findings (3-5 points)
   - Charts and figures
   - Citations

3. **Visualization**:
   - Reconstruct charts as SVG/React components
   - Generate explanatory diagrams
   - Screenshot important figures

4. **Save to** `deep-research/content/`:
   - `paper.md` - Extracted text
   - `summary.md` - Condensed summary
   - `figures/` - Extracted images

### GitHub Repositories

1. **Analyze Structure**:
   - README content
   - Main code files
   - Package dependencies
   - Architecture patterns

2. **Key Information**:
   - Project purpose
   - Main features (3-5)
   - Code examples
   - Installation/usage

3. **Visualization**:
   - Directory tree diagrams
   - Architecture diagrams
   - Code flow diagrams

4. **Save to** `deep-research/content/`:
   - `readme.md` - README content
   - `analysis.md` - Code analysis
   - `examples/` - Code snippets

## Resource Collection

### Sources

### Additional Resource Generation and Creation

- Use AgentBrowser to navigate web pages (GitHub, product homepages, papers), record videos from source pages (scroll to bottom, or select important text), see agent-browser skill for details
- You can use scripts to capture important charts from papers or web pages and save them, or download related videos
- Search for comments from platforms like x.com posts, Xiaohongshu, Reddit, etc., and generate corresponding HTML or SVG to convert to images
- You can generate various SVG animations for video use, such as formulas, charts, Lottie, etc.
- Mermaid, Excalidraw for explanations

### Organization

Save to `deep-research/images/`:
```
images/
├── downloaded/
│   ├── figure-1.png
│   └── screenshot-1.png
├── generated/
│   ├── concept-1.png
│   └── illustration-1.png
└── metadata.json
```

### Metadata Tracking

```json
{
  "images": [
    {
      "filename": "figure-1.png",
      "source": "downloaded",
      "url": "https://...",
      "attribution": "Whether attribution is required",
      "session_usage": "session-01"
    }
  ]
}
```

## BGM Research

### Sources

- BGM options (search and download, see free-resource skill for details)

### Selection Criteria

1. **Mood**: Match video tone (upbeat, calm, dramatic)
2. **Tempo**: Align with voiceover pacing
3. **Duration**: Or loopable sections
4. **Licensing**: Verify commercial use if needed

### Save to `deep-research/bgm/`:
```
bgm/
├── track-1.mp3
├── track-2.mp3
└── options.json
```

## Research Notes

Maintain `deep-research/notes.md` containing:

```markdown
# Research Notes: [Topic]

## Main Argument
[1-2 sentences]

## Key Points
1. Point 1
   - Supporting details
   - Visual suggestions
2. Point 2
   ...

## Target Audience
[Description]

## Video Approach
[Style, tone, format]

## Session Ideas
- Session 1: Introduction - Cover main argument
- Session 2: Deep dive - Points 1 & 2
- ...

## Questions/Unknowns
- [Things needing clarification]
```

## Token Management

**Problem**: Research can consume many tokens.

**Solutions**:
1. **Incremental Saving**: Write to files frequently
2. **Summarization**: Create condensed versions early
3. **Selective Loading**: Only load content needed for current step
4. **External Tools**: Use Task agents for parallel research

## Quality Checklist

Before moving to session planning:
- [ ] Main content extracted and saved
- [ ] 3-5 key points identified
- [ ] 10+ potential images collected
- [ ] 2-3 BGM options available
- [ ] Research notes completed
- [ ] Target audience defined
- [ ] Video style determined
