# Skills Store

A collection of reusable skills for AI assistants.

## Available Skills

### free-resource

Search and download royalty-free images, videos, sound effects, and music from Pixabay, Freesound, and Jamendo.

**Features:**
- Pixabay: Images and videos search & download
- Freesound: Audio effects search & download
- Jamendo: Music/BGM search & download

**Quick Start:**
```bash
cd skills/free-resource
cp config.example.json config.json
# Edit config.json with your API keys
bun ./scripts/pixabay.ts search-images --query "nature"
```

### qwen3-audio

High-performance audio processing library optimized

**Features:**
- Text-to-Speech (TTS) with voice cloning
- Speech-to-Text (STT) / Automatic Speech Recognition

**Prerequisites:**
- Python 3.10+

**Quick Start:**
```bash
uv run --project "/<qwen-audio-skill-path>" python "./scripts/qwen-audio.py" tts --text "hello world" --output "output.wav"
```

## Structure

```
skills-store/
├── skills/
│   ├── free-resource/     # Royalty-free media search
│   │   ├── SKILL.md       # Skill documentation
│   │   ├── scripts/       # CLI tools
│   │   └── config.json    # API keys configuration
│   └── qwen-audio/       # Audio processing (TTS/STT)
│       ├── SKILL.md       # Skill documentation
│       ├── scripts/       # Python scripts
│       └── .venv/         # Python virtual environment
└── LICENSE
```

## License

[MIT](LICENSE)
