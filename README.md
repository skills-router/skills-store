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

High-performance audio processing library optimized for Apple Silicon (M1/M2/M3/M4).

**Features:**
- Text-to-Speech (TTS) with voice cloning
- Speech-to-Text (STT) / Automatic Speech Recognition
- Voice design from text descriptions
- Emotion/style control with predefined speakers

**Prerequisites:**
- Python 3.10+
- Apple Silicon Mac (M1/M2/M3/M4)

**Quick Start:**
```bash
cd skills/qwen3-audio
uv run --python ".venv/bin/python" "./scripts/mlx-audio.py" tts --text "hello world" --output "output.wav"
```

## Structure

```
skills-store/
├── skills/
│   ├── free-resource/     # Royalty-free media search
│   │   ├── SKILL.md       # Skill documentation
│   │   ├── scripts/       # CLI tools
│   │   └── config.json    # API keys configuration
│   └── qwen3-audio/       # Audio processing (TTS/STT)
│       ├── SKILL.md       # Skill documentation
│       ├── scripts/       # Python scripts
│       └── .venv/         # Python virtual environment
└── LICENSE
```

## License

[MIT](LICENSE)
