---
name: qwen-audio
description: "High-performance audio library with text-to-speech (TTS), speech-to-text (STT), voice design, and voice cloning. Use when Codex needs to transcribe audio, generate speech audio, create reusable voice profiles, or clone a voice from reference audio."
version: 0.0.5
---

# Qwen-Audio

## Overview

Qwen-Audio provides local audio capabilities through `scripts/qwen-audio.py`: text-to-speech, speech-to-text, voice design, and voice cloning.

Always treat the skill root as the runtime project. The Python virtual environment must live at:

```text
<qwen-audio-skill-path>/.venv
```

Do not create a separate `qwen-audio-runtime` directory. Do not run `uv init` for this skill. The installed skill already contains `pyproject.toml`.

## Runtime Setup

Complete setup before using TTS or STT. Run commands from the skill root directory unless a command explicitly says otherwise.

### Windows

Use PowerShell:

```powershell
$SkillDir = "<qwen-audio-skill-path>"
Set-Location $SkillDir
$env:UV_PYPI_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"

if (-not (Test-Path ".venv")) {
  uv venv ".venv" --python 3.12
}

uv sync --prerelease=allow --no-cache
```

If `uv sync` fails, stop and report the exact failing package and error. Do not repeatedly retry with different cache directories. Do not fall back to `pip` unless the user explicitly approves a manual repair.

For Windows GPU detection, run:

```powershell
nvidia-smi
```

If `nvidia-smi` succeeds and prints `NVIDIA-SMI`, keep the CUDA torch source in `pyproject.toml`. If no NVIDIA GPU is available, prefer a CPU-only `pyproject.toml` shape before syncing:

```toml
[project]
name = "qwen3-audio"
version = "0.1.0"
description = "Qwen audio runtime"
requires-python = ">=3.10"
dependencies = [
    "qwen-asr",
    "qwen-tts>=0.1.1",
]

[tool.uv]
extra-index-url = ["https://pypi.org/simple"]
override-dependencies = ["transformers==4.57.6"]
```

### macOS

Use the skill root and install the MLX backend into the skill-local `.venv`:

```bash
cd "<qwen-audio-skill-path>"
export UV_PYPI_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

if [ ! -d .venv ]; then
  uv venv .venv --python 3.12
fi

uv add mlx-audio --prerelease=allow --no-cache
```

### Verify

After setup, verify imports with the skill-local environment:

```bash
uv run --no-sync --project "<qwen-audio-skill-path>" python -c "import sys; print(sys.version)"
```

Windows backend check:

```powershell
uv run --no-sync --project "<qwen-audio-skill-path>" python -c "import qwen_asr, qwen_tts, torch; print('qwen-asr/qwen-tts/torch ok'); print(torch.__version__, torch.cuda.is_available())"
```

macOS backend check:

```bash
uv run --no-sync --project "<qwen-audio-skill-path>" python -c "import mlx_audio; print('mlx-audio ok')"
```

Use `--no-sync` for normal skill commands after setup so Codex does not unexpectedly re-resolve or reinstall dependencies.

## Command Pattern

Use this command shape for all capabilities:

```bash
uv run --no-sync --project "<qwen-audio-skill-path>" python "<qwen-audio-skill-path>/scripts/qwen-audio.py" <command> ...
```

## Voice Management

Voices are stored in `./voices/` at the skill root. Each voice folder contains:

- `ref_audio.wav` - Reference audio file
- `ref_text.txt` - Reference text transcript
- `ref_instruct.txt` - Voice style description

### Create a Voice

Create a reusable voice profile with the VoiceDesign model. `--instruct` is required:

```bash
uv run --no-sync --project "<qwen-audio-skill-path>" python "<qwen-audio-skill-path>/scripts/qwen-audio.py" voice create --text "This is a sample voice reference text." --instruct "A warm, friendly female voice with a professional tone." --id "my-voice-id"
```

Optional: pass `--id "my-voice-id"` to choose a stable voice ID.

### List Voices

```bash
uv run --no-sync --project "<qwen-audio-skill-path>" python "<qwen-audio-skill-path>/scripts/qwen-audio.py" voice list
```

## Text to Speech

### TTS Voice Pre-check

Before `tts` generation:

1. Run `voice list`.
2. If the list is empty, ask the user what kind of voice to create first. Offer concise choices such as warm female narrator, deep male broadcast voice, young energetic neutral voice, or calm professional service voice.
3. If voices exist, show the available `id` values and ask which one to use as `--ref_voice`.

Only run `tts` after the user confirms the voice choice.

```bash
uv run --no-sync --project "<qwen-audio-skill-path>" python "<qwen-audio-skill-path>/scripts/qwen-audio.py" tts --text "hello world" --output "/path/to/save.wav" --ref_voice "my-voice-id"
```

### Voice Cloning

Provide both reference audio and its transcript:

```bash
uv run --no-sync --project "<qwen-audio-skill-path>" python "<qwen-audio-skill-path>/scripts/qwen-audio.py" tts --text "hello world" --output "/path/to/save.wav" --ref_audio "sample_audio.wav" --ref_text "This is what my voice sounds like."
```

## Speech to Text

```bash
uv run --no-sync --project "<qwen-audio-skill-path>" python "<qwen-audio-skill-path>/scripts/qwen-audio.py" stt --audio "/sample_audio.wav" --output "/path/to/save.txt" --output-format txt
```

`--output-format` accepts `txt`, `ass`, `srt`, or `all`.

Test audio:

```text
https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav
```
