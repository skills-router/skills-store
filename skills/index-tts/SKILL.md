---
name: index-tts
description: Use when the user wants to install, set up, run, or troubleshoot IndexTTS / IndexTTS2 for text-to-speech, zero-shot voice cloning, emotion-controlled speech synthesis, or duration-controlled speech generation. Prefer ModelScope as the default model download source, especially when Hugging Face access is slow or blocked.
---

# IndexTTS

Use this skill for the official IndexTTS repository: `https://github.com/index-tts/index-tts`.

Focus on:

- installing and running the official repo
- using `uv` instead of `pip` or `conda`
- defaulting model download to **ModelScope**
- generating speech with a reference speaker audio prompt
- using emotion audio, emotion vectors, or emotion text when needed
- launching the WebUI when the user wants an interactive workflow

## Bundled helper script

This skill includes `./scripts/index_tts.py` for common repo operations.

Prefer these commands when the user wants an operational wrapper instead of raw shell steps:

```bash
python ./skills/index-tts/scripts/index_tts.py setup --repo-dir ./index-tts
python ./skills/index-tts/scripts/index_tts.py gpu-check --repo-dir ./index-tts
python ./skills/index-tts/scripts/index_tts.py webui --repo-dir ./index-tts
python ./skills/index-tts/scripts/index_tts.py infer-basic --repo-dir ./index-tts --speaker ./ref.wav --text "你好，欢迎使用 IndexTTS" --output ./gen.wav
```

The helper script defaults model download to **ModelScope** and prefers `IndexTTS-2`.

## Core rules

1. Prefer the official repo only.
2. Treat `uv` as mandatory. Do not recommend `pip install` or conda-based setup for the repo workflow unless the user explicitly asks for a non-official workaround.
3. Default model download to **ModelScope**:

```bash
uv tool install "modelscope"
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

4. Prefer `IndexTTS-2` unless the user explicitly asks for another version.
5. If the user wants browser-based interaction, prefer:

```bash
uv run webui.py
```

6. If the user wants scripting or integration, prefer the Python API.

## Recommended setup flow

Use this order unless the user already has a working checkout:

```bash
git clone https://github.com/index-tts/index-tts.git
cd index-tts
git lfs pull
uv sync --all-extras
uv tool install "modelscope"
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
uv run tools/gpu_check.py
```

If the user only wants WebUI and wants to reduce dependency surface, mention the repo's optional extras pattern and keep the answer aligned with the official `uv` workflow.

## Environment requirements

Check these before blaming the model or code:

- `git` installed
- `git-lfs` installed and enabled
- `uv` installed
- NVIDIA CUDA 12.8 or newer for the intended GPU path
- enough disk space for repo + checkpoints

If the user is on macOS or CPU-only hardware, be explicit that the upstream repo emphasizes CUDA GPU usage and that performance or compatibility may not match the recommended setup.

## Default model download guidance

Always present ModelScope first:

```bash
uv tool install "modelscope"
modelscope download --model IndexTeam/IndexTTS-2 --local_dir checkpoints
```

Only mention Hugging Face as a fallback or alternative:

```bash
uv tool install "huggingface-hub[cli,hf_xet]"
hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
```

If the user says Hugging Face is slow, prefer keeping the answer on ModelScope instead of adding mirror complexity.

## Common usage patterns

### 1. Basic zero-shot voice cloning

Use a reference speaker audio file and target text:

```python
from indextts.infer_v2 import IndexTTS2

tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints")
tts.infer(
    spk_audio_prompt="examples/voice_01.wav",
    text="Hello world!",
    output_path="gen.wav",
)
```

### 2. Emotion control with an emotion reference audio

```python
tts.infer(
    spk_audio_prompt="examples/voice_07.wav",
    text="文本内容",
    output_path="gen.wav",
    emo_audio_prompt="examples/emo_sad.wav",
    emo_alpha=0.9,
)
```

### 3. Emotion control with an emotion vector

Vector order:

`[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`

```python
tts.infer(
    spk_audio_prompt="examples/09.wav",
    text="文本",
    output_path="gen.wav",
    emo_vector=[0, 0, 0.8, 0, 0, 0, 0, 0],
)
```

### 4. Emotion control from emotion text

Use this when the user describes a style like “fearful”, “sad”, or “very calm” instead of giving an emotion audio file.

```python
tts.infer(
    spk_audio_prompt="examples/voice_12.wav",
    text="快躲起来！",
    output_path="gen.wav",
    emo_alpha=0.6,
    use_emo_text=True,
)
```

For text-driven emotion, prefer `emo_alpha` around `0.6` unless the user explicitly wants a stronger effect.

## WebUI workflow

Use this when the user wants to test voices interactively:

```bash
uv run webui.py
```

Then open the local Gradio address shown by the repo, commonly `http://127.0.0.1:7860`.

## Troubleshooting rules

When setup fails, check these in order:

1. Was the repo installed with `uv` rather than `pip`/conda?
2. Were large files pulled with `git lfs pull`?
3. Is the model actually present under `checkpoints/`?
4. Does `uv run tools/gpu_check.py` show a usable GPU environment?
5. Is the user trying to run on unsupported or weak hardware?

## Important cautions

- Do not recommend unofficial mirrors or third-party forks as the primary source.
- The upstream repo explicitly warns that non-`uv` installs can cause random bugs and unsupported states.
- `use_random=True` may reduce voice-cloning fidelity.
- DeepSpeed is optional and may improve or worsen performance depending on hardware.
- FP16 is generally the practical default recommendation when the user asks about speed or VRAM tradeoffs.

## Response style

When helping the user:

- give copy-paste-ready commands
- prefer the shortest working path
- default to ModelScope for model download
- use WebUI examples for interactive users
- use Python snippets for integration users
- mention hardware constraints early if they are likely to matter
