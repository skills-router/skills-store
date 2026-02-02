检查以下项目, 每次完成后请修改该文件,标记完 [x] 完成
- [ ] 检查在该技能目录中是否创建了uv环境.venv, python版本需要>=3.10, 如果没有则创建uv环境 `uv sync --prerelease=allow`
- [ ] 检查是否安装mlx-audio, 如无则安装mlx-audio 执行 `uv add mlx-audio --prerelease=allow`
- [ ] 检查模型是否已下载到huggingface缓存目录(需要查看环境变量中是否设置了HF_HOME,默认路径为"~/.cache/huggingface/hub"),否则模型下载 "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16", "mlx-community/Qwen3-ASR-0.6B-bf16", "mlx-community/Qwen3-ForcedAligner-0.6B-8bit", "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"
- [ ] tts测试是否正常
- [ ] stt测试是否正常
