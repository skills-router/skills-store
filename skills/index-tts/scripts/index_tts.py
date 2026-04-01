#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_REPO_URL = "https://github.com/index-tts/index-tts.git"
DEFAULT_MODEL = "IndexTeam/IndexTTS-2"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_repo(repo_dir: Path, repo_url: str) -> None:
    if repo_dir.exists():
        git_dir = repo_dir / ".git"
        if not git_dir.exists():
            raise RuntimeError(f"repo_dir exists but is not a git repo: {repo_dir}")
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", repo_url, str(repo_dir)])


def setup_repo(repo_dir: Path, repo_url: str, model: str, skip_sync: bool, skip_download: bool) -> None:
    ensure_repo(repo_dir, repo_url)
    run(["git", "lfs", "pull"], cwd=repo_dir)
    if not skip_sync:
        run(["uv", "sync", "--all-extras"], cwd=repo_dir)
    run(["uv", "tool", "install", "modelscope"])
    if not skip_download:
        run(["modelscope", "download", "--model", model, "--local_dir", "checkpoints"], cwd=repo_dir)


def gpu_check(repo_dir: Path) -> None:
    run(["uv", "run", "tools/gpu_check.py"], cwd=repo_dir)


def launch_webui(repo_dir: Path, host: str, port: int) -> None:
    run(["uv", "run", "webui.py", "--server-name", host, "--server-port", str(port)], cwd=repo_dir)


def infer_basic(repo_dir: Path, speaker: Path, text: str, output: Path, config: str, model_dir: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    python_code = "\n".join(
        [
            "from indextts.infer_v2 import IndexTTS2",
            f'tts = IndexTTS2(cfg_path={json.dumps(config)}, model_dir={json.dumps(model_dir)})',
            "tts.infer(",
            f'    spk_audio_prompt={json.dumps(str(speaker))},',
            f'    text={json.dumps(text)},',
            f'    output_path={json.dumps(str(output))},',
            ")",
        ]
    )
    run(["uv", "run", "python", "-c", python_code], cwd=repo_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Helper for the official IndexTTS repo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser("setup", help="clone repo, sync deps, and download model via ModelScope")
    setup_parser.add_argument("--repo-dir", default="./index-tts", help="Path to the IndexTTS repo checkout")
    setup_parser.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="Official IndexTTS git URL")
    setup_parser.add_argument("--model", default=DEFAULT_MODEL, help="ModelScope model id")
    setup_parser.add_argument("--skip-sync", action="store_true", help="Skip uv sync")
    setup_parser.add_argument("--skip-download", action="store_true", help="Skip modelscope download")

    gpu_parser = subparsers.add_parser("gpu-check", help="run the upstream GPU check")
    gpu_parser.add_argument("--repo-dir", default="./index-tts", help="Path to the IndexTTS repo checkout")

    webui_parser = subparsers.add_parser("webui", help="launch IndexTTS WebUI")
    webui_parser.add_argument("--repo-dir", default="./index-tts", help="Path to the IndexTTS repo checkout")
    webui_parser.add_argument("--host", default="127.0.0.1", help="WebUI host")
    webui_parser.add_argument("--port", type=int, default=7860, help="WebUI port")

    infer_parser = subparsers.add_parser("infer-basic", help="run a basic zero-shot inference")
    infer_parser.add_argument("--repo-dir", default="./index-tts", help="Path to the IndexTTS repo checkout")
    infer_parser.add_argument("--speaker", required=True, help="Reference speaker wav path")
    infer_parser.add_argument("--text", required=True, help="Text to synthesize")
    infer_parser.add_argument("--output", required=True, help="Output wav path")
    infer_parser.add_argument("--config", default="checkpoints/config.yaml", help="Config path inside repo")
    infer_parser.add_argument("--model-dir", default="checkpoints", help="Model directory inside repo")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "setup":
            setup_repo(
                repo_dir=Path(args.repo_dir).resolve(),
                repo_url=args.repo_url,
                model=args.model,
                skip_sync=args.skip_sync,
                skip_download=args.skip_download,
            )
        elif args.command == "gpu-check":
            gpu_check(Path(args.repo_dir).resolve())
        elif args.command == "webui":
            launch_webui(Path(args.repo_dir).resolve(), args.host, args.port)
        elif args.command == "infer-basic":
            infer_basic(
                repo_dir=Path(args.repo_dir).resolve(),
                speaker=Path(args.speaker).resolve(),
                text=args.text,
                output=Path(args.output).resolve(),
                config=args.config,
                model_dir=args.model_dir,
            )
        else:
            parser.error(f"unknown command: {args.command}")
    except subprocess.CalledProcessError as exc:
        print(f"command failed with exit code {exc.returncode}", file=sys.stderr)
        return exc.returncode or 1
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
