from __future__ import annotations

import argparse
import ipaddress
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = "downloads/Lance_3B_Video"
DEFAULT_RESOLUTION = "video_480p"


def _sanitize_no_proxy_for_httpx() -> None:
    for env_key in ("no_proxy", "NO_PROXY"):
        value = os.environ.get(env_key)
        if not value:
            continue
        items = []
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if "://" not in item and item.startswith("["):
                end = item.find("]")
                if end > 1:
                    host = item[1:end]
                    rest = item[end + 1 :]
                    try:
                        if ipaddress.ip_address(host).version == 6 and (
                            not rest or (rest.startswith(":") and rest[1:].isdigit())
                        ):
                            item = host
                    except ValueError:
                        pass
            if ":" in item:
                continue
            items.append(item)
        os.environ[env_key] = ",".join(items)


_sanitize_no_proxy_for_httpx()

import gradio as gr


def _write_config(task: str, prompt: str, image_path: str) -> Path:
    config_dir = REPO_ROOT / "tmp" / "gradio_ff2v_idip"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{task}_{int(time.time() * 1000)}.json"
    sample = {
        "0001": {
            "interleave_array": [prompt, image_path],
            "element_dtype_array": ["text", "image"],
            "istarget_in_interleave": [0, 0],
        }
    }
    config_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    return config_path


def _latest_video(save_dir: Path, started_at: float) -> Optional[Path]:
    if not save_dir.exists():
        return None
    videos = [
        path
        for path in save_dir.rglob("*.mp4")
        if path.is_file() and path.stat().st_mtime >= started_at - 1
    ]
    if not videos:
        videos = [path for path in save_dir.rglob("*.mp4") if path.is_file()]
    if not videos:
        return None
    return max(videos, key=lambda path: path.stat().st_mtime)


def _run_inference(
    *,
    task: str,
    prompt: str,
    image_path: str,
    save_path: str,
    model_path: str,
    resolution: str,
    num_frames: int,
    video_height: Optional[int] = None,
    video_width: Optional[int] = None,
) -> tuple[Optional[str], str]:
    prompt = (prompt or "").strip()
    if not prompt:
        return None, "Please enter a prompt."
    if not image_path:
        return None, "Please upload an input image."
    if num_frames <= 0:
        return None, "NUM_FRAMES must be greater than 0."

    save_dir = (REPO_ROOT / save_path).resolve() if not Path(save_path).is_absolute() else Path(save_path)
    config_path = _write_config(task, prompt, image_path)

    cmd = [
        "bash",
        "inference_lance.sh",
        "--TASK_NAME",
        task,
        "--MODEL_PATH",
        model_path,
        "--RESOLUTION",
        resolution,
        "--NUM_FRAMES",
        str(int(num_frames)),
        "--SAVE_PATH_GEN",
        str(save_dir),
        "--CONFIG_PATH",
        str(config_path),
    ]
    if task == "ti2v":
        if not video_height or not video_width:
            return None, "VIDEO_HEIGHT and VIDEO_WIDTH are required for IDIP."
        cmd.extend(["--VIDEO_HEIGHT", str(int(video_height)), "--VIDEO_WIDTH", str(int(video_width))])

    started_at = time.time()
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    output = "\n".join(part for part in [result.stdout, result.stderr] if part).strip()
    if result.returncode != 0:
        return None, output[-8000:] or f"Inference failed with exit code {result.returncode}."

    video_path = _latest_video(save_dir, started_at)
    if video_path is None:
        return None, (output[-8000:] + "\n\nNo generated mp4 found.").strip()
    return str(video_path), output[-8000:]


def run_ff2v(prompt: str, image_path: str, num_frames: int, save_path: str, model_path: str, resolution: str):
    return _run_inference(
        task="ff2v",
        prompt=prompt,
        image_path=image_path,
        save_path=save_path,
        model_path=model_path,
        resolution=resolution,
        num_frames=int(num_frames),
    )


def run_idip(
    prompt: str,
    image_path: str,
    num_frames: int,
    video_height: int,
    video_width: int,
    save_path: str,
    model_path: str,
    resolution: str,
):
    return _run_inference(
        task="ti2v",
        prompt=prompt,
        image_path=image_path,
        save_path=save_path,
        model_path=model_path,
        resolution=resolution,
        num_frames=int(num_frames),
        video_height=int(video_height),
        video_width=int(video_width),
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Lance FF2V / IDIP") as demo:
        gr.Markdown("# Lance FF2V / IDIP")
        with gr.Tabs():
            with gr.Tab("FF2V"):
                ff_prompt = gr.Textbox(label="Prompt", lines=5)
                ff_image = gr.Image(label="First frame", type="filepath")
                ff_frames = gr.Slider(label="NUM_FRAMES", minimum=1, maximum=121, step=1, value=50)
                ff_save = gr.Textbox(label="SAVE_PATH_GEN", value="results/ti2v_test/ff2v")
                ff_model = gr.Textbox(label="MODEL_PATH", value=DEFAULT_MODEL_PATH)
                ff_resolution = gr.Textbox(label="RESOLUTION", value=DEFAULT_RESOLUTION)
                ff_button = gr.Button("Generate FF2V", variant="primary")
                ff_video = gr.Video(label="Result")
                ff_log = gr.Textbox(label="Log", lines=12)
                ff_button.click(
                    run_ff2v,
                    inputs=[ff_prompt, ff_image, ff_frames, ff_save, ff_model, ff_resolution],
                    outputs=[ff_video, ff_log],
                )

            with gr.Tab("IDIP"):
                idip_prompt = gr.Textbox(label="Prompt", lines=5)
                idip_image = gr.Image(label="Reference image", type="filepath")
                idip_frames = gr.Slider(label="NUM_FRAMES", minimum=1, maximum=121, step=1, value=50)
                with gr.Row():
                    idip_height = gr.Number(label="VIDEO_HEIGHT", value=480, precision=0)
                    idip_width = gr.Number(label="VIDEO_WIDTH", value=848, precision=0)
                idip_save = gr.Textbox(label="SAVE_PATH_GEN", value="results/ti2v_test/idip2")
                idip_model = gr.Textbox(label="MODEL_PATH", value=DEFAULT_MODEL_PATH)
                idip_resolution = gr.Textbox(label="RESOLUTION", value=DEFAULT_RESOLUTION)
                idip_button = gr.Button("Generate IDIP", variant="primary")
                idip_video = gr.Video(label="Result")
                idip_log = gr.Textbox(label="Log", lines=12)
                idip_button.click(
                    run_idip,
                    inputs=[
                        idip_prompt,
                        idip_image,
                        idip_frames,
                        idip_height,
                        idip_width,
                        idip_save,
                        idip_model,
                        idip_resolution,
                    ],
                    outputs=[idip_video, idip_log],
                )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()
    build_demo().queue().launch(server_name=args.server_name, server_port=args.server_port)


if __name__ == "__main__":
    main()
