from __future__ import annotations

import html
import json
import os
import random
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import gradio as gr

from .settings import *

def get_aspect_ratio_choices_for_task(task: str) -> list[tuple[str, str]]:
    """Get Aspect Ratio choices with default/recommended marker for the given task."""
    internal_task = normalize_task(task)
    default_ratio = DEFAULT_IMAGE_ASPECT_RATIO if internal_task in IMAGE_TASKS else DEFAULT_VIDEO_ASPECT_RATIO
    return [
        (f"{ratio}" if ratio == default_ratio else ratio, ratio)
        for ratio in ASPECT_RATIO_CHOICES
    ]

def get_video_duration_choices() -> list[tuple[str, int]]:
    return [(f"{seconds}s", seconds) for seconds in range(1, 11)]

def display_path(path: Path) -> str:
    path_text = path.as_posix()
    if path.is_absolute():
        try:
            path_text = path.relative_to(Path.cwd()).as_posix()
        except ValueError:
            return path_text
    if path_text == "." or path_text.startswith("./"):
        return path_text
    return f"./{path_text}"

def get_model_base_dir() -> Path:
    """Return the local model directory only.

    Local-only mode never selects remote storage and never downloads
    model assets from remote repositories. Override with LANCE_MODEL_BASE_DIR
    when your local weights live somewhere else.
    """
    configured = os.getenv("LANCE_MODEL_BASE_DIR")
    return Path(configured).expanduser() if configured else LOCAL_MODEL_BASE_DIR

def normalize_model_variant(model_variant: Optional[str] = None) -> str:
    variant = (model_variant or os.getenv("LANCE_MODEL_VARIANT", DEFAULT_MODEL_VARIANT)).strip().lower()
    if variant in {"image", "t2i", "i2t"}:
        return MODEL_VARIANT_IMAGE
    return MODEL_VARIANT_VIDEO

def get_model_path(model_variant: Optional[str] = None) -> Path:
    variant = normalize_model_variant(model_variant)
    variant_env_name = "LANCE_IMAGE_MODEL_PATH" if variant == MODEL_VARIANT_IMAGE else "LANCE_VIDEO_MODEL_PATH"
    variant_configured = os.getenv(variant_env_name)
    if variant_configured:
        return Path(variant_configured).expanduser()

    configured = os.getenv("LANCE_MODEL_PATH")
    if configured:
        return Path(configured).expanduser()

    model_dir_name = MODEL_VARIANT_TO_DIR[variant]
    return get_model_base_dir() / model_dir_name

def get_required_model_asset_paths(model_base_dir: Path, model_path: Path) -> list[Path]:
    return [
        model_path / "llm_config.json",
        model_path / "model.safetensors",
        model_base_dir / "Qwen2.5-VL-ViT" / "vit.safetensors",
        model_base_dir / "Wan2.2_VAE.pth",
    ]

def ensure_model_assets(model_variant: Optional[str] = None) -> Path:
    """Verify that all required model assets exist locally.

    Expected layout by default:
      downloads/
        Lance_3B_Video/
        Lance_3B/
        Qwen2.5-VL-ViT/
        Wan2.2_VAE.pth

    Set LANCE_MODEL_BASE_DIR, LANCE_MODEL_PATH, LANCE_VIDEO_MODEL_PATH or
    LANCE_IMAGE_MODEL_PATH to point at local files. No remote download is
    attempted.
    """
    model_base_dir = get_model_base_dir()
    os.environ["LANCE_MODEL_BASE_DIR"] = display_path(model_base_dir)
    model_path = get_model_path(model_variant)
    required_paths = get_required_model_asset_paths(model_base_dir, model_path)

    if all(path.exists() for path in required_paths):
        return model_path

    missing = "\n".join(f"- {display_path(path)}" for path in required_paths if not path.exists())
    raise FileNotFoundError(
        "Local Lance model assets are missing. This local-only build does not "
        "download from remote repositories. Set LANCE_MODEL_BASE_DIR "
        "or the model path environment variables to your local weights.\n"
        f"Missing files:\n{missing}"
    )

def ensure_dirs() -> None:
    TMP_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

def save_generation_record(record: dict, save_dir: Path) -> None:
    ensure_dirs()
    run_record_path = save_dir / RUN_RECORD_FILENAME
    with run_record_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    with RECORD_WRITE_LOCK:
        with GLOBAL_RECORDS_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def normalize_seed(seed: int) -> int:
    return random.randint(0, 2**31 - 1) if seed == -1 else seed

def video_seconds_to_num_frames(seconds: int) -> int:
    seconds = max(1, min(10, int(seconds)))
    return 12 * seconds + 1

def normalize_task(task: str) -> str:
    task_key = (task or TASK_LABEL_VIDEO_GENERATION).strip()
    task = TASK_LABEL_TO_INTERNAL.get(task_key, TASK_LABEL_TO_INTERNAL.get(task_key.lower(), ""))
    if task not in GENERATION_TASKS | UNDERSTANDING_TASKS:
        raise ValueError(f"Unsupported task type: {task}")
    return task

def normalize_resolution_choice_value(resolution: str, task: str) -> str:
    resolution_text = str(resolution or "").strip()
    for choice in get_resolution_choices_for_task(task):
        if isinstance(choice, tuple):
            label, value = choice
            if resolution_text in {str(label), str(value)}:
                return str(value)
        elif resolution_text == str(choice):
            return str(choice)
    return resolution_text

def get_resolution_choice_values_for_task(task: str) -> list[str]:
    choices = get_resolution_choices_for_task(task)
    values = []
    for choice in choices:
        values.append(choice[1] if isinstance(choice, tuple) else choice)
    return values

def get_resolution_choices_for_task(task: str) -> list[str | tuple[str, str]]:
    internal_task = normalize_task(task)
    if internal_task in IMAGE_TASKS:
        return IMAGE_RESOLUTION_CHOICES
    if internal_task == TASK_T2V:
        return VIDEO_RESOLUTION_DISPLAY_CHOICES
    if internal_task == TASK_VIDEO_EDIT:
        return VIDEO_EDIT_RESOLUTION_CHOICES
    if internal_task in VIDEO_TASKS:
        return VIDEO_EDIT_RESOLUTION_CHOICES
    return VIDEO_RESOLUTION_CHOICES

def get_default_resolution_for_task(task: str) -> str:
    internal_task = normalize_task(task)
    if internal_task in IMAGE_TASKS:
        return DEFAULT_IMAGE_RESOLUTION
    # Video Generation should default to the lightweight/recommended 360p profile.
    # This is used by both task switching and recommended-case click handlers
    # through reset_generation_defaults_for_task(), so every Video Generation
    # example fill now returns video_360p instead of falling through to 480p.
    if internal_task == TASK_T2V:
        return DEFAULT_RESOLUTION
    if internal_task == TASK_VIDEO_EDIT:
        return DEFAULT_VIDEO_EDIT_RESOLUTION
    if internal_task in VIDEO_TASKS:
        return DEFAULT_VIDEO_EDIT_RESOLUTION
    return DEFAULT_RESOLUTION

def normalize_resolution_for_backend(resolution: str, task: str) -> str:
    internal_task = normalize_task(task)
    normalized_resolution = normalize_resolution_choice_value(resolution, internal_task)
    choices = get_resolution_choice_values_for_task(internal_task)
    if normalized_resolution in choices:
        return normalized_resolution
    return get_default_resolution_for_task(internal_task)

def get_default_aspect_ratio(task: str) -> str:
    internal_task = normalize_task(task)
    return DEFAULT_IMAGE_ASPECT_RATIO if internal_task in IMAGE_TASKS else DEFAULT_VIDEO_ASPECT_RATIO

def normalize_video_resolution(resolution: Optional[str], task: Optional[str] = None) -> str:
    if task is None:
        return resolution if resolution in VIDEO_RESOLUTION_CHOICES else DEFAULT_RESOLUTION
    normalized_resolution = normalize_resolution_choice_value(resolution, task)
    choices = get_resolution_choice_values_for_task(task)
    return normalized_resolution if normalized_resolution in choices else get_default_resolution_for_task(task)

def get_size_for_aspect_ratio(task: str, aspect_ratio: str, video_resolution: Optional[str] = None) -> tuple[int, int]:
    internal_task = normalize_task(task)
    aspect_ratio = aspect_ratio if aspect_ratio in ASPECT_RATIO_CHOICES else get_default_aspect_ratio(internal_task)
    if internal_task in IMAGE_TASKS:
        size_map = IMAGE_ASPECT_RATIO_TO_SIZE
    else:
        size_map = VIDEO_RESOLUTION_TO_SIZE_MAP[normalize_video_resolution(video_resolution, internal_task)]
    return size_map[aspect_ratio]

def format_size_markdown(task: str, width: int, height: int) -> str:
    internal_task = normalize_task(task)
    if internal_task in UNDERSTANDING_TASKS:
        return ""
    #return f"**Output Resolution:** `{width} x {height}`"
    return f"{width} x {height}"

def get_size_map_for_task(task: str, video_resolution: Optional[str] = None) -> dict[str, tuple[int, int]]:
    internal_task = normalize_task(task)
    if internal_task in IMAGE_TASKS:
        return IMAGE_ASPECT_RATIO_TO_SIZE
    return VIDEO_RESOLUTION_TO_SIZE_MAP[normalize_video_resolution(video_resolution, internal_task)]

def get_output_resolution_choices_for_task(task: str, video_resolution: Optional[str] = None) -> list[tuple[str, str]]:
    """Get Output Resolution choices with a one-to-one mapping to aspect ratios."""
    internal_task = normalize_task(task)
    default_ratio = get_default_aspect_ratio(internal_task)
    size_map = get_size_map_for_task(internal_task, video_resolution)
    choices = []
    for ratio in ASPECT_RATIO_CHOICES:
        width, height = size_map[ratio]
        resolution_text = format_size_markdown(internal_task, width, height)
        label = f"{resolution_text}" if ratio == default_ratio else resolution_text
        choices.append((label, resolution_text))
    return choices

def build_lance_label_html(text: str, *extra_classes: str) -> str:
    class_names = " ".join(["lance-section-label", *extra_classes]).strip()
    return f'<div class="{class_names}">{html.escape(text)}</div>'

def build_lance_icon_label_html(text: str, icon: str, *extra_classes: str) -> str:
    icon_map = {
        "video": """
            <span class="lance-label-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3.5" y="6" width="11" height="12" rx="2.2"></rect>
                    <path d="M15 10.2 20.5 7v10L15 13.8z" fill="currentColor" stroke="none"></path>
                </svg>
            </span>
        """,
        "image": """
            <span class="lance-label-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3.5" y="5.5" width="17" height="13" rx="2.2"></rect>
                    <circle cx="9" cy="10" r="1.5" fill="currentColor" stroke="none"></circle>
                    <path d="M5.5 16.5 10 12l2.7 2.7 2.1-2.1 3.7 3.9"></path>
                </svg>
            </span>
        """,
        "text": """
            <span class="lance-label-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3.5" y="5.5" width="17" height="13" rx="2.2"></rect>
                    <path d="M7 9h10"></path>
                    <path d="M7 12h7.5"></path>
                    <path d="M7 15h5.5"></path>
                </svg>
            </span>
        """,
        "logs": """
            <span class="lance-label-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3.5" y="5.5" width="17" height="13" rx="2.2"></rect>
                    <path d="M7 10.2 10 12l-3 1.8"></path>
                    <path d="M12.5 15h4"></path>
                </svg>
            </span>
        """,
    }
    icon_html = icon_map.get(icon, "")
    class_names = " ".join(["lance-section-label", "lance-icon-label", *extra_classes]).strip()
    return f'<div class="{class_names}">{icon_html}<span class="lance-output-label-text">{html.escape(text)}</span></div>'

def update_size_from_aspect_ratio(task: str, aspect_ratio: str, video_resolution: Optional[str] = None):
    width, height = get_size_for_aspect_ratio(task, aspect_ratio, video_resolution)
    return height, width, gr.update(
        choices=get_output_resolution_choices_for_task(task, video_resolution),
        value=format_size_markdown(task, width, height),
    )

def update_output_resolution_from_video_profile(task: str, aspect_ratio: str, video_resolution: str):
    width, height = get_size_for_aspect_ratio(task, aspect_ratio, video_resolution)
    return (
        gr.update(
            choices=get_output_resolution_choices_for_task(task, video_resolution),
            value=format_size_markdown(task, width, height),
        ),
        height,
        width,
    )

def reset_generation_defaults_for_task(task: str):
    internal_task = normalize_task(task)
    aspect_ratio = get_default_aspect_ratio(internal_task)
    resolution = get_default_resolution_for_task(internal_task)
    width, height = get_size_for_aspect_ratio(internal_task, aspect_ratio, resolution)
    num_frames = DEFAULT_VIDEO_DURATION_SECONDS
    return aspect_ratio, height, width, num_frames, resolution, gr.update(
        choices=get_output_resolution_choices_for_task(internal_task, resolution),
        value=format_size_markdown(internal_task, width, height),
    )

def make_prompt_example_click_handler(prompt_text: str):
    """Create a click handler for custom text-to-visual prompt-example rows."""

    def _handler(task: str):
        defaults = reset_generation_defaults_for_task(task)
        return (prompt_text, "", *defaults)

    return _handler

def make_media_prompt_example_click_handler(
    prompt_text: str,
    input_video_path: Optional[str] = None,
    input_image_path: Optional[str] = None,
):
    """Create a click handler for edit/understanding example rows."""

    def _handler(task: str):
        internal_task = normalize_task(task)
        defaults = reset_generation_defaults_for_task(internal_task)
        system_prompt = normalize_understanding_system_prompt(internal_task, None) if internal_task in UNDERSTANDING_TASKS else ""
        return (prompt_text, input_video_path, input_image_path, system_prompt, *defaults)

    return _handler

def get_understanding_system_prompt_choices(task: str) -> list[str]:
    internal_task = normalize_task(task)
    if internal_task == TASK_X2T_IMAGE:
        return [I2T_QA_SYSTEM_PROMPT]
    return [V2T_QA_SYSTEM_PROMPT]

def normalize_understanding_system_prompt(task: str, system_prompt: Optional[str]) -> str:
    return get_understanding_system_prompt_choices(task)[0]

def create_request_json(
    task: str,
    prompt: str,
    input_video: Optional[str],
    input_image: Optional[str],
    system_prompt: Optional[str] = None,
) -> Path:
    ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prompt_file = TMP_INPUT_DIR / f"{task}_{timestamp}.json"

    if task == TASK_T2V:
        payload = {"000000.mp4": prompt}
    elif task == TASK_T2I:
        payload = {"000000.png": prompt}
    elif task == TASK_VIDEO_EDIT:
        if not input_video:
            raise ValueError("The video edit task requires an input video.")
        payload = {
            "000000": {
                "interleave_array": [prompt, input_video, input_video],
                "element_dtype_array": ["text", "video", "video"],
                "istarget_in_interleave": [0, 0, 1],
            }
        }
    elif task == TASK_IMAGE_EDIT:
        if not input_image:
            raise ValueError("The image edit task requires an input image.")
        payload = {
            "000000": {
                "interleave_array": [prompt, input_image, input_image],
                "element_dtype_array": ["text", "image", "image"],
                "istarget_in_interleave": [0, 0, 1],
            }
        }
    elif task == TASK_X2T_VIDEO:
        if not input_video:
            raise ValueError("The video understanding task requires an input video.")
        system_prompt = normalize_understanding_system_prompt(task, system_prompt)
        payload = {
            "000000": {
                "interleave_array": [input_video, [system_prompt, prompt, ""]],
                "element_dtype_array": ["video", "text"],
                "istarget_in_interleave": [0, 1],
            }
        }
    elif task == TASK_X2T_IMAGE:
        if not input_image:
            raise ValueError("The image understanding task requires an input image.")
        system_prompt = normalize_understanding_system_prompt(task, system_prompt)
        payload = {
            "000000": {
                "interleave_array": [input_image, [system_prompt, prompt, ""]],
                "element_dtype_array": ["image", "text"],
                "istarget_in_interleave": [0, 1],
            }
        }
    else:
        raise ValueError(f"Unsupported task type: {task}")

    with prompt_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return prompt_file

def resolve_example_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    repo_candidate = (REPO_ROOT / candidate)
    if repo_candidate.exists():
        return str(repo_candidate.resolve())
    if candidate.exists():
        return str(candidate.resolve())
    return path

def resolve_video_example_paths(path: str) -> tuple[str, str]:
    """Return (browser_preview_path, model_input_path).

    Model input keeps the original sample path. Browser preview uses a
    H.264/yuv420p copy when the source codec is not reliably playable.
    """
    original_path = resolve_example_path(path)
    return prepare_browser_preview_video(original_path), original_path

def _probe_video_stream(video_path: Path) -> dict[str, str]:
    if not shutil.which("ffprobe"):
        return {}
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        data = json.loads(result.stdout or "{}")
        streams = data.get("streams") or []
        return streams[0] if streams else {}
    except Exception:
        return {}

def _is_browser_playable_mp4(video_path: Path) -> bool:
    stream = _probe_video_stream(video_path)
    return stream.get("codec_name") == "h264" and stream.get("pix_fmt") == "yuv420p"

def prepare_browser_preview_video(video_path: str) -> str:
    source = _resolve_existing_media_path(video_path)
    if source is None:
        return video_path
    if _is_browser_playable_mp4(source):
        return str(source)
    if not shutil.which("ffmpeg"):
        return str(source)

    PREVIEW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    preview_path = PREVIEW_VIDEO_DIR / f"{source.stem}_h264.mp4"
    if preview_path.exists() and preview_path.stat().st_mtime >= source.stat().st_mtime:
        return str(preview_path)

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(source),
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(preview_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return str(preview_path)
    except Exception:
        return str(source)

def _resolve_existing_media_path(media_path: Optional[str]) -> Optional[Path]:
    if not media_path:
        return None
    candidate = Path(str(media_path))
    candidates = [candidate] if candidate.is_absolute() else [REPO_ROOT / candidate, candidate]
    for item in candidates:
        try:
            resolved = item.expanduser().resolve()
        except Exception:
            continue
        if resolved.exists():
            return resolved
    return None

def build_gradio_media_url(media_path: Optional[str]) -> str:
    """Build a Gradio file-serving URL for local recommended-case media."""
    existing = _resolve_existing_media_path(media_path)
    source = str(existing if existing else media_path or "")
    if not source:
        return ""
    try:
        from gradio.route_utils import API_PREFIX
    except Exception:
        API_PREFIX = ""
    return f"{API_PREFIX or ''}/file={quote(source, safe='/:')}"

def build_example_media_html(media_path: Optional[str], media_type: str, fallback_media_path: Optional[str] = None) -> str:
    """Build a lightweight complete-fit media preview for recommended cases."""
    if media_type == "video":
        sources = []
        for candidate in (media_path, fallback_media_path):
            url = build_gradio_media_url(candidate)
            if url and url not in sources:
                sources.append(url)
        if not sources:
            return '<div class="reference-media-fallback">Video file not found</div>'
        source_tags = "".join(
            f'<source src="{html.escape(url, quote=True)}" type="video/mp4">'
            for url in sources
        )
        return (
            '<video class="example-preview-video" controls muted preload="metadata" playsinline>'
            + source_tags
            + 'Your browser cannot play this reference video.</video>'
        )

    url = build_gradio_media_url(media_path)
    if not url:
        return '<div class="reference-media-fallback">Image file not found</div>'
    alt_text = html.escape(Path(str(media_path)).name or "example image", quote=True)
    return f'<img class="example-preview-image" src="{html.escape(url, quote=True)}" alt="{alt_text}" loading="lazy" />'

def load_json_examples(relative_path: str) -> dict:
    path = REPO_ROOT / relative_path
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

T2V_EXAMPLE_SUMMARIES = {
    "000000.mp4": "Red panda surfing on a bright seaside wave.",
    "000002.mp4": "Panda cub skateboarding in a creative loft.",
    "000004.mp4": "Young woman shaping clay in a sunlit pottery workshop.",
    "000005.mp4": "Panda boxing a robot in a luxurious palace ring.",
    "000008.mp4": "Fantasy pastel horse stepping through a glowing cloud valley.",
}

def make_generation_examples(
    task_label: str,
    relative_path: str,
    limit: int,
    image_task: bool,
    selected_keys: Optional[list[str]] = None,
    summaries: Optional[dict[str, str]] = None,
) -> list[list]:
    data = load_json_examples(relative_path)
    items = [(key, data[key]) for key in selected_keys if key in data] if selected_keys else list(data.items())[:limit]
    return [[prompt] for _output_name, prompt in items]

def make_edit_examples(task_label: str, relative_path: str, limit: int, media_type: str) -> list[list]:
    data = load_json_examples(relative_path)
    examples = []
    for sample in list(data.values())[:limit]:
        interleave = sample["interleave_array"]
        prompt = interleave[0]
        if media_type == "video":
            preview_video_path, input_video_path = resolve_video_example_paths(interleave[1])
            examples.append([prompt, preview_video_path, input_video_path, None, None])
        else:
            image_path = resolve_example_path(interleave[1])
            examples.append([prompt, None, None, image_path, image_path])
    return examples

def make_understanding_examples(task_label: str, relative_path: str, limit: int, media_type: str) -> list[list]:
    data = load_json_examples(relative_path)
    examples = []
    for sample in list(data.values())[:limit]:
        interleave = sample["interleave_array"]
        text_payload = interleave[1]
        question = text_payload[1] if isinstance(text_payload, list) and len(text_payload) > 1 else ""
        if media_type == "video":
            preview_video_path, input_video_path = resolve_video_example_paths(interleave[0])
            examples.append([question, preview_video_path, input_video_path, None, None])
        else:
            image_path = resolve_example_path(interleave[0])
            examples.append([question, None, None, image_path, image_path])
    return examples

VIDEO_GENERATION_EXAMPLES = make_generation_examples(
    TASK_LABEL_VIDEO_GENERATION,
    "config/examples/t2v_example.json",
    limit=7,
    image_task=False,
    #selected_keys=["000000.mp4", "000002.mp4", "000005.mp4", "000004.mp4", "000008.mp4"],
    selected_keys=["000004.mp4", "000002.mp4", "000000.mp4", "000005.mp4", "000008.mp4", "000007.mp4", "000001.mp4"],
    summaries=T2V_EXAMPLE_SUMMARIES,
)
VIDEO_EDIT_EXAMPLES = make_edit_examples(
    TASK_LABEL_VIDEO_EDIT,
    "config/examples/video_edit_example.json",
    limit=3,
    media_type="video",
)
VIDEO_UNDERSTANDING_EXAMPLES = make_understanding_examples(
    TASK_LABEL_VIDEO_UNDERSTANDING,
    "config/examples/x2t_video_example.json",
    limit=3,
    media_type="video",
)
IMAGE_GENERATION_EXAMPLES = make_generation_examples(
    TASK_LABEL_IMAGE_GENERATION,
    "config/examples/t2i_example.json",
    limit=9,
    image_task=True,
    selected_keys=["000000.png", "000003.png", "000002.png", "000005.png", "000006.png", "000007.png", "000008.png", "000009.png", "000010.png"],
)
IMAGE_EDIT_EXAMPLES = make_edit_examples(
    TASK_LABEL_IMAGE_EDIT,
    "config/examples/image_edit_example.json",
    limit=5,
    media_type="image",
)
IMAGE_UNDERSTANDING_EXAMPLES = make_understanding_examples(
    TASK_LABEL_IMAGE_UNDERSTANDING,
    "config/examples/x2t_image_example.json",
    limit=6,
    media_type="image",
)

def build_save_dir(task: str) -> Path:
    ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RESULTS_ROOT / f"{task}_{timestamp}_{int(time.time() * 1000) % 1000:03d}"

def find_generated_video(save_dir: Path) -> Optional[Path]:
    videos = sorted(save_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return videos[0] if videos else None

def find_generated_image(save_dir: Path) -> Optional[Path]:
    images = sorted(save_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    return images[0] if images else None

def extract_text_result(save_dir: Path) -> str:
    prompt_result_path = save_dir / PROMPT_JSON_FILENAME
    if not prompt_result_path.exists():
        return ""
    with prompt_result_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return ""
    first_value = next(iter(data.values()))
    return first_value if isinstance(first_value, str) else json.dumps(first_value, ensure_ascii=False)
