from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GRADIO_TMP_ROOT = Path(os.getenv("LANCE_GRADIO_TMP_ROOT", "/tmp/lance_gradio")).expanduser()
TMP_INPUT_DIR = GRADIO_TMP_ROOT / "inputs"
RESULTS_ROOT = GRADIO_TMP_ROOT / "results"
PREVIEW_VIDEO_DIR = GRADIO_TMP_ROOT / "preview_videos"
GLOBAL_RECORDS_FILE = GRADIO_TMP_ROOT / "generation_records.jsonl"
RUN_RECORD_FILENAME = "generation_record.json"
PROMPT_JSON_FILENAME = "prompt.json"

LOCAL_MODEL_BASE_DIR = Path("downloads")
DEFAULT_MODEL_VARIANT = "video"
MODEL_VARIANT_VIDEO = "video"
MODEL_VARIANT_IMAGE = "image"
MODEL_VARIANT_TO_DIR = {
    MODEL_VARIANT_VIDEO: "Lance_3B_Video",
    MODEL_VARIANT_IMAGE: "Lance_3B",
}
DEFAULT_VIT_TYPE = "qwen_2_5_vl_original"
DEFAULT_TASK = "t2v"
DEFAULT_TIMESTEPS = 30
DEFAULT_TIMESTEP_SHIFT = 3.5
DEFAULT_CFG_TEXT_SCALE = 4.0
DEFAULT_RESOLUTION = "video_480p"
DEFAULT_VIDEO_EDIT_RESOLUTION = "video_480p"
DEFAULT_IMAGE_RESOLUTION = "image_768res"
DEFAULT_BASIC_SEED = 42
DEFAULT_HEIGHT = 352
DEFAULT_WIDTH = 640
DEFAULT_VIDEO_DURATION_SECONDS = 8
MAX_VIDEO_DURATION_SECONDS = 10
MAX_VIDEO_NUM_FRAMES = 12 * MAX_VIDEO_DURATION_SECONDS + 1
DEFAULT_NUM_FRAMES = 12 * DEFAULT_VIDEO_DURATION_SECONDS + 1
DEFAULT_VIDEO_ASPECT_RATIO = "16:9"
DEFAULT_IMAGE_ASPECT_RATIO = "1:1"
ASPECT_RATIO_CHOICES = ["21:9", "16:9", "3:2", "4:3", "1:1", "3:4", "2:3", "9:16"]

VIDEO_360P_ASPECT_RATIO_TO_SIZE = {
    "21:9": (672, 288),
    "16:9": (640, 352),
    "3:2": (528, 352),
    "4:3": (560, 416),
    "1:1": (480, 480),
    "3:4": (416, 560),
    "2:3": (352, 528),
    "9:16": (352, 640),
}

VIDEO_480P_ASPECT_RATIO_TO_SIZE = {
    "21:9": (976, 416),
    "16:9": (848, 480),
    "3:2": (784, 528),
    "4:3": (736, 560),
    "1:1": (640, 640),
    "3:4": (560, 736),
    "2:3": (528, 784),
    "9:16": (480, 848),
}

VIDEO_RESOLUTION_TO_SIZE_MAP = {
    "video_360p": VIDEO_360P_ASPECT_RATIO_TO_SIZE,
    "video_480p": VIDEO_480P_ASPECT_RATIO_TO_SIZE,
}

IMAGE_ASPECT_RATIO_TO_SIZE = {
    "21:9": (1168, 496),
    "16:9": (1024, 576),
    "3:2": (944, 624),
    "4:3": (880, 672),
    "1:1": (768, 768),
    "3:4": (672, 880),
    "2:3": (624, 944),
    "9:16": (576, 1024),
}
DEFAULT_GPUS = "0"
DEFAULT_QUEUE_SIZE = 32
USE_KVCACHE = True
TEXT_TEMPLATE = True
RECORD_WRITE_LOCK = threading.Lock()

LANCE_HOMEPAGE_URL = "https://lance-project.github.io/"
LANCE_PAPER_URL = "http://arxiv.org/abs/2605.18678"
LANCE_HUGGING_FACE_URL = "https://huggingface.co/bytedance-research/Lance"
LANCE_GITHUB_URL = "https://github.com/bytedance/Lance"
LANCE_LOGO_PATH = REPO_ROOT / "assets" / "logo" / "lance-logo.png"

TASK_T2V = "t2v"
TASK_T2I = "t2i"
TASK_V2T = "v2t"
TASK_X2T = "x2t"
TASK_X2T_VIDEO = "x2t_video"
TASK_X2T_IMAGE = "x2t_image"
TASK_IMAGE_EDIT = "image_edit"
TASK_VIDEO_EDIT = "video_edit"
TASK_LABEL_VIDEO_GENERATION = "Video Generation"
TASK_LABEL_VIDEO_EDIT = "Video Edit"
TASK_LABEL_VIDEO_UNDERSTANDING = "Video Understanding"
TASK_LABEL_IMAGE_GENERATION = "Image Generation"
TASK_LABEL_IMAGE_EDIT = "Image Edit"
TASK_LABEL_IMAGE_UNDERSTANDING = "Image Understanding"
TASK_CHOICES = [
    TASK_LABEL_VIDEO_GENERATION,
    TASK_LABEL_VIDEO_EDIT,
    TASK_LABEL_VIDEO_UNDERSTANDING,
    TASK_LABEL_IMAGE_GENERATION,
    TASK_LABEL_IMAGE_EDIT,
    TASK_LABEL_IMAGE_UNDERSTANDING,
]
TASK_LABEL_TO_INTERNAL = {
    TASK_LABEL_VIDEO_GENERATION: TASK_T2V,
    TASK_LABEL_VIDEO_EDIT: TASK_VIDEO_EDIT,
    TASK_LABEL_VIDEO_UNDERSTANDING: TASK_X2T_VIDEO,
    TASK_LABEL_IMAGE_GENERATION: TASK_T2I,
    TASK_LABEL_IMAGE_EDIT: TASK_IMAGE_EDIT,
    TASK_LABEL_IMAGE_UNDERSTANDING: TASK_X2T_IMAGE,
    TASK_T2V: TASK_T2V,
    TASK_VIDEO_EDIT: TASK_VIDEO_EDIT,
    TASK_V2T: TASK_X2T_VIDEO,
    TASK_X2T: TASK_X2T_VIDEO,
    TASK_X2T_VIDEO: TASK_X2T_VIDEO,
    TASK_T2I: TASK_T2I,
    TASK_IMAGE_EDIT: TASK_IMAGE_EDIT,
    TASK_X2T_IMAGE: TASK_X2T_IMAGE,
}
GENERATION_TASKS = {TASK_T2V, TASK_T2I, TASK_IMAGE_EDIT, TASK_VIDEO_EDIT}
UNDERSTANDING_TASKS = {TASK_X2T_VIDEO, TASK_X2T_IMAGE}
IMAGE_TASKS = {TASK_T2I, TASK_IMAGE_EDIT, TASK_X2T_IMAGE}
VIDEO_TASKS = {TASK_T2V, TASK_VIDEO_EDIT, TASK_X2T_VIDEO}
EDIT_TASKS = {TASK_IMAGE_EDIT, TASK_VIDEO_EDIT}
VIDEO_RESOLUTION_CHOICES = [DEFAULT_RESOLUTION]
VIDEO_EDIT_RESOLUTION_CHOICES = [DEFAULT_VIDEO_EDIT_RESOLUTION]
IMAGE_RESOLUTION_CHOICES = [DEFAULT_IMAGE_RESOLUTION]
VIDEO_RESOLUTION_DISPLAY_CHOICES = [("360p", "video_360p"), ("480p", "video_480p")]
V2T_QA_SYSTEM_PROMPT = "View the video  attentively and provide a suitable answer to the posed question."
I2T_QA_SYSTEM_PROMPT = "View the image attentively and provide a suitable answer to the posed question."

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lance multimodal Gradio")
    parser.add_argument("--server-name", default=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--server-port", type=int, default=int(os.getenv("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument(
        "--gpus",
        default=os.getenv("LANCE_GPUS", DEFAULT_GPUS),
        help="Comma-separated GPU list, for example: 0,1,2,3,4,5,6",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=int(os.getenv("LANCE_QUEUE_SIZE", str(DEFAULT_QUEUE_SIZE))),
        help="Maximum number of queued Gradio requests.",
    )
    return parser.parse_args()

def parse_gpu_ids(gpu_string: str) -> list[int]:
    gpu_ids: list[int] = []
    for item in gpu_string.split(","):
        item = item.strip()
        if not item:
            continue
        gpu_ids.append(int(item))
    if not gpu_ids:
        raise ValueError("No valid GPU IDs were parsed.")
    return gpu_ids
