from __future__ import annotations

import argparse
import base64
import concurrent.futures
import gc
import html
import math
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import spaces
except ImportError:  # pragma: no cover - keeps local CPU runs working
    class _SpacesShim:
        @staticmethod
        def GPU(*args, **kwargs):
            if args and callable(args[0]) and not kwargs:
                return args[0]

            def decorator(fn):
                return fn

            return decorator

    spaces = _SpacesShim()

import gradio as gr
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import set_seed
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig

from common.utils.logging import get_logger
from common.utils.misc import AutoEncoderParams, tuple_mul
from config.config_factory import DataArguments, InferenceArguments, ModelArguments
from data.data_utils import add_special_tokens
from data.dataset_base import DataConfig, simple_custom_collate
from data.datasets_custom import ValidationDataset
from inference_lance import (
    PROMPT_JSON_FILENAME,
    apply_inference_defaults,
    clean_memory,
    init_from_model_path_if_needed,
    save_prompt_results,
    validate_on_fixed_batch,
)
from modeling.lance import Lance, LanceConfig, Qwen2ForCausalLM
from modeling.qwen2 import Qwen2Tokenizer
from modeling.qwen2.modeling_qwen2 import Qwen2Config
from modeling.vae.wan.model import WanVideoVAE
from modeling.vit.qwen2_5_vl_vit import Qwen2_5_VisionTransformerPretrainedModel


REPO_ROOT = Path(__file__).resolve().parent
RIFE_DIR = REPO_ROOT / "RIFE"
RIFE_SCRIPT_PATH = RIFE_DIR / "inference_video.py"
RIFE_MODEL_DIR = RIFE_DIR / "train_log"
RIFE_AVAILABLE = RIFE_SCRIPT_PATH.exists()
GRADIO_TMP_ROOT = Path(os.getenv("LANCE_GRADIO_TMP_ROOT", "/tmp/lance_gradio")).expanduser()
TMP_INPUT_DIR = GRADIO_TMP_ROOT / "inputs"
RESULTS_ROOT = GRADIO_TMP_ROOT / "results"
GLOBAL_RECORDS_FILE = GRADIO_TMP_ROOT / "generation_records.jsonl"
RUN_RECORD_FILENAME = "generation_record.json"

LOCAL_MODEL_BASE_DIR = Path("downloads")
SPACE_MODEL_BASE_DIR = Path("/data/lance_models")
DEFAULT_MODEL_REPO_ID = "bytedance-research/Lance"
DEFAULT_FLASH_ATTN_VERSION = "2.8.3"
DEFAULT_FLASH_ATTN_WHEEL_URL = "https://huggingface.co/strangertoolshf/flash_attention_2_wheelhouse/resolve/main/wheelhouse-flash_attn-2.8.3/linux_x86_64/torch2.8/cu12/abiTRUE/cp310/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl"
DEFAULT_MODEL_VARIANT = "video"
MODEL_VARIANT_VIDEO = "video"
MODEL_VARIANT_IMAGE = "image"
MODEL_VARIANT_TO_DIR = {
    MODEL_VARIANT_VIDEO: "Lance_3B_Video",
    MODEL_VARIANT_IMAGE: "Lance_3B",
}
DEFAULT_MODEL_PATH = LOCAL_MODEL_BASE_DIR / MODEL_VARIANT_TO_DIR[MODEL_VARIANT_VIDEO]
DEFAULT_VIT_TYPE = "qwen_2_5_vl_original"
DEFAULT_TASK = "t2v"
DEFAULT_TIMESTEPS = 30
DEFAULT_TIMESTEP_SHIFT = 3.5
DEFAULT_CFG_TEXT_SCALE = 4.0
DEFAULT_RESOLUTION = "video_360p"
DEFAULT_VIDEO_EDIT_RESOLUTION = "video_480p"
DEFAULT_IMAGE_RESOLUTION = "image_768x768"
DEFAULT_BASIC_SEED = 42
DEFAULT_HEIGHT = 352
DEFAULT_WIDTH = 640
DEFAULT_IMAGE_SIZE = 768
DEFAULT_VIDEO_DURATION_SECONDS = 3
MAX_VIDEO_DURATION_SECONDS = 360
MAX_VIDEO_NUM_FRAMES = 12 * MAX_VIDEO_DURATION_SECONDS + 1
DEFAULT_NUM_FRAMES = 12 * DEFAULT_VIDEO_DURATION_SECONDS + 1
DEFAULT_VIDEO_ASPECT_RATIO = "16:9"
DEFAULT_IMAGE_ASPECT_RATIO = "1:1"
FRAME_INTERPOLATION_YES = "Yes"
FRAME_INTERPOLATION_NO = "No"
DEFAULT_FRAME_INTERPOLATION = FRAME_INTERPOLATION_YES
ASPECT_RATIO_CHOICES = ["21:9", "16:9", "3:2", "4:3", "1:1", "3:4", "2:3", "9:16", "9:21"]

VIDEO_360P_ASPECT_RATIO_TO_SIZE = {
    "21:9": (672, 288),
    "16:9": (640, 352),
    "3:2": (528, 352),
    "4:3": (560, 416),
    "1:1": (480, 480),
    "3:4": (416, 560),
    "2:3": (352, 528),
    "9:16": (352, 640),
    "9:21": (288, 672),
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
    "9:21": (416, 976),
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
    "9:21": (496, 1168),
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

APP_CSS = """
.gradio-container {
    max-width: 1680px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

.contain {
    max-width: 1680px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

.lance-hero {
    text-align: center;
    padding: 8px 12px 6px;
}

.lance-logo {
    width: min(160px, 36vw);
    height: auto;
    display: block;
    margin: 0 auto 4px;
}

.lance-title {
    margin: 0 auto 5px;
    font-size: clamp(22px, 2.5vw, 32px);
    line-height: 1.08;
    font-weight: 800;
    letter-spacing: 0;
}

.lance-authors {
    margin: 0 auto 6px;
    max-width: 1280px;
    font-size: 20px;
    line-height: 1.24;
    color: var(--body-text-color-subdued);
}

.lance-authors a {
    color: inherit;
    text-decoration: none;
}

.lance-authors a:hover {
    text-decoration: underline;
}

.lance-badges {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 5px;
    margin: 4px auto 0;
}

.lance-badges a {
    line-height: 0;
}

.lance-badges img {
    height: 20px;
    width: auto;
    display: block;
}

.lance-status {
    max-width: 1180px;
    margin: 0 auto 18px;
}

.lance-run-status {
    margin: 0 0 8px 0 !important;
    min-height: 0 !important;
}

.lance-run-status p {
    margin: 0 !important;
}

.lance-run-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 999px;
    border: 1px solid var(--border-color-primary);
    background: rgba(255, 255, 255, 0.03);
    color: var(--body-text-color-subdued);
    font-size: 14px;
    font-weight: 700;
    line-height: 1;
}

.lance-run-status-chip {
    width: 8px;
    height: 8px;
    border-radius: 999px;
    background: var(--primary-500, #f97316);
    box-shadow: 0 0 0 4px rgba(249, 115, 22, 0.12);
    flex: 0 0 auto;
}

.lance-run-status-dots {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    margin-left: 2px;
}

.lance-run-status-dots i {
    width: 4px;
    height: 4px;
    border-radius: 999px;
    background: currentColor;
    opacity: 0.3;
    animation: lance-dot-pulse 1.1s infinite ease-in-out;
}

.lance-run-status-dots i:nth-child(2) {
    animation-delay: 0.15s;
}

.lance-run-status-dots i:nth-child(3) {
    animation-delay: 0.3s;
}

@keyframes lance-dot-pulse {
    0%, 80%, 100% {
        transform: translateY(0);
        opacity: 0.25;
    }
    40% {
        transform: translateY(-1px);
        opacity: 1;
    }
}

/* Lance UI labels rendered as explicit HTML nodes.
   Typography is controlled here, while panels/cards restore the original boxed visual hierarchy. */
.lance-panel,
.lance-control-field {
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 10px !important;
    background: var(--block-background-fill) !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.14) !important;
}

.lance-panel {
    padding: 14px 14px 12px !important;
    margin: 0 0 14px 0 !important;
}

.lance-output-panel {
    padding: 4px 10px 4px !important;
    margin: 0 0 4px 0 !important;
    width: 100% !important;
}

.lance-output-panel .lance-display-frame {
    margin: 0 !important;
}

.lance-output-panel .lance-display-frame > .form,
.lance-output-panel .lance-display-frame > div {
    background: transparent !important;
}

.lance-panel > .form,
.lance-control-field > .form {
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.lance-section-label,
.lance-generation-label {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    padding: 0 !important;
    color: var(--body-text-color) !important;
    white-space: normal !important;
}

.lance-icon-label {
    gap: 10px !important;
}

.lance-section-label::before,
.lance-generation-label::before {
    content: "";
    display: inline-block;
    width: 4px;
    height: 16px;
    border-radius: 999px;
    background: var(--primary-500, #f97316);
    flex: 0 0 auto;
}

.lance-icon-label::before {
    display: none !important;
    content: none !important;
}

.lance-label-icon {
    width: 24px;
    height: 24px;
    flex: 0 0 auto;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 8px;
    border: 1px solid rgba(249, 115, 22, 0.18);
    background: rgba(249, 115, 22, 0.1);
    color: var(--primary-500, #f97316);
}

.lance-label-icon svg {
    width: 14px;
    height: 14px;
    display: block;
}

.lance-section-label {
    margin: 0 0 10px 0 !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    line-height: 1.15 !important;
}

.lance-prompt-label {
    margin-top: 16px !important;
}

.lance-output-label {
    margin: 0 0 2px 0 !important;
}

.lance-generation-label {
    margin: 0 0 8px 0 !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    line-height: 1.15 !important;
}

.lance-control-field {
    min-width: 0 !important;
    gap: 0 !important;
    padding: 12px 14px !important;
}

.lance-label-html,
.lance-label-html > div,
.lance-label-html .wrap {
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
    min-height: 0 !important;
}

.lance-task-prompt-panel .task-selector {
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.lance-task-prompt-panel .task-selector > .wrap {
    padding: 0 !important;
}

.task-selector {
    overflow-x: auto;
}

.task-selector .wrap {
    display: grid;
    grid-template-columns: repeat(3, minmax(220px, 1fr));
    gap: 8px;
    min-width: 680px;
}

.task-selector label {
    justify-content: center;
    min-height: 38px;
    white-space: nowrap;
    border-radius: 10px !important;
}

.task-selector .wrap label span {
    font-size: 16px !important;
}

.main-prompt-control label span,
.main-prompt-control .block-label,
.main-prompt-control .label-wrap span,
.output-media-control label span,
.output-media-control .block-label,
.output-media-control .label-wrap span {
    font-size: 20px !important;
    font-weight: 700 !important;
    line-height: 1.15 !important;
}

.generation-controls-row .generation-two-line-label label,
.generation-controls-row .generation-two-line-label > label,
.generation-controls-row .generation-two-line-label label span,
.generation-controls-row .generation-two-line-label .block-label,
.generation-controls-row .generation-two-line-label .block-title,
.generation-controls-row .generation-two-line-label .label-wrap,
.generation-controls-row .generation-two-line-label .label-wrap span {
    font-size: 18px !important;
    font-weight: 700 !important;
    line-height: 1.1 !important;
    white-space: normal !important;
    max-width: 100% !important;
}

.lance-generation-label {
    font-size: 18px !important;
    font-weight: 700 !important;
    line-height: 1.1 !important;
}

.generation-control-stack {
    display: flex !important;
    flex-direction: column !important;
    gap: 12px !important;
    width: 100% !important;
    min-width: 0 !important;
}

.generation-controls-row {
    width: 100% !important;
}

.generation-controls-row > .form {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) !important;
    gap: 12px !important;
    align-items: start !important;
    width: 100% !important;
    min-width: 0 !important;
}

.frame-interpolation-row > .form,
.aspect-ratio-row > .form,
.output-resolution-row > .form,
.video-duration-row > .form {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) !important;
    gap: 12px !important;
    align-items: start !important;
    width: 100% !important;
    min-width: 0 !important;
}

.generation-choice-grid .wrap {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)) !important;
    gap: 8px !important;
    min-width: 0 !important;
    width: 100% !important;
}

.aspect-ratio-row .generation-choice-grid .wrap {
    justify-content: flex-start !important;
}

.generation-choice-grid label {
    justify-content: center;
    min-height: 38px;
    white-space: nowrap;
    border-radius: 10px !important;
}

.aspect-ratio-row .generation-choice-grid label,
.video-duration-row .generation-choice-grid label {
    justify-content: flex-start !important;
    text-align: left !important;
    padding-left: 14px !important;
}

.generation-choice-grid .wrap label span {
    font-size: 16px !important;
    white-space: nowrap !important;
}

.recommended-title {
    text-align: center !important;
    margin: 14px auto 10px !important;
}

.recommended-title h3,
.recommended-title p {
    text-align: center !important;
    font-size: 22px !important;
    font-weight: 800 !important;
    color: var(--body-text-color) !important;
}

.example-panel {
    margin-top: 14px !important;
    padding: 10px 12px !important;
    border-radius: 8px !important;
    background: rgba(248, 250, 252, 0.72) !important;
    border: 1px solid var(--border-color-primary) !important;
}

.prompt-examples table,
.prompt-examples th,
.prompt-examples td {
    border: 1px solid var(--border-color-primary) !important;
}

.prompt-examples table {
    border-collapse: collapse !important;
    width: 100% !important;
}

.prompt-examples td {
    border-bottom: 1px solid var(--border-color-primary) !important;
    padding: 12px !important;
    vertical-align: top !important;
}

.example-panel th,
.example-panel .block-label,
.example-panel label span,
.example-panel .label-wrap span {
    font-size: 18px !important;
    font-weight: 700 !important;
}

.prompt-dataset {
    max-height: 420px !important;
    overflow-y: auto !important;
    overscroll-behavior: contain !important;
    scrollbar-gutter: stable !important;
}

.prompt-dataset button {
    height: auto !important;
    min-height: 48px !important;
    font-size: 17px !important;
    line-height: 1.35 !important;
    white-space: normal !important;
    text-align: left !important;
    align-items: flex-start !important;
}

.prompt-dataset button span,
.prompt-dataset button p {
    font-size: 17px !important;
    line-height: 1.35 !important;
}

.prompt-dataset button,
.example-panel table td:first-child button {
    max-height: 180px !important;
    overflow-y: auto !important;
    overscroll-behavior: contain !important;
}

.prompt-dataset button,
.example-panel table td:first-child button,
.prompt-dataset button span,
.prompt-dataset button p,
.example-panel table td:first-child span,
.example-panel table td:first-child p {
    white-space: pre-wrap !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
    text-overflow: clip !important;
    -webkit-line-clamp: unset !important;
    line-clamp: unset !important;
}

.prompt-dataset button span,
.prompt-dataset button p,
.example-panel table td:first-child span,
.example-panel table td:first-child p {
    overflow: visible !important;
    display: block !important;
}

.lance-recommended-section .example-panel td,
.lance-recommended-section .example-panel td *,
.lance-recommended-section .example-panel button,
.lance-recommended-section .example-panel button *,
.lance-recommended-section .example-panel label,
.lance-recommended-section .example-panel label *,
.lance-recommended-section .example-panel span,
.lance-recommended-section .example-panel p {
    white-space: pre-wrap !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
    text-overflow: clip !important;
    -webkit-line-clamp: unset !important;
    line-clamp: unset !important;
}

.lance-recommended-section .example-panel button,
.lance-recommended-section .example-panel td {
    height: auto !important;
    max-height: none !important;
    overflow: visible !important;
}

.lance-recommended-section .example-panel [style*="ellipsis"],
.lance-recommended-section .example-panel [style*="nowrap"],
.lance-recommended-section .example-panel [style*="hidden"] {
    white-space: pre-wrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
}

.lance-recommended-section .example-panel {
    overflow: visible !important;
}

.lance-recommended-section .example-panel table {
    width: 100% !important;
    table-layout: fixed !important;
    border-collapse: collapse !important;
}

.lance-recommended-section .example-panel tr,
.lance-recommended-section .example-panel th,
.lance-recommended-section .example-panel td {
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
}

.lance-recommended-section .example-panel td:first-child,
.lance-recommended-section .example-panel td:first-child *,
.prompt-dataset td,
.prompt-dataset td *,
.prompt-dataset button,
.prompt-dataset button * {
    white-space: pre-wrap !important;
    overflow: visible !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
    text-overflow: clip !important;
    -webkit-line-clamp: unset !important;
    line-clamp: unset !important;
}

.lance-recommended-section .example-panel td:first-child button,
.prompt-dataset button {
    width: 100% !important;
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
    padding: 12px 14px !important;
    text-align: center !important;
    justify-content: center !important;
    align-items: center !important;
    line-height: 1.35 !important;
}

.prompt-dataset .paginate {
    display: none !important;
}

.video-edit-examples .block-label::before,
.video-edit-examples .label-wrap::before,
.video-edit-examples .label-wrap span::before,
.video-edit-examples .example-label::before,
.video-edit-examples .examples-label::before {
    display: none !important;
    content: none !important;
}

.example-no-icon .block-label::before,
.example-no-icon .label-wrap::before,
.example-no-icon .label-wrap span::before,
.example-no-icon .example-label::before,
.example-no-icon .examples-label::before {
    display: none !important;
    content: none !important;
}

.example-no-icon .label svg {
    display: none !important;
}

.lance-advanced-panel {
    margin-top: 0 !important;
}

.lance-advanced-accordion .block-title,
.lance-advanced-accordion .label-wrap,
.lance-advanced-accordion .label-wrap span,
.lance-advanced-accordion .block-label,
.lance-advanced-accordion summary span,
.lance-advanced-accordion summary,
.lance-advanced-accordion button span {
    font-size: 18px !important;
    font-weight: 700 !important;
    line-height: 1.15 !important;
}

.lance-recommended-section {
    min-width: 0 !important;
}

.lance-recommended-section > .form {
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
    min-width: 0 !important;
}

.lance-recommended-section .lance-section-label {
    margin: 0 !important;
}

.lance-recommended-section .example-panel {
    margin-top: 0 !important;
}

.prompt-example-proxy {
    display: none !important;
}

.lance-main-row {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) !important;
    gap: 16px !important;
    align-items: stretch !important;
}

.lance-main-column {
    min-width: 0 !important;
    width: 100% !important;
}

.lance-display-frame,
.lance-display-frame > div,
.lance-display-frame textarea {
    width: 100% !important;
}

.lance-display-frame textarea {
    min-height: 170px !important;
}

.lance-output-column,
.lance-output-column > .form {
    display: flex !important;
    flex-direction: column !important;
    min-height: 0 !important;
}

.lance-output-column {
    height: var(--lance-input-column-height, 100%) !important;
    max-height: var(--lance-input-column-height, none) !important;
}

.run-logs {
    display: flex !important;
    flex: 1 1 auto !important;
    flex-direction: column !important;
    min-height: 180px !important;
}

.run-logs-field {
    flex: 1 1 auto !important;
    min-height: 0 !important;
}

.run-logs-field > .form {
    display: flex !important;
    flex: 1 1 auto !important;
    flex-direction: column !important;
    min-height: 0 !important;
}

.run-logs > div,
.run-logs .wrap {
    display: flex !important;
    flex: 1 1 auto !important;
    flex-direction: column !important;
    min-height: 0 !important;
}

.run-logs textarea {
    flex: 1 1 auto !important;
    height: 100% !important;
    min-height: 180px !important;
    resize: none !important;
}

.lance-run-button {
    font-size: 18px !important;
    font-weight: 800 !important;
}



/* Prompt example tables: Gradio Dataset renders Textbox cells with an inline
   max-width: 35ch and a single-line preview, which causes long prompts to be
   clipped with an ellipsis. These rules expand the Prompt column, wrap text,
   and keep very long rows usable through scrolling. */
.prompt-dataset,
.prompt-dataset .table-wrap {
    width: 100% !important;
    max-width: 100% !important;
    overflow-x: auto !important;
    overflow-y: auto !important;
}

.prompt-dataset .table-wrap {
    max-height: 420px !important;
    overscroll-behavior: contain !important;
    scrollbar-gutter: stable !important;
}

.prompt-dataset table {
    width: 100% !important;
    min-width: 720px !important;
    max-width: none !important;
    table-layout: fixed !important;
    border-collapse: collapse !important;
}

.prompt-dataset thead,
.prompt-dataset tbody,
.prompt-dataset tr,
.prompt-dataset th,
.prompt-dataset td,
.prompt-dataset td.textbox,
.prompt-dataset td[style*="35ch"] {
    height: auto !important;
    min-height: 0 !important;
    max-height: none !important;
    max-width: none !important;
    width: 100% !important;
    min-width: 0 !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    vertical-align: top !important;
}

.prompt-dataset th,
.prompt-dataset td {
    padding: 12px 14px !important;
}

.prompt-dataset td > * {
    width: 100% !important;
    max-width: none !important;
    min-width: 0 !important;
    height: auto !important;
    min-height: 0 !important;
    max-height: 260px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    overscroll-behavior: contain !important;
    white-space: pre-wrap !important;
    text-align: left !important;
}

.prompt-dataset td *,
.prompt-dataset td [class*="truncate"],
.prompt-dataset td [class*="ellipsis"],
.prompt-dataset td [class*="line-clamp"],
.prompt-dataset td [style*="nowrap"],
.prompt-dataset td [style*="ellipsis"],
.prompt-dataset td [style*="line-clamp"],
.prompt-dataset td span,
.prompt-dataset td p,
.prompt-dataset td div,
.prompt-dataset td button {
    max-width: none !important;
    white-space: pre-wrap !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
    text-overflow: clip !important;
    -webkit-line-clamp: unset !important;
    line-clamp: unset !important;
}

.prompt-dataset td span,
.prompt-dataset td p {
    display: block !important;
}



/* Full prompt example rows.  Do not use gr.Dataset for these two generation
   sections: Dataset table cells are rendered as compact previews and the
   actual DOM text may already contain "...".  These button rows keep and render
   the original prompt string, wrap it fully, and make very long rows scrollable. */
.prompt-example-full-table,
.prompt-example-full-table > .form,
.prompt-example-full-table > div {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
}

.prompt-example-full-table {
    max-height: 460px !important;
    overflow-x: auto !important;
    overflow-y: auto !important;
    overscroll-behavior: contain !important;
    scrollbar-gutter: stable !important;
    border: 1px solid var(--border-color-primary) !important;
    border-radius: 8px !important;
}

.prompt-example-table-header,
.prompt-example-table-header > div,
.prompt-example-table-header .wrap {
    position: sticky !important;
    top: 0 !important;
    z-index: 3 !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 12px 14px !important;
    border: 0 !important;
    border-bottom: 1px solid var(--border-color-primary) !important;
    background: var(--block-title-background-fill, var(--block-background-fill)) !important;
    color: var(--body-text-color) !important;
    font-size: 18px !important;
    font-weight: 800 !important;
    line-height: 1.25 !important;
    text-align: center !important;
    box-shadow: none !important;
}

.prompt-example-table-body,
.prompt-example-table-body > .form {
    gap: 0 !important;
    width: 100% !important;
    min-width: 720px !important;
}

.prompt-examples .prompt-example-row-button,
.prompt-examples .prompt-example-row-button > button,
.prompt-examples .prompt-example-row-button button {
    width: 100% !important;
    max-width: none !important;
    min-width: 0 !important;
    height: auto !important;
    min-height: 54px !important;
    max-height: 220px !important;
    margin: 0 !important;
    padding: 12px 14px !important;
    border-radius: 0 !important;
    border: 0 !important;
    border-bottom: 1px solid var(--border-color-primary) !important;
    background: var(--block-background-fill) !important;
    color: var(--body-text-color) !important;
    display: flex !important;
    justify-content: flex-start !important;
    align-items: flex-start !important;
    text-align: left !important;
    overflow-x: hidden !important;
    overflow-y: auto !important;
    white-space: normal !important;
    cursor: pointer !important;
}

.prompt-examples .prompt-example-row-button span,
.prompt-examples .prompt-example-row-button p,
.prompt-examples .prompt-example-row-button div {
    width: 100% !important;
    max-width: none !important;
    display: block !important;
    overflow: visible !important;
    white-space: pre-wrap !important;
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
    text-overflow: clip !important;
    -webkit-line-clamp: unset !important;
    line-clamp: unset !important;
    font-size: 16px !important;
    line-height: 1.38 !important;
    text-align: left !important;
}

.prompt-examples .prompt-example-row-button:last-child,
.prompt-examples .prompt-example-row-button:last-child > button,
.prompt-examples .prompt-example-row-button:last-child button {
    border-bottom: 0 !important;
}


.prompt-example-table-header-with-media,
.prompt-example-table-header-with-media > div,
.prompt-example-table-header-with-media .wrap {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) minmax(180px, 260px) !important;
    gap: 0 !important;
    text-align: center !important;
}

.prompt-example-multimodal-row,
.prompt-example-multimodal-row > .form {
    width: 100% !important;
    min-width: 720px !important;
    margin: 0 !important;
    gap: 0 !important;
    align-items: stretch !important;
    border-bottom: 1px solid var(--border-color-primary) !important;
}

.prompt-example-multimodal-row > .form {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) minmax(180px, 260px) !important;
}

.prompt-example-prompt-cell,
.prompt-example-prompt-cell > .form,
.prompt-example-media-cell,
.prompt-example-media-cell > .form {
    width: 100% !important;
    min-width: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

.prompt-example-multimodal-row .prompt-example-row-button,
.prompt-example-multimodal-row .prompt-example-row-button > button,
.prompt-example-multimodal-row .prompt-example-row-button button {
    height: 100% !important;
    min-height: 150px !important;
    max-height: 260px !important;
    border-bottom: 0 !important;
}

.prompt-example-media-cell {
    border-left: 1px solid var(--border-color-primary) !important;
}

.prompt-example-media-preview,
.prompt-example-media-preview > div,
.prompt-example-media-preview .wrap {
    width: 100% !important;
    height: 150px !important;
    min-height: 150px !important;
    max-height: 150px !important;
    margin: 0 !important;
    border: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: hidden !important;
}

.prompt-example-media-preview video,
.prompt-example-media-preview img {
    width: 100% !important;
    height: 150px !important;
    object-fit: cover !important;
    border-radius: 0 !important;
}

/* Keep the prompt column unchanged. Video examples fill the current row height,
   keep their original aspect ratio, and adapt their width inside the media column. */
.prompt-example-video-cell,
.prompt-example-video-cell > .form {
    display: flex !important;
    align-items: stretch !important;
    justify-content: center !important;
    padding: 0 !important;
    height: 100% !important;
    min-height: 150px !important;
    max-height: 260px !important;
    overflow: hidden !important;
}

.prompt-example-video-preview,
.prompt-example-video-preview > div,
.prompt-example-video-preview .wrap {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    min-width: 0 !important;
    max-width: 100% !important;
    height: 100% !important;
    min-height: 150px !important;
    max-height: 260px !important;
    margin: 0 auto !important;
    border-radius: 0 !important;
    overflow: hidden !important;
}

.prompt-example-video-preview video {
    width: auto !important;
    max-width: 100% !important;
    height: 100% !important;
    min-height: 150px !important;
    max-height: 260px !important;
    object-fit: contain !important;
    border-radius: 0 !important;
}

.prompt-example-multimodal-row:last-child,
.prompt-example-multimodal-row:last-child > .form {
    border-bottom: 0 !important;
}

@media (max-width: 900px) {
    .prompt-example-table-header-with-media,
    .prompt-example-table-header-with-media > div,
    .prompt-example-table-header-with-media .wrap,
    .prompt-example-multimodal-row > .form {
        grid-template-columns: minmax(0, 1fr) minmax(140px, 180px) !important;
    }
}

@media (max-width: 900px) {
    .lance-main-row {
        grid-template-columns: minmax(0, 1fr) !important;
    }
}
"""

APP_JS = """
() => {
    const applyImportantStyle = (element, property, value) => {
        if (!element) {
            return;
        }
        if (element.style.getPropertyValue(property) !== value || element.style.getPropertyPriority(property) !== "important") {
            element.style.setProperty(property, value, "important");
        }
    };

    const enforceLanceLabelTypography = () => {
        document.querySelectorAll(".lance-section-label").forEach((element) => {
            applyImportantStyle(element, "font-size", "20px");
            applyImportantStyle(element, "font-weight", "700");
            applyImportantStyle(element, "line-height", "1.15");
            const sectionMargin = element.classList.contains("lance-prompt-label")
                ? "16px 0 10px 0"
                : "0 0 10px 0";
            applyImportantStyle(element, "margin", sectionMargin);
            applyImportantStyle(element, "padding", "0");
        });

        document.querySelectorAll(".lance-generation-label").forEach((element) => {
            applyImportantStyle(element, "font-size", "18px");
            applyImportantStyle(element, "font-weight", "700");
            applyImportantStyle(element, "line-height", "1.15");
            applyImportantStyle(element, "margin", "0 0 8px 0");
            applyImportantStyle(element, "padding", "0");
        });
    };

    const enforceRecommendedCaseText = () => {
        document.querySelectorAll(".lance-recommended-section .example-panel").forEach((panel) => {
            applyImportantStyle(panel, "overflow", "visible");
            panel.querySelectorAll("table, tbody, tr, th, td, button, label, span, p, div").forEach((element) => {
                applyImportantStyle(element, "white-space", "pre-wrap");
                applyImportantStyle(element, "overflow-wrap", "anywhere");
                applyImportantStyle(element, "word-break", "break-word");
                applyImportantStyle(element, "text-overflow", "clip");
                applyImportantStyle(element, "-webkit-line-clamp", "unset");
                applyImportantStyle(element, "line-clamp", "unset");
            });
            panel.querySelectorAll("td, button").forEach((element) => {
                applyImportantStyle(element, "height", "auto");
                applyImportantStyle(element, "max-height", "none");
                applyImportantStyle(element, "overflow", "visible");
            });
            panel.querySelectorAll("button").forEach((element) => {
                applyImportantStyle(element, "width", "100%");
                applyImportantStyle(element, "text-align", "center");
                applyImportantStyle(element, "justify-content", "center");
                applyImportantStyle(element, "align-items", "center");
            });
        });
    };



    const enforcePromptDatasetText = () => {
        document.querySelectorAll(".prompt-dataset").forEach((dataset) => {
            applyImportantStyle(dataset, "width", "100%");
            applyImportantStyle(dataset, "max-width", "100%");
            applyImportantStyle(dataset, "overflow-x", "auto");
            applyImportantStyle(dataset, "overflow-y", "auto");

            dataset.querySelectorAll(".table-wrap").forEach((element) => {
                applyImportantStyle(element, "width", "100%");
                applyImportantStyle(element, "max-width", "100%");
                applyImportantStyle(element, "max-height", "420px");
                applyImportantStyle(element, "overflow-x", "auto");
                applyImportantStyle(element, "overflow-y", "auto");
                applyImportantStyle(element, "overscroll-behavior", "contain");
            });

            dataset.querySelectorAll("table").forEach((element) => {
                applyImportantStyle(element, "width", "100%");
                applyImportantStyle(element, "min-width", "720px");
                applyImportantStyle(element, "max-width", "none");
                applyImportantStyle(element, "table-layout", "fixed");
                applyImportantStyle(element, "border-collapse", "collapse");
            });

            dataset.querySelectorAll("thead, tbody, tr, th, td, td.textbox, td[style*='35ch']").forEach((element) => {
                applyImportantStyle(element, "height", "auto");
                applyImportantStyle(element, "min-height", "0");
                applyImportantStyle(element, "max-height", "none");
                applyImportantStyle(element, "max-width", "none");
                applyImportantStyle(element, "width", "100%");
                applyImportantStyle(element, "min-width", "0");
                applyImportantStyle(element, "white-space", "normal");
                applyImportantStyle(element, "overflow", "visible");
                applyImportantStyle(element, "text-overflow", "clip");
                applyImportantStyle(element, "vertical-align", "top");
            });

            dataset.querySelectorAll("td *").forEach((element) => {
                applyImportantStyle(element, "max-width", "none");
                applyImportantStyle(element, "white-space", "pre-wrap");
                applyImportantStyle(element, "overflow-wrap", "anywhere");
                applyImportantStyle(element, "word-break", "break-word");
                applyImportantStyle(element, "text-overflow", "clip");
                applyImportantStyle(element, "-webkit-line-clamp", "unset");
                applyImportantStyle(element, "line-clamp", "unset");
            });

            dataset.querySelectorAll("td > *").forEach((element) => {
                applyImportantStyle(element, "width", "100%");
                applyImportantStyle(element, "max-width", "none");
                applyImportantStyle(element, "min-width", "0");
                applyImportantStyle(element, "height", "auto");
                applyImportantStyle(element, "min-height", "0");
                applyImportantStyle(element, "max-height", "260px");
                applyImportantStyle(element, "overflow-y", "auto");
                applyImportantStyle(element, "overflow-x", "hidden");
                applyImportantStyle(element, "overscroll-behavior", "contain");
                applyImportantStyle(element, "white-space", "pre-wrap");
                applyImportantStyle(element, "text-align", "left");
            });

            dataset.querySelectorAll("td span, td p").forEach((element) => {
                applyImportantStyle(element, "display", "block");
            });
        });
    };

    const enforcePromptExampleRows = () => {
        document.querySelectorAll(".prompt-example-full-table").forEach((table) => {
            applyImportantStyle(table, "width", "100%");
            applyImportantStyle(table, "max-width", "100%");
            applyImportantStyle(table, "max-height", "460px");
            applyImportantStyle(table, "overflow-x", "auto");
            applyImportantStyle(table, "overflow-y", "auto");
        });

        document.querySelectorAll(".prompt-example-table-body, .prompt-example-table-body > .form").forEach((element) => {
            applyImportantStyle(element, "width", "100%");
            applyImportantStyle(element, "min-width", "720px");
            applyImportantStyle(element, "gap", "0");
        });

        document.querySelectorAll(".prompt-example-row-button, .prompt-example-row-button button").forEach((element) => {
            applyImportantStyle(element, "width", "100%");
            applyImportantStyle(element, "max-width", "none");
            applyImportantStyle(element, "height", "auto");
            applyImportantStyle(element, "min-height", "54px");
            applyImportantStyle(element, "max-height", "220px");
            applyImportantStyle(element, "margin", "0");
            applyImportantStyle(element, "padding", "12px 14px");
            applyImportantStyle(element, "border-radius", "0");
            applyImportantStyle(element, "border", "0");
            applyImportantStyle(element, "border-bottom", "1px solid var(--border-color-primary)");
            applyImportantStyle(element, "display", "flex");
            applyImportantStyle(element, "justify-content", "flex-start");
            applyImportantStyle(element, "align-items", "flex-start");
            applyImportantStyle(element, "text-align", "left");
            applyImportantStyle(element, "overflow-x", "hidden");
            applyImportantStyle(element, "overflow-y", "auto");
            applyImportantStyle(element, "white-space", "normal");
        });

        document.querySelectorAll(".prompt-example-row-button span, .prompt-example-row-button p, .prompt-example-row-button div").forEach((element) => {
            applyImportantStyle(element, "width", "100%");
            applyImportantStyle(element, "max-width", "none");
            applyImportantStyle(element, "display", "block");
            applyImportantStyle(element, "overflow", "visible");
            applyImportantStyle(element, "white-space", "pre-wrap");
            applyImportantStyle(element, "overflow-wrap", "anywhere");
            applyImportantStyle(element, "word-break", "break-word");
            applyImportantStyle(element, "text-overflow", "clip");
            applyImportantStyle(element, "-webkit-line-clamp", "unset");
            applyImportantStyle(element, "line-clamp", "unset");
            applyImportantStyle(element, "font-size", "16px");
            applyImportantStyle(element, "line-height", "1.38");
            applyImportantStyle(element, "text-align", "left");
        });

        document.querySelectorAll(".prompt-example-table-header-with-media, .prompt-example-table-header-with-media > div, .prompt-example-table-header-with-media .wrap, .prompt-example-multimodal-row > .form").forEach((element) => {
            applyImportantStyle(element, "display", "grid");
            applyImportantStyle(element, "grid-template-columns", "minmax(0, 1fr) minmax(180px, 260px)");
            applyImportantStyle(element, "gap", "0");
        });

        document.querySelectorAll(".prompt-example-multimodal-row, .prompt-example-multimodal-row > .form").forEach((element) => {
            applyImportantStyle(element, "width", "100%");
            applyImportantStyle(element, "min-width", "720px");
            applyImportantStyle(element, "margin", "0");
            applyImportantStyle(element, "border-bottom", "1px solid var(--border-color-primary)");
        });

        document.querySelectorAll(".prompt-example-multimodal-row .prompt-example-row-button, .prompt-example-multimodal-row .prompt-example-row-button button").forEach((element) => {
            applyImportantStyle(element, "height", "100%");
            applyImportantStyle(element, "min-height", "150px");
            applyImportantStyle(element, "max-height", "260px");
            applyImportantStyle(element, "border-bottom", "0");
        });

        document.querySelectorAll(".prompt-example-media-preview, .prompt-example-media-preview > div, .prompt-example-media-preview .wrap, .prompt-example-media-preview video, .prompt-example-media-preview img").forEach((element) => {
            applyImportantStyle(element, "width", "100%");
            applyImportantStyle(element, "height", "150px");
            applyImportantStyle(element, "max-height", "150px");
            applyImportantStyle(element, "border-radius", "0");
            applyImportantStyle(element, "overflow", "hidden");
        });

        document.querySelectorAll(".prompt-example-video-cell, .prompt-example-video-cell > .form").forEach((element) => {
            applyImportantStyle(element, "display", "flex");
            applyImportantStyle(element, "align-items", "stretch");
            applyImportantStyle(element, "justify-content", "center");
            applyImportantStyle(element, "padding", "0");
            applyImportantStyle(element, "height", "100%");
            applyImportantStyle(element, "min-height", "150px");
            applyImportantStyle(element, "max-height", "260px");
            applyImportantStyle(element, "overflow", "hidden");
        });

        document.querySelectorAll(".prompt-example-video-preview, .prompt-example-video-preview > div, .prompt-example-video-preview .wrap").forEach((element) => {
            applyImportantStyle(element, "display", "flex");
            applyImportantStyle(element, "align-items", "center");
            applyImportantStyle(element, "justify-content", "center");
            applyImportantStyle(element, "width", "100%");
            applyImportantStyle(element, "min-width", "0");
            applyImportantStyle(element, "max-width", "100%");
            applyImportantStyle(element, "height", "100%");
            applyImportantStyle(element, "min-height", "150px");
            applyImportantStyle(element, "max-height", "260px");
            applyImportantStyle(element, "margin", "0 auto");
            applyImportantStyle(element, "border-radius", "0");
            applyImportantStyle(element, "overflow", "hidden");
        });

        document.querySelectorAll(".prompt-example-video-preview video").forEach((element) => {
            applyImportantStyle(element, "width", "auto");
            applyImportantStyle(element, "max-width", "100%");
            applyImportantStyle(element, "height", "100%");
            applyImportantStyle(element, "min-height", "150px");
            applyImportantStyle(element, "max-height", "260px");
            applyImportantStyle(element, "object-fit", "contain");
            applyImportantStyle(element, "border-radius", "0");
        });
    };

    const syncOutputColumnHeight = () => {
        const row = document.querySelector(".lance-main-row");
        const inputColumn = document.querySelector(".lance-input-column");
        const outputColumn = document.querySelector(".lance-output-column");
        if (!row || !inputColumn || !outputColumn) {
            return;
        }

        if (window.matchMedia("(max-width: 900px)").matches) {
            row.style.removeProperty("--lance-input-column-height");
            outputColumn.style.removeProperty("height");
            outputColumn.style.removeProperty("min-height");
            outputColumn.style.removeProperty("max-height");
            return;
        }

        const height = Math.ceil(inputColumn.getBoundingClientRect().height);
        if (height <= 0) {
            return;
        }
        const heightPx = `${height}px`;
        row.style.setProperty("--lance-input-column-height", heightPx);
        outputColumn.style.height = heightPx;
        outputColumn.style.minHeight = heightPx;
        outputColumn.style.maxHeight = heightPx;
    };

    const scheduleSync = () => requestAnimationFrame(() => {
        enforceLanceLabelTypography();
        enforceRecommendedCaseText();
        enforcePromptDatasetText();
        enforcePromptExampleRows();
        syncOutputColumnHeight();
    });
    const attachObservers = () => {
        const inputColumn = document.querySelector(".lance-input-column");
        const row = document.querySelector(".lance-main-row");
        if (!inputColumn || !row || row.dataset.lanceHeightObserverAttached === "true") {
            return;
        }
        row.dataset.lanceHeightObserverAttached = "true";
        new ResizeObserver(scheduleSync).observe(inputColumn);
        new MutationObserver(scheduleSync).observe(inputColumn, {
            attributes: true,
            childList: true,
            subtree: true,
        });
        window.addEventListener("resize", scheduleSync);
        scheduleSync();
        setTimeout(scheduleSync, 250);
        setTimeout(scheduleSync, 1000);
    };

    enforceLanceLabelTypography();
    enforceRecommendedCaseText();
    enforcePromptDatasetText();
    enforcePromptExampleRows();
    attachObservers();
    new MutationObserver(() => {
        enforceLanceLabelTypography();
        enforceRecommendedCaseText();
        enforcePromptDatasetText();
        enforcePromptExampleRows();
        attachObservers();
    }).observe(document.body, {
        childList: true,
        subtree: true,
    });
}
"""

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
VIDEO_RESOLUTION_CHOICES = ["video_360p", "video_480p"]
VIDEO_RESOLUTION_DISPLAY_CHOICES = [
    ("video_360p", "video_360p"),
    ("video_480p（Higher quota usage. Use sparingly.）", "video_480p"),
]
VIDEO_EDIT_RESOLUTION_CHOICES = [DEFAULT_VIDEO_EDIT_RESOLUTION]
IMAGE_RESOLUTION_CHOICES = [DEFAULT_IMAGE_RESOLUTION]
RESOLUTION_CHOICES = VIDEO_RESOLUTION_CHOICES + IMAGE_RESOLUTION_CHOICES
CAPTION_SYSTEM_PROMPT_TEMPLATE = (
    "Describe the key features of the input {vision_type}, including color, shape, size, texture, objects, background."
)
V2T_CAPTION_SYSTEM_PROMPT = CAPTION_SYSTEM_PROMPT_TEMPLATE.format(vision_type="video")
I2T_CAPTION_SYSTEM_PROMPT = CAPTION_SYSTEM_PROMPT_TEMPLATE.format(vision_type="image")
V2T_QA_SYSTEM_PROMPT = "View the video  attentively and provide a suitable answer to the posed question."
I2T_QA_SYSTEM_PROMPT = "View the image attentively and provide a suitable answer to the posed question."
 
 
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

def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def running_on_space() -> bool:
    return bool(os.getenv("SPACE_ID") or os.getenv("SPACE_HOST"))


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
    configured = os.getenv("LANCE_MODEL_BASE_DIR")
    if configured:
        configured_path = Path(configured).expanduser()
        if _path_can_be_created_or_written(configured_path):
            return configured_path
    if LOCAL_MODEL_BASE_DIR.exists():
        return LOCAL_MODEL_BASE_DIR
    if running_on_space() and SPACE_MODEL_BASE_DIR.exists() and os.access(SPACE_MODEL_BASE_DIR, os.W_OK):
        return SPACE_MODEL_BASE_DIR
    return LOCAL_MODEL_BASE_DIR


def _path_can_be_created_or_written(path: Path) -> bool:
    if path.exists():
        return path.is_dir() and os.access(path, os.W_OK)
    probe = path.parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return probe.exists() and os.access(probe, os.W_OK)


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


def get_model_download_allow_patterns(model_variant: Optional[str] = None) -> list[str]:
    variant = normalize_model_variant(model_variant)
    model_dir_name = MODEL_VARIANT_TO_DIR[variant]
    return [
        f"{model_dir_name}/**",
        "Qwen2.5-VL-ViT/**",
        "Wan2.2_VAE.pth",
        "generation_config.json",
        "llm_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "config.json",
    ]


def _get_safetensors_first_tensor_dtype(path: Path) -> Optional[torch.dtype]:
    if not path.exists():
        return None
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        if not keys:
            return None
        return f.get_tensor(keys[0]).dtype


def convert_model_weights_to_bf16_inplace(model_path: Path) -> bool:
    weight_path = model_path / "model.safetensors"
    if not weight_path.exists():
        return False

    first_dtype = _get_safetensors_first_tensor_dtype(weight_path)
    if first_dtype is None or first_dtype == torch.bfloat16:
        return False

    if first_dtype != torch.float32:
        print(
            f"[startup] Skipping bf16 conversion for {weight_path} because the first tensor dtype is {first_dtype}.",
            flush=True,
        )
        return False

    temp_path = weight_path.with_suffix(".bf16.safetensors.tmp")
    print(f"[startup] Converting {weight_path} to bf16 to reduce disk usage.", flush=True)
    with safe_open(str(weight_path), framework="pt", device="cpu") as f:
        metadata = f.metadata()
        tensor_names = list(f.keys())
        tensors = {}
        for name in tensor_names:
            tensor = f.get_tensor(name)
            tensors[name] = tensor.to(torch.bfloat16) if tensor.dtype == torch.float32 else tensor
        save_file(tensors, str(temp_path), metadata=metadata)

    os.replace(temp_path, weight_path)
    print(f"[startup] Replaced original fp32 weights with bf16 weights at {weight_path}.", flush=True)
    return True


def compact_downloaded_model_weights(model_base_dir: Path, variants: Optional[list[str]] = None) -> None:
    model_dir_names = variants or [MODEL_VARIANT_TO_DIR[MODEL_VARIANT_IMAGE], MODEL_VARIANT_TO_DIR[MODEL_VARIANT_VIDEO]]
    for model_dir_name in model_dir_names:
        model_path = model_base_dir / model_dir_name
        try:
            convert_model_weights_to_bf16_inplace(model_path)
        except Exception as exc:
            print(f"[startup] bf16 compaction skipped for {display_path(model_path)}: {exc}", flush=True)


def ensure_model_assets(model_variant: Optional[str] = None) -> Path:
    model_base_dir = get_model_base_dir()
    os.environ["LANCE_MODEL_BASE_DIR"] = display_path(model_base_dir)
    model_path = get_model_path(model_variant)

    required_paths = get_required_model_asset_paths(model_base_dir, model_path)
    if all(path.exists() for path in required_paths):
        compact_downloaded_model_weights(model_base_dir, [MODEL_VARIANT_TO_DIR[normalize_model_variant(model_variant)]])
        return model_path

    downloads_model_base_dir = Path("downloads")
    if model_base_dir == Path(".") and downloads_model_base_dir.exists():
        downloads_model_path = downloads_model_base_dir / MODEL_VARIANT_TO_DIR[normalize_model_variant(model_variant)]
        downloads_required_paths = get_required_model_asset_paths(downloads_model_base_dir, downloads_model_path)
        if all(path.exists() for path in downloads_required_paths):
            model_base_dir = downloads_model_base_dir
            model_path = downloads_model_path
            required_paths = downloads_required_paths
            os.environ["LANCE_MODEL_BASE_DIR"] = display_path(model_base_dir)
            compact_downloaded_model_weights(model_base_dir, [MODEL_VARIANT_TO_DIR[normalize_model_variant(model_variant)]])
            return model_path

    auto_download = env_flag("LANCE_AUTO_DOWNLOAD", running_on_space())
    if not auto_download:
        missing = "\n".join(f"- {display_path(path)}" for path in required_paths if not path.exists())
        raise FileNotFoundError(
            "Lance model assets are missing. Set LANCE_MODEL_BASE_DIR or enable "
            f"LANCE_AUTO_DOWNLOAD=1.\nMissing files:\n{missing}"
        )

    model_base_dir.mkdir(parents=True, exist_ok=True)
    repo_id = os.getenv("LANCE_MODEL_REPO_ID", DEFAULT_MODEL_REPO_ID)
    print(f"[startup] Downloading Lance model assets from {repo_id} to {display_path(model_base_dir)}", flush=True)
    hub_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    snapshot_path = Path(
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_base_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            token=hub_token,
            allow_patterns=get_model_download_allow_patterns(model_variant),
        )
    )
    if snapshot_path != model_base_dir and not model_path.exists():
        os.environ["LANCE_MODEL_BASE_DIR"] = display_path(snapshot_path)
        model_path = get_model_path(model_variant)
    compact_downloaded_model_weights(model_base_dir, [MODEL_VARIANT_TO_DIR[normalize_model_variant(model_variant)]])
    return model_path


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


def normalize_frame_interpolation(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "open"}


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


def get_aspect_ratio_for_output_resolution(task: str, output_resolution: str, video_resolution: Optional[str] = None) -> str:
    internal_task = normalize_task(task)
    resolution_text = str(output_resolution or "").strip()
    size_map = get_size_map_for_task(internal_task, video_resolution)
    for ratio in ASPECT_RATIO_CHOICES:
        width, height = size_map[ratio]
        if resolution_text == format_size_markdown(internal_task, width, height):
            return ratio
    return get_default_aspect_ratio(internal_task)


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
    return f'<div class="{class_names}">{icon_html}<span>{html.escape(text)}</span></div>'


def update_size_from_aspect_ratio(task: str, aspect_ratio: str, video_resolution: Optional[str] = None):
    width, height = get_size_for_aspect_ratio(task, aspect_ratio, video_resolution)
    return height, width, gr.update(
        choices=get_output_resolution_choices_for_task(task, video_resolution),
        value=format_size_markdown(task, width, height),
    )


def update_aspect_ratio_from_output_resolution(task: str, output_resolution: str, video_resolution: Optional[str] = None):
    aspect_ratio = get_aspect_ratio_for_output_resolution(task, output_resolution, video_resolution)
    width, height = get_size_for_aspect_ratio(task, aspect_ratio, video_resolution)
    return aspect_ratio, height, width


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


def apply_prompt_example(task: str, evt: gr.SelectData):
    prompt_text = ""
    if isinstance(evt.row_value, list) and evt.row_value:
        prompt_text = str(evt.row_value[0])
    elif isinstance(evt.value, list) and evt.value:
        prompt_text = str(evt.value[0])
    elif evt.value is not None:
        prompt_text = str(evt.value)
    defaults = reset_generation_defaults_for_task(task)
    return (prompt_text, *defaults)


def make_prompt_example_click_handler(prompt_text: str):
    """Create a click handler for custom text-to-visual prompt-example rows.

    gr.Dataset and gr.Examples render long text through compact preview cells, so
    long prompts/instructions/questions can be truncated before CSS gets a chance
    to wrap them. The custom rows below use normal buttons for display and keep
    the full prompt string in this closure for click-to-fill behavior.
    """

    def _handler(task: str):
        defaults = reset_generation_defaults_for_task(task)
        return (prompt_text, *defaults)

    return _handler


def make_media_prompt_example_click_handler(
    prompt_text: str,
    input_video_path: Optional[str] = None,
    input_image_path: Optional[str] = None,
):
    """Create a click handler for edit/understanding example rows.

    The row button renders the complete prompt/instruction/question, while the
    closure also carries the matching media path so one click still fills every
    required input component.
    """

    def _handler(task: str):
        defaults = reset_generation_defaults_for_task(task)
        return (prompt_text, input_video_path, input_image_path, *defaults)

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


def resolve_browser_video_example_path(path: str) -> str:
    candidate = Path(path)
    compatible_candidate = candidate.with_name(f"{candidate.stem}_h264{candidate.suffix}")
    repo_compatible_candidate = REPO_ROOT / compatible_candidate
    if not compatible_candidate.is_absolute() and repo_compatible_candidate.exists():
        return str(repo_compatible_candidate.resolve())
    if compatible_candidate.is_absolute() and compatible_candidate.exists():
        return str(compatible_candidate.resolve())
    repo_candidate = REPO_ROOT / candidate
    if not candidate.is_absolute() and repo_candidate.exists():
        return str(repo_candidate.resolve())
    if candidate.is_absolute() and candidate.exists():
        return str(candidate.resolve())
    return resolve_example_path(path)


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
    examples = []
    for output_name, prompt in items:
        examples.append([prompt])
    return examples


def make_edit_examples(task_label: str, relative_path: str, limit: int, media_type: str) -> list[list]:
    data = load_json_examples(relative_path)
    examples = []
    for sample in list(data.values())[:limit]:
        interleave = sample["interleave_array"]
        prompt = interleave[0]
        media_path = resolve_example_path(interleave[1])
        examples.append([
            prompt,
            media_path if media_type == "video" else None,
            media_path if media_type == "image" else None,
        ])
    return examples


def make_understanding_examples(task_label: str, relative_path: str, limit: int, media_type: str) -> list[list]:
    data = load_json_examples(relative_path)
    examples = []
    for sample in list(data.values())[:limit]:
        interleave = sample["interleave_array"]
        media_path = (
            resolve_browser_video_example_path(interleave[0])
            if media_type == "video"
            else resolve_example_path(interleave[0])
        )
        text_payload = interleave[1]
        question = text_payload[1] if isinstance(text_payload, list) and len(text_payload) > 1 else ""
        examples.append([
            question,
            media_path if media_type == "video" else None,
            media_path if media_type == "image" else None,
        ])
    return examples


def make_understanding_system_prompt_map(relative_path: str, task: str) -> dict[str, str]:
    data = load_json_examples(relative_path)
    system_prompts = {}
    for sample in data.values():
        interleave = sample["interleave_array"]
        text_payload = interleave[1]
        if not isinstance(text_payload, list) or len(text_payload) < 2:
            continue
        system_prompts[text_payload[1]] = normalize_understanding_system_prompt(task, text_payload[0])
    return system_prompts


VIDEO_GENERATION_EXAMPLES = make_generation_examples(
    TASK_LABEL_VIDEO_GENERATION,
    "config/examples/t2v_example.json",
    limit=6,
    image_task=False,
    #selected_keys=["000000.mp4", "000002.mp4", "000005.mp4", "000004.mp4", "000008.mp4"],
    selected_keys=["000004.mp4", "000002.mp4", "000000.mp4", "000005.mp4", "000008.mp4", "000007.mp4"],
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
VIDEO_UNDERSTANDING_SYSTEM_PROMPTS = make_understanding_system_prompt_map(
    "config/examples/x2t_video_example.json",
    TASK_X2T_VIDEO,
)
IMAGE_GENERATION_EXAMPLES = make_generation_examples(
    TASK_LABEL_IMAGE_GENERATION,
    "config/examples/t2i_example.json",
    limit=5,
    image_task=True,
    selected_keys=["000000.png", "000003.png", "000006.png", "000008.png", "000009.png"],
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
    limit=3,
    media_type="image",
)
IMAGE_UNDERSTANDING_SYSTEM_PROMPTS = make_understanding_system_prompt_map(
    "config/examples/x2t_image_example.json",
    TASK_X2T_IMAGE,
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


def run_rife_interpolation(video_path: Path, device_id: int, exp: int = 1) -> tuple[Path, str]:
    rife_script = RIFE_SCRIPT_PATH
    if not rife_script.exists():
        return video_path, ""

    output_path = video_path.with_name(f"{video_path.stem}_rife_{2 ** exp}x{video_path.suffix}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device_id)
    command = [
        "python3",
        str(rife_script),
        "--exp",
        str(exp),
        "--video",
        str(video_path),
        "--output",
        str(output_path),
        "--model",
        str(RIFE_MODEL_DIR),
    ]
    try:
        subprocess.run(
            command,
            cwd=str(video_path.parent),
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return video_path, ""
    if not output_path.exists():
        return video_path, ""
    return output_path, ""


def filter_run_logs(log_text: str) -> str:
    if not log_text:
        return ""
    blocked_tokens = (
        "[rife]",
        "frame_interpolation=",
        "original_video_path=",
        "rife_error=",
        "interpolation",
        "rife",
        "Traceback (most recent call last):",
        "During handling of the above exception",
        "RuntimeError: RIFE failed",
        "ffmpeg version",
        "built with gcc",
        "configuration:",
        "libavutil",
        "libavcodec",
        "libavformat",
        "libavdevice",
        "libavfilter",
        "libswscale",
        "libswresample",
        "libpostproc",
        "input #",
        "output #",
        "metadata:",
        "stream #",
        "duration:",
        "output file #0 does not contain any stream",
        "./temp/audio.mkv",
        "./temp/audio.m4a",
        "audio transfer failed",
        "lossless audio transfer failed",
        "will not merge audio",
    )
    kept_lines = []
    for line in log_text.splitlines():
        normalized = line.strip().lower()
        if any(token in normalized for token in blocked_tokens):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines).strip()


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


class LanceT2VV2TPipeline:
    def __init__(self, device_id: int, model_variant: str = MODEL_VARIANT_VIDEO) -> None:
        self._init_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self.initialized = False
        self.device = device_id
        self.model_variant = normalize_model_variant(model_variant)
        self.logger = get_logger(f"lance_{self.model_variant}_gpu{device_id}")

        self.model: Optional[Lance] = None
        self.vae_model: Optional[WanVideoVAE] = None
        self.vae_config: Optional[AutoEncoderParams] = None
        self.tokenizer: Optional[Qwen2Tokenizer] = None
        self.new_token_ids: Optional[dict] = None
        self.image_token_id: Optional[int] = None
        self.base_model_args: Optional[ModelArguments] = None
        self.base_data_args: Optional[DataArguments] = None
        self.base_inference_args: Optional[InferenceArguments] = None

    def _log_stage(self, stage_name: str, start_time: float, extra: str = "") -> None:
        elapsed = time.perf_counter() - start_time
        suffix = f" | {extra}" if extra else ""
        print(f"[startup][gpu:{self.device}] {stage_name} done in {elapsed:.2f}s{suffix}", flush=True)

    def _build_base_model_args(self) -> ModelArguments:
        model_path = str(get_model_path(self.model_variant))
        return ModelArguments(
            model_path=model_path,
            vit_type=DEFAULT_VIT_TYPE,
            llm_qk_norm=True,
            llm_qk_norm_und=True,
            llm_qk_norm_gen=True,
            tie_word_embeddings=False,
            max_num_frames=MAX_VIDEO_NUM_FRAMES,
            max_latent_size=64,
            latent_patch_size=[1, 1, 1],
        )

    def _build_base_inference_args(self) -> InferenceArguments:
        return InferenceArguments(
            validation_num_timesteps=DEFAULT_TIMESTEPS,
            validation_timestep_shift=DEFAULT_TIMESTEP_SHIFT,
            copy_init_moe=True,
            visual_und=True,
            visual_gen=True,
            vae_model_type="wan",
            apply_qwen_2_5_vl_pos_emb=True,
            apply_chat_template=False,
            cfg_type=0,
            validation_data_seed=42,
            video_height=DEFAULT_HEIGHT,
            video_width=DEFAULT_WIDTH,
            num_frames=DEFAULT_NUM_FRAMES,
            task=DEFAULT_TASK,
            save_path_gen=str(RESULTS_ROOT),
            resolution=DEFAULT_RESOLUTION,
            text_template=TEXT_TEMPLATE,
            use_KVcache=USE_KVCACHE,
        )

    def initialize(self) -> None:
        with self._init_lock:
            if self.initialized:
                return

            ensure_dirs()
            resolved_model_path = ensure_model_assets(self.model_variant)
            print(
                f"[startup][gpu:{self.device}][{self.model_variant}] Using Lance model path: {resolved_model_path}",
                flush=True,
            )
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Lance T2V/V2T Gradio requires a GPU environment.")
            if self.device >= torch.cuda.device_count():
                raise RuntimeError(
                    f"GPU {self.device} is unavailable. Detected {torch.cuda.device_count()} GPU(s)."
                )
            torch.cuda.set_device(self.device)

            model_args = self._build_base_model_args()
            data_args = DataArguments()
            inference_args = self._build_base_inference_args()
            apply_inference_defaults(model_args, data_args, inference_args)
            inference_args.validation_noise_seed = inference_args.validation_data_seed

            self.base_model_args = model_args
            self.base_data_args = data_args
            self.base_inference_args = inference_args

            set_seed(inference_args.global_seed)

            stage_start = time.perf_counter()
            print(
                f"[startup][gpu:{self.device}] Loading LLM config: {Path(model_args.model_path) / 'llm_config.json'}",
                flush=True,
            )
            llm_config: Qwen2Config = Qwen2Config.from_json_file(str(Path(model_args.model_path) / "llm_config.json"))
            self._log_stage("LLM config load", stage_start)

            llm_config.layer_module = model_args.layer_module
            llm_config.qk_norm = model_args.llm_qk_norm
            llm_config.qk_norm_und = model_args.llm_qk_norm_und
            llm_config.qk_norm_gen = model_args.llm_qk_norm_gen
            llm_config.tie_word_embeddings = model_args.tie_word_embeddings
            llm_config.freeze_und = inference_args.freeze_und
            llm_config.apply_qwen_2_5_vl_pos_emb = inference_args.apply_qwen_2_5_vl_pos_emb

            stage_start = time.perf_counter()
            print(f"[startup][gpu:{self.device}] Initializing LLM weights: {model_args.model_path}", flush=True)
            language_model: Qwen2ForCausalLM = Qwen2ForCausalLM(llm_config)
            self._log_stage("LLM weight init", stage_start)

            vit_model = None
            vit_config = None
            if inference_args.visual_und:
                if model_args.vit_type not in ("qwen2_5_vl", "qwen_2_5_vl_original"):
                    raise ValueError(f"Unsupported vit_type: {model_args.vit_type}")
                stage_start = time.perf_counter()
                print(f"[startup][gpu:{self.device}] Loading VIT config: {model_args.vit_path}", flush=True)
                vit_config = Qwen2_5_VLVisionConfig.from_pretrained(model_args.vit_path)
                self._log_stage("VIT config load", stage_start)

                stage_start = time.perf_counter()
                print(
                    f"[startup][gpu:{self.device}] Loading VIT weights: {Path(model_args.vit_path) / 'vit.safetensors'}",
                    flush=True,
                )
                vit_model = Qwen2_5_VisionTransformerPretrainedModel(vit_config)
                vit_weights = load_file(str(Path(model_args.vit_path) / "vit.safetensors"))
                vit_model.load_state_dict(vit_weights, strict=True)
                self._log_stage("VIT weight load", stage_start)
                clean_memory(vit_weights)

            if inference_args.visual_gen:
                stage_start = time.perf_counter()
                print(f"[startup][gpu:{self.device}] Initializing VAE", flush=True)
                vae_model = WanVideoVAE()
                vae_config = deepcopy(vae_model.vae_config)
                self._log_stage("VAE init", stage_start)
            else:
                vae_model = None
                vae_config = None

            config = LanceConfig(
                visual_gen=inference_args.visual_gen,
                visual_und=inference_args.visual_und,
                llm_config=llm_config,
                vit_config=vit_config if inference_args.visual_und else None,
                vae_config=vae_config if inference_args.visual_gen else None,
                latent_patch_size=model_args.latent_patch_size,
                max_num_frames=model_args.max_num_frames,
                max_latent_size=model_args.max_latent_size,
                vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
                connector_act=model_args.connector_act,
                interpolate_pos=model_args.interpolate_pos,
                timestep_shift=inference_args.timestep_shift,
            )
            model: Lance = Lance(
                language_model=language_model,
                vit_model=vit_model if inference_args.visual_und else None,
                vit_type=model_args.vit_type,
                config=config,
                training_args=inference_args,
            )

            stage_start = time.perf_counter()
            print(f"[startup][gpu:{self.device}] Casting Lance model to bf16 on CPU", flush=True)
            model = model.to(dtype=torch.bfloat16)
            self._log_stage("Lance model bf16 cast", stage_start)

            stage_start = time.perf_counter()
            print(f"[startup][gpu:{self.device}] Loading tokenizer: {model_args.model_path}", flush=True)
            tokenizer: Qwen2Tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path)
            tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
            self._log_stage("tokenizer load and special token init", stage_start, extra=f"num_new_tokens={num_new_tokens}")

            if inference_args.copy_init_moe:
                language_model.init_moe()

            init_from_model_path_if_needed(model, model_args)

            if num_new_tokens > 0:
                model.language_model.resize_token_embeddings(len(tokenizer))
                model.config.llm_config.vocab_size = len(tokenizer)
                model.language_model.config.vocab_size = len(tokenizer)

            if model_args.vit_type.lower() == "qwen2_5_vl":
                from common.model.hacks import hack_qwen2_5_vl_config

                language_model = hack_qwen2_5_vl_config(language_model)

            image_token_id = language_model.config.video_token_id
            new_token_ids.update({"image_token_id": image_token_id})
            model.update_tokenizer(tokenizer=tokenizer)

            if model_args.tie_word_embeddings:
                model.language_model.untie_lm_head()
                model.language_model.copy_new_token_rows_to_lm_head(num_new_tokens)
                model_args.tie_word_embeddings = False
                llm_config.tie_word_embeddings = False
            else:
                assert (
                    model.language_model.get_input_embeddings().weight.data.data_ptr()
                    != model.language_model.get_output_embeddings().weight.data.data_ptr()
                ), "tie_word_embeddings conflict"

            stage_start = time.perf_counter()
            print(f"[startup][gpu:{self.device}] Moving Lance model to GPU {self.device}", flush=True)
            model = model.to(device=self.device)
            self._log_stage("Lance model move to GPU", stage_start)
            model.eval()
            if vae_model is not None and hasattr(vae_model, "eval"):
                vae_model.eval()

            self.model = model
            self.vae_model = vae_model
            self.vae_config = vae_config
            self.tokenizer = tokenizer
            self.new_token_ids = new_token_ids
            self.image_token_id = image_token_id
            self.initialized = True
            print(
                f"[startup][gpu:{self.device}][{self.model_variant}] Lance multimodal Gradio model loaded and ready for reuse.",
                flush=True,
            )

    def unload(self) -> None:
        with self._init_lock:
            if self.model is not None:
                self.model.cpu()
            if self.vae_model is not None and hasattr(self.vae_model, "vae"):
                vae_inner = self.vae_model.vae
                if hasattr(vae_inner, "model"):
                    vae_inner.model.cpu()

            self.model = None
            self.vae_model = None
            self.vae_config = None
            self.tokenizer = None
            self.new_token_ids = None
            self.image_token_id = None
            self.base_model_args = None
            self.base_data_args = None
            self.base_inference_args = None
            self.initialized = False
            gc.collect()
            if torch.cuda.is_available():
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

    def _build_request_batch(
        self,
        prompt_file: Path,
        model_args: ModelArguments,
        data_args: DataArguments,
        inference_args: InferenceArguments,
    ):
        assert self.tokenizer is not None
        assert self.new_token_ids is not None
        assert self.vae_config is not None

        dataset_config = DataConfig.from_yaml(str(prompt_file))
        if inference_args.visual_und:
            dataset_config.vit_patch_size = model_args.vit_patch_size
            dataset_config.vit_patch_size_temporal = model_args.vit_patch_size_temporal
            dataset_config.vit_max_num_patch_per_side = model_args.vit_max_num_patch_per_side
        if inference_args.visual_gen:
            vae_downsample = tuple_mul(
                tuple(model_args.latent_patch_size),
                (
                    self.vae_config.downsample_temporal,
                    self.vae_config.downsample_spatial,
                    self.vae_config.downsample_spatial,
                ),
            )
            dataset_config.latent_patch_size = model_args.latent_patch_size
            dataset_config.vae_downsample = vae_downsample
            dataset_config.max_latent_size = model_args.max_latent_size
            dataset_config.max_num_frames = model_args.max_num_frames

        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob

        dataset_config.num_frames = inference_args.num_frames
        dataset_config.H = inference_args.video_height
        dataset_config.W = inference_args.video_width
        dataset_config.task = inference_args.task
        dataset_config.resolution = inference_args.resolution
        dataset_config.text_template = inference_args.text_template

        val_dataset = ValidationDataset(
            jsonl_path=str(prompt_file),
            tokenizer=self.tokenizer,
            data_args=data_args,
            model_args=model_args,
            training_args=inference_args,
            new_token_ids=self.new_token_ids,
            dataset_config=dataset_config,
            local_rank=0,
            world_size=1,
        )
        return simple_custom_collate([val_dataset[0]])

    def generate(
        self,
        task: str,
        prompt: str,
        system_prompt: Optional[str],
        input_video: Optional[str],
        input_image: Optional[str],
        height: int,
        width: int,
        num_frames: int,
        seed: int,
        resolution: str,
        validation_num_timesteps: int,
        validation_timestep_shift: float,
        cfg_text_scale: float,
        enable_frame_interpolation: bool,
    ):
        self.initialize()
        internal_task = normalize_task(task)
        prompt = (prompt or "").strip()
        input_video = str(input_video).strip() if input_video else ""
        input_image = str(input_image).strip() if input_image else ""

        if internal_task in GENERATION_TASKS and not prompt:
            return None, None, "", "Please enter a prompt.", ""
        if internal_task in UNDERSTANDING_TASKS and not prompt:
            return None, None, "", "Please enter a question.", ""
        if internal_task in {TASK_VIDEO_EDIT, TASK_X2T_VIDEO} and not input_video:
            return None, None, "", "Please upload an input video.", ""
        if internal_task in {TASK_IMAGE_EDIT, TASK_X2T_IMAGE} and not input_image:
            return None, None, "", "Please upload an input image.", ""
        if height <= 0 or width <= 0:
            return None, None, "", "Height and width must be greater than 0.", ""
        if num_frames <= 0:
            return None, None, "", "The number of frames must be greater than 0.", ""

        assert self.model is not None
        assert self.tokenizer is not None
        assert self.new_token_ids is not None
        assert self.image_token_id is not None
        assert self.base_model_args is not None
        assert self.base_data_args is not None
        assert self.base_inference_args is not None
        active_model_path = self.base_model_args.model_path

        with self._generate_lock:
            torch.cuda.set_device(self.device)
            actual_seed = normalize_seed(int(seed))
            prompt_file = create_request_json(
                task=internal_task,
                prompt=prompt,
                input_video=input_video,
                input_image=input_image,
                system_prompt=system_prompt,
            )
            save_dir = build_save_dir(internal_task)
            save_dir.mkdir(parents=True, exist_ok=True)
            request_started_at = datetime.now().isoformat(timespec="seconds")

            request_model_args = deepcopy(self.base_model_args)
            request_model_args.cfg_text_scale = float(cfg_text_scale)

            request_data_args = deepcopy(self.base_data_args)
            request_data_args.val_dataset_config_file = str(prompt_file)

            request_inference_args = deepcopy(self.base_inference_args)
            request_inference_args.validation_num_timesteps = int(validation_num_timesteps)
            request_inference_args.validation_timestep_shift = float(validation_timestep_shift)
            request_inference_args.validation_data_seed = actual_seed
            request_inference_args.validation_noise_seed = actual_seed
            request_inference_args.video_height = int(height)
            request_inference_args.video_width = int(width)
            request_inference_args.num_frames = int(num_frames)
            display_resolution = str(resolution)
            backend_resolution = normalize_resolution_for_backend(display_resolution, internal_task)
            request_inference_args.resolution = backend_resolution
            request_inference_args.save_path_gen = str(save_dir)
            request_inference_args.task = internal_task
            request_inference_args.text_template = TEXT_TEMPLATE
            request_inference_args.prompt_data_dict = {}

            try:
                print(
                    "[lance_gradio_t2v_v2t] Start generation "
                    f"| task={internal_task} | gpu={self.device} | seed={actual_seed} | "
                    f"size={height}x{width} | frames={num_frames} | resolution={display_resolution}",
                    flush=True,
                )
                val_data_cpu = self._build_request_batch(
                    prompt_file=prompt_file,
                    model_args=request_model_args,
                    data_args=request_data_args,
                    inference_args=request_inference_args,
                )
                generate_start = time.perf_counter()
                validate_on_fixed_batch(
                    fsdp_model=self.model,
                    vae_model=self.vae_model,
                    tokenizer=self.tokenizer,
                    val_data_cpu=val_data_cpu,
                    training_args=request_inference_args,
                    model_args=request_model_args,
                    inference_args=request_inference_args,
                    new_token_ids=self.new_token_ids,
                    image_token_id=self.image_token_id,
                    device=self.device,
                    save_source_video=False,
                    save_path_gen=request_inference_args.save_path_gen,
                    save_path_gt="",
                )
                elapsed = time.perf_counter() - generate_start
                save_prompt_results(request_inference_args.prompt_data_dict, request_inference_args.save_path_gen, self.logger)
                clean_memory()

                video_path = find_generated_video(save_dir) if internal_task in {TASK_T2V, TASK_VIDEO_EDIT} else None
                original_video_path = video_path
                rife_log = ""
                rife_error = ""
                frame_interpolation_enabled = normalize_frame_interpolation(enable_frame_interpolation) and internal_task in {TASK_T2V, TASK_VIDEO_EDIT} and RIFE_AVAILABLE
                if frame_interpolation_enabled and video_path is not None:
                    try:
                        clean_memory()
                        print(
                            "[rife] Start frame interpolation "
                            f"| task={internal_task} | gpu={self.device} | input={video_path}",
                            flush=True,
                        )
                        video_path, rife_log = run_rife_interpolation(video_path, self.device, exp=1)
                    except Exception:
                        rife_error = traceback.format_exc()
                        print(rife_error, flush=True)
                image_path = find_generated_image(save_dir) if internal_task in {TASK_T2I, TASK_IMAGE_EDIT} else None
                text_result = extract_text_result(save_dir) if internal_task in UNDERSTANDING_TASKS else ""
                record = {
                    "request_started_at": request_started_at,
                    "request_finished_at": datetime.now().isoformat(timespec="seconds"),
                    "status": "success",
                    "task": internal_task,
                    "model_variant": self.model_variant,
                    "model_path": active_model_path,
                    "gpu": self.device,
                    "prompt": prompt,
                    "system_prompt": normalize_understanding_system_prompt(internal_task, system_prompt)
                    if internal_task in UNDERSTANDING_TASKS
                    else "",
                    "input_video": input_video,
                    "input_image": input_image,
                    "seed": actual_seed,
                    "height": int(height),
                    "width": int(width),
                    "num_frames": int(num_frames),
                    "resolution": display_resolution,
                    "backend_resolution": backend_resolution,
                    "validation_num_timesteps": int(validation_num_timesteps),
                    "validation_timestep_shift": float(validation_timestep_shift),
                    "cfg_text_scale": float(cfg_text_scale),
                    "frame_interpolation": frame_interpolation_enabled,
                    "elapsed_seconds": round(elapsed, 3),
                    "prompt_file": str(prompt_file),
                    "output_dir": str(save_dir),
                    "original_video_path": str(original_video_path) if original_video_path is not None else "",
                    "video_path": str(video_path) if video_path is not None else "",
                    "image_path": str(image_path) if image_path is not None else "",
                    "text_result": text_result,
                    "rife_error": rife_error,
                }
                if internal_task in {TASK_T2V, TASK_VIDEO_EDIT} and video_path is None:
                    record["status"] = "completed_without_video"
                if internal_task in {TASK_T2I, TASK_IMAGE_EDIT} and image_path is None:
                    record["status"] = "completed_without_image"
                if internal_task in UNDERSTANDING_TASKS and not text_result:
                    record["status"] = "completed_without_text"
                save_generation_record(record, save_dir)

                logs = "\n".join(
                    [
                        "[lance_gradio_t2v_v2t] Inference finished in-process.",
                        f"task={internal_task}",
                        f"model_variant={self.model_variant}",
                        f"model_path={active_model_path}",
                        f"gpu={self.device}",
                        f"seed={actual_seed}",
                        f"height={height}",
                        f"width={width}",
                        f"num_frames={num_frames}",
                        f"resolution={display_resolution}",
                        f"backend_resolution={backend_resolution}",
                        f"validation_num_timesteps={validation_num_timesteps}",
                        f"validation_timestep_shift={validation_timestep_shift}",
                        f"cfg_text_scale={cfg_text_scale}",
                        f"elapsed={elapsed:.2f}s",
                        f"output_dir={save_dir}",
                    ]
                )
                logs = filter_run_logs(logs)
                if rife_log:
                    logs = "\n".join([logs, filter_run_logs(rife_log)]).strip()

                if internal_task in {TASK_T2V, TASK_VIDEO_EDIT}:
                    if video_path is None:
                        status = (
                            "Inference completed, but no output video was found.\n\n"
                            f"- Task: `{internal_task}`\n"
                            f"- Model: `{self.model_variant}`\n"
                            f"- Model path: `{active_model_path}`\n"
                            f"- GPU: `{self.device}`\n"
                            f"- Actual seed: `{actual_seed}`\n"
                            f"- Output directory: `{save_dir}`"
                        )
                        return None, None, "", status, logs
                    # status = (
                    #     "Inference completed.\n\n"
                    #     f"- Task: `{internal_task}`\n"
                    #     f"- Model: `{self.model_variant}`\n"
                    #     f"- Model path: `{active_model_path}`\n"
                    #     f"- GPU: `{self.device}`\n"
                    #     f"- Actual seed: `{actual_seed}`\n"
                    #     f"- Output directory: `{save_dir}`\n"
                    #     f"- Result file: `{video_path}`"
                    # )
                    status = ""
                    return str(video_path), None, "", status, logs

                if internal_task in {TASK_T2I, TASK_IMAGE_EDIT}:
                    if image_path is None:
                        status = (
                            "Inference completed, but no output image was found.\n\n"
                            f"- Task: `{internal_task}`\n"
                            f"- Model: `{self.model_variant}`\n"
                            f"- Model path: `{active_model_path}`\n"
                            f"- GPU: `{self.device}`\n"
                            f"- Actual seed: `{actual_seed}`\n"
                            f"- Output directory: `{save_dir}`"
                        )
                        return None, None, "", status, logs
                    # status = (
                    #     "Inference completed.\n\n"
                    #     f"- Task: `{internal_task}`\n"
                    #     f"- Model: `{self.model_variant}`\n"
                    #     f"- Model path: `{active_model_path}`\n"
                    #     f"- GPU: `{self.device}`\n"
                    #     f"- Actual seed: `{actual_seed}`\n"
                    #     f"- Output directory: `{save_dir}`\n"
                    #     f"- Result file: `{image_path}`"
                    # )
                    status = ""
                    return None, str(image_path), "", status, logs

                # status = (
                #     "Understanding completed.\n\n"
                #     f"- Task: `{task}`\n"
                #     f"- Model: `{self.model_variant}`\n"
                #     f"- Model path: `{active_model_path}`\n"
                #     f"- GPU: `{self.device}`\n"
                #     f"- Actual seed: `{actual_seed}`\n"
                #     f"- Output directory: `{save_dir}`"
                # )
                status = ""
                return None, None, text_result, status, logs
            except Exception:
                error_trace = traceback.format_exc()
                print(error_trace, flush=True)
                record = {
                    "request_started_at": request_started_at,
                    "request_finished_at": datetime.now().isoformat(timespec="seconds"),
                    "status": "failed",
                    "task": internal_task,
                    "model_variant": self.model_variant,
                    "model_path": active_model_path,
                    "gpu": self.device,
                    "prompt": prompt,
                    "input_video": input_video,
                    "input_image": input_image,
                    "seed": actual_seed,
                    "height": int(height),
                    "width": int(width),
                    "num_frames": int(num_frames),
                    "resolution": display_resolution,
                    "backend_resolution": backend_resolution,
                    "validation_num_timesteps": int(validation_num_timesteps),
                    "validation_timestep_shift": float(validation_timestep_shift),
                    "cfg_text_scale": float(cfg_text_scale),
                    "prompt_file": str(prompt_file),
                    "output_dir": str(save_dir),
                    "video_path": "",
                    "image_path": "",
                    "text_result": "",
                    "error": error_trace,
                }
                save_generation_record(record, save_dir)
                status = (
                    "Inference failed.\n\n"
                    f"- Task: `{internal_task}`\n"
                    f"- Model: `{self.model_variant}`\n"
                    f"- Model path: `{active_model_path}`\n"
                    f"- GPU: `{self.device}`\n"
                    f"- Actual seed: `{actual_seed}`\n"
                    f"- Resolution: `{display_resolution}`\n"
                    f"- Output directory: `{save_dir}`"
                )
                return None, None, "", status, error_trace


class PipelinePool:
    def __init__(self, gpu_ids: list[int], model_variant: str = MODEL_VARIANT_VIDEO) -> None:
        if not gpu_ids:
            raise ValueError("At least one GPU must be configured.")
        self.gpu_ids = gpu_ids
        self.model_variant = normalize_model_variant(model_variant)
        self.pipelines = [
            LanceT2VV2TPipeline(device_id=gpu_id, model_variant=self.model_variant)
            for gpu_id in gpu_ids
        ]
        self._available = deque(self.pipelines)
        self._condition = threading.Condition()

    @property
    def size(self) -> int:
        return len(self.pipelines)

    @property
    def gpu_summary(self) -> str:
        return ",".join(str(gpu_id) for gpu_id in self.gpu_ids)

    def initialize_all(self) -> None:
        print(f"[startup][{self.model_variant}] Preparing parallel GPU preload: {self.gpu_ids}", flush=True)
        exceptions: list[Exception] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.size) as executor:
            futures = {
                executor.submit(pipeline.initialize): pipeline.device for pipeline in self.pipelines
            }
            for future in concurrent.futures.as_completed(futures):
                gpu_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    print(f"[startup][gpu:{gpu_id}][{self.model_variant}] Preload failed: {exc}", flush=True)
                    exceptions.append(exc)
        if exceptions:
            raise RuntimeError(
                f"{self.model_variant} preload failed on {len(exceptions)} GPU(s). Please check the terminal logs."
            ) from exceptions[0]
        print(
            f"[startup][{self.model_variant}] GPU preload finished. Ready to handle {self.size} concurrent request(s).",
            flush=True,
        )

    def acquire(self) -> LanceT2VV2TPipeline:
        with self._condition:
            while not self._available:
                self._condition.wait()
            return self._available.popleft()

    def release(self, pipeline: LanceT2VV2TPipeline) -> None:
        with self._condition:
            self._available.append(pipeline)
            self._condition.notify()

    def unload_all(self) -> None:
        print(f"[runtime][{self.model_variant}] Unloading model pool from GPU(s): {self.gpu_ids}", flush=True)
        with self._condition:
            while len(self._available) != len(self.pipelines):
                self._condition.wait()

        for pipeline in self.pipelines:
            pipeline.unload()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print(f"[runtime][{self.model_variant}] Model pool unloaded.", flush=True)

    def generate(
        self,
        task: str,
        prompt: str,
        system_prompt: Optional[str],
        input_video: Optional[str],
        input_image: Optional[str],
        height: int,
        width: int,
        num_frames: int,
        seed: int,
        resolution: str,
        validation_num_timesteps: int,
        validation_timestep_shift: float,
        cfg_text_scale: float,
        enable_frame_interpolation: bool,
    ):
        pipeline = self.acquire()
        try:
            return pipeline.generate(
                task=task,
                prompt=prompt,
                system_prompt=system_prompt,
                input_video=input_video,
                input_image=input_image,
                height=height,
                width=width,
                num_frames=num_frames,
                seed=seed,
                resolution=resolution,
                validation_num_timesteps=validation_num_timesteps,
                validation_timestep_shift=validation_timestep_shift,
                cfg_text_scale=cfg_text_scale,
                enable_frame_interpolation=enable_frame_interpolation,
            )
        finally:
            self.release(pipeline)


ACTIVE_PIPELINE_POOL: Optional[PipelinePool] = None
ACTIVE_POOL_LOCK = threading.Lock()
QUEUE_MAX_SIZE = DEFAULT_QUEUE_SIZE


def get_task_model_variant(task: str) -> str:
    internal_task = normalize_task(task)
    return MODEL_VARIANT_IMAGE if internal_task in IMAGE_TASKS else MODEL_VARIANT_VIDEO


def get_env_int(name: str, default: int) -> int:
    """Read an integer environment variable, falling back safely on invalid values."""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def get_env_float(name: str, default: float) -> float:
    """Read a float environment variable, falling back safely on invalid values."""
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def ensure_flash_attn_installed() -> None:
    try:
        from importlib.metadata import PackageNotFoundError, version as package_version
        current_version = package_version("flash_attn")
        if current_version == DEFAULT_FLASH_ATTN_VERSION:
            print(f"[startup] flash-attn {current_version} already installed.", flush=True)
            return
        print(
            f"[startup] flash-attn {current_version} detected; reinstalling {DEFAULT_FLASH_ATTN_VERSION} from wheel.",
            flush=True,
        )
    except Exception:
        print(
            f"[startup] flash-attn not available; installing {DEFAULT_FLASH_ATTN_VERSION} from wheel.",
            flush=True,
        )

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--no-deps",
        "--force-reinstall",
        DEFAULT_FLASH_ATTN_WHEEL_URL,
    ]
    subprocess.check_call(command)
    print(f"[startup] flash-attn {DEFAULT_FLASH_ATTN_VERSION} installed from wheel.", flush=True)


def get_zerogpu_duration_cap() -> int:
    """Fixed duration requested from ZeroGPU for each run.

    The duration value is a ZeroGPU reservation/timeout hint. Shorter values can
    improve queue priority and reduce wasted quota, but the value must still cover
    model warm-up plus inference. Override per deployment when needed:
        LANCE_ZEROGPU_MAX_DURATION_SECONDS=300
    """
    return max(1, get_env_int("LANCE_ZEROGPU_MAX_DURATION_SECONDS", 300))


def clamp_zerogpu_duration(seconds: int) -> int:
    return max(1, min(int(seconds), get_zerogpu_duration_cap()))


ZERO_GPU_RUN_TASK_DURATION_SECONDS = get_zerogpu_duration_cap()

def is_pipeline_pool_ready_for_task(task: str) -> bool:
    """Retained for compatibility with earlier duration logic."""
    return False


def finalize_zerogpu_duration(estimated_seconds: float, task: str) -> int:
    """Clamp a heuristic duration to the deployment cap with a small safety margin."""
    task_key = normalize_task(task)
    raw_seconds = float(estimated_seconds)
    if raw_seconds <= 0:
        raw_seconds = _estimate_zerogpu_duration_seconds(
            task_key,
            prompt="",
            system_prompt=None,
            input_video=None,
            input_image=None,
            height=0,
            width=0,
            num_frames=0,
            seed=0,
            resolution="",
            validation_num_timesteps=0,
            validation_timestep_shift=0.0,
            cfg_text_scale=0.0,
            enable_frame_interpolation=False,
        )
    return clamp_zerogpu_duration(math.ceil(raw_seconds * 1.15) + 5)


def _estimate_zerogpu_duration_seconds(
    task: str,
    prompt: str,
    system_prompt: Optional[str],
    input_video: Optional[str],
    input_image: Optional[str],
    height: int,
    width: int,
    num_frames: int,
    seed: int,
    resolution: str,
    validation_num_timesteps: int,
    validation_timestep_shift: float,
    cfg_text_scale: float,
    enable_frame_interpolation: bool,
) -> int:
    internal_task = normalize_task(task)
    prompt_length = len((prompt or "").strip())
    has_video_input = bool((input_video or "").strip())
    has_image_input = bool((input_image or "").strip())
    is_video_task = internal_task in {TASK_T2V, TASK_VIDEO_EDIT, TASK_X2T_VIDEO}
    is_image_task = internal_task in {TASK_T2I, TASK_IMAGE_EDIT, TASK_X2T_IMAGE}

    if internal_task == TASK_T2I:
        return 150

    if internal_task == TASK_IMAGE_EDIT:
        return 150

    if internal_task == TASK_X2T_IMAGE:
        return 150

    if internal_task == TASK_X2T_VIDEO:
        return 200

    if internal_task == TASK_VIDEO_EDIT:
        base = 300
        base += min(48, max(0, num_frames - 37) // 2)
        base += 32 if enable_frame_interpolation else 0
        base += 20 if has_video_input else 0
        base += 16 if resolution == "video_480p" else 0
        return base

    if internal_task == TASK_T2V:
        base = 224 if resolution == "video_360p" else 264
        base += min(56, max(0, num_frames - 37) // 2)
        base += 28 if enable_frame_interpolation else 0
        base += min(20, prompt_length // 260)
        return base

    if is_video_task:
        base = 240
        base += min(40, max(0, num_frames - 37) // 2)
        base += 24 if enable_frame_interpolation else 0
        return base

    if is_image_task:
        return 120

    return 160


def get_run_task_gpu_duration(
    task: str,
    prompt: str,
    system_prompt: Optional[str],
    input_video: Optional[str],
    input_image: Optional[str],
    height: int,
    width: int,
    num_frames: int,
    seed: int,
    resolution: str,
    validation_num_timesteps: int,
    validation_timestep_shift: float,
    cfg_text_scale: float,
    enable_frame_interpolation: bool,
) -> int:
    estimated_seconds = _estimate_zerogpu_duration_seconds(
        task=task,
        prompt=prompt,
        system_prompt=system_prompt,
        input_video=input_video,
        input_image=input_image,
        height=height,
        width=width,
        num_frames=num_frames,
        seed=seed,
        resolution=resolution,
        validation_num_timesteps=validation_num_timesteps,
        validation_timestep_shift=validation_timestep_shift,
        cfg_text_scale=cfg_text_scale,
        enable_frame_interpolation=enable_frame_interpolation,
    )
    return finalize_zerogpu_duration(estimated_seconds, task)


def get_pipeline_pool(task: str) -> PipelinePool:
    global ACTIVE_PIPELINE_POOL
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Lance inference requires a GPU. The Gradio UI can start on CPU, but generation is disabled "
            "until GPU hardware is attached."
        )
    model_variant = get_task_model_variant(task)
    with ACTIVE_POOL_LOCK:
        if ACTIVE_PIPELINE_POOL is not None and ACTIVE_PIPELINE_POOL.model_variant == model_variant:
            return ACTIVE_PIPELINE_POOL

        gpu_ids = parse_gpu_ids(os.getenv("LANCE_GPUS", DEFAULT_GPUS))
        if ACTIVE_PIPELINE_POOL is not None:
            previous_variant = ACTIVE_PIPELINE_POOL.model_variant
            print(
                f"[runtime] Switching Lance model from {previous_variant} to {model_variant}.",
                flush=True,
            )
            ACTIVE_PIPELINE_POOL.unload_all()
            ACTIVE_PIPELINE_POOL = None

        ACTIVE_PIPELINE_POOL = PipelinePool(gpu_ids, model_variant=model_variant)
        ACTIVE_PIPELINE_POOL.initialize_all()
        return ACTIVE_PIPELINE_POOL


@spaces.GPU(size="large", duration=get_run_task_gpu_duration)
def run_task(
    task: str,
    prompt: str,
    system_prompt: Optional[str],
    input_video: Optional[str],
    input_image: Optional[str],
    height: int,
    width: int,
    num_frames: int,
    seed: int,
    resolution: str,
    validation_num_timesteps: int,
    validation_timestep_shift: float,
    cfg_text_scale: float,
    enable_frame_interpolation: bool,
):
    internal_task = normalize_task(task)
    if internal_task == TASK_T2V:
        num_frames = video_seconds_to_num_frames(num_frames)
    pipeline_pool = get_pipeline_pool(task)
    return pipeline_pool.generate(
        task=task,
        prompt=prompt,
        system_prompt=system_prompt,
        input_video=input_video,
        input_image=input_image,
        height=height,
        width=width,
        num_frames=num_frames,
        seed=seed,
        resolution=resolution,
        validation_num_timesteps=validation_num_timesteps,
        validation_timestep_shift=validation_timestep_shift,
        cfg_text_scale=cfg_text_scale,
        enable_frame_interpolation=enable_frame_interpolation,
    )


def build_status_markdown() -> str:
    gpu_text = "unknown"
    concurrency = 1
    active_variant = "none"
    if ACTIVE_PIPELINE_POOL is not None:
        active_variant = ACTIVE_PIPELINE_POOL.model_variant
        gpu_text = ACTIVE_PIPELINE_POOL.gpu_summary
        concurrency = ACTIVE_PIPELINE_POOL.size
    return (
        f"**Status**  GPU: `{gpu_text}`  |  Max concurrency: `{concurrency}`  |  "
        f"Queue limit: `{QUEUE_MAX_SIZE}`  |  Active model: `{active_variant}`  |  "
        f"Switch mode: `unload then load`"
    )


def build_running_status_markdown() -> str:
    return "Running..."


def get_logo_data_uri() -> str:
    if not LANCE_LOGO_PATH.exists():
        return ""
    encoded_logo = base64.b64encode(LANCE_LOGO_PATH.read_bytes()).decode("ascii")
    return f"data:image/webp;base64,{encoded_logo}"


def build_header_html() -> str:
    logo_data_uri = get_logo_data_uri()
    logo_html = (
        f'<img class="lance-logo" src="{logo_data_uri}" alt="Lance logo">'
        if logo_data_uri
        else ""
    )
    return f"""
    <div class="lance-hero">
        {logo_html}
        <h1 class="lance-title">Lance: Unified Multimodal Modeling by Multi-Task Synergy</h1>
        <div class="lance-authors">
            <strong>
                <a href="https://scholar.google.com.hk/citations?user=FXxoQlsAAAAJ&hl=zh-CN&oi=ao" target="_blank">Fengyi Fu</a><sup>*</sup>,
                <a href="https://corleone-huang.github.io/" target="_blank">Mengqi Huang</a><sup>*,✉</sup>,
                <a href="https://scholar.google.com.hk/citations?user=9ER6nVkAAAAJ&hl=zh-CN&oi=ao" target="_blank">Shaojin Wu</a><sup>*</sup>,
                Yunsheng Jiang<sup>*</sup>,
                Yufei Huo,
                <a href="https://guojianzhu.com/" target="_blank">Jianzhu Guo</a><sup>✉,§</sup>
            </strong><br>
            Hao Li, Yinghang Song, Fei Ding, Qian He, Zheren Fu, Zhendong Mao, Yongdong Zhang<br>
            <em>ByteDance</em>
        </div>
        <div class="lance-badges">
            <a href="{LANCE_HOMEPAGE_URL}" target="_blank" rel="noopener noreferrer">
                <img alt="Homepage" src="https://img.shields.io/badge/Homepage-Lance-blue?style=flat">
            </a>
            <a href="{LANCE_PAPER_URL}" target="_blank" rel="noopener noreferrer">
                <img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-red?style=flat&logo=arxiv">
            </a>
            <a href="{LANCE_HUGGING_FACE_URL}" target="_blank" rel="noopener noreferrer">
                <img alt="Hugging Face" src="https://img.shields.io/badge/Model-HuggingFace-yellow?style=flat&logo=huggingface">
            </a>
            <a href="{LANCE_GITHUB_URL}" target="_blank" rel="noopener noreferrer">
                <img alt="GitHub" src="https://img.shields.io/badge/Code-GitHub-536af5?color=536af5&logo=github">
            </a>
        </div>
    </div>
    """


def update_task_ui(task: str):
    internal_task = normalize_task(task)
    is_image_task = internal_task in IMAGE_TASKS
    is_video_task = internal_task in VIDEO_TASKS
    is_edit_task = internal_task in EDIT_TASKS
    is_understanding_task = internal_task in UNDERSTANDING_TASKS
    is_generation_task = internal_task in GENERATION_TASKS
    is_text_to_visual_task = internal_task in {TASK_T2V, TASK_T2I}
    show_media_input = is_edit_task or is_understanding_task
    resolution_choices = get_resolution_choice_values_for_task(internal_task)
    resolution_value = get_default_resolution_for_task(internal_task)
    aspect_ratio_value = DEFAULT_IMAGE_ASPECT_RATIO if is_image_task else DEFAULT_VIDEO_ASPECT_RATIO
    width_value, height_value = get_size_for_aspect_ratio(internal_task, aspect_ratio_value, resolution_value)
    size_markdown = format_size_markdown(internal_task, width_value, height_value)
    system_prompt_choices = get_understanding_system_prompt_choices(internal_task)

    if is_text_to_visual_task:
        text_label = "Prompt"
        text_placeholder = "Describe what you want to generate..."
    elif is_edit_task:
        text_label = "Instruction"
        text_placeholder = "Describe the edit you want..."
    else:
        text_label = "Question"
        text_placeholder = "Ask a question about the input..."

    if internal_task in {TASK_T2V, TASK_VIDEO_EDIT}:
        output_label = "Output Video"
    elif internal_task in {TASK_T2I, TASK_IMAGE_EDIT}:
        output_label = "Output Image"
    else:
        output_label = "Output Text"

    output_icon = "video" if output_label == "Output Video" else "image" if output_label == "Output Image" else "text"
    show_generation_settings = is_generation_task or is_edit_task
    show_aspect_ratio = is_text_to_visual_task
    show_input_video = internal_task in {TASK_VIDEO_EDIT, TASK_X2T_VIDEO}
    show_input_image = internal_task in {TASK_IMAGE_EDIT, TASK_X2T_IMAGE}
    show_frame_interpolation_settings = internal_task in {TASK_T2V, TASK_VIDEO_EDIT} and RIFE_AVAILABLE
    show_video_resolution_settings = internal_task == TASK_T2V

    return (
        gr.update(value=build_lance_label_html(text_label, "lance-prompt-label")),
        gr.update(
            label=text_label,
            placeholder=text_placeholder,
            visible=True,
            value="",
        ),
        gr.update(
            choices=system_prompt_choices,
            value=system_prompt_choices[0],
            visible=False,
        ),
        # Switching task pages should always start from a clean input state.
        # Clear both visual input boxes even if one of them stays visible across tasks.
        gr.update(label="Input Video", visible=show_input_video, value=None),
        gr.update(label="Input Image", visible=show_input_image, value=None),
        gr.update(visible=show_frame_interpolation_settings),
        gr.update(visible=show_aspect_ratio),
        gr.update(visible=False),
        gr.update(visible=internal_task == TASK_T2V),
        gr.update(visible=show_video_resolution_settings),
        gr.update(choices=get_aspect_ratio_choices_for_task(internal_task), value=aspect_ratio_value, visible=show_aspect_ratio),
        gr.update(value=height_value),
        gr.update(value=width_value),
        gr.update(visible=show_frame_interpolation_settings, value=DEFAULT_FRAME_INTERPOLATION if RIFE_AVAILABLE else FRAME_INTERPOLATION_NO),
        gr.update(choices=get_output_resolution_choices_for_task(internal_task, resolution_value), value=size_markdown, visible=False),
        gr.update(visible=internal_task == TASK_T2V, value=DEFAULT_VIDEO_DURATION_SECONDS),
        gr.update(choices=resolution_choices, value=resolution_value, visible=show_video_resolution_settings),
        gr.update(value=build_lance_icon_label_html(output_label, output_icon, "lance-output-label")),
        gr.update(visible=internal_task in {TASK_T2V, TASK_VIDEO_EDIT}),
        gr.update(visible=internal_task in {TASK_T2I, TASK_IMAGE_EDIT}),
        gr.update(visible=is_understanding_task, value=""),
        gr.update(visible=internal_task == TASK_T2V),
        gr.update(visible=internal_task == TASK_VIDEO_EDIT),
        gr.update(visible=internal_task == TASK_X2T_VIDEO),
        gr.update(visible=internal_task == TASK_T2I),
        gr.update(visible=internal_task == TASK_IMAGE_EDIT),
        gr.update(visible=internal_task == TASK_X2T_IMAGE),
    )


def keep_example_clicks_from_changing_visibility(*examples_components) -> None:
    for examples_component in examples_components:
        dataset = getattr(examples_component, "dataset", None)
        component_props = getattr(dataset, "component_props", None)
        if not component_props:
            continue
        for props in component_props:
            props.pop("visible", None)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Lance", css=APP_CSS, js=APP_JS) as demo:
        gr.HTML(build_header_html())
        gr.Markdown(build_status_markdown(), elem_classes=["lance-status"], visible=False)

        with gr.Row(elem_classes=["lance-main-row"]):
            with gr.Column(scale=1, elem_classes=["lance-main-column", "lance-input-column"]):
                with gr.Column(elem_classes=["lance-panel", "lance-task-prompt-panel"]):
                    gr.HTML('<div class="lance-section-label">Task</div>', elem_classes=["lance-label-html"])
                    task = gr.Radio(
                        label="Task",
                        show_label=False,
                        choices=TASK_CHOICES,
                        value=TASK_LABEL_VIDEO_GENERATION,
                        elem_classes=["task-selector"],
                    )
                    prompt_label = gr.HTML(build_lance_label_html("Prompt", "lance-prompt-label"), elem_classes=["lance-label-html"])
                    prompt = gr.Textbox(
                        label="Prompt",
                        show_label=False,
                        lines=6,
                        placeholder="Describe the video you want to generate...",
                        elem_classes=["main-prompt-control"],
                    )
                system_prompt = gr.Dropdown(
                    label="System Prompt",
                    choices=get_understanding_system_prompt_choices(TASK_X2T_VIDEO),
                    value=V2T_QA_SYSTEM_PROMPT,
                    visible=False,
                )
                input_video = gr.Video(label="Input Video", visible=False, elem_classes=["lance-display-frame"])
                input_image = gr.Image(label="Input Image", type="filepath", visible=False, elem_classes=["lance-display-frame"])
                with gr.Column(elem_classes=["generation-control-stack"]):
                    with gr.Row(elem_classes=["generation-controls-row", "frame-interpolation-row"]) as frame_interpolation_row:
                        with gr.Column(elem_classes=["lance-control-field"]):
                            gr.HTML('<div class="lance-generation-label">Frame Interpolation</div>', elem_classes=["lance-label-html"])
                            enable_frame_interpolation = gr.Dropdown(
                                label="Frame Interpolation",
                                show_label=False,
                                choices=[FRAME_INTERPOLATION_YES, FRAME_INTERPOLATION_NO],
                                value=DEFAULT_FRAME_INTERPOLATION if RIFE_AVAILABLE else FRAME_INTERPOLATION_NO,
                                elem_classes=["generation-control", "generation-two-line-label"],
                            )
                    with gr.Row(elem_classes=["generation-controls-row", "aspect-ratio-row"]) as aspect_ratio_row:
                        with gr.Column(elem_classes=["lance-control-field"]):
                            gr.HTML('<div class="lance-generation-label">Aspect Ratio</div>', elem_classes=["lance-label-html"])
                            aspect_ratio = gr.Radio(
                                label="Aspect Ratio",
                                show_label=False,
                                # choices=ASPECT_RATIO_CHOICES, # 原始版本，不显示 是否为 default
                                choices=get_aspect_ratio_choices_for_task(TASK_T2V),
                                value=DEFAULT_VIDEO_ASPECT_RATIO,
                                elem_classes=["generation-control", "generation-choice-grid", "generation-two-line-label"],
                            )
                    with gr.Row(elem_classes=["generation-controls-row", "output-resolution-row"], visible=False) as output_resolution_row:
                        with gr.Column(elem_classes=["lance-control-field"]):
                            gr.HTML('<div class="lance-generation-label">Output Resolution</div>', elem_classes=["lance-label-html"])
                            real_size = gr.Radio(
                                label="Output Resolution",
                                show_label=False,
                                choices=get_output_resolution_choices_for_task(TASK_T2V),
                                value=format_size_markdown(TASK_T2V, DEFAULT_WIDTH, DEFAULT_HEIGHT),
                                interactive=True,
                                visible=False,
                                elem_classes=["generation-control", "generation-choice-grid", "generation-two-line-label"],
                            )
                with gr.Row(elem_classes=["generation-controls-row", "video-duration-row"]) as video_duration_row:
                    with gr.Column(elem_classes=["lance-control-field"]):
                        gr.HTML(build_lance_label_html("Video Duration (seconds)", "lance-generation-label"), elem_classes=["lance-label-html"])
                        num_frames = gr.Radio(
                            label="Video Duration (seconds)",
                            show_label=False,
                            choices=get_video_duration_choices(),
                            value=DEFAULT_VIDEO_DURATION_SECONDS,
                            elem_classes=["generation-control", "generation-choice-grid", "generation-two-line-label"],
                        )
                with gr.Row(elem_classes=["generation-controls-row", "video-resolution-row"]) as video_resolution_row:
                    with gr.Column(elem_classes=["lance-control-field"]):
                        gr.HTML(build_lance_label_html("Video Resolution", "lance-generation-label"), elem_classes=["lance-label-html"])
                        resolution = gr.Dropdown(
                            label="Video Resolution",
                            show_label=False,
                            choices=VIDEO_RESOLUTION_DISPLAY_CHOICES,
                            value=DEFAULT_RESOLUTION,
                            allow_custom_value=True,
                            elem_classes=["generation-control"],
                        )
                height = gr.Number(value=DEFAULT_HEIGHT, precision=0, visible=False)
                width = gr.Number(value=DEFAULT_WIDTH, precision=0, visible=False)

                with gr.Accordion("Advanced Parameters", open=False, elem_classes=["lance-advanced-accordion"]):
                    with gr.Column(elem_classes=["lance-control-field"]):
                        gr.HTML(build_lance_label_html("Seed (-1 for random seed)", "lance-generation-label"), elem_classes=["lance-label-html"])
                        seed = gr.Number(
                            label="Seed (-1 for random seed)",
                            show_label=False,
                            value=DEFAULT_BASIC_SEED,
                            precision=0,
                        )
                    with gr.Column(elem_classes=["lance-control-field"]):
                        gr.HTML(build_lance_label_html("Validation Num Timesteps", "lance-generation-label"), elem_classes=["lance-label-html"])
                        validation_num_timesteps = gr.Slider(
                            minimum=1,
                            maximum=50,
                            step=1,
                            value=DEFAULT_TIMESTEPS,
                            label="Validation Num Timesteps",
                            show_label=False,
                        )
                    with gr.Row(elem_classes=["generation-controls-row"]):
                        with gr.Column(elem_classes=["lance-control-field"]):
                            gr.HTML(build_lance_label_html("Validation Timestep Shift", "lance-generation-label"), elem_classes=["lance-label-html"])
                            validation_timestep_shift = gr.Number(
                                label="Validation Timestep Shift",
                                value=DEFAULT_TIMESTEP_SHIFT,
                                show_label=False,
                            )
                        with gr.Column(elem_classes=["lance-control-field"]):
                            gr.HTML(build_lance_label_html("CFG Text Scale", "lance-generation-label"), elem_classes=["lance-label-html"])
                            cfg_text_scale = gr.Number(
                                label="CFG Text Scale",
                                value=DEFAULT_CFG_TEXT_SCALE,
                                show_label=False,
                            )

                generation_example_inputs = [
                    prompt,
                    input_video,
                    input_image,
                ]

            with gr.Column(scale=1, elem_classes=["lance-main-column", "lance-output-column"]):
                with gr.Column(elem_classes=["lance-panel", "lance-output-panel"]):
                    output_label = gr.HTML(
                        build_lance_icon_label_html("Output Video", "video", "lance-output-label"),
                        elem_classes=["lance-label-html"],
                    )
                    output_video = gr.Video(label="Output Video", show_label=False, elem_classes=["lance-display-frame", "output-media-control"])
                    output_image = gr.Image(label="Output Image", show_label=False, type="filepath", visible=False, elem_classes=["lance-display-frame", "output-media-control"])
                    output_text = gr.Textbox(label="Output Text", show_label=False, lines=3, visible=False, elem_classes=["lance-display-frame"])
                status = gr.Markdown("", elem_classes=["lance-run-status"])
                with gr.Column(elem_classes=["lance-control-field", "run-logs-field"]):
                    gr.HTML(build_lance_icon_label_html("Run Logs", "logs", "lance-output-label"), elem_classes=["lance-label-html"])
                    logs = gr.Textbox(
                        label="Run Logs",
                        show_label=False,
                        lines=16,
                        max_lines=40,
                        elem_classes=["run-logs"],
                    )

        run_button = gr.Button("🚀 Generate", variant="primary", elem_classes=["lance-run-button"])

        def build_prompt_example_table(examples: list[list], media_type: Optional[str] = None):
            """Render examples with full prompt text instead of Gradio compact previews."""
            example_buttons = []
            with gr.Column(elem_classes=["prompt-example-full-table"]):
                if media_type == "video":
                    gr.HTML("<div>Prompt / Instruction / Question</div><div>Input Video</div>", elem_classes=["prompt-example-table-header", "prompt-example-table-header-with-media"])
                elif media_type == "image":
                    gr.HTML("<div>Prompt / Instruction / Question</div><div>Input Image</div>", elem_classes=["prompt-example-table-header", "prompt-example-table-header-with-media"])
                else:
                    gr.HTML("<div>Prompt</div>", elem_classes=["prompt-example-table-header"])

                with gr.Column(elem_classes=["prompt-example-table-body"]):
                    for example_row in examples:
                        example_prompt = str(example_row[0]) if example_row else ""
                        video_path = str(example_row[1]) if len(example_row) > 1 and example_row[1] else None
                        image_path = str(example_row[2]) if len(example_row) > 2 and example_row[2] else None

                        if media_type == "video" and video_path:
                            with gr.Row(elem_classes=["prompt-example-multimodal-row", "prompt-example-video-row"]):
                                with gr.Column(elem_classes=["prompt-example-prompt-cell"]):
                                    example_button = gr.Button(
                                        example_prompt,
                                        variant="secondary",
                                        elem_classes=["prompt-example-row-button"],
                                    )
                                with gr.Column(elem_classes=["prompt-example-media-cell", "prompt-example-video-cell"]):
                                    gr.Video(
                                        value=video_path,
                                        label="Input Video",
                                        show_label=False,
                                        interactive=False,
                                        elem_classes=["prompt-example-media-preview", "prompt-example-video-preview"],
                                    )
                            example_buttons.append((example_button, example_prompt, video_path, None))
                        elif media_type == "image" and image_path:
                            with gr.Row(elem_classes=["prompt-example-multimodal-row"]):
                                with gr.Column(elem_classes=["prompt-example-prompt-cell"]):
                                    example_button = gr.Button(
                                        example_prompt,
                                        variant="secondary",
                                        elem_classes=["prompt-example-row-button"],
                                    )
                                with gr.Column(elem_classes=["prompt-example-media-cell"]):
                                    gr.Image(
                                        value=image_path,
                                        label="Input Image",
                                        show_label=False,
                                        interactive=False,
                                        type="filepath",
                                        elem_classes=["prompt-example-media-preview"],
                                    )
                            example_buttons.append((example_button, example_prompt, None, image_path))
                        else:
                            example_button = gr.Button(
                                example_prompt,
                                variant="secondary",
                                elem_classes=["prompt-example-row-button"],
                            )
                            example_buttons.append((example_button, example_prompt, None, None))
            return example_buttons

        with gr.Column(visible=True, elem_classes=["lance-recommended-section"]) as video_generation_examples_group:
            gr.HTML(build_lance_label_html("Video generation recommended cases", "lance-section-label"), elem_classes=["lance-label-html"])
            with gr.Group(elem_classes=["example-panel", "prompt-examples"]):
                video_generation_example_buttons = build_prompt_example_table(VIDEO_GENERATION_EXAMPLES)

        with gr.Column(visible=False, elem_classes=["lance-recommended-section"]) as video_edit_examples_group:
            gr.HTML(build_lance_label_html("Video edit recommended cases", "lance-section-label"), elem_classes=["lance-label-html"])
            with gr.Group(elem_classes=["example-panel", "prompt-examples", "video-edit-examples"]):
                video_edit_example_buttons = build_prompt_example_table(VIDEO_EDIT_EXAMPLES, media_type="video")

        with gr.Column(visible=False, elem_classes=["lance-recommended-section"]) as video_understanding_examples_group:
            gr.HTML(build_lance_label_html("Video understanding recommended cases", "lance-section-label"), elem_classes=["lance-label-html"])
            with gr.Group(elem_classes=["example-panel", "prompt-examples"]):
                video_understanding_example_buttons = build_prompt_example_table(VIDEO_UNDERSTANDING_EXAMPLES, media_type="video")

        with gr.Column(visible=False, elem_classes=["lance-recommended-section"]) as image_generation_examples_group:
            gr.HTML(build_lance_label_html("Image generation recommended cases", "lance-section-label"), elem_classes=["lance-label-html"])
            with gr.Group(elem_classes=["example-panel", "prompt-examples"]):
                image_generation_example_buttons = build_prompt_example_table(IMAGE_GENERATION_EXAMPLES)

        with gr.Column(visible=False, elem_classes=["lance-recommended-section"]) as image_edit_examples_group:
            gr.HTML(build_lance_label_html("Image edit recommended cases", "lance-section-label"), elem_classes=["lance-label-html"])
            with gr.Group(elem_classes=["example-panel", "prompt-examples"]):
                image_edit_example_buttons = build_prompt_example_table(IMAGE_EDIT_EXAMPLES, media_type="image")

        with gr.Column(visible=False, elem_classes=["lance-recommended-section"]) as image_understanding_examples_group:
            gr.HTML(build_lance_label_html("Image understanding recommended cases", "lance-section-label"), elem_classes=["lance-label-html"])
            with gr.Group(elem_classes=["example-panel", "prompt-examples"]):
                image_understanding_example_buttons = build_prompt_example_table(IMAGE_UNDERSTANDING_EXAMPLES, media_type="image")

        task.change(
            fn=update_task_ui,
            inputs=[task],
            outputs=[
                prompt_label,
                prompt,
                system_prompt,
                input_video,
                input_image,
                frame_interpolation_row,
                aspect_ratio_row,
                output_resolution_row,
                video_duration_row,
                video_resolution_row,
                aspect_ratio,
                height,
                width,
                enable_frame_interpolation,
                real_size,
                num_frames,
                resolution,
                output_label,
                output_video,
                output_image,
                output_text,
                video_generation_examples_group,
                video_edit_examples_group,
                video_understanding_examples_group,
                image_generation_examples_group,
                image_edit_examples_group,
                image_understanding_examples_group,
            ],
        )

        aspect_ratio.change(
            fn=update_size_from_aspect_ratio,
            inputs=[task, aspect_ratio, resolution],
            outputs=[height, width, real_size],
            queue=False,
            show_api=False,
        )

        real_size.change(
            fn=update_aspect_ratio_from_output_resolution,
            inputs=[task, real_size, resolution],
            outputs=[aspect_ratio, height, width],
            queue=False,
            show_api=False,
        )

        resolution.change(
            fn=update_output_resolution_from_video_profile,
            inputs=[task, aspect_ratio, resolution],
            outputs=[real_size, height, width],
            queue=False,
            show_api=False,
        )

        for example_button, example_prompt, _, _ in video_generation_example_buttons + image_generation_example_buttons:
            example_button.click(
                fn=make_prompt_example_click_handler(example_prompt),
                inputs=[task],
                outputs=[prompt, aspect_ratio, height, width, num_frames, resolution, real_size],
                queue=False,
                show_api=False,
            )

        for example_button, example_prompt, example_video, example_image in (
            video_edit_example_buttons
            + video_understanding_example_buttons
            + image_edit_example_buttons
            + image_understanding_example_buttons
        ):
            example_button.click(
                fn=make_media_prompt_example_click_handler(example_prompt, example_video, example_image),
                inputs=[task],
                outputs=[prompt, input_video, input_image, aspect_ratio, height, width, num_frames, resolution, real_size],
                queue=False,
                show_api=False,
            )

        run_button.click(
            fn=build_running_status_markdown,
            inputs=[],
            outputs=[status],
            queue=False,
            show_api=False,
        ).then(
            fn=run_task,
            inputs=[
                task,
                prompt,
                system_prompt,
                input_video,
                input_image,
                height,
                width,
                num_frames,
                seed,
                resolution,
                validation_num_timesteps,
                validation_timestep_shift,
                cfg_text_scale,
                enable_frame_interpolation,
            ],
            outputs=[output_video, output_image, output_text, status, logs],
            show_progress="minimal",
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lance multimodal Gradio")
    parser.add_argument("--server-name", default=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"))
    parser.add_argument("--server-port", type=int, default=int(os.getenv("GRADIO_SERVER_PORT", "7860")))
    parser.add_argument("--share", action="store_true", default=env_flag("GRADIO_SHARE", False))
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


def prefetch_model_assets_before_launch() -> None:
    """Download and compact model files before the first ZeroGPU request.

    On ZeroGPU, time spent downloading model snapshots inside @spaces.GPU burns
    the first user's GPU reservation. Prefetching only touches CPU/disk and keeps
    the visible UI unchanged. Set LANCE_PREFETCH_MODEL_ASSETS=0 to skip this at
    Space startup, or LANCE_PREFETCH_MODEL_VARIANTS=video to prefetch less.
    """
    if running_on_space() or env_flag("LANCE_INSTALL_FLASH_ATTN_ON_STARTUP", False):
        try:
            ensure_flash_attn_installed()
        except Exception as exc:
            print(f"[startup] flash-attn startup install failed and will be retried lazily during inference: {exc}", flush=True)

    if not env_flag("LANCE_PREFETCH_MODEL_ASSETS", running_on_space()):
        print("[startup] Model asset prefetch disabled.", flush=True)
        return

    variants_text = os.getenv("LANCE_PREFETCH_MODEL_VARIANTS", f"{MODEL_VARIANT_VIDEO},{MODEL_VARIANT_IMAGE}")
    variants: list[str] = []
    for raw_variant in variants_text.split(","):
        raw_variant = raw_variant.strip()
        if not raw_variant:
            continue
        variant = normalize_model_variant(raw_variant)
        if variant not in variants:
            variants.append(variant)

    for variant in variants:
        try:
            start = time.perf_counter()
            model_path = ensure_model_assets(variant)
            elapsed = time.perf_counter() - start
            print(
                f"[startup][{variant}] Model assets are ready at {display_path(model_path)} "
                f"before ZeroGPU inference. elapsed={elapsed:.2f}s",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[startup][{variant}] Model asset prefetch failed and will be retried lazily during inference: {exc}",
                flush=True,
            )


if __name__ == "__main__":
    args = parse_args()
    os.environ["LANCE_GPUS"] = args.gpus
    QUEUE_MAX_SIZE = args.queue_size
    prefetch_model_assets_before_launch()
    print(
        "[startup] Skipping GPU model preload. UI will launch first, and Lance weights will be prefetched on CPU before ZeroGPU inference. If that prefetch fails, inference will fall back to lazy loading.",
        flush=True,
    )
    concurrency_limit = 1
    demo = build_demo()
    demo.queue(
        max_size=args.queue_size,
        default_concurrency_limit=concurrency_limit,
    ).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
    )
