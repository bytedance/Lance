# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# coding: utf-8

"""
Configuration definitions and lightweight factory helpers.

- TemplateArguments  : chat templates
- ModelArguments     : model structure and loading parameters
- DataArguments      : validation dataset input parameters
- TrainingArguments  : model loading, sampling, and compatibility fields still used at inference time
- InferenceArguments : inference-only parameters inherited from TrainingArguments
- EvaluationArguments: evaluation-only parameters inherited from InferenceArguments

This module also handles:
- shared training/inference dataclass definitions
- `path_default.yaml` parsing
- lightweight configuration factory helpers for the inference pipeline
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import yaml

# ==============================================
# Model path configuration management
# ==============================================

# Global cache to avoid repeated loads
_MODEL_PATH_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_DEFAULT_PATH_FILE = Path(__file__).with_name("path_default.yaml")
_PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _get_nested_value(config: Dict[str, Any], path_key: str) -> Any:
    """
    Get a value from a nested config using a dot-separated path, e.g. "vit.qwen2_5_vl".
    """
    value: Any = config
    for key in path_key.split("."):
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            raise ValueError(f"Path key '{path_key}' not found in {_DEFAULT_PATH_FILE.name}")
    return value


def _resolve_config_values(value: Any, config: Dict[str, Any]) -> Any:
    """
    Recursively resolve placeholders in the config while preserving the original nested structure.
    """
    if isinstance(value, dict):
        return {k: _resolve_config_values(v, config) for k, v in value.items()}
    if isinstance(value, str):
        return _resolve_placeholders(value, config)
    return value


def _resolve_placeholders(path: str, config: Dict[str, Any]) -> str:
    """
    Recursively resolve placeholders in a path, e.g. ${base_dir} or ${vit.qwen2_5_vl}.
    """
    matches = _PLACEHOLDER_PATTERN.findall(path)

    if not matches:
        return path

    result = path
    for match in matches:
        try:
            value = _get_nested_value(config, match)
        except ValueError as exc:
            raise ValueError(f"Placeholder ${match} not found in {_DEFAULT_PATH_FILE.name}") from exc

        # Recursively resolve placeholders in the value.
        resolved_value = _resolve_placeholders(str(value), config)
        result = result.replace(f"${{{match}}}", resolved_value)

    return result


def get_model_path_config(reload: bool = False) -> Dict[str, Any]:
    """
    Load and resolve the path_default.yaml configuration file.
    :param reload: Force reload and ignore the cache.
    :return: Resolved configuration dictionary.
    """
    global _MODEL_PATH_CONFIG_CACHE

    if _MODEL_PATH_CONFIG_CACHE is not None and not reload:
        return _MODEL_PATH_CONFIG_CACHE

    if not _DEFAULT_PATH_FILE.exists():
        raise FileNotFoundError(
            f"Model path configuration file not found: {_DEFAULT_PATH_FILE}"
        )

    with _DEFAULT_PATH_FILE.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    resolved_config = _resolve_config_values(config, config)
    _MODEL_PATH_CONFIG_CACHE = resolved_config

    return resolved_config


def get_model_path(path_key: str) -> str:
    """
    Get a configured path value.
    :param path_key: Path key with nested keys supported, e.g. "vit.qwen2_5_vl", "data.t2i".
    :return: Resolved full path.
    """
    config = get_model_path_config()
    value = _get_nested_value(config, path_key)

    return str(value) if value is not None else ""

@dataclass
class TemplateArguments:
    chat_template: List[str] = (
        '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n',
        'Describe this image.<|im_end|>\n<|im_start|>assistant\n',
    )  # NOTE: the instruction should adapt to different data types; insert VIT tokens in the middle and text tokens at the end.
    chat_template_T2I: List[str] = (
        '<|im_start|>system\nDescribe the image by detailing the color, quantity, text, shape, size, texture, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n<|quad_start|><|im_end|>\n<|im_start|>assistant\n',
    )  # NOTE: insert text tokens in the middle of the template and VAE tokens at the end.
    pad_token_template_T2I: str = "<|quad_start|>"
    pad_token_template: str = "<|quad_end|>"


@dataclass
class ModelArguments:
    model_path:                 str = ""
    llm_path:                   str = ""
    llm_qk_norm:                bool = True
    llm_qk_norm_und:            bool = True
    llm_qk_norm_gen:            bool = True
    tie_word_embeddings:        bool = False
    layer_module:               str = "Qwen2MoTDecoderLayer"
    vit_path:                   str = ""
    max_num_frames:             int = 25
    max_latent_size:            int = 64
    latent_patch_size:          List[int] = (1, 2, 2)  # pt ph pw
    vit_patch_size:             int = 14
    vit_patch_size_temporal:    int = 2
    vit_max_num_patch_per_side: int = 70
    connector_act:              str = "gelu_pytorch_tanh"
    interpolate_pos:            bool = False
    vit_select_layer:           int = -2
    vit_rope:                   bool = False

    text_cond_dropout_prob:     float = 0.1
    vae_cond_dropout_prob:      float = 0.3
    vit_cond_dropout_prob:      float = 0.3
    vit_type:                   str = "qwen2_5_vl"  # options: qwen2_5_vl

    val_text_cond_dropout_prob: float = 0
    val_vae_cond_dropout_prob:  float = 0
    val_vit_cond_dropout_prob:  float = 0

    cfg_text_scale:             float = 4.0  # for validation


@dataclass
class DataArguments:
    val_dataset_config_file:    Optional[str] = None


@dataclass
class TrainingArguments:
    # Inference runtime switches
    apply_chat_template:        bool = False  # Whether to apply the Qwen2.5-VL chat template to input text.
    apply_qwen_2_5_vl_pos_emb:  bool = False  # Whether to enable Qwen2.5-VL position embeddings.

    vae_model_type:             str = "seedance"
    visual_gen:                 bool = True
    visual_und:                 bool = True
    freeze_und:                 bool = False
    copy_init_moe:              bool = False
    finetune_from_hf:           bool = False
    finetune_from_vlm:          bool = False
    use_flex:                   bool = False
    num_replicate:              int = 1
    num_shard:                  int = 1

    global_seed:                int = 2025

    # Sampling settings
    timestep_shift:             float = 1.0
    validation_data_seed:       int = 42
    validation_num_timesteps:   int = 30
    validation_timestep_shift:  float = 3.0
    validation_max_samples:     int = 8
    validation_noise_seed:      int = 2025
    validation_video_saving_fps:int = 12
    validation_log_type:        str = "direct"

    # CFG and text condition control
    cfg_type:                   int = 0       # 0: remove all text conditions; 1: keep only special tokens; 2: keep special tokens and replace middle text tokens with <NULL>.
    cfg_uncond_token_id:        int = 151643  # Only used when cfg_type=2.
    cfg_interval:               List[float] = field(default_factory=lambda: [0.4, 1.0])
    cfg_renorm_min:             float = 0
    cfg_renorm_type:            str = "global"  # global | channel | ""

@dataclass
class InferenceArguments(TrainingArguments):
    save_path_gen:              str = "tmp/results/inference/generation"  # Save path for generated videos/images.
    save_path_gt:               str = ""    # Save path for ground-truth videos/images; not saved by default.
    video_height:               int = 480
    video_width:                int = 480
    num_frames:                 int = 50
    task:                       str = "t2v"  # t2v / t2i / edit / idip ...
    resolution:                 str = "video_360p"  # image_256res, image_512res, video_192p, video_360p, etc.
    text_template:              bool = False  # Whether to use the system_prompt text template.
    max_duration:               float = 6.0  # Maximum video duration in seconds.

    system_prompt_type:         str = "SP0"  # options: SP1, SP2 ...
    use_KVcache:                bool = False


@dataclass
class EvaluationArguments(InferenceArguments):
    config_json_path:           str = field(default="", metadata={"help": "Path to the JSON config file."})  # Override arguments with this config when provided.
    sample_num_per_prompt:      int = field(default=4, metadata={"help": "Number of samples per case."})
    max_eval_cases:             int = field(default=0, metadata={"help": "Limit the number of evaluation cases; 0 means all cases."})
    do_sample:                  bool = False  # Whether UND tasks use sampling.
    evaluation_seed:            int = 42
    quick_debug:                bool = False  # Quick debug mode.
