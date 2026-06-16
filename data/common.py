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
import json
import torch
import random
import io
# from data.tos import TosWrapper
from einops import rearrange
from typing import List
import torch.nn.functional as F
import re
import numpy as np

RE_ZH = re.compile(r"[\u4e00-\u9fff]")
RE_EN = re.compile(r"[A-Za-z]")

def generate_system_prompt(system_prompt_type="caption", vision_type="video"):
    if system_prompt_type == "caption":
        str_list = [
            f"Generate a detailed and accurate description of the {vision_type}, including all the key moments and visual details.",
            f"Write an in-depth depiction of the {vision_type}, covering all its aspects.",
            f"Write an exhaustive depiction of the given {vision_type}, capturing its essence and key moments.",
            f"Describe the key features of the input {vision_type}, including color, shape, size, texture, objects, background.",
        ]
    elif system_prompt_type == "t2v" or system_prompt_type == "i2v":
        str_list = [f"Describe the {vision_type} by detailing the color, quantity, visible text, shape, size, texture, spatial relationships and motion/camera movements of the objects and background:"]
    elif system_prompt_type == "t2i":
        str_list = [f"Describe the {vision_type} by detailing the color, quantity, text, shape, size, texture, spatial relationships of the objects and background:"]
    elif "edit" in system_prompt_type:
        str_list = [f"Describe the key features of the input {vision_type} (color, shape, size, texture, objects, background), then explain how the user’s text instruction should alter or modify the {vision_type}. Generate a new {vision_type} that meets the user’s requirements while maintaining consistency with the original input where appropriate."]
    elif "idip" in system_prompt_type:
        str_list = [f"Describe the key features of the input image (color, shape, size, texture, objects, background, style), then incorporate the user’s text description to generate a new {vision_type} that satisfies the user’s requirements while preserving the essential identity and object or style information from the reference input."]
    elif 'maze' in system_prompt_type:
        str_list = [
            "Describe the key elements of the input maze image (layout, white path, black walls, blue star, red flag, and overall background), then generate a 2D animation. The blue star should slide smoothly along the white path, stop exactly on the red flag, and then acquire a trophy. Ensure the blue star never crosses or enters the black maze walls. Keep the camera as a static top-down view showing the entire maze."
        ]

    return random.choice(str_list)


def shift_position_ids(
    position_ids: torch.Tensor,
    pos_shift: any,
    attn_modes: List[str],
    split_lens: int,
    shift_attn_mode=["full_noise", "full"],
    pro_type=None,
    i_sample_task=None,
    i_sample_modality=None,
) -> torch.Tensor:
    curr_split = 0
    for i, attn_mode in enumerate(attn_modes):
        if attn_mode in shift_attn_mode:
            if pro_type == 10:  # Related to sample_modality.
                if position_ids[:, :, i_sample_modality == 4].sum() != 0:
                    pos_shift_type4 = 1000 - position_ids[:, :, i_sample_modality == 4][0, 0, 0]
                    position_ids[0, :, i_sample_modality == 4] += pos_shift_type4
                if position_ids[:, :, i_sample_modality == 3].sum() != 0:
                    pos_shift_type3 = 2000 - position_ids[:, :, i_sample_modality == 3][0, 0, 0]
                    position_ids[0, :, i_sample_modality == 3] += pos_shift_type3
                if position_ids[:, :, i_sample_modality == 2].sum() != 0 and sum(i_sample_modality == 2) == sum(i_sample_modality == 1):
                    position_ids[:, :, i_sample_modality == 1] = position_ids[:, :, i_sample_modality == 2]

        curr_split += split_lens[i]

    return position_ids

def detect_lang_simple(s: str) -> str:
    """
    Fast heuristic: return 'zh' if Chinese is present, 'en' if English letters are present, otherwise 'other'.
    Useful for quick routing. If both are present, this returns 'zh'; adjust if needed.
    """
    # Remove digits before detection
    s_without_digits = re.sub(r'\d+', '', s)

    if RE_ZH.search(s_without_digits):
        return "zh"
    if RE_EN.search(s_without_digits):
        return "en"
    return "other"

def map_to_nearest_aspect_ratio(h, w, target_resolution=256):
    """
    Map h and w to the closest preset aspect ratio and return adjusted h and w near the target resolution.

    Preset ratios: ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"].
    target_resolution: Base target resolution, default 256.
    """
    # Precompute all preset aspect ratios as width / height
    PRESET_RATIOS = [21 / 9, 16 / 9, 4 / 3, 1 / 1, 3 / 4, 9 / 16]

    # Compute the original aspect ratio
    original_ratio = w / h

    # Find the closest preset ratio
    min_index = min(range(len(PRESET_RATIOS)), key=lambda i: abs(original_ratio - PRESET_RATIOS[i]))
    best_ratio = PRESET_RATIOS[min_index]

    # Compute scale so the longer side is close to the target resolution
    if best_ratio >= 1:  # Landscape: width >= height
        scale = target_resolution / best_ratio
        adjusted_w = round(target_resolution)
        adjusted_h = round(scale)
    else:  # Portrait: height > width
        scale = target_resolution
        adjusted_h = round(target_resolution)
        adjusted_w = round(scale * best_ratio)

    return adjusted_h, adjusted_w


def concat_resize_tensor_list(video_latents: List[torch.Tensor], dim: int = 0, is_offline: bool = False, max_num_frames: int = 121) -> torch.Tensor:
    """
    Concatenate tensors along dim; resize H/W of tensors with different sizes to match target.
    - tensors: Non-empty list; all tensors must have the same ndim.
    - dim: Concatenation axis; negative values are supported.
    - pad_value: Padding value, default 0.0.
    Returns: Concatenated tensor.
    """

    if is_offline:
        H, W = video_latents[-1].shape[-3], video_latents[-1].shape[-2]
    else:
        H, W = video_latents[-1].shape[-2], video_latents[-1].shape[-1]

    padded_video_latents = []
    num_frames_target = video_latents[-1].shape[dim]

    num_frames_all = num_frames_target
    for index, video_latent in enumerate(video_latents):
        if index != len(video_latents) - 1 and num_frames_all + video_latent.shape[dim] > max_num_frames:  # Avoid producing videos longer than MAX_NUM_FRAMES
            continue
        num_frames_all += video_latent.shape[dim]
        if is_offline:
            # video_latent:[t,h,w,c] -> [t,c,h,w]
            video_latent = rearrange(video_latent, "t h w c -> t c h w")

        h, w = video_latent.shape[-2], video_latent.shape[-1]
        if h != H or w != W:
            video_latent = F.interpolate(video_latent, size=(H, W), mode="bilinear", align_corners=False)
        padded_video_latents.append(video_latent)

    padded_video_latents = torch.cat(padded_video_latents, dim=dim)

    if is_offline:
        # padded_video_latents: [t,c,h,w] -> [t,h,w,c]
        padded_video_latents = rearrange(padded_video_latents, "t c h w -> t h w c")

    return padded_video_latents


def concat_pad_tensor_list(video_latents: List[torch.Tensor], dim: int = 0, pad_value: float = 0.0, max_num_frames: int = 121) -> torch.Tensor:
    """
    Concatenate tensors along dim; pad other axes to the maximum length on each axis with pad_value.
    - tensors: Non-empty list; all tensors must have the same ndim.
    - dim: Concatenation axis; negative values are supported.
    - pad_value: Padding value, default 0.0.
    Returns: Concatenated tensor.
    """
    video_sizes = [item.shape for item in video_latents]
    max_video_size = [max(item) for item in list(zip(*video_sizes))]
    padded_video_latents = []
    num_frames_target = video_latents[-1].shape[dim]

    num_frames_all = num_frames_target
    for index, video_latent in enumerate(video_latents):
        if index != len(video_latents) - 1 and num_frames_all + video_latent.shape[dim] > max_num_frames:  # Avoid producing videos longer than MAX_NUM_FRAMES
            continue
        num_frames_all += video_latent.shape[dim]
        max_video_size[dim] = video_latent.shape[dim]
        padded_video_latent = torch.zeros(max_video_size)
        n1, n2, n3, n4 = video_latent.shape
        padded_video_latent[:n1, :n2, :n3, :n4] = video_latent
        padded_video_latents.append(padded_video_latent)

    padded_video_latents = torch.cat(padded_video_latents, dim=dim)

    return padded_video_latents


def parse_videochat2it_doubao_caption(row):
    try:
        IQA_i = "View the video attentively and provide a suitable answer to the posed question."
        rewrite_VQA = json.loads(row['rewrite_VQA'])

        IQA_q = rewrite_VQA['question'] if 'question' in rewrite_VQA.keys() else rewrite_VQA['Question']
        IQA_a = rewrite_VQA['final_answer']
        IQA_resoning = rewrite_VQA['reasoning']
        # Combine reasoning and final_answer as the final answer

        # if random.random() < 0.5: # 50% chance to include the reasoning process
        #     IQA_a = IQA_a + '\n' + IQA_resoning
        #     IQA_i = IQA_i + ' Please provide the reasoning process for selecting the correct answer.'

        if 'options' not in IQA_q and 'Options' not in IQA_q: # When the question does not contain options
            try:
                options = rewrite_VQA['options']
            except:
                options = rewrite_VQA['Options']
            if options == []:
                return [IQA_i, IQA_q, IQA_a]
            elif isinstance(options,list):
                options = '\n'.join(options)
            elif isinstance(options,dict):
                options_str  = [key + ' ' + value if value not in key else key for key,value in options.items()]
                options = '\n'.join(options_str)

            IQA_q = IQA_q + '\nOptions:\n' + options # Add options to the question
        return [IQA_i, IQA_q, IQA_a]
    except:
        if 'rewrite_VQA' in row.keys():
            raise ValueError(f"wrong rewrite_VQA in {row['rewrite_VQA']}")
        else:
            raise ValueError(f"wrong rewrite_VQA in {row}")