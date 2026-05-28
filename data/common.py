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

import random
from typing import List

import torch


def generate_system_prompt(system_prompt_type="caption", vision_type="video"):
    if system_prompt_type == "caption":
        str_list = [
            f"Generate a detailed and accurate description of the {vision_type}, including all the key moments and visual details.",
            f"Write an in-depth depiction of the {vision_type}, covering all its aspects.",
            f"Write an exhaustive depiction of the given {vision_type}, capturing its essence and key moments.",
            f"Describe the key features of the input {vision_type}, including color, shape, size, texture, objects, background.",
        ]
    elif system_prompt_type == "t2v" or system_prompt_type == "ff2v":
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
