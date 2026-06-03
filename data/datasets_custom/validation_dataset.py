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
import os
from typing import Any, Dict, List
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
import decord
from decord import VideoReader
from PIL import Image

from data.video.sampler.utils import FRAME_SAMPLER_TYPES
from data.video.sampler.frames import FrameSamplerOutput
from data.transforms import VideoTransform
from data.data_utils import (
    get_flattened_position_ids_extrapolate_video,
    len2weight,
    patchify_video_with_merge,
)
from data.system_prompt_render import render_qwenvl_prompt, expand_and_index_by_token_ids_new
from data.common import generate_system_prompt
from modeling.qwen2 import Qwen2Tokenizer
from config.config_factory import ModelArguments, DataArguments, TrainingArguments

sample_task_map = {
    't2v': 0,
    'idip': 1,
    'edit': 2,
    'refedit': 3,
}
modality_map = {
    'system_prompt': -1,
    'text': 0,
    'noise': 1,
    'ref_source': 2,
    'ref_image': 3,
    'ref_vit': 4
}


class ValidationDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: Qwen2Tokenizer,
        data_args: DataArguments,
        model_args: ModelArguments,
        training_args: TrainingArguments,
        new_token_ids: Dict[str, int],
        dataset_config: None,
        local_rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize the validation dataset.

        Args:
            jsonl_path: Path to the JSONL file.
            tokenizer: Tokenizer instance.
        """
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.new_token_ids = new_token_ids

        try:
            full_data = self._read_jsonl()
        except:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                full_data = json.load(f)
            if isinstance(full_data, dict):
                full_data = [{"index": self.pro_index(index), "data": prompt} for index, prompt in full_data.items()]

        if world_size > 1:
            self.data = full_data[local_rank::world_size]
            print(f"Rank {local_rank}/{world_size} will process {len(self.data)} samples")
        else:
            self.data = full_data

        self.data_config = dataset_config

        self.bos_token_id = self.new_token_ids["bos_token_id"]
        self.eos_token_id = self.new_token_ids["eos_token_id"]
        self.start_of_image = self.new_token_ids["start_of_image"]
        self.end_of_image = self.new_token_ids["end_of_image"]
        self.image_token_id = self.new_token_ids["image_token_id"]

        try:
            max_duration = self.data_config.max_duration
        except:
            max_duration = 6.0

        video_frame_sampler_params = {"temporal": 4, "sample_fps": 12, "max_duration": max_duration, "assert_seconds": False, "truncate": False}

        self.frame_sampler = FRAME_SAMPLER_TYPES["multi_clips"](**video_frame_sampler_params)
        self.cpu_count = os.cpu_count() or 1

        if self.data_config.resolution in ["video_192p", "image_256res"]:
            resolution_vae = 256
            resolution_vit = 224
        elif self.data_config.resolution == "image_512res":
            resolution_vae = 512
            resolution_vit = 448
        elif self.data_config.resolution == "image_768res":
            resolution_vae = 768
            resolution_vit = 672
        elif self.data_config.resolution == "video_360p":
            resolution_vae = 480
            resolution_vit = 476
        elif self.data_config.resolution == "video_480p":
            resolution_vae = 640
            resolution_vit = 616
        else:
            raise ValueError(f"Unknown resolution: {self.data_config.resolution}")

        video_transform_args = {
            "resolution": resolution_vae,
            "mode": "bucket",
            "divisible_crop_size": 16,
            "stride_spatial": 16,
            "stride_temporal": 4,
            "aspect_ratios": ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
            "mean": 0.5,
            "std": 0.5,
        }
        self.transform = VideoTransform(**video_transform_args)

        vit_video_transform_args = {
            "resolution": resolution_vit,
            "mode": "bucket",
            "divisible_crop_size": 28,
            "aspect_ratios": ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"],
            "mean": [0.48145466, 0.4578275, 0.40821073],
            "std": [0.26862954, 0.26130258, 0.27577711],
        }
        self.vit_transform = VideoTransform(**vit_video_transform_args)

        self.sample = self.set_sequence_status()

        self.frame_condition_idx = []

        if hasattr(self.data_config, 'system_prompt_type'):
            self.system_prompt_type = self.data_config.system_prompt_type
        else:
            self.system_prompt_type = 'SP0'

    def pro_index(self, index: int):
        if isinstance(index, str):
            for x in ['.mp4', '.jpg', '.png', '.jpeg']:
                index = index.replace(x, "")
        return int(index)

    def set_sequence_status(self):
        sequence_status = dict(
            curr=0,
            sample_lens=[],
            sample_type=[],
            sample_N_target=[],
            packed_position_ids=[],
            nested_attention_masks=[],
            split_lens=[],
            attn_modes=[],
            packed_text_ids=[],
            packed_text_indexes=[],
            packed_label_ids=[],
            ce_loss_indexes=[],
            ce_loss_weights=[],
            vae_image_tensors=[],
            vae_video_tensors=[],
            packed_latent_position_ids=[],
            vae_latent_shapes=[],
            packed_vae_token_indexes=[],
            packed_timesteps=[],
            mse_loss_indexes=[],
            packed_vit_tokens=[],
            vit_token_seqlens=[],
            packed_vit_position_ids=[],
            packed_vit_token_indexes=[],
            vit_video_grid_thw=[],
            vae_video_grid_thw=[],
            video_grid_thw=[],
            vit_video_tensors=[],
            vae_video_latent=[],
            vae_data_mode=[],
            vit_data_mode=[],
            sample_task=[],
            sample_modality=[],
            save_fps=12,
        )
        return sequence_status

    def _read_jsonl(self) -> List[Dict[str, Any]]:
        """Read the JSONL file."""
        data = []
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def _maybe_enhance_t2v_prompt(self, prompt: str) -> str:
        if self.data_config.task != "t2v":
            return prompt
        if not getattr(self.data_config, "enhance_prompt", False):
            return prompt

        from common.utils.caption_rewrite import has_rewrite_api_key, rewrite_prompt

        if not has_rewrite_api_key():
            return prompt

        try:
            enhanced_prompt = rewrite_prompt(prompt)
        except Exception as exc:
            print(f"[enhance_prompt][t2v][warning] prompt rewrite failed, use original prompt. error={exc}")
            return prompt
        print(f"[enhance_prompt][t2v][original] {prompt}")
        print(f"[enhance_prompt][t2v][rewritten] {enhanced_prompt}")
        return enhanced_prompt

    def _maybe_enhance_i2v_prompt(self, prompt: str, image_path: str) -> str:
        if "i2v" not in self.data_config.task:
            return prompt
        if not getattr(self.data_config, "enhance_prompt", False):
            return prompt

        from common.utils.caption_rewrite import has_rewrite_api_key, rewrite_i2v_prompt

        if not has_rewrite_api_key():
            return prompt

        try:
            enhanced_prompt = rewrite_i2v_prompt(prompt, image_path=image_path)
        except Exception as exc:
            print(f"[enhance_prompt][i2v][warning] prompt rewrite failed, use original prompt. error={exc}")
            return prompt
        print(f"[enhance_prompt][i2v][image] {image_path}")
        print(f"[enhance_prompt][i2v][original] {prompt}")
        print(f"[enhance_prompt][i2v][rewritten] {enhanced_prompt}")
        return enhanced_prompt

    def __len__(self) -> int:
        return len(self.data)


    @staticmethod
    def _read_decord(video: VideoReader, frame_idx: List[int]) -> List[Image.Image]:
        frames_np = video.get_batch(frame_idx).asnumpy()
        return [Image.fromarray(frame) for frame in frames_np]

    def get_video_tensor_online(self, media_url, vision_stream, worker_id=0, element_dtype="image") -> torch.Tensor:
        self.vision_stream = vision_stream
        video_stream = media_url

        if element_dtype == "image":
            image = Image.open(video_stream)
            if image.mode == "P":
                image = image.convert("RGBA")
            if image.mode == "RGBA":
                bg = Image.new("RGB", image.size, (255, 255, 255))
                bg.paste(image, mask=image.split()[3])
                image = bg
            else:
                image = image.convert("RGB")
            video_frames = [image]
        else:
            video_reader = VideoReader(video_stream, ctx=decord.cpu(worker_id % self.cpu_count))
            total_frames = len(video_reader)

            try:
                fps = int(round(float(video_reader.get_avg_fps())))
            except Exception:
                fps = 24
            frames_info = {
                    "clip_indices": [(0, total_frames)],
                    "fps": fps,
                }

            frames_sampler_output: FrameSamplerOutput = self.frame_sampler(frames_info)
            video_frames = self._read_decord(video_reader, frames_sampler_output.indices)

        if vision_stream == "vae_video":
            video_tensor = self.transform(video_frames)
        elif vision_stream == "vit_video":
            video_tensor = self.vit_transform(video_frames)
            if element_dtype == "image":
                video_tensor = video_tensor.repeat(1, 2, 1, 1)
            if video_tensor.shape[1] % 2 == 1:
                last_frame = video_tensor[:, -1:, :, :]
                video_tensor = torch.cat([video_tensor, last_frame], dim=1)

        else:
            raise ValueError(f"Unknown vision_stream: {vision_stream}")
        return video_tensor

    def process_vit_video(self, video_tensor, curr: int, curr_rope_id: int, curr_split_len: int, curr_video_grid_thw: None, item_loss=0):
        if not self.data_config.text_template:
            self.sample["packed_text_ids"].append(self.start_of_image)
            self.sample["packed_text_indexes"].append(curr)
            curr += 1
            curr_split_len += 1

        if isinstance(video_tensor, torch.Tensor):
            self.sample["vit_video_tensors"].append(video_tensor)
            vit_tokens = patchify_video_with_merge(
                video_tensor, self.data_config.vit_patch_size, self.data_config.vit_patch_size_temporal
            )
            num_video_tokens = vit_tokens.shape[0] // 4
            t, h, w = video_tensor.size(1), video_tensor.size(2), video_tensor.size(3)

            self.sample["packed_vit_tokens"].append(vit_tokens)
            self.sample["vit_data_mode"].append("online")

        if t is not None:
            vit_video_grid_thw = [
                t // self.data_config.vit_patch_size_temporal,
                h // self.data_config.vit_patch_size,
                w // self.data_config.vit_patch_size,
            ]
        self.sample["vit_video_grid_thw"].append(vit_video_grid_thw)
        curr_video_grid_thw.append(vit_video_grid_thw)

        self.sample["vit_token_seqlens"].append(num_video_tokens)
        self.sample["packed_vit_position_ids"].append(
            torch.zeros(num_video_tokens)
        )

        if not self.data_config.text_template:
            self.sample["packed_vit_token_indexes"].extend(range(curr, curr + num_video_tokens))
            curr += num_video_tokens
            curr_split_len += num_video_tokens

            self.sample["packed_text_ids"].extend([self.image_token_id] * num_video_tokens)
            self.sample["packed_text_ids"].append(self.end_of_image)
            self.sample["packed_text_indexes"].append(curr)
            curr += 1
            curr_split_len += 1
            self.sample["packed_position_ids"].extend([curr_rope_id] * curr_split_len)
            curr_rope_id += 1

            self.sample["attn_modes"].append("full")
            self.sample["split_lens"].append(curr_split_len)

        return self.sample, curr, curr_rope_id, curr_split_len, curr_video_grid_thw, num_video_tokens

    def process_text(self, caption: str, curr: int, curr_rope_id: int, curr_split_len: int, item_loss=0):
        """Process text and append special tokens."""
        text_ids = self.tokenizer.encode(caption)
        shifted_text_ids = [self.bos_token_id] + text_ids
        self.sample["packed_text_ids"].extend(shifted_text_ids)
        self.sample["packed_text_indexes"].extend(range(curr, curr + len(shifted_text_ids)))

        if item_loss == 1:
            loss_token_shift = 0
            self.sample["ce_loss_indexes"].extend(range(curr - loss_token_shift, curr + len(shifted_text_ids)))
            self.sample["ce_loss_weights"].extend([len2weight(len(shifted_text_ids) + loss_token_shift)] * (len(shifted_text_ids) + loss_token_shift))
            self.sample["packed_label_ids"].extend(text_ids + [self.eos_token_id])
        curr += len(shifted_text_ids)
        curr_split_len += len(shifted_text_ids)

        # Append the <|im_end|> end token.
        self.sample["packed_text_ids"].append(self.eos_token_id)
        self.sample["packed_text_indexes"].append(curr)
        curr += 1
        curr_split_len += 1
        self.sample["attn_modes"].append("causal")
        self.sample["packed_position_ids"].extend(range(curr_rope_id, curr_rope_id + curr_split_len))
        curr_rope_id += curr_split_len
        self.sample["split_lens"].append(curr_split_len)
        return self.sample, curr, curr_rope_id, curr_split_len


    def process_vae_video(self, video_tensor, curr: int, curr_rope_id: int, curr_split_len: int, curr_video_grid_thw: None, video_sizes: list, item_loss=0):
        if not self.data_config.text_template:
            num_special_tokens = 0
            self.sample["packed_text_ids"].append(self.start_of_image)
            self.sample["packed_text_indexes"].append(curr)
            curr += 1
            curr_split_len += 1
            num_special_tokens += 1

        if isinstance(video_tensor, torch.Tensor):
            self.sample["vae_video_tensors"].append(video_tensor)
            _, T, H, W = video_tensor.shape
            _T, _H, _W = self.data_config.vae_downsample
            t = (T - 1) // _T + 1
            h = H // _H
            w = W // _W
            self.sample["vae_data_mode"].append("online")

            spatial_merge_size = 2
            vae_video_grid_thw = [
                t,
                h * spatial_merge_size,
                w * spatial_merge_size,
            ]

            self.sample["vae_video_grid_thw"].append(vae_video_grid_thw)
            curr_video_grid_thw.append(vae_video_grid_thw)
            self.sample["vae_latent_shapes"].append((t, h, w))
            packed_latent_position_ids = get_flattened_position_ids_extrapolate_video(t, h, w, max_latent_size=self.data_config.max_latent_size)
            self.sample["packed_latent_position_ids"].append(packed_latent_position_ids)

            num_vid_tokens = t * h * w
            if not self.data_config.text_template:
                self.sample["packed_vae_token_indexes"].extend(range(curr, curr + num_vid_tokens))

            if item_loss == 1:
                timestep = np.random.randn()

                frame_condition_idx = self.frame_condition_idx
                packed_timesteps = [timestep] * num_vid_tokens

                mse_loss_indexes = list(range(curr, curr + num_vid_tokens))
                frame_condition_indexes = []
                for idx in frame_condition_idx:
                    if idx == -1:
                        idx = t - 1
                        if idx == 1:
                            continue
                    frame_condition_indexes.extend(mse_loss_indexes[idx * h * w : (idx + 1) * h * w])
                    packed_timesteps[idx * h * w : (idx + 1) * h * w] = [-sys.float_info.max] * (h * w)
                if frame_condition_idx:
                    mse_loss_indexes = sorted(list(set(mse_loss_indexes) - set(frame_condition_indexes)))

                if not self.data_config.text_template:
                    self.sample["mse_loss_indexes"].extend(mse_loss_indexes)
            else:
                timestep = float("-inf")
                packed_timesteps = [timestep] * num_vid_tokens

            self.sample["packed_timesteps"].extend(packed_timesteps)

            if not self.data_config.text_template:
                curr += num_vid_tokens
                curr_split_len += num_vid_tokens

                self.sample["packed_text_ids"].extend([self.image_token_id] * num_vid_tokens)

                # Append the <|endofimage|> image end token.
                self.sample["packed_text_ids"].append(self.end_of_image)
                self.sample["packed_text_indexes"].append(curr)
                curr += 1
                curr_split_len += 1
                num_special_tokens += 1

                # Update sequence state.
                if item_loss == 1:
                    self.sample["attn_modes"].append("noise")
                else:
                    self.sample["attn_modes"].append("full_noise")

                self.sample["packed_position_ids"].extend([curr_rope_id] * (num_vid_tokens + num_special_tokens))
                curr_rope_id += 1
                self.sample["split_lens"].append(curr_split_len)

            video_sizes.append([T, H, W])

        return self.sample, curr, curr_rope_id, curr_split_len, curr_video_grid_thw, video_sizes, num_vid_tokens

    def process_text_template(
        self,
        text_ids,
        spans_index,
        tgt_index,
        caption_index,
        video_types: list[str],
        curr: int,
        curr_rope_id: int,
        curr_split_len: int,
        item_loss=0,
    ):
        self.sample["packed_text_ids"].extend(text_ids)
        self.sample["sample_lens"] = len(text_ids)
        curr_split_idx = curr

        for video_id, span_index in enumerate(spans_index):
            vision_start, vision_end = curr_split_idx + span_index[0], curr_split_idx + span_index[-1]
            self.sample["packed_text_indexes"].extend(range(curr, vision_start))
            if (vision_start - 1) - curr != 0:
                curr_split_len = (vision_start - 1) - curr
                self.sample["packed_position_ids"].extend(
                    range(curr_rope_id, curr_rope_id + curr_split_len)
                )
                curr_rope_id += curr_split_len
                self.sample["sample_modality"].extend([modality_map["system_prompt"]] * curr_split_len)

                if caption_index != [] and caption_index[0] in range(curr, curr + curr_split_len):
                    split_len_1 = caption_index[0] - curr
                    split_len_2 = len(caption_index)
                    split_len_3 = curr_split_len - split_len_1 - split_len_2

                    split_len_text = [split_len_1, split_len_2, split_len_3]
                    split_len_text = [x for x in split_len_text if x != 0]
                    self.sample["attn_modes"].extend(["causal"] * len(split_len_text))
                    self.sample["split_lens"].extend(split_len_text)
                else:
                    self.sample["attn_modes"].append("causal")
                    self.sample["split_lens"].append(curr_split_len)

            curr_split_len = len(span_index) + 2
            if video_types[video_id] == "vit_video":
                self.sample["packed_vit_token_indexes"].extend(range(vision_start, vision_end + 1))
                self.sample["attn_modes"].append("full")
                self.sample["sample_modality"].extend([modality_map["ref_vit"]] * curr_split_len)
            elif "vae_video" in video_types[video_id]:
                self.sample["packed_vae_token_indexes"].extend(range(vision_start, vision_end + 1))
                if "cond" in video_types[video_id]:
                    self.sample["attn_modes"].append("full_noise")
                    if self.sample_task == "edit":
                        self.sample["sample_modality"].extend([modality_map["ref_source"]] * curr_split_len)
                    elif self.sample_task == "idip":
                        self.sample["sample_modality"].extend([modality_map["ref_image"]] * curr_split_len)
                elif "target" in video_types[video_id]:
                    self.sample["mse_loss_indexes"].extend(range(vision_start, vision_end + 1))
                    self.sample["attn_modes"].append("noise")
                    self.sample["sample_modality"].extend([modality_map["noise"]] * curr_split_len)
                else:
                    raise ValueError(f"video_types {video_types[video_id]} not supported")

            self.sample["packed_position_ids"].extend([curr_rope_id] * curr_split_len)
            self.sample["split_lens"].append(len(span_index) + 2)
            curr = vision_end + 1
            curr_rope_id += 1
            self.sample["packed_text_indexes"].append(curr)
            curr += 1

        len_split_last = self.sample["sample_lens"] - (curr - curr_split_idx) if spans_index != [] else len(text_ids)
        if len_split_last != 0:
            self.sample["split_lens"].append(len_split_last)
            self.sample["packed_text_indexes"].extend(range(curr, curr + len_split_last))
            self.sample["packed_position_ids"].extend(range(curr_rope_id, curr_rope_id + len_split_last))
            self.sample["attn_modes"].append("causal")
            self.sample["sample_modality"].extend([modality_map["system_prompt"]] * len_split_last)

        if item_loss == 1:
            packed_label_index = tgt_index
            self.sample["packed_label_ids"].extend(text_ids[packed_label_index[0] :])
            packed_label_index = np.asarray(packed_label_index, dtype=np.int64) + curr_split_idx
            ce_loss_indexes = (packed_label_index - 1).tolist()
            self.sample["ce_loss_indexes"].extend(ce_loss_indexes)
            self.sample["ce_loss_weights"].extend([len2weight(len(packed_label_index))] * (len(packed_label_index)))

        if caption_index != []:
            self.sample["sample_modality"][caption_index[0] : caption_index[-1] + 1] = [modality_map["text"]] * (caption_index[-1] - caption_index[0] + 1)

        curr_split_idx += len(text_ids)
        curr = curr_split_idx
        return self.sample, curr, curr_rope_id, curr_split_len
    def process_und_template(self, system_prompt, user_prompt, answer, vit_video_tensor):
        curr = 0
        sample_lens = 0
        curr_rope_id = 0
        curr_video_grid_thw = []

        prompt_prefix = "<|im_start|>" + "system\n" + system_prompt + "<|im_end|>" + "\n" + "<|im_start|>" + "user\n"
        text_ids_prompt_prefix = self.tokenizer.encode(prompt_prefix)
        self.sample["packed_text_ids"].extend(text_ids_prompt_prefix)
        self.sample["packed_text_indexes"].extend(range(curr, curr + len(text_ids_prompt_prefix)))
        curr += len(text_ids_prompt_prefix)
        split_len_prefix = len(text_ids_prompt_prefix)

        # Update sequence state.
        self.sample["attn_modes"].append("causal")
        self.sample["packed_position_ids"].extend(range(curr_rope_id, curr_rope_id + split_len_prefix))
        self.sample["split_lens"].append(split_len_prefix)
        curr_rope_id += split_len_prefix

        self.sample["packed_text_ids"].append(self.start_of_image)
        self.sample["packed_text_indexes"].append(curr)
        curr += 1
        split_len_vision_token = 1

        if isinstance(vit_video_tensor, torch.Tensor):
            self.sample["vit_video_tensors"].append(vit_video_tensor)

            # Preprocess the video.
            vit_tokens = patchify_video_with_merge(
                vit_video_tensor, self.data_config.vit_patch_size, self.data_config.vit_patch_size_temporal
            )
            num_video_tokens = vit_tokens.shape[0] // 4
            t, h, w = vit_video_tensor.size(1), vit_video_tensor.size(2), vit_video_tensor.size(3)

            self.sample["packed_vit_tokens"].append(vit_tokens)
            self.sample["vit_data_mode"].append("online")

        if t is not None:
            vit_video_grid_thw = [
                t // self.data_config.vit_patch_size_temporal,
                h // self.data_config.vit_patch_size,
                w // self.data_config.vit_patch_size,
            ]
        self.sample["vit_video_grid_thw"].append(vit_video_grid_thw)
        curr_video_grid_thw.append(vit_video_grid_thw)

        self.sample["vit_token_seqlens"].append(num_video_tokens)
        self.sample["packed_vit_position_ids"].append(
            torch.zeros(num_video_tokens)
        )

        self.sample["packed_vit_token_indexes"].extend(range(curr, curr + num_video_tokens))
        curr += num_video_tokens
        split_len_vision_token += num_video_tokens

        # Fill placeholder position_ids.
        self.sample["packed_text_ids"].extend([self.image_token_id] * num_video_tokens)

        # Append the <|endofimage|> image end token.
        self.sample["packed_text_ids"].append(self.end_of_image)
        self.sample["packed_text_indexes"].append(curr)
        curr += 1
        split_len_vision_token += 1

        # Update sequence state.
        self.sample["attn_modes"].append("full")
        self.sample["packed_position_ids"].extend([curr_rope_id] * split_len_vision_token)
        self.sample["split_lens"].append(split_len_vision_token)
        curr_rope_id += 1

        prompt_postfix = user_prompt + "<|im_end|>" + "\n" + "<|im_start|>" + "assistant"
        text_ids_prompt_postfix = self.tokenizer.encode(prompt_postfix)
        self.sample["packed_text_ids"].extend(text_ids_prompt_postfix)
        self.sample["packed_text_indexes"].extend(range(curr, curr + len(text_ids_prompt_postfix)))
        curr += len(text_ids_prompt_postfix)
        split_len_postfix = len(text_ids_prompt_postfix)

        self.sample["attn_modes"].append("causal")
        self.sample["packed_position_ids"].extend(range(curr_rope_id, curr_rope_id + split_len_postfix))
        self.sample["split_lens"].append(split_len_postfix)
        curr_rope_id += split_len_postfix

        answer = "\n" + answer
        answer_ids = self.tokenizer.encode(answer)
        shifted_text_ids_answer = answer_ids + [self.eos_token_id]
        self.sample["packed_text_ids"].extend(shifted_text_ids_answer)
        self.sample["packed_text_indexes"].extend(range(curr, curr + len(shifted_text_ids_answer)))

        self.sample["ce_loss_indexes"].extend(range(curr, curr + len(shifted_text_ids_answer)))
        self.sample["ce_loss_weights"].extend([len2weight(len(shifted_text_ids_answer))] * (len(shifted_text_ids_answer)))
        self.sample["packed_label_ids"].extend(shifted_text_ids_answer)

        curr += len(shifted_text_ids_answer)
        split_len_answer = len(shifted_text_ids_answer)

        self.sample["attn_modes"].append("causal")
        self.sample["packed_position_ids"].extend(range(curr_rope_id, curr_rope_id + split_len_answer))
        self.sample["split_lens"].append(split_len_answer)
        curr_rope_id += split_len_answer

        sample_lens = len(self.sample["packed_text_ids"])

        return sample_lens, curr_video_grid_thw

    def _finalize_sample(self, sample_lens, curr_video_grid_thw, sample_type, sample=None, additional_fields=None, video_sizes=None):
        self.sample["sample_lens"] = [sample_lens]
        self.sample["video_grid_thw"] = torch.tensor([curr_video_grid_thw])
        self.sample["packed_text_ids"] = torch.tensor(self.sample["packed_text_ids"])
        self.sample["packed_text_indexes"] = torch.tensor(self.sample["packed_text_indexes"])

        self.sample["packed_vae_token_indexes"] = torch.tensor(self.sample["packed_vae_token_indexes"])
        self.sample["packed_position_ids"] = torch.tensor(self.sample["packed_position_ids"])
        self.sample["vae_video_grid_thw"] = torch.tensor(self.sample["vae_video_grid_thw"])

        self.sample["vit_video_grid_thw"] = torch.tensor(self.sample["vit_video_grid_thw"])
        self.sample["packed_vit_token_indexes"] = torch.tensor(self.sample["packed_vit_token_indexes"])

        self.sample["sample_N_target"] = torch.tensor([[1]])
        self.sample["sample_type"] = [sample_type]
        self.sample["padded_videos"] = self.sample["vae_video_tensors"]

        if "ce_loss_indexes" in self.sample and len(self.sample["ce_loss_indexes"]) > 0:
            self.sample["ce_loss_indexes"] = torch.tensor(self.sample["ce_loss_indexes"])
        self.sample["mse_loss_indexes"] = torch.tensor(self.sample["mse_loss_indexes"])
        if video_sizes is not None:
            self.sample["video_sizes"] = torch.tensor(video_sizes)
        elif "video_sizes" in self.sample:
            self.sample["video_sizes"] = torch.tensor(self.sample["video_sizes"])
        if "sample_modality" in self.sample and len(self.sample["sample_modality"]) > 0:
            self.sample["sample_modality"] = torch.tensor(self.sample["sample_modality"])

        if sample is not None:
            for key in ["index", "category", "question", "gt"]:
                if key in sample:
                    self.sample[key] = sample[key]

        if additional_fields is not None:
            for key, value in additional_fields.items():
                self.sample[key] = value

        return self.sample

    def ti2t_sample(self, idx: int) -> Dict[str, Any]:
        self.sample = self.set_sequence_status()
        sample = self.data[idx]

        system_prompt = sample["system_prompt"]
        user_prompt = sample["user_prompt"]
        answer = sample["gt"]
        image_path = sample["image_path"]
        vit_image_tensor = self.get_video_tensor_online(image_path, vision_stream="vit_video", element_dtype="image")

        sample_lens, curr_video_grid_thw = self.process_und_template(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            answer=answer,
            vit_video_tensor=vit_image_tensor,
        )

        self.sample["system_prompt"] = system_prompt
        self.sample["user_prompt"] = user_prompt
        self.sample["image_path"] = image_path
        self.sample["instruction"] = user_prompt

        return self._finalize_sample(
            sample_lens, curr_video_grid_thw,
            sample_type="und",
            sample=sample
        )

    def t2v_sample(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        thw_video, thw_downsample = self.get_thw()
        t, h, w = thw_downsample
        num_vid_tokens = t * h * w
        spatial_merge_size = 2

        self.sample = self.set_sequence_status()
        packed_text_indexes, packed_position_ids, sample_modality = [], [], []
        sample = self.data[idx]
        if "prompt_en" in sample.keys():
            user_prompt = "".join(sample["prompt_en"][0])
        else:
            user_prompt = sample["data"]
        user_prompt = self._maybe_enhance_t2v_prompt(user_prompt)

        if self.data_config.text_template:
            caption_instruction = generate_system_prompt(system_prompt_type=self.data_config.task, vision_type=self.data_config.target_modality)

            text_template_user, text_template_assistant, vit_num_tokens, video_types = [], [], [], []
            if self.system_prompt_type == 'SP2':
                user_prompt = caption_instruction + " " + user_prompt
                caption_instruction = "You are a helpful assistant. "
            elif self.system_prompt_type == 'SP1':
                caption_instruction = "You are a helpful assistant. " + caption_instruction

            text_template_user.append({"type": "text", "text": user_prompt})
        else:
            text_ids = self.tokenizer.encode(user_prompt)
            text_ids = [self.new_token_ids["bos_token_id"]] + text_ids + [self.new_token_ids["eos_token_id"]]
            text_split_len = len(text_ids)
            packed_text_indexes.extend(range(0, text_split_len))
            packed_position_ids.extend(range(0, text_split_len))
            sample_modality.extend([modality_map['text']] * text_split_len)

        if self.data_config.text_template:
            text_template_assistant.append({"type":self.data_config.target_modality})
        else:
            text_ids.append(self.new_token_ids["start_of_image"])
            packed_text_indexes.append(text_split_len)
            packed_vae_token_indexes = torch.tensor(range(len(text_ids), len(text_ids) + num_vid_tokens))
            text_ids.extend([self.image_token_id] * num_vid_tokens)
            text_ids.append(self.new_token_ids["end_of_image"])
            packed_text_indexes.append(len(text_ids) - 1)
            video_split_len = num_vid_tokens + 2
            packed_position_ids.extend([text_split_len] * video_split_len)
            sample_modality.extend([modality_map['noise']] * video_split_len)

        if self.data_config.text_template:
            all_token_id, spans_index, tgt_index, search_index = self.render_template(caption_instruction, text_template_assistant, text_template_user, [num_vid_tokens], search_text=user_prompt)

            self.sample, curr, curr_rope_id, curr_split_len = self.process_text_template(
                all_token_id,
                spans_index,
                tgt_index,
                search_index,
                video_types=['target_vae_video'],
                curr=0,
                curr_rope_id=0,
                curr_split_len=0,
                item_loss=0,
                )

        return {
            "packed_text_ids": torch.tensor(text_ids) if not self.data_config.text_template else torch.tensor(self.sample["packed_text_ids"]),
            "packed_text_indexes": torch.tensor(packed_text_indexes) if not self.data_config.text_template else torch.tensor(self.sample["packed_text_indexes"]),
            "packed_vae_token_indexes": packed_vae_token_indexes if not self.data_config.text_template else torch.tensor(self.sample["packed_vae_token_indexes"]),
            "vae_video_grid_thw": torch.tensor([[t, h * spatial_merge_size, w * spatial_merge_size]]),
            "video_grid_thw": torch.tensor([[[t, h * spatial_merge_size, w * spatial_merge_size]]]),
            "sample_N_target": torch.tensor([[1]]),
            "split_lens": [text_split_len, video_split_len] if not self.data_config.text_template else self.sample["split_lens"],
            "attn_modes": ["causal", "noise"] if not self.data_config.text_template else self.sample["attn_modes"],
            "sample_lens": [text_split_len + video_split_len] if not self.data_config.text_template else [self.sample["sample_lens"]],
            "val_sample_type": ["gen"],
            "padded_latent": None,
            "mse_loss_indexes": packed_vae_token_indexes if not self.data_config.text_template else torch.tensor(self.sample["mse_loss_indexes"]),
            "video_sizes": torch.tensor([thw_video]),
            "packed_position_ids": torch.tensor(packed_position_ids) if not self.data_config.text_template else torch.tensor(self.sample["packed_position_ids"]),
            "caption": user_prompt,
            "sample_type": ["gen"],
            "index": sample["index"],
            "caption_cn": user_prompt,
            "original_prompt_en": sample["original_prompt_en"] if "original_prompt_en" in sample.keys() else user_prompt,
            "sample_task": torch.zeros(text_split_len + video_split_len) if not self.data_config.text_template else torch.zeros(self.sample["sample_lens"]),
            "sample_modality": torch.tensor(sample_modality) if not self.data_config.text_template else torch.tensor(self.sample["sample_modality"]),
            "additional_info": sample["additional_info"] if "additional_info" in sample.keys() else None,
        }

    def get_thw(self):
        _T, _H, _W = self.data_config.vae_downsample
        if self.data_config.target_modality == "image":
            t = 1
            t_ = 1
        elif self.data_config.target_modality == "video":
            t = (self.data_config.num_frames - 1) // _T + 1
            t_ = self.data_config.num_frames

        h = self.data_config.H // _H
        w = self.data_config.W // _W
        return [t_, self.data_config.H, self.data_config.W], [t, h, w]  # Original video size and downsampled size.


    def gen_timesteps(self, t, h, w, curr, num_vid_tokens):
        timestep = np.random.randn()
        frame_condition_idx = self.frame_condition_idx
        packed_timesteps = [timestep] * num_vid_tokens
        mse_loss_indexes = list(range(curr, curr + num_vid_tokens))
        frame_condition_indexes = []
        for idx in frame_condition_idx:
            frame_condition_indexes.extend(mse_loss_indexes[idx * h * w : (idx + 1) * h * w])
            packed_timesteps[idx * h * w : (idx + 1) * h * w] = [-sys.float_info.max] * (h * w)
        if frame_condition_idx:
            mse_loss_indexes = sorted(list(set(mse_loss_indexes) - set(frame_condition_indexes)))
        return packed_timesteps, mse_loss_indexes

    def tiv2v_sample(self, idx: int) -> Dict[str, Any]:
        sample_modality, text_template_user, text_template_assistant, vit_num_tokens, video_types = [], [], [], [], []
        self.sample = self.set_sequence_status()
        sample_lens = 0
        sample = self.data[idx]

        index = sample["index"]
        data_sample = sample["data"]
        additional_info = sample["data"]["additional_info"] if "additional_info" in sample["data"] else []

        interleave_array, element_dtype_array, istarget_in_interleave = data_sample["interleave_array"], data_sample["element_dtype_array"], data_sample["istarget_in_interleave"]

        curr, curr_rope_id, curr_split_len, curr_video_grid_thw, video_sizes, caption_all = 0, 0, 0, [], [], ''
        for element, element_dtype, is_target in zip(interleave_array, element_dtype_array, istarget_in_interleave):
            if element_dtype == "text":
                caption_all += element
                if self.data_config.text_template:
                    text_template_user.append({"type": "text", "text": element})
                    search_text = element
                else:
                    self.sample, curr, curr_rope_id, curr_split_len = self.process_text(element, curr=curr, curr_rope_id=curr_rope_id, curr_split_len=0, item_loss=is_target)
                    sample_lens += curr_split_len
                    sample_modality.extend([modality_map['text']] * curr_split_len)
            elif element_dtype in ["image", "video"]:
                if is_target == 0:
                    vit_image_tensor = self.get_video_tensor_online(element, vision_stream="vit_video", element_dtype=element_dtype)
                    self.sample, curr, curr_rope_id, curr_split_len, curr_video_grid_thw, num_tokens_ = self.process_vit_video(
                        vit_image_tensor, curr=curr, curr_rope_id=curr_rope_id, curr_split_len=0, curr_video_grid_thw=curr_video_grid_thw, item_loss=0
                        )
                    if self.data_config.text_template:
                        text_template_user.append({"type": element_dtype})
                        vit_num_tokens.append(num_tokens_)
                        video_types.append("vit_video")
                    else:
                        sample_lens += curr_split_len
                        sample_modality.extend([modality_map['ref_vit']] * curr_split_len)

                    # Process VAE conditioning input.
                    vae_image_tensor = self.get_video_tensor_online(element, vision_stream="vae_video", element_dtype=element_dtype)
                    self.sample, curr, curr_rope_id, curr_split_len, curr_video_grid_thw, video_sizes, num_tokens_ = self.process_vae_video(
                        vae_image_tensor, curr=curr, curr_rope_id=curr_rope_id, curr_split_len=0, curr_video_grid_thw=curr_video_grid_thw, video_sizes=video_sizes, item_loss=is_target
                    )
                    vit_num_tokens.append(num_tokens_)
                    text_template_user.append({"type": element_dtype})
                    video_types.append("cond_vae_video")

                    if self.sample_task == 'edit':
                        self.data_config.num_frames, self.data_config.H, self.data_config.W = vae_image_tensor.shape[1], vae_image_tensor.shape[2], vae_image_tensor.shape[3]

        # Process the VAE target input.
        thw_video, thw_downsample = self.get_thw()
        video_sizes.append(thw_video)
        t, h, w = thw_downsample
        num_vid_tokens = t * h * w
        self.sample["vae_data_mode"].append("online")
        spatial_merge_size = 2
        vae_video_grid_thw = [
                t,
                h * spatial_merge_size,
                w * spatial_merge_size,
            ]
        curr_video_grid_thw.append(vae_video_grid_thw)
        self.sample["vae_video_grid_thw"].append(vae_video_grid_thw)
        self.sample["vae_latent_shapes"].append((t, h, w))
        # Use 3D-aware extrapolated position encoding.
        packed_latent_position_ids = get_flattened_position_ids_extrapolate_video(t, h, w, max_latent_size=self.data_config.max_latent_size)
        self.sample["packed_latent_position_ids"].append(packed_latent_position_ids)
        packed_timesteps, mse_loss_indexes = self.gen_timesteps(t, h, w, curr, num_vid_tokens)
        self.sample["packed_timesteps"].extend(packed_timesteps)
        vae_tensor = torch.randn([3, thw_video[0], thw_video[1], thw_video[2]], dtype=torch.float32) # Raw CTHW video, not latent.
        self.sample["vae_video_tensors"].append(vae_tensor)
        if self.data_config.text_template:
            vit_num_tokens.append(num_vid_tokens)
            text_template_assistant.append({"type": self.data_config.target_modality})
            video_types.append("target_vae_video")

            if text_template_user[0]['type']=='text':
                text_template_user = text_template_user[1:] + text_template_user[:1]
            caption_instruction = generate_system_prompt(system_prompt_type=self.data_config.task, vision_type=element_dtype)
            all_token_id, spans_index, tgt_index, search_index = self.render_template(caption_instruction, text_template_assistant, text_template_user, vit_num_tokens, search_text=search_text)
            self.sample, curr, curr_rope_id, curr_split_len = self.process_text_template(
                all_token_id,
                spans_index,
                tgt_index,
                search_index,
                video_types=video_types,
                curr=0,
                curr_rope_id=0,
                curr_split_len=0,
                item_loss=0,
                )
            sample_lens = len(all_token_id)
            sample_modality = self.sample["sample_modality"]

        additional_fields = {
            "caption": caption_all,
            "caption_cn": caption_all,
            "index": sample["index"],
            "additional_info": additional_info
        }

        if self.sample_task == 'edit':
            self.sample["sample_task"] = torch.ones(sample_lens) * sample_task_map['edit']
        elif self.sample_task == 'idip':
            self.sample["sample_task"] = torch.ones(sample_lens) * sample_task_map['idip']

        return self._finalize_sample(
            sample_lens, curr_video_grid_thw,
            sample_type="gen",
            sample=sample,
            additional_fields=additional_fields,
            video_sizes=video_sizes
        )

    def ff2v_sample(self, idx: int) -> Dict[str, Any]:
        """Get a single sample."""
        sample_modality, text_template_user, text_template_assistant, vit_num_tokens, video_types, search_text = [], [], [], [], [], ''
        self.sample = self.set_sequence_status()
        sample_lens = 0
        sample = self.data[idx]

        index = sample["index"]
        data_sample = sample["data"]
        additional_info = sample["data"]["additional_info"] if "additional_info" in sample["data"] else []

        interleave_array, element_dtype_array, istarget_in_interleave = data_sample["interleave_array"], data_sample["element_dtype_array"], data_sample["istarget_in_interleave"]
        interleave_array = list(interleave_array)

        text_idx = next((i for i, dtype in enumerate(element_dtype_array) if dtype == "text"), None)
        image_idx = next(
            (
                i
                for i, (dtype, is_target) in enumerate(zip(element_dtype_array, istarget_in_interleave))
                if dtype == "image" and is_target == 0
            ),
            None,
        )
        if text_idx is not None and image_idx is not None:
            interleave_array[text_idx] = self._maybe_enhance_i2v_prompt(
                interleave_array[text_idx],
                image_path=interleave_array[image_idx],
            )

        curr, curr_rope_id, curr_split_len, curr_video_grid_thw, video_sizes, caption_all, vae_image_tensor = 0, 0, 0, [], [], '', None
        for element, element_dtype, is_target in zip(interleave_array, element_dtype_array, istarget_in_interleave):
            if element_dtype == "text":
                caption_all += element
                if self.data_config.text_template:
                    text_template_user.append({"type": "text", "text": element})
                    search_text = element
                else:
                    self.sample, curr, curr_rope_id, curr_split_len = self.process_text(element, curr=curr, curr_rope_id=curr_rope_id, curr_split_len=0, item_loss=is_target)
                    sample_lens += curr_split_len
                    sample_modality.extend([modality_map['text']] * curr_split_len)
            elif element_dtype in ["image", "video"]:
                if is_target == 0:
                    vae_image_tensor = self.get_video_tensor_online(element, vision_stream="vae_video", element_dtype=element_dtype)
                    self.data_config.H, self.data_config.W = vae_image_tensor.shape[2], vae_image_tensor.shape[3]
                    self.frame_condition_idx = [0]

        # Add the target VAE latent.
        thw_video, thw_downsample = self.get_thw()
        video_sizes.append(thw_video)
        t, h, w = thw_downsample
        num_vid_tokens = t * h * w
        self.sample["vae_data_mode"].append("online")
        spatial_merge_size = 2
        vae_video_grid_thw = [
                t,
                h * spatial_merge_size,
                w * spatial_merge_size,
            ]
        curr_video_grid_thw.append(vae_video_grid_thw)
        self.sample["vae_video_grid_thw"].append(vae_video_grid_thw)
        self.sample["vae_latent_shapes"].append((t, h, w))
        packed_latent_position_ids = get_flattened_position_ids_extrapolate_video(t, h, w, max_latent_size=self.data_config.max_latent_size)
        self.sample["packed_latent_position_ids"].append(packed_latent_position_ids)
        packed_timesteps, mse_loss_indexes = self.gen_timesteps(t, h, w, curr, num_vid_tokens)
        self.sample["packed_timesteps"].extend(packed_timesteps)
        vae_tensor = torch.randn([3, thw_video[0], thw_video[1], thw_video[2]], dtype=torch.float32)
        if vae_image_tensor is not None:  # Fill in the first frame.
            vae_tensor[:, :4, :, :] = vae_image_tensor[:, 0:1, :, :].repeat(1, 4, 1, 1)
        else:
            raise ValueError("vae_image_tensor of first frame is None")
        self.sample["vae_video_tensors"].append(vae_tensor)
        if self.data_config.text_template:
            vit_num_tokens.append(num_vid_tokens)
            text_template_assistant.append({"type": self.data_config.target_modality})
            video_types.append("target_vae_video")

            if len(text_template_user) > 0 and text_template_user[0]['type'] == 'text':
                text_template_user = text_template_user[1:] + text_template_user[:1]
            caption_instruction = generate_system_prompt(system_prompt_type=self.data_config.task, vision_type=self.data_config.target_modality)
            all_token_id, spans_index, tgt_index, search_index = self.render_template(caption_instruction, text_template_assistant, text_template_user, vit_num_tokens, search_text=search_text)
            self.sample, curr, curr_rope_id, curr_split_len = self.process_text_template(
                all_token_id,
                spans_index,
                tgt_index,
                search_index,
                video_types=video_types,
                curr=0,
                curr_rope_id=0,
                curr_split_len=0,
                item_loss=0,
                )
            sample_lens = len(all_token_id)
            sample_modality = self.sample["sample_modality"]

        self.sample["sample_task"] = torch.ones(sample_lens) * sample_task_map[self.sample_task]
        self.sample["sample_modality"] = sample_modality
        has_vit_video_grid = self.sample["vit_video_grid_thw"] != []
        has_packed_vit_token_indexes = self.sample["packed_vit_token_indexes"] != []
        if self.frame_condition_idx != []:
            mse_loss_indexes_first = self.sample["mse_loss_indexes"][0]
            self.sample["mse_loss_indexes"] = [idx + mse_loss_indexes_first for idx in mse_loss_indexes]

        finalized_sample = self._finalize_sample(
            sample_lens,
            curr_video_grid_thw,
            sample_type="gen",
            sample=sample,
            additional_fields={
                "caption": caption_all,
                "caption_cn": caption_all,
                "additional_info": additional_info,
            },
            video_sizes=video_sizes,
        )
        if not has_vit_video_grid:
            finalized_sample["vit_video_grid_thw"] = None
        if not has_packed_vit_token_indexes:
            finalized_sample["packed_vit_token_indexes"] = None
        return finalized_sample

    def render_template(self, instruction, text_template_assistant, text_template_user, vit_num_tokens, search_text=""):
        messages = [
            {
                "role": "user",
                "content": text_template_user,
            },
            {
                "role": "assistant",
                "content": text_template_assistant,
            },
        ]
        caption_all = render_qwenvl_prompt(messages, default_system=instruction, include_assistant_content=True)

        all_token_id, spans_index, tgt_index, search_index = expand_and_index_by_token_ids_new(
            rendered_text=caption_all.strip(), tokens=vit_num_tokens, target_text=f"assistant\n", tokenizer=self.tokenizer, search_text=search_text
        )
        assert len(all_token_id[tgt_index[0] :]) == len(tgt_index)
        return all_token_id, spans_index, tgt_index, search_index

    def x2t_sample(self, idx: int) -> Dict[str, Any]:
        sample_modality = []
        self.sample = self.set_sequence_status()
        sample_lens = 0
        sample = self.data[idx]
        index = sample["index"]
        data_sample = sample["data"]

        interleave_array, element_dtype_array, istarget_in_interleave = data_sample["interleave_array"], data_sample["element_dtype_array"], data_sample["istarget_in_interleave"]

        curr, curr_rope_id, curr_split_len, curr_video_grid_thw, video_sizes, caption_all = 0, 0, 0, [], [], ""
        if self.data_config.text_template:
            text_template_user, text_template_assistant, vit_num_tokens, video_types = [], [], [], []
        for element, element_dtype, is_target in zip(interleave_array, element_dtype_array, istarget_in_interleave):
            if element_dtype == "text":
                if is_target == 1:
                    if self.data_config.text_template:
                        if isinstance(element, str):
                            caption_a = element
                            caption_i = generate_system_prompt(system_prompt_type="caption", vision_type=element_dtype_array[0])
                            caption_q = ""
                            element = [caption_i, caption_q, caption_a]

                        caption_i, caption_q, caption_a = element[0], element[1], element[2]
                        if self.system_prompt_type == 'SP2':
                            caption_q = caption_i + " " + caption_q
                            caption_i = "You are a helpful assistant. "
                        elif self.system_prompt_type == 'SP1':
                            caption_i = "You are a helpful assistant. " + caption_i
                        element = [caption_i, caption_q, caption_a]

                        caption_i, caption_q, caption_a = element[0], element[1], element[2]

                        text_template_assistant.append({"type": "text", "text": caption_a})
                        if caption_q != "":
                            text_template_user.append({"type": "text", "text": caption_q})

                        all_token_id, spans_index, tgt_index, search_index = self.render_template(caption_i, text_template_assistant, text_template_user, vit_num_tokens)
                        self.sample, curr, curr_rope_id, curr_split_len = self.process_text_template(
                            all_token_id,
                            spans_index,
                            tgt_index,
                            search_index,
                            video_types,
                            curr=curr,
                            curr_rope_id=curr_rope_id,
                            curr_split_len=0,
                            item_loss=is_target,
                        )
                        sample_lens += curr_split_len

                        caption_all += "\n".join(element)
                        caption_answer = element[-1]
                    else:
                        if isinstance(element, list):
                            element = element[-1]
                        self.sample, curr, curr_rope_id, curr_split_len = self.process_text(
                            element, curr=curr, curr_rope_id=curr_rope_id, curr_split_len=0, item_loss=is_target
                        )
                        sample_lens += curr_split_len
                        sample_modality.extend([modality_map["text"]] * curr_split_len)
                        caption_all += element
                        caption_answer = element

            elif element_dtype in ["image", "video"]:

                vit_image_tensor = self.get_video_tensor_online(element, vision_stream="vit_video", element_dtype=element_dtype)
                self.sample, curr, curr_rope_id, curr_split_len, curr_video_grid_thw, num_tokens_ = self.process_vit_video(
                    vit_image_tensor, curr=curr, curr_rope_id=curr_rope_id, curr_split_len=0, curr_video_grid_thw=curr_video_grid_thw, item_loss=0
                )
                sample_lens += curr_split_len
                sample_modality.extend([modality_map["ref_vit"]] * curr_split_len)
                index_video_path_name = element.split("/")[-1]

                if self.data_config.text_template:
                    text_template_user.append({"type": element_dtype})
                    vit_num_tokens.append(num_tokens_)
                    video_types.append("vit_video")

        if self.sample["sample_lens"] != []:
            sample_lens = self.sample["sample_lens"]

        if self.sample["sample_modality"] != []:
            sample_modality = self.sample["sample_modality"]
        self.sample["sample_modality"] = sample_modality
        self.sample["sample_task"] = torch.ones(self.sample["sample_lens"]) * sample_task_map["t2v"]

        additional_fields = {
            "caption": caption_all,
            "caption_cn": caption_all,
            "caption_answer": caption_answer,
            "index_item": index,
            "index": index_video_path_name,
            "additional_information": data_sample["additional_information"] if "additional_information" in data_sample.keys() else {},
            "visual_path": data_sample["interleave_array"][0],
            "question": data_sample["interleave_array"][1][1] if isinstance(data_sample["interleave_array"][1], list) and len(data_sample["interleave_array"][1]) > 1 else None,
            "answer": data_sample["interleave_array"][1][2] if isinstance(data_sample["interleave_array"][1], list) and len(data_sample["interleave_array"][1]) > 2 else None
        }

        return self._finalize_sample(
            sample_lens, curr_video_grid_thw,
            sample_type="und",
            additional_fields=additional_fields
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        task = self.data_config.task
        # Get target modality
        if '_t' in task:
            self.data_config.target_modality = 'text'
        elif '2i' in task or 'image' in task:
            self.data_config.target_modality = 'image'
        else:
            self.data_config.target_modality = 'video'

        # Get sample
        if task in ["t2i", "t2v"]:  # Text-to-image or text-to-video
            return self.t2v_sample(idx)
        elif 'edit' in task:  # Video Editing or Image Editing
            self.sample_task = 'edit'
            return self.tiv2v_sample(idx)
        elif 'idip' in task:  # Video IDIP, Image IDIP
            self.sample_task = 'idip'
            return self.tiv2v_sample(idx)
        elif "i2v" in task:  # Text-Image-to-Video
            self.sample_task = 't2v'
            return self.ff2v_sample(idx)
        elif task in ["x2t", "x2t_image", "x2t_video"]:  # Multi-modal Understanding
            return self.x2t_sample(idx)
        else:
            raise ValueError(f"Unknown task: {task}")
