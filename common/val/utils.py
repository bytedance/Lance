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

import os
EXP_HW_20250819 = os.environ.get("EXP_HW_20250819", "False").lower() == "true"
from einops import rearrange
import torch
from typing import List
import imageio
import glob
import numpy as np


def _vit_denorm_uint8_thwc(video_tensor_c_first: torch.Tensor) -> np.ndarray:
    """
    输入: T C H W float，范围近似标准化(mean/std)。输出: T H W C uint8
    固定用 Qwen2.5-VL vit 的 mean/std，保持与原实现一致。
    """
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    mean_t = torch.tensor(mean, device=video_tensor_c_first.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=video_tensor_c_first.device).view(1, 3, 1, 1)
    x = torch.clamp(video_tensor_c_first * std_t + mean_t, 0, 1)
    x = (x * 255).round().clamp(0, 255).to(torch.uint8)  # T C H W
    return x.permute(0, 2, 3, 1).cpu().numpy()


def pad_video_list(video_tensor):  # video_tensor: List[Tensor], each tensor shape is [C T H W]
    video_sizes = [item.shape for item in video_tensor]
    max_video_size = [max(item) for item in list(zip(*video_sizes))]
    padded_videos_latent = torch.zeros(size=(len(video_tensor), *max_video_size))
    for i, video_tensor_ in enumerate(video_tensor):
        c, t, h, w = video_tensor_.shape
        padded_videos_latent[i, :c, :t, :h, :w] = video_tensor_
    return padded_videos_latent


def decode_video_tensor(video_tensor, video_type="vae", save_path="", save_half=False, idx="", max_save_num=100000, save_item_name="", save_fps=12):
    # video_tensor: list [N], each item shape is [C T H W]
    # video_type: vae, vit
    N_target = len(video_tensor)
    if N_target != 1:  # TODO: update this when saving multiple target videos is supported.
        padded_videos_latent = pad_video_list(video_tensor)
        v_tc_hw = rearrange(padded_videos_latent, "n c t h w -> t c h (n w)")  # T C H' W
    else:
        v_tc_hw = video_tensor[0].permute(1, 0, 2, 3)
    if video_type == "vae":
        v_thwc = v_tc_hw.float().clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    elif video_type == "vit":
        v_thwc = _vit_denorm_uint8_thwc(v_tc_hw)
    else:
        raise ValueError(f"video_type {video_type} not supported")
    if save_path != "":
        os.makedirs(save_path, exist_ok=True)
        if save_half:
            w = v_thwc.shape[2]
            v_thwc_save = v_thwc[:, :, w // 2:, :]
        else:
            v_thwc_save = v_thwc
        if v_thwc.shape[0] > 1:  # Save video.
            existing_files = glob.glob(f"{save_path}/*.mp4")
            if len(existing_files) > max_save_num:
                quit()
            save_path_i = f"{save_path}/{save_item_name}.mp4"
            imageio.mimsave(save_path_i, v_thwc_save, fps=save_fps, format="mp4")
        else:  # Save image.
            existing_files = glob.glob(f"{save_path}/*.png")
            if len(existing_files) > max_save_num:
                quit()
            save_path_i = f"{save_path}/{save_item_name}.png"
            imageio.imwrite(save_path_i, v_thwc_save[0], format="png")
        print(f"video or image saved to {save_path_i}")
    return v_thwc


def map_splits_to_samples(sample_lens: List[int], split_lens: List[int]) -> List[List[int]]:
    """
    将split索引映射到对应的样本

    参数:
        val_sample_lens: 每个样本的总长度列表
        val_split_lens: 每个split的长度列表

    返回:
        列表，其中每个元素是一个列表，包含属于对应样本的split索引
    """
    sample_splits = []
    current_split_idx = 0
    remaining_length = 0

    for sample_len in sample_lens:
        splits = []
        remaining_length = sample_len

        while remaining_length > 0 and current_split_idx < len(split_lens):
            # Add the current split index to this sample.
            splits.append(current_split_idx)

            # Consume the current split length and move to the next split.
            remaining_length -= split_lens[current_split_idx]
            current_split_idx += 1

        sample_splits.append(splits)

    return sample_splits


@torch.no_grad()
def make_padded_latent(padded_videos, data_mode, vae_model):  # Support both online and offline modes.
    """
    for vae:
    data_mode = data['vae_data_mode']
    padded_videos = data.pop("padded_videos")
    """
    if data_mode.count("offline") == 0:  # All entries are online.
        padded_latent = vae_model.vae_encode(padded_videos)
    elif data_mode.count("online") == 0:  # All entries are offline.
        padded_latent = padded_videos
    else:  # Mixed online/offline modes.
        online_buf, idxs = [], []
        padded_latent = [None] * len(padded_videos)

        for i, (x, m) in enumerate(zip(padded_videos, data_mode)):
            if m.lower().startswith("off"):  # offline: use the latent directly.
                padded_latent[i] = x
            else:  # online: collect video tensors for encoding.
                online_buf.append(x)
                idxs.append(i)

        lat = vae_model.vae_encode(online_buf)  # Batch VAE encoding for efficiency.
        for i, idx in enumerate(idxs):
            padded_latent[idx] = lat[i]

    del padded_videos
    torch.cuda.empty_cache()
    return padded_latent


@torch.no_grad()
def make_packed_vit_token_embed(packed_vit_tokens, vit_data_mode, vit_video_grid_thw, vit_model):  # Support both online and offline modes.
    """
    for vit:
    vit_data_mode = vit_data_mode
    packed_vit_tokens = packed_vit_tokens
    """
    if vit_data_mode.count("offline") == 0:  # All entries are online.
        packed_vit_tokens = torch.cat(packed_vit_tokens, dim=0)
        packed_vit_token_embed = vit_model(
            hidden_states=packed_vit_tokens,  # L x 1176 or 2048
            grid_thw=vit_video_grid_thw,  # t, h, w
        )  # L x 1176 or 2048 -> L//4 x 2048
    elif vit_data_mode.count("online") == 0:  # All entries are offline.
        packed_vit_token_embed = torch.cat(packed_vit_tokens, dim=0)  # L x 1176 or 2048
    else:  # Mixed online/offline modes.
        packed_vit_token_embed, i_online = [], 0
        for i, (x, m) in enumerate(zip(packed_vit_tokens, vit_data_mode)):
            if m.lower().startswith("off"):  # offline: use the latent directly.
                packed_vit_token_embed.append(x)
            else:
                if vit_video_grid_thw.shape[0] == len(packed_vit_tokens):  # Offline videos are also present in vit_video_grid_thw.
                    i_online = i
                thw = vit_video_grid_thw[i_online:i_online+1]
                packed_vit_token_embed.append(
                    vit_model(
                        hidden_states=x,
                        grid_thw=thw,
                    )
                )
                i_online += 1
        packed_vit_token_embed = torch.cat(packed_vit_token_embed, dim=0)  # L x 1176 or 2048

    return packed_vit_token_embed


def uncond_split_pro(
    language_model,
    current_attn_modes,
    current_split_lens,
    vae_video_grid_thw,
    vit_video_grid_thw,
    curr_vae_split_idx,
    curr_vit_split_idx,
    device,
    dtype,
    start_id,
    image_token_id,
    end_id,
    BLOCK_SIZE,
    is_text_uncond=True,
    is_vit_uncond=False,
):
    uncond_split, uncond_pos_ids = [], []
    (
        curr_vae_split_idx_,
        curr_vit_split_idx_,
        uncond_vae_index,
        uncond_vit_index,
        uncond_packed_gen_token_indexes,
        uncond_packed_und_token_indexes,
        uncond_split_lens,
        uncond_attn_modes,
    ) = (
        curr_vae_split_idx,
        curr_vit_split_idx,
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for i_visual, attn_mode_ in enumerate(current_attn_modes):
        split_len_ = current_split_lens[i_visual]
        if attn_mode_ == "causal" and is_text_uncond:
            continue
        elif attn_mode_ == "full" and is_vit_uncond:
            continue
        elif attn_mode_ in ["noise", "full_noise"]:
            t, h, w = vae_video_grid_thw[curr_vae_split_idx_]
            num_visual = int(t * h * w / 4)  # 4 is merge_size 2 squared.
            uncond_vae_index.extend(range(len(uncond_split) + 1, len(uncond_split) + 1 + num_visual))
            uncond_packed_und_token_indexes.extend([len(uncond_split), len(uncond_split) + 1 + num_visual])
            uncond_packed_gen_token_indexes.extend(range(len(uncond_split) + 1, len(uncond_split) + 1 + num_visual))
            curr_vae_split_idx_ += 1
        elif attn_mode_ == "full":
            t, h, w = vit_video_grid_thw[curr_vit_split_idx_]
            num_visual = int(t * h * w / 4)
            uncond_vit_index.extend(range(len(uncond_split) + 1, len(uncond_split) + 1 + num_visual))
            uncond_packed_und_token_indexes.extend(range(len(uncond_split), len(uncond_split) + 2 + num_visual))
            curr_vit_split_idx_ += 1
        uncond_split += [start_id] + [image_token_id] * num_visual + [end_id]

        uncond_split_lens.append(split_len_)
        uncond_attn_modes.append(attn_mode_)
        uncond_pos_ids += [curr_vae_split_idx_ + curr_vit_split_idx_ - 1] * split_len_
    uncond_vae_index = torch.tensor(uncond_vae_index, dtype=torch.long, device=device)
    uncond_vit_index = torch.tensor(uncond_vit_index, dtype=torch.long, device=device)
    uncond_packed_gen_token_indexes = torch.tensor(uncond_packed_gen_token_indexes, dtype=torch.long, device=device)
    uncond_packed_und_token_indexes = torch.tensor(uncond_packed_und_token_indexes, dtype=torch.long, device=device)

    # ---- Create unconditional condition. ----
    uncond_text_ids = torch.tensor(uncond_split, device=device, dtype=torch.long)
    uncond_sequence = language_model.model.embed_tokens(uncond_text_ids).to(dtype=dtype)

    # 2) Match training behavior by padding the tail block as well.
    uncond_seq_len = len(uncond_text_ids)
    uncond_seq_len_pad = (uncond_seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE
    uncond_pad = uncond_seq_len_pad - uncond_seq_len
    if uncond_pad > 0:
        uncond_split_lens.append(uncond_pad)
        uncond_attn_modes.append("causal")

    return (
        uncond_sequence,
        uncond_attn_modes,
        uncond_split_lens,
        uncond_vae_index,
        uncond_vit_index,
        uncond_packed_gen_token_indexes,
        uncond_packed_und_token_indexes,
        uncond_text_ids,
        uncond_seq_len,
        uncond_pad,
    )
