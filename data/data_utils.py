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
Data helpers used by inference (`inference_lance.py`, `ValidationDataset`) and the
Lance model core (`modeling/lance/lance.py`).

Exported utilities:
    - Position id helpers (image / video, interpolate / extrapolate)
    - Patchify helpers (image + video-with-merge)
    - create_sparse_mask       : flex-attention sparse mask builder
    - add_special_tokens       : register chat / vision tokens on a tokenizer
    - len2weight               : CE loss reweighting factor
"""

from einops import rearrange

import torch
from torch.nn.attention.flex_attention import or_masks, and_masks


# ------------------------------------------------------------------
# Position id helpers
# ------------------------------------------------------------------

def get_flattened_position_ids_interpolate_video(num_frames, img_h, img_w, patch_size, max_num_frames, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    # temporal
    boundaries_t = torch.arange(1 / max_num_frames, 1.0, 1 / max_num_frames)
    fractional_coords_t = torch.arange(0, 1 - 1e-6, 1 / num_frames)
    bucket_coords_t = torch.bucketize(fractional_coords_t, boundaries_t, right=True)
    # spatial
    boundaries_s = torch.arange(1 / max_num_patches_per_side, 1.0, 1 / max_num_patches_per_side)
    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / num_patches_h)
    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / num_patches_w)
    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries_s, right=True)
    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries_s, right=True)
    pos_ids = (
        bucket_coords_t[:, None, None] * max_num_patches_per_side * max_num_patches_per_side
        + bucket_coords_h[None, :, None] * max_num_patches_per_side
        + bucket_coords_w[None, None, :]
    ).flatten()
    return pos_ids


def get_flattened_position_ids_extrapolate_video(t, h, w, max_latent_size):
    """
    Defaults:
        num_frames = 7 (corresponding to 25 frames)
        max_num_patches_per_side = 64
    """
    coords_t = torch.arange(0, t)
    coords_h = torch.arange(0, h)
    coords_w = torch.arange(0, w)
    pos_ids = (
        coords_t[:, None, None] * max_latent_size * max_latent_size
        + coords_h[None, :, None] * max_latent_size
        + coords_w[None, None, :]
    ).flatten()
    return pos_ids


def get_flattened_position_ids_extrapolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    coords_h = torch.arange(0, num_patches_h)
    coords_w = torch.arange(0, num_patches_w)
    pos_ids = (coords_h[:, None] * max_num_patches_per_side + coords_w).flatten()
    return pos_ids


def get_flattened_position_ids_interpolate(img_h, img_w, patch_size, max_num_patches_per_side):
    num_patches_h, num_patches_w = img_h // patch_size, img_w // patch_size
    boundaries = torch.arange(1 / max_num_patches_per_side, 1.0, 1 / max_num_patches_per_side)
    fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / num_patches_h)
    fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / num_patches_w)
    bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
    bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)
    pos_ids = (bucket_coords_h[:, None] * max_num_patches_per_side + bucket_coords_w).flatten()
    return pos_ids


# ------------------------------------------------------------------
# Patchify helpers
# ------------------------------------------------------------------

def patchify(image, patch_size):
    p = patch_size
    c, h, w = image.shape
    assert h % p == 0 and w % p == 0
    image = image.reshape(c, h // p, p, w // p, p)
    image = torch.einsum("chpwq->hwpqc", image)
    image = image.reshape(-1, p**2 * c)
    return image


def patchify_video_with_merge(video, spatial_patch_size, temporal_patch_size, merge_size=2):
    """
    Args:
        video: Tensor of shape [C, T, H, W]
        spatial_patch_size: patch size for H/W
        temporal_patch_size: patch size for T
        merge_size: merging factor for spatial grid (fixed at 2)

    Returns:
        patches: Tensor of shape [num_patches, patch_dim]
    """
    video = rearrange(video, "C T H W -> T C H W")
    T, C, H, W = video.shape
    p, tp, ms = spatial_patch_size, temporal_patch_size, merge_size

    gt, gh, gw = T // tp, H // p, W // p
    video = video.reshape(gt, tp, C, gh // ms, ms, p, gw // ms, ms, p)
    video = video.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    patches = video.reshape(gt * gh * gw, C * tp * p * p)
    return patches


# ------------------------------------------------------------------
# Sparse attention mask (flex-attention)
# ------------------------------------------------------------------

def create_sparse_mask(document_lens, split_lens, attn_modes, device):
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    def full_and_noise_mask(b, h, q_idx, kv_idx):
        return (full_and_noise_seq_id[q_idx] == full_and_noise_seq_id[kv_idx]) & (full_and_noise_seq_id[q_idx] >= 0)

    def remove_noise_mask(b, h, q_idx, kv_idx):
        return ~((noise_seq_id[kv_idx] >= 0) & (noise_seq_id[q_idx] != noise_seq_id[kv_idx]))

    def sample_mask(b, h, q_idx, kv_idx):
        return document_id[q_idx] == document_id[kv_idx]

    full_and_noise_tmp = []
    noise_tmp = []

    for i, (length, mode) in enumerate(zip(split_lens, attn_modes)):
        value = i if mode in ["full", "noise"] else -1
        full_and_noise_tmp.extend([value] * length)
        value_noise = i if mode == "noise" else -1
        noise_tmp.extend([value_noise] * length)

    full_and_noise_seq_id = torch.Tensor(full_and_noise_tmp).to(device)
    noise_seq_id = torch.Tensor(noise_tmp).to(device)

    document_id = torch.cat([torch.full((l,), i) for i, l in enumerate(document_lens, start=1)]).to(device)

    return and_masks(or_masks(causal_mask, full_and_noise_mask), remove_noise_mask, sample_mask)


# ------------------------------------------------------------------
# Tokenizer / loss helpers
# ------------------------------------------------------------------

def add_special_tokens(tokenizer):
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []
    for tok in ("<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>"):
        if tok not in all_special_tokens:
            new_tokens.append(tok)

    num_new_tokens = tokenizer.add_tokens(new_tokens)
    new_token_ids = dict(
        bos_token_id=tokenizer.convert_tokens_to_ids("<|im_start|>"),
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
        start_of_image=tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        end_of_image=tokenizer.convert_tokens_to_ids("<|vision_end|>"),
    )
    return tokenizer, new_token_ids, num_new_tokens


def len2weight(x, loss_reduction="square"):
    if x == 0:
        return x
    if loss_reduction == "token":
        return 1
    if loss_reduction == "sample":
        return 1 / x
    if loss_reduction == "square":
        return 1 / (x**0.5)
    raise NotImplementedError(loss_reduction)
