# coding: utf-8

import json
import torch
import random
import io
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
        "Describe the key elements of the input maze image (layout, white path, black walls, blue star, red flag, and overall background), then generate a 2D animation. The blue star should slide smoothly along the white path, stop exactly on the red flag, and then acquire a trophy. Ensure the blue star never crosses or enters the black maze walls. Keep the camera as a static top-down view showing the entire maze."]


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
    first_pro = True
    i_ref = 1
    for i, attn_mode in enumerate(attn_modes):
        if attn_mode in shift_attn_mode:
            if pro_type == 1:  # 基于特定安全位移进行shift，如 pos_shift=1000
                position_ids[:, :, curr_split : curr_split + split_lens[i]] += pos_shift
            elif pro_type == 2:  # 基于特定安全位移进行shift，并令起始token pos id为特定值pos_shift
                if first_pro:
                    pos_shift -= position_ids[0, 0, curr_split].clone()
                    first_pro = False
                position_ids[:, :, curr_split : curr_split + split_lens[i]] += pos_shift
            elif pro_type == 3:  # 把 reference image 视作特殊 “frame(s)” 放到 t 轴之外（例如 t = -1, -2 ...）
                if first_pro:
                    N_shift = position_ids[0, 0, curr_split].clone()
                    first_pro = False
                position_ids[0, :, curr_split : curr_split + split_lens[i]] = -(position_ids[0, :, curr_split : curr_split + split_lens[i]] - N_shift + 1)
            elif pro_type == 4:  # 把 reference image 的thw 都放到负轴上
                if first_pro:
                    N_shift = position_ids[0, 0, curr_split].clone()
                    first_pro = False
                position_ids[:, :, curr_split : curr_split + split_lens[i]] = -(position_ids[:, :, curr_split : curr_split + split_lens[i]] - N_shift + 1)
            elif pro_type == 5:  # mrope2 的UNO改良版， 基于特定安全位移进行shift，并令起始token pos id为特定值pos_shift, 同时面对N个ref_image，呈现倍数趋势，
                if (
                    attn_mode == "full" and attn_modes[i + 1] == "full_noise"
                ):  # 即对 ['causal', 'full', 'full_noise', 'full', 'full_noise', 'noise'] ， 'full', 'full_noise'为同一ref imgae/video 特征
                    pos_shift_i = pos_shift * i_ref
                    pos_shift_i -= position_ids[0, 0, curr_split].clone()
                    # first_pro = False
                    i_ref += 1
                position_ids[:, :, curr_split : curr_split + split_lens[i]] += pos_shift_i

            elif pro_type == 6:  # mrope1 的UNO改良版， 基于特定安全位移进行shift，如 pos_shift=1000 *N_ref
                if (
                    attn_mode == "full" and attn_modes[i + 1] == "full_noise"
                ):  # 即对 ['causal', 'full', 'full_noise', 'full', 'full_noise', 'noise'] ， 'full', 'full_noise'为同一ref imgae/video 特征
                    pos_shift_i = pos_shift * i_ref
                    i_ref += 1
                position_ids[:, :, curr_split : curr_split + split_lens[i]] += pos_shift_i

            elif pro_type == 7:  # mrope5 的改良版，只在t维度上进行shift
                if attn_mode == "full" and attn_modes[i + 1] in [
                    "full_noise",
                    "causal",
                ]:  # 即对 ['causal', 'full', 'full_noise', 'full', 'full_noise', 'noise'] ， 'full', 'full_noise'为同一ref imgae/video 特征
                    pos_shift_i = pos_shift * i_ref
                    pos_shift_i -= position_ids[0, 0, curr_split].clone()
                    # first_pro = False
                    i_ref += 1
                position_ids[0, :, curr_split : curr_split + split_lens[i]] += pos_shift_i
            # elif pro_type == 10:  # 与sample_modality 有关
            #     # i_sample_modality==4 对应 ref_vit 的 position_ids:  使用1000 offset （ ref_vit 起始id均为1000）
            #     if position_ids[:, :, i_sample_modality == 4].sum() != 0:
            #         pos_shift_type4 = 1000 - position_ids[:, :, i_sample_modality == 4][0, 0, 0]
            #         position_ids[0, :, i_sample_modality == 4] += pos_shift_type4
            #     # i_sample_modality==3 对应 ref_image 的 position_ids:  使用2000 offset （ ref_image 起始id均为2000）
            #     if position_ids[:, :, i_sample_modality == 3].sum() != 0:
            #         pos_shift_type3 = 2000 - position_ids[:, :, i_sample_modality == 3][0, 0, 0]
            #         position_ids[0, :, i_sample_modality == 3] += pos_shift_type3
            #     # i_sample_modality==2 对应 ref_video 的 position_ids, i_sample_modality==1 对应 noise 的 position_ids: 'ref_source' 与 'noise' 特征共享 pos_id
            #     if position_ids[:, :, i_sample_modality == 2].sum() != 0:
            #         position_ids[:, :, i_sample_modality == 1] = position_ids[:, :, i_sample_modality == 2]
            elif pro_type == 10:  # 与sample_modality 有关
                # i_sample_modality==4 对应 ref_vit 的 position_ids:  使用1000 offset （ ref_vit 起始id均为1000）
                if position_ids[:, :, i_sample_modality == 4].sum() != 0:
                    pos_shift_type4 = 1000 - position_ids[:, :, i_sample_modality == 4][0, 0, 0]
                    position_ids[0, :, i_sample_modality == 4] += pos_shift_type4
                # i_sample_modality==3 对应 ref_image 的 position_ids:  使用2000 offset （ ref_image 起始id均为2000）
                if position_ids[:, :, i_sample_modality == 3].sum() != 0:
                    pos_shift_type3 = 2000 - position_ids[:, :, i_sample_modality == 3][0, 0, 0]
                    position_ids[0, :, i_sample_modality == 3] += pos_shift_type3
                # i_sample_modality==2 对应 ref_video 的 position_ids, i_sample_modality==1 对应 noise 的 position_ids: 'ref_source' 与 'noise' 特征共享 pos_id
                if position_ids[:, :, i_sample_modality == 2].sum() != 0 and sum(i_sample_modality == 2) == sum(i_sample_modality == 1):
                    position_ids[:, :, i_sample_modality == 1] = position_ids[:, :, i_sample_modality == 2]
            elif pro_type == "10-thw":  # 与sample_modality 有关
                # i_sample_modality==4 对应 ref_vit 的 position_ids:  使用1000 offset （ ref_vit 起始id均为1000）
                if position_ids[:, :, i_sample_modality == 4].sum() != 0:
                    pos_shift_type4 = 1000 - position_ids[:, :, i_sample_modality == 4][0, 0, 0]
                    position_ids[:, :, i_sample_modality == 4] += pos_shift_type4
                # i_sample_modality==3 对应 ref_image 的 position_ids:  使用2000 offset （ ref_image 起始id均为2000）
                if position_ids[:, :, i_sample_modality == 3].sum() != 0:
                    pos_shift_type3 = 2000 - position_ids[:, :, i_sample_modality == 3][0, 0, 0]
                    position_ids[:, :, i_sample_modality == 3] += pos_shift_type3
                # i_sample_modality==2 对应 ref_video 的 position_ids, i_sample_modality==1 对应 noise 的 position_ids: 'ref_source' 与 'noise' 特征共享 pos_id
                if position_ids[:, :, i_sample_modality == 2].sum() != 0:
                    position_ids[:, :, i_sample_modality == 1] = position_ids[:, :, i_sample_modality == 2]

            elif pro_type == 11: # 使用system_prompt的时候不能用
                # i_sample_modality==4 对应 ref_vit 的 position_ids:  使用1000 offset （ ref_vit 起始id均为1000）
                if position_ids[:, :, i_sample_modality == 4].sum() != 0:
                    pos_shift_type4 = 1000 - position_ids[:, :, i_sample_modality == 4][0, 0, 0]
                    position_ids[:, :, i_sample_modality == 4] += pos_shift_type4
                # i_sample_modality==3 对应 ref_image 的 position_ids:  使用2000 offset （ ref_image 起始id均为2000）
                if position_ids[:, :, i_sample_modality == 3].sum() != 0:
                    pos_shift_type3 = 2000 - position_ids[:, :, i_sample_modality == 3][0, 0, 0]
                    position_ids[:, :, i_sample_modality == 3] += pos_shift_type3
                # i_sample_modality==2 对应 ref_video 的 position_ids, i_sample_modality==1 对应 noise 的 position_ids: 'ref_source' 与 'noise' 特征共享 pos_id
                if position_ids[:, :, i_sample_modality == 2].sum() != 0:
                    try:
                        pos_shift = position_ids[:, :, i_sample_modality == 0][0, 0, -1] + 1 - position_ids[:, :, i_sample_modality == 2][0, 0, 0]
                    except:
                        pos_shift = -position_ids[:, :, i_sample_modality == 2][0, 0, 0]
                    position_ids[:, :, i_sample_modality == 2] += pos_shift
                    position_ids[:, :, i_sample_modality == 1] = position_ids[:, :, i_sample_modality == 2]
                else:
                    try:
                        pos_shift = position_ids[:, :, i_sample_modality == 0][0, 0, -1] + 1 - position_ids[:, :, i_sample_modality == 1][0, 0, 0]
                    except:
                        pos_shift = -position_ids[:, :, i_sample_modality == 1][0, 0, 0]
                    position_ids[:, :, i_sample_modality == 1] += pos_shift
            elif pro_type == 12: # 使用system_prompt的时候不能用
                # i_sample_modality==4 对应 ref_vit 的 position_ids:  使用1000 offset （ ref_vit 起始id均为1000）
                if position_ids[:, :, i_sample_modality == 4].sum() != 0:
                    pos_shift_type4 = 1000 - position_ids[:, :, i_sample_modality == 4][0, 0, 0]
                    position_ids[0, :, i_sample_modality == 4] += pos_shift_type4
                # i_sample_modality==3 对应 ref_image 的 position_ids:  使用2000 offset （ ref_image 起始id均为2000）
                if position_ids[:, :, i_sample_modality == 3].sum() != 0:
                    pos_shift_type3 = 2000 - position_ids[:, :, i_sample_modality == 3][0, 0, 0]
                    position_ids[0, :, i_sample_modality == 3] += pos_shift_type3
                # i_sample_modality==2 对应 ref_video 的 position_ids, i_sample_modality==1 对应 noise 的 position_ids: 'ref_source' 与 'noise' 特征共享 pos_id
                if position_ids[:, :, i_sample_modality == 2].sum() != 0:
                    try:
                        pos_shift = position_ids[:, :, i_sample_modality == 0][0, 0, -1] + 1 - position_ids[:, :, i_sample_modality == 2][0, 0, 0]
                    except:
                        pos_shift = -position_ids[:, :, i_sample_modality == 2][0, 0, 0]
                    position_ids[0, :, i_sample_modality == 2] += pos_shift
                    position_ids[:, :, i_sample_modality == 1] = position_ids[:, :, i_sample_modality == 2]
                else:
                    try:
                        pos_shift = position_ids[:, :, i_sample_modality == 0][0, 0, -1] + 1 - position_ids[:, :, i_sample_modality == 1][0, 0, 0]
                    except:
                        pos_shift = -position_ids[:, :, i_sample_modality == 1][0, 0, 0]
                    position_ids[0, :, i_sample_modality == 1] += pos_shift

            # 对thw 基于 Tmax, Hmax, Wmax  开始位移
        curr_split += split_lens[i]

    return position_ids

def detect_lang_simple(s: str) -> str:
    """
    快速判断：若含中文返回 'zh'，否则若含英文字母返回 'en'，否则 'other'。
    适用于快速路由（注意：含两者时返回 'zh'，可按需调整）。
    """
    # 先移除数字再作判断
    s_without_digits = re.sub(r'\d+', '', s)

    if RE_ZH.search(s_without_digits):
        return "zh"
    if RE_EN.search(s_without_digits):
        return "en"
    return "other"

def map_to_nearest_aspect_ratio(h, w, target_resolution=256):
    """
    将h和w映射到最接近的预设宽高比，返回调整后的h和w，且保持分辨率在目标值左右

    预设比例: ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]
    target_resolution: 目标分辨率基准值（默认256）
    """
    # 预计算所有预设宽高比的浮点值 (宽度/高度)
    PRESET_RATIOS = [21 / 9, 16 / 9, 4 / 3, 1 / 1, 3 / 4, 9 / 16]

    # 计算原始宽高比
    original_ratio = w / h

    # 找到最接近的预设比例
    min_index = min(range(len(PRESET_RATIOS)), key=lambda i: abs(original_ratio - PRESET_RATIOS[i]))
    best_ratio = PRESET_RATIOS[min_index]

    # 计算缩放因子，使较长边接近目标分辨率
    if best_ratio >= 1:  # 宽屏 (宽度 >= 高度)
        scale = target_resolution / best_ratio
        adjusted_w = round(target_resolution)
        adjusted_h = round(scale)
    else:  # 竖屏 (高度 > 宽度)
        scale = target_resolution
        adjusted_h = round(target_resolution)
        adjusted_w = round(scale * best_ratio)

    return adjusted_h, adjusted_w


def concat_resize_tensor_list(video_latents: List[torch.Tensor], dim: int = 0, is_offline: bool = False, max_num_frames: int = 121) -> torch.Tensor:
    """
    把 tensors 列表在轴 dim 上 concat；将size不同的tensor的H/W resize到与target 相同。
    - tensors: 非空 list，所有 tensor 的 ndim 必须相同
    - dim: concat 轴（支持负值）
    - pad_value: 填充值（默认 0.0）
    返回：拼接后的 tensor
    """

    if is_offline:
        H, W = video_latents[-1].shape[-3], video_latents[-1].shape[-2]
    else:
        H, W = video_latents[-1].shape[-2], video_latents[-1].shape[-1]

    padded_video_latents = []
    num_frames_target = video_latents[-1].shape[dim]

    num_frames_all = num_frames_target
    for index, video_latent in enumerate(video_latents):
        if index != len(video_latents) - 1 and num_frames_all + video_latent.shape[dim] > max_num_frames:  # 避免产生超过MAX_NUM_FRAMES的视频
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
    把 tensors 列表在轴 dim 上 concat；对其它轴按该轴的最大长度用 pad_value 填充。
    - tensors: 非空 list，所有 tensor 的 ndim 必须相同
    - dim: concat 轴（支持负值）
    - pad_value: 填充值（默认 0.0）
    返回：拼接后的 tensor
    """
    video_sizes = [item.shape for item in video_latents]
    max_video_size = [max(item) for item in list(zip(*video_sizes))]
    padded_video_latents = []
    num_frames_target = video_latents[-1].shape[dim]

    num_frames_all = num_frames_target
    for index, video_latent in enumerate(video_latents):
        if index != len(video_latents) - 1 and num_frames_all + video_latent.shape[dim] > max_num_frames:  # 避免产生超过MAX_NUM_FRAMES的视频
            continue
        num_frames_all += video_latent.shape[dim]
        max_video_size[dim] = video_latent.shape[dim]
        padded_video_latent = torch.zeros(max_video_size)
        n1, n2, n3, n4 = video_latent.shape
        padded_video_latent[:n1, :n2, :n3, :n4] = video_latent
        padded_video_latents.append(padded_video_latent)

    padded_video_latents = torch.cat(padded_video_latents, dim=dim)

    return padded_video_latents


def dump_url_to_latent(url: str, tos_cli):
    mean_and_logvar = torch.load(io.BytesIO(tos_cli.get_obj(url)), map_location="cpu")
    mean_and_logvar = rearrange(mean_and_logvar, "t h w c -> c t h w")
    u, log_var = mean_and_logvar.chunk(2, dim=0)  # [c t h w]
    u = reparameterize(u, log_var)  # [1,48,t,h,w]
    latents = rearrange(u, "c t h w -> t h w c")
    return latents


def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return eps * std + mu


def dump_data(properties, element_dtype_array, interleave_array, res_dump, tos_cli):
    pass
    # properties = video_meta["properties"]
    wan22_vae_latent, qwen25vl_vit_latent = properties["wan22_vae_latent"], properties["qwen25vl_vit_latent"]

    for index, element_dtype in enumerate(element_dtype_array):
        res_dump_ = res_dump[-1] if element_dtype == "video" else res_dump[0]

        try:
            url = json.loads(wan22_vae_latent)[index][res_dump_]
        except:
            url = wan22_vae_latent[index][res_dump_]
        vae_latents = dump_url_to_latent(url, tos_cli)
        try:
            url = json.loads(qwen25vl_vit_latent)[index][res_dump_]
        except:
            url = qwen25vl_vit_latent[index][res_dump_]
        try:
            vit_tensor = torch.load(io.BytesIO(tos_cli.get_obj(url)), map_location="cpu")  # [L, D = 2048]
        except:
            raise ValueError(f"Wrong dump vit url: {url}")
        vit_shape = json.loads(properties["qwen25vl_vit_videoShape"])[index][res_dump_]

        interleave_array[index] = (vae_latents, vit_tensor, vit_shape)

    return interleave_array


def parse_caption_audio_human(video_meta_url: str, tos_cli):
    """
    解析 caption 数组，并将 en 和 zh 的信息分别存入独立的键中。
    """
    video_meta = json.loads(tos_cli.get_obj_by_url(video_meta_url))

    video_url = video_meta["target_video_url"]

    caption_dct = video_meta["caption"]
    caption_zh = json.loads(caption_dct["distill_pd_caption_zh"])[0]
    caption_en = json.loads(caption_dct["distill_pd_caption_en"])[0]

    # # NOTE: 先只取 en，后续再考虑 zh
    caption = caption_en["image_caption"] + caption_en["video_caption"]

    return video_url, caption


def parse_caption_live_vertical(caption_array):
    """
    解析 caption 数组，并将 en 和 zh 的信息分别存入独立的键中。
    """
    separated_data = {}
    for key, json_str in caption_array:
        # 提取语言后缀 ('en' 或 'zh')
        lang_code = key.split("_")[-1]

        # 解析JSON字符串，并直接获取列表中的字典
        content_dict = json.loads(json_str)[0]

        # 以语言后缀为键，存入解析后的字典
        separated_data[lang_code] = content_dict

    # parse caption
    info_zh, info_en = separated_data.get("zh"), separated_data.get("en")

    # NOTE: 先只取 en，后续再考虑 zh
    if info_en is None:
        return None

    caption = info_en["image_caption"] + info_en["video_caption"]
    return caption


def parse_caption_vertical_dump(video_meta_url: str, tos_cli, res_dump: str):  # 可以作为 dump 数据的 统一处理
    """
    res_dump 为分辨率， 类似 "12fps_192p", "12fps_480p", "fixed25_360p", "fixed25_480p"
    """
    video_meta = json.loads(tos_cli.get_obj_by_url(video_meta_url))
    properties = video_meta["properties"]

    if "wan22_vae_latent" in properties.keys():  # vae dump
        wan22_vae_latent = properties["wan22_vae_latent"]
        url = json.loads(wan22_vae_latent)[0][res_dump]
        mean_and_logvar = torch.load(io.BytesIO(tos_cli.get_obj(url)), map_location="cpu")
        # mean, logvar = mean_and_logvar.chunk(2, dim=-1) # [t,h,w,c]
        mean_and_logvar = rearrange(mean_and_logvar, "t h w c -> c t h w")
        u, log_var = mean_and_logvar.chunk(2, dim=0)  # [c t h w]
        u = reparameterize(u, log_var)  # [1,48,t,h,w]
        latents = rearrange(u, "c t h w -> t h w c")

    if "qwen25vl_vit_latent" in properties.keys():  # vit dump
        qwen25vl_vit_latent = properties["qwen25vl_vit_latent"]
        url = json.loads(qwen25vl_vit_latent)[0][res_dump]
        vit_tensor = torch.load(io.BytesIO(tos_cli.get_obj(url)), map_location="cpu")  # [L, D = 2048]

        vit_shape = json.loads(properties["qwen25vl_vit_videoShape"])[0][res_dump]

    caption_zh = json.loads(properties["distill_pd_caption_zh"])[0]
    caption_en = json.loads(properties["distill_pd_caption_en"])[0]

    # # NOTE: 先只取 en，后续再考虑 zh
    caption = caption_en["image_caption"] + caption_en["video_caption"]

    url = video_meta["interleave_array"][0]

    return latents, vit_tensor, caption, vit_shape


def parse_caption_vertical_online(video_meta_url: str, tos_cli):  # 可以作为 dump 数据的 统一处理
    video_meta = json.loads(tos_cli.get_obj_by_url(video_meta_url))
    url = json.loads(video_meta["element_meta_array"][0])["original_path"]
    properties = video_meta["properties"]

    caption_zh = json.loads(properties["distill_pd_caption_zh"])[0]
    caption_en = json.loads(properties["distill_pd_caption_en"])[0]

    # # NOTE: 先只取 en，后续再考虑 zh
    caption = caption_en["image_caption"] + caption_en["video_caption"]

    return url, caption


def parse_caption_llava(video_meta_url: str, tos_cli):  # 可以作为 dump 数据的 统一处理
    video_meta = json.loads(tos_cli.get_obj_by_url(video_meta_url))
    properties = video_meta["properties"]
    conversations = json.loads(properties["conversations"])
    L_VQA = len(conversations) // 2
    choice_idx = random.choice(range(0, L_VQA))
    condition_text = conversations[choice_idx * 2]["value"]
    target_text = conversations[choice_idx * 2 + 1]["value"]

    QA_q = condition_text
    QA_a = target_text
    try:
        QA_i = "View the image attentively and provide a suitable answer to the posed question." #"You are a helpful assistant."
        url = video_meta["interleave_array"][0]
        interleave_array = [url, (QA_i,QA_q,QA_a)]
        element_dtype_array =  ["image","text"]
    except:
        QA_i = "Provide a suitable answer to the posed question." #"You are a helpful assistant."
        interleave_array = [(QA_i,QA_q,QA_a)]
        element_dtype_array =  ["text"]

    return interleave_array, element_dtype_array

def parse_caption_nemotron(video_meta_url: str, tos_cli):  # 可以作为 dump 数据的 统一处理
    video_meta = json.loads(tos_cli.get_obj_by_url(video_meta_url))

    QA_i = "Read the text attentively and provide an appropriate response."
    QA_q = ''
    QA_a = video_meta['interleave_array'][0]

    interleave_array = [(QA_i,QA_q,QA_a)]
    element_dtype_array =  ["text"]

    return interleave_array, element_dtype_array


def parse_caption_vfm_action_clips_online(video_meta_url: str, tos_cli, caption_key: str):  # 可以作为 dump 数据的 统一处理
    video_meta = json.loads(tos_cli.get_obj_by_url(video_meta_url))
    url = video_meta["interleave_array"][0]
    properties = video_meta["properties"]

    # NOTE: 先只取 en，后续再考虑 zh
    caption = properties[caption_key]  # caption_key 为 'i2v_caption_en'

    return url, caption

def parse_caption_vfm_action_clips_dump(video_meta_url: str, tos_cli, caption_key: str, res_dump: str):  # 可以作为 dump 数据的 统一处理
    """
    res_dump 为分辨率， 类似 "12fps_192p", "12fps_480p", "fixed25_360p", "fixed25_480p"
    """
    video_meta = json.loads(tos_cli.get_obj_by_url(video_meta_url))
    properties = video_meta["properties"]

    # pd caption 获取：
    try:
        captions = json.loads(properties["distill_pd_caption_en"])
        if captions[0]["image_caption"] == "" or captions[0]["video_caption"] == "":
            raise ValueError(f"Wrong caption: empty pd_caption in data vfm_action_clips","--"*80)
        caption = captions[0]["image_caption"] + captions[0]["video_caption"]
    except:
        raise ValueError(f"Wrong caption: no pd_caption in data vfm_action_clips in {video_meta_url}")
        # 原caption 获取 NOTE: 先只取 en，后续再考虑 zh
        #caption = properties[caption_key]  # caption_key 为 'i2v_caption_en'

    if len(caption) > 1500:
        raise ValueError(f"Wrong caption: len(condition_text) = {len(caption)} in data vfm_action_clips" )

    if "wan22_vae_latent" in properties.keys():  # vae dump
        wan22_vae_latent = properties["wan22_vae_latent"]
        url = json.loads(wan22_vae_latent)[0][res_dump]
        mean_and_logvar = torch.load(io.BytesIO(tos_cli.get_obj(url)), map_location="cpu")
        # mean, logvar = mean_and_logvar.chunk(2, dim=-1) # [t,h,w,c]
        mean_and_logvar = rearrange(mean_and_logvar, "t h w c -> c t h w")
        u, log_var = mean_and_logvar.chunk(2, dim=0)  # [c t h w]
        u = reparameterize(u, log_var)  # [1,48,t,h,w]
        latents = rearrange(u, "c t h w -> t h w c")
    if "qwen25vl_vit_latent" in properties.keys():  # vit dump
        qwen25vl_vit_latent = properties["qwen25vl_vit_latent"]
        url = json.loads(qwen25vl_vit_latent)[0][res_dump]
        vit_tensor = torch.load(io.BytesIO(tos_cli.get_obj(url)), map_location="cpu")  # [L, D = 2048]
        vit_shape = json.loads(properties["qwen25vl_vit_videoShape"])[0][res_dump]

    # 识别caption 为中文 or 英文？
    if detect_lang_simple(caption) != "en":
        print(f'uid: {video_meta["uid"]} wrong caption: {caption}')
        return None

    return latents, vit_tensor, caption, vit_shape

def parse_caption_dreamedit(video_meta_url: str, tos_cli):
    video_meta = json.loads(tos_cli.get_obj_by_url(video_meta_url))
    interleave_array = video_meta["interleave_array"]
    element_dtype_array = video_meta["element_dtype_array"]
    # 一般 len 为 3 or 2 or 1， 第0个为 source_image_url， 第1个为 mask_url (可选) ,第2个为 target_image_url (可选,此时source_image_url同为target_image_url)
    if len(interleave_array) == 3:
        del interleave_array[1]
        del element_dtype_array[1]
    messages = json.loads(video_meta["properties"]["messages"])

    # caption + instruction
    # condition_text = (
    #     messages[-1]["content"][-1]["info"]["caption"]["base_caption_en"]
    #     + f'. The difference between original and target image is: {messages[-1]["content"][-1]["info"]["caption"]["base_instruction_en"]}'
    # )  # base_caption_en or short_caption_en or base_instruction_en
    # 仅 caption
    # condition_text = messages[-1]["content"][-1]["info"]["caption"]["base_caption_en"]
    # 仅 instruction
    condition_text = messages[-1]["content"][-1]["info"]["caption"]["base_instruction_en"]

    interleave_array = [condition_text] + interleave_array
    element_dtype_array = ["text"] + element_dtype_array
    return interleave_array, element_dtype_array


def parse_caption_dreamO(video_meta_url: str, tos_cli):
    video_meta = json.loads(tos_cli.get_obj_by_url(video_meta_url))
    interleave_array = video_meta["interleave_array"]
    element_dtype_array = video_meta["element_dtype_array"]
    # 一般 len 为 3 or 2 or 1， 第0个为 source_image_url， 第1个为 mask_url (可选) ,第2个为 target_image_url (可选,此时source_image_url同为target_image_url)
    if len(interleave_array) >= 3:
        raise ValueError(f"Wrong sample: {len(interleave_array)-1} (more than 1) reference image in sample")

    messages = json.loads(video_meta["properties"]["messages"])
    gen_cap = False
    # caption
    for item in ["t2i_caption", "clean_caption", "caption_en"]: # NOTE： 使用的都是caption
        try:
            try:
                condition_text = messages[-1]["content"][-1]["info"]["caption"][item]
            except:
                condition_text = messages[0]["content"][-1]["info"]["caption"][item]
                if video_meta["data_name"] == 'STYLE_IMAGE-GEN_30L': # 这个数据集中的gt 和cond img 位置反了
                    interleave_array = interleave_array[1:] + interleave_array[:1] # 交换 cond img 和 gt img 的位置


            gen_cap = True
            break
        except:
            pass
    if not gen_cap:
        raise ValueError(f"Wrong sample: no caption in dreamO sample")

    interleave_array = [condition_text] + interleave_array
    element_dtype_array = ["text"] + element_dtype_array
    return interleave_array, element_dtype_array


def parse_caption_ocr(video_meta_url: str, tos_cli, data_filter: dict):
    video_meta = json.loads(tos_cli.get_obj_by_url(video_meta_url))
    condition_text = video_meta["properties"]["recaption"]

    if data_filter != {}:  # 启动过滤
        doubao_pred = json.loads(video_meta["properties"]["doubao_pred"])

        # 用于过滤的字段
        is_simple = data_filter.get("is_simple", False)
        is_high_clarity = data_filter.get("is_high_clarity", False)
        max_len_cap = data_filter.get("max_len_cap", 100000)

        if (is_simple and doubao_pred["simple_or_complex"] != "simple") or (is_high_clarity and doubao_pred["clarity"] != "high") or (len(condition_text) > max_len_cap):
            return None

    interleave_array = video_meta["interleave_array"]
    element_dtype_array = video_meta["element_dtype_array"]
    interleave_array = [condition_text] + interleave_array
    element_dtype_array = ["text"] + element_dtype_array
    return interleave_array, element_dtype_array


def parse_caption_video_idip_online(
    video_meta_url: str, tos_cli, res_dump: list = ["12fps_192p"], caption_key: str = "", dataset_type: str = ""
):  # 可以作为 dump 数据的 统一处理
    video_meta = json.loads(tos_cli.get_obj_by_url(video_meta_url))
    element_dtype_array = video_meta["element_dtype_array"]
    interleave_array = video_meta["interleave_array"]
    # condition_text = json.loads(video_meta["properties"]['interleaved_caption'])["prompt_en"] # 带 [Ref#1] 指示

    ref_image_first_items = []
    if "refedit" in dataset_type:
        # onlyins
        try:
            condition_text = json.loads(video_meta["properties"]["instruction"])["prompt_short_en"]
        except:
            condition_text = json.loads(video_meta["properties"]["instruction"])["prompt_en"]

    elif "edit" in dataset_type:
        # 使用caption + instruction
        try:
            instruction = json.loads(video_meta["properties"]["instruction"])
            if "prompt_en" in instruction.keys():
                # condition_text = json.loads(video_meta["properties"][caption_key])[0]["video_caption"] + f". The difference between original and target video is: {instruction['prompt_en']}"
                condition_text = instruction["prompt_en"]
            else:
                # condition_text = json.loads(video_meta["properties"][caption_key])[0]["video_caption"] +  f". The difference between original and target video is: {instruction['combinedDescription_en']}"
                condition_text = instruction["combinedDescription_en"]

        except:
            # condition_text = json.loads(video_meta["properties"][caption_key])[0]["video_caption"] + f". The difference between original and target video is: {video_meta['properties']['instruction_en']}"
            condition_text = video_meta["properties"]["instruction_en"]

        # 使用更统一的caption
        # condition_text = json.loads(video_meta["properties"][caption_key])[0]["image_caption"] + json.loads(video_meta["properties"][caption_key])[0]["video_caption"]

    elif "idip" in dataset_type:  # for video_idip, 使用 video_caption + image_caption
        try:
            condition_text = json.loads(video_meta["properties"][caption_key])[0]["image_caption"] + json.loads(video_meta["properties"][caption_key])[0]["video_caption"]
        except:
            condition_text = (
                json.loads(video_meta["properties"]["distill_pd_caption_en"])[0]["image_caption"]
                + json.loads(video_meta["properties"]["distill_pd_caption_en"])[0]["video_caption"]
            )

        # 获取 ref_image_ip_index
        ref_image_ip_index = json.loads(video_meta["extra"]["ref_image_ip_index"])
        # 仅提取第一项
        ref_image_first_items = [values[0] for values in ref_image_ip_index.values()]

    # 创建新的数组来存储需要保留的元素
    new_interleave_array = []
    new_element_dtype_array = []

    if "offline" in dataset_type:  # 离线数据处理
        properties = video_meta["properties"]
        wan22_vae_latent, qwen25vl_vit_latent = properties["wan22_vae_latent"], properties["qwen25vl_vit_latent"]

        for index, element_dtype in enumerate(element_dtype_array):
            if element_dtype != "video" and (ref_image_first_items != [] and index not in ref_image_first_items):
                continue
            res_dump_ = res_dump[-1] if element_dtype == "video" else res_dump[0]

            try:
                url = json.loads(wan22_vae_latent)[index][res_dump_]
            except:
                url = wan22_vae_latent[index][res_dump_]
            vae_latents = dump_url_to_latent(url, tos_cli)
            try:
                url = json.loads(qwen25vl_vit_latent)[index][res_dump_]
            except:
                url = qwen25vl_vit_latent[index][res_dump_]
            original_video_url = interleave_array[index]
            try:
                vit_tensor = torch.load(io.BytesIO(tos_cli.get_obj(url)), map_location="cpu")  # [L, D = 2048]
            except:
                raise ValueError(f"Wrong url: {url} in dataset_type {dataset_type}")
            vit_shape = json.loads(properties["qwen25vl_vit_videoShape"])[index][res_dump_]

            new_interleave_array.append((vae_latents, vit_tensor, vit_shape))  # , original_video_url)
            new_element_dtype_array.append(element_dtype)
    else:
        for index, element_dtype in enumerate(element_dtype_array):
            if element_dtype != "video" and (ref_image_first_items != [] and index not in ref_image_first_items):
                continue
            new_interleave_array.append(interleave_array[index])
            new_element_dtype_array.append(element_dtype)

    interleave_array = [condition_text] + new_interleave_array
    element_dtype_array = ["text"] + new_element_dtype_array

    return interleave_array, element_dtype_array


def parse_vfm_videos_and_clips_join(
    video_meta_url: str, video_unimodel_url: str, tos_cli, res_dump: list = ["12fps_192p"], caption_key: str = "", dataset_type: str = ""
):  # 可以作为 dump 数据的 统一处理
    video_meta = json.loads(tos_cli.get_obj_by_url(video_meta_url))
    element_dtype_array = video_meta["element_dtype_array"]
    interleave_array = video_meta["interleave_array"]
    # condition_text = json.loads(video_meta["properties"]['interleaved_caption'])["prompt_en"] # 带 [Ref#1] 指示

    # 获取 pd_caption
    video_unimodel = json.loads(tos_cli.get_obj_by_url(video_unimodel_url))
    captions = json.loads(video_unimodel["caption"]["distill_pd_caption_en"])
    if len(captions) == 1:
        if captions[0]["image_caption"] == "" or captions[0]["video_caption"] == "":
            raise ValueError(f"Wrong caption: empty pd_caption in data vfm_action_clips")
        condition_text = captions[0]["image_caption"]  + captions[0]["video_caption"]  # ! ! ! ! ! !
    elif len(captions) != 1:
        raise ValueError(f"Invalid caption length: {len(captions)}, expected 1 in parse_vfm_videos_and_clips_join")

    if len(condition_text) > 1500:
        raise ValueError(f"Wrong caption: len(condition_text) = {len(condition_text)}in data {dataset_type}" )

    if "offline" in dataset_type:  # 离线数据处理
        interleave_array = dump_data(video_meta["properties"], element_dtype_array, interleave_array, res_dump, tos_cli)

    interleave_array = [condition_text] + interleave_array
    element_dtype_array = ["text"] + element_dtype_array

    return interleave_array, element_dtype_array


def parse_caption_video_maze(
    video_meta_url: str, tos_cli, res_dump: list = ["12fps_192p"], caption_key: str = "", dataset_type: str = ""
):  # 可以作为 dump 数据的 统一处理
    video_meta = json.loads(tos_cli.get_obj_by_url(video_meta_url))
    if int(video_meta["properties"]["path_length"]) < 10 and random.random() < 0.5:  # 按0.5 的比例 过滤掉路径长度小于10的视频
        pass
        # raise ValueError(f"filter maze data with path_length: {int(video_meta['properties']['path_length'])}")

    element_dtype_array = video_meta["element_dtype_array"]
    interleave_array = video_meta["interleave_array"]

    # 自定义迷宫任务caption
    # condition_text = "Create a 2D animation based on the provided image of a maze. The blue star slides smoothly along the white path, stopping perfectly on the red flag and then acquiring a trophy. The blue star never slides or crosses into the black segments of the maze. The camera is a static, top-down view showing the entire maze."

    if "offline" in dataset_type:  # 离线数据处理
        interleave_array = dump_data(video_meta["properties"], element_dtype_array, interleave_array, res_dump, tos_cli)

    # interleave_array = [condition_text] + interleave_array
    # element_dtype_array = ["text"] + element_dtype_array

    return interleave_array, element_dtype_array

def parse_vfm_videos(video_meta_url: str, tos_cli, res_dump: list = ["12fps_192p"], caption_key: str = "", dataset_type: str = ""):  # 可以作为 dump 数据的 统一处理
    video_meta = json.loads(tos_cli.get_obj_by_url(video_meta_url))
    element_dtype_array = video_meta["element_dtype_array"]
    interleave_array = video_meta["interleave_array"]
    # 获取 pd_caption
    if "video_sft2" in dataset_type:
        # 仅保留通用 3k 数据：
        if video_meta['extra']['version'] != 'V1.0.0-sft_3k-0404':
            raise ValueError(f"filter video_meta with version: {video_meta['extra']['version']}")

        condition_text = json.loads(video_meta["properties"][caption_key])[0]["image_caption"] + json.loads(video_meta["properties"][caption_key])[0]["video_caption"]
    else:
        try: # 对应 vfm_videos 数据 pd caption 获取
            captions = json.loads(video_meta["properties"]["distill_pd_caption_en"])
            if len(captions) == 1:
                if captions[0]["image_caption"] == "" or captions[0]["video_caption"] == "":
                    raise ValueError(f"Wrong caption: empty pd_caption in data vfm_action_clips")
                condition_text = captions[0]["image_caption"] + captions[0]["video_caption"]  # ! ! ! ! ! !
            elif len(captions) != 1:
                raise ValueError(f"Invalid caption length: {len(captions)}, expected 1 in parse_vfm_videos")
        except:
            raise ValueError(f"Wrong caption: no pd_caption in data {dataset_type} in {video_meta_url}")

    if len(condition_text) > 1500:
        raise ValueError(f"Wrong caption: len(condition_text) = {len(condition_text)}in data {dataset_type}" )

    if not isinstance(condition_text, str):
        raise ValueError(f"Invalid caption: {condition_text}, expected str")

    if "offline" in dataset_type:  # 离线数据处理
        interleave_array = dump_data(video_meta["properties"], element_dtype_array, interleave_array, res_dump, tos_cli)

    interleave_array = [condition_text] + interleave_array
    element_dtype_array = ["text"] + element_dtype_array

    return interleave_array, element_dtype_array


def parse_videochat2it(video_meta_url: str, tos_cli, res_dump: list = ["12fps_192p"], caption_key: str = "", dataset_type: str = ""):  # 可以作为 dump 数据的 统一处理
    video_meta = json.loads(tos_cli.get_obj_by_url(video_meta_url))
    element_dtype_array = video_meta["element_dtype_array"]
    interleave_array = video_meta["interleave_array"]

    data_name = video_meta["data_name"]  # "data_name":
    QA = json.loads(video_meta["properties"]["QA"])
    QA = QA[random.randint(0, len(QA) - 1)]  # ['i', 'q', 'a'] , 对应 instruction, question, answer
    if "i" not in QA.keys():
        if "conversation" in data_name:
            QA_i = "View the video attentively and provide a suitable answer to the posed question."  # "You are a helpful assistant."
        else:
            raise ValueError(f"Invalid QA: no i in {QA} in {data_name}")
    else:
        QA_i = QA["i"]

    QA_q, QA_a = QA["q"], QA["a"]

    if "offline" in dataset_type:  # 离线数据处理
        interleave_array = dump_data(video_meta["properties"], element_dtype_array, interleave_array, res_dump, tos_cli)

    interleave_array = interleave_array + [(QA_i, QA_q, QA_a)]
    element_dtype_array = element_dtype_array + ["text"]

    return interleave_array, element_dtype_array


def parse_caption_mmpr(video_meta_url: str, tos_cli):  # 可以作为 dump 数据的 统一处理
    video_meta = json.loads(tos_cli.get_obj_by_url(video_meta_url))
    #print('video_meta',video_meta)
    properties = video_meta["properties"]

    QA_question = properties['question']
    QA_chosen, QA_rejected = properties['chosen'], properties['rejected']
    if 'answer_gt' in properties.keys():
        QA_answer_gt = properties['answer_gt']
    elif 'answer' in properties.keys():
        QA_answer_gt = properties['answer']
    else:
        QA_answer_gt = None

    if QA_answer_gt is not None:
        if 'answer' not in QA_chosen or 'Answer' not in QA_chosen:
            QA_chosen = QA_chosen + f'\nFinal Answer:{QA_answer_gt}'


    try:
        QA_i = "View the image attentively and provide a suitable answer to the posed question." #"You are a helpful assistant."
        url = video_meta["interleave_array"][0]
        interleave_array = [url, (QA_i,QA_question,QA_chosen)]
        element_dtype_array =  ["image","text"]
    except:
        QA_i = "Provide a suitable answer to the posed question." #"You are a helpful assistant."
        interleave_array = [(QA_i,QA_question,QA_chosen)]
        element_dtype_array =  ["text"]

    return interleave_array, element_dtype_array



def parse_videochat2it_doubao_caption(row):
    try:
        IQA_i = "View the video attentively and provide a suitable answer to the posed question."
        rewrite_VQA = json.loads(row['rewrite_VQA'])

        IQA_q = rewrite_VQA['question'] if 'question' in rewrite_VQA.keys() else rewrite_VQA['Question']
        IQA_a = rewrite_VQA['final_answer']
        IQA_resoning = rewrite_VQA['reasoning']
        # 把 resoning 和 final_answer 组合作为 最终answer

        # if random.random() < 0.5: # 50% 的概率，加入 reasoning process
        #     IQA_a = IQA_a + '\n' + IQA_resoning
        #     IQA_i = IQA_i + ' Please provide the reasoning process for selecting the correct answer.'

        if 'options' not in IQA_q and 'Options' not in IQA_q: # 即question 中不含选项时
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

            IQA_q = IQA_q + '\nOptions:\n' + options # 将选项加入到question 中
        return [IQA_i, IQA_q, IQA_a]
    except:
        if 'rewrite_VQA' in row.keys():
            raise ValueError(f"wrong rewrite_VQA in {row['rewrite_VQA']}")
        else:
            raise ValueError(f"wrong rewrite_VQA in {row}")
