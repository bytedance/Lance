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

from PIL import Image

import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode, Compose, Normalize

from .video.transforms.na_resize import NaResize
from .video.transforms.divisible_crop import DivisibleCrop
from .video.transforms.rearrange import Rearrange


class MaxLongEdgeMinShortEdgeResize(torch.nn.Module):
    """Resize the input image so that its longest side and shortest side are within a specified range,
    ensuring that both sides are divisible by a specified stride.

    Args:
        max_size (int): Maximum size for the longest edge of the image.
        min_size (int): Minimum size for the shortest edge of the image.
        stride (int): Value by which the height and width of the image must be divisible.
        max_pixels (int): Maximum pixels for the full image.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.NEAREST_EXACT``,
            ``InterpolationMode.BILINEAR``, and ``InterpolationMode.BICUBIC`` are supported.
        The corresponding Pillow integer constants, e.g., ``PIL.Image.BILINEAR`` are also accepted.
        antialias (bool, optional): Whether to apply antialiasing (default is True).
    """

    def __init__(
        self,
        max_size: int,
        min_size: int,
        stride: int,
        max_pixels: int,
        interpolation=InterpolationMode.BICUBIC,
        antialias=True
    ):
        super().__init__()
        self.max_size = max_size
        self.min_size = min_size
        self.stride = stride
        self.max_pixels = max_pixels
        self.interpolation = interpolation
        self.antialias = antialias

    def _make_divisible(self, value, stride):
        """Ensure the value is divisible by the stride."""
        return max(stride, int(round(value / stride) * stride))

    def _apply_scale(self, width, height, scale):
        new_width = round(width * scale)
        new_height = round(height * scale)
        new_width = self._make_divisible(new_width, self.stride)
        new_height = self._make_divisible(new_height, self.stride)
        return new_width, new_height

    def forward(self, img, img_num=1):
        """
        Args:
            img (PIL Image): Image to be resized.
            img_num (int): Number of images, used to change max_tokens.
        Returns:
            PIL Image or Tensor: Rescaled image with divisible dimensions.
        """
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size

        scale = min(self.max_size / max(width, height), 1.0)
        scale = max(scale, self.min_size / min(width, height))
        new_width, new_height = self._apply_scale(width, height, scale)

        # Ensure the number of pixels does not exceed max_pixels
        if new_width * new_height > self.max_pixels / img_num:
            scale = self.max_pixels / img_num / (new_width * new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        # Ensure longest edge does not exceed max_size
        if max(new_width, new_height) > self.max_size:
            scale = self.max_size / max(new_width, new_height)
            new_width, new_height = self._apply_scale(new_width, new_height, scale)

        return F.resize(img, (new_height, new_width), self.interpolation, antialias=self.antialias)


class ImageTransform:
    def __init__(
        self,
        max_image_size,
        min_image_size,
        image_stride,
        max_pixels=14*14*9*1024,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5]
    ):
        self.stride = image_stride

        self.resize_transform = MaxLongEdgeMinShortEdgeResize(
            max_size=max_image_size,
            min_size=min_image_size,
            stride=image_stride,
            max_pixels=max_pixels,
        )
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(mean=image_mean, std=image_std, inplace=True)

    def __call__(self, img, img_num=1):
        img = self.resize_transform(img, img_num=img_num)
        img = self.to_tensor_transform(img)
        img = self.normalize_transform(img)
        return img


class VideoTransform:
    def __init__(
        self,
        resolution=640,
        mode="area",
        divisible_crop_size=16,
        aspect_ratios=("21:9", "16:9", "4:3", "1:1", "3:4", "9:16"),
        stride_spatial=16,
        stride_temporal=4,
        mean=0.5,
        std=0.5,
        **kwargs
    ):
        self.transform = Compose(
            [
                NaResize(
                    resolution=resolution,
                    mode=mode,
                    downsample_only=True,
                    stride=stride_spatial,
                    # NOTE: aspect_ratios are only for `bucket` resize.
                    aspect_ratios=aspect_ratios,
                ),
                DivisibleCrop(divisible_crop_size),
                Normalize(mean, std),
                Rearrange("t c h w -> c t h w"),
            ]
        )
        # self.stride = divisible_crop_size if isinstance(divisible_crop_size, int) else divisible_crop_size[0]
        self.stride_spatial = stride_spatial
        self.stride_temporal = stride_temporal

    def __call__(self, video):
        return self.transform(video)


class VisualTransform:
    def __init__(
        self,
        max_frame_size,
        min_frame_size,
        image_stride,
        max_pixels=14*14*9*1024,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5]
    ):
        self.stride = image_stride
        self.resize_transform = MaxLongEdgeMinShortEdgeResize(
            max_size=max_frame_size,
            min_size=min_frame_size,
            stride=image_stride,
            max_pixels=max_pixels,
        )
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(mean=image_mean, std=image_std, inplace=True)

    def _process_single(self, img, img_num=1):
        img = self.resize_transform(img, img_num=img_num)
        img = self.to_tensor_transform(img)
        img = self.normalize_transform(img)
        return img

    def __call__(self, img, img_num=1):
        # --- Video sequence handling ---
        if isinstance(img, (list, tuple)):
            # List of PIL.Image or tensors
            out = torch.stack([self._process_single(frame, img_num=img_num) for frame in img])  # [T, C, H, W]
            out = out.permute(1, 0, 2, 3)  # [C, T, H, W]
            return out
        elif isinstance(img, np.ndarray) and img.ndim == 4:
            # numpy array: [T, H, W, C] or [T, C, H, W]
            frames = [img[i] for i in range(img.shape[0])]
            processed_frames = [self._process_single(Image.fromarray(frame) if frame.shape[-1] in [3, 4] else frame, img_num=img_num)
                                for frame in frames]
            out = torch.stack(processed_frames)  # [T, C, H, W]
            out = out.permute(1, 0, 2, 3)  # [C, T, H, W]
            return out
        elif isinstance(img, torch.Tensor) and img.ndim == 4:
            # torch tensor: [T, C, H, W] or [T, H, W, C]
            frames = [img[i] for i in range(img.shape[0])]
            processed_frames = [self._process_single(frame, img_num=img_num) for frame in frames]
            out = torch.stack(processed_frames)  # [T, C, H, W]
            out = out.permute(1, 0, 2, 3)  # [C, T, H, W]
            return out
        else:
            # Single frame.
            return self._process_single(img, img_num=img_num)
