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

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import torch
import yaml


@dataclass
class DataConfig:
    """
    DataConfig 版本，其中 vae_downsample 是一个三元组。
    """
    grouped_datasets: Dict[str, Any] = field(default_factory=dict)
    text_cond_dropout_prob: float = 0.1
    vit_cond_dropout_prob: float = 0.4
    vae_cond_dropout_prob: float = 0.1

    # vae_downsample stores temporal, height, and width downsampling factors.
    vae_downsample: Tuple[int, int, int] = (4, 16, 16)

    max_latent_size: int = 64             # by ModelArguments
    vit_patch_size: int = 14              # by ModelArguments
    vit_patch_size_temporal: int = 2      # by ModelArguments
    vit_max_num_patch_per_side: int = 70  # by ModelArguments
    max_num_frames: int = 25              # by ModelArguments

    latent_patch_size: int = None         # by ModelArguments

    @classmethod
    def from_yaml(cls, file_path: str) -> 'DataConfig':
        """从 YAML/JSON 文件创建 DataConfig 实例"""
        with open(file_path, "r") as stream:
            data = yaml.safe_load(stream)
        return cls(grouped_datasets=data)


class SimpleCustomBatch:
    def __init__(self, batch):
        data = batch[0]
        for key, value in data.items():
            setattr(self, key, value)

    def pin_memory(self):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.pin_memory())
            elif isinstance(value, list) and value and all(isinstance(i, torch.Tensor) for i in value):
                setattr(self, key, [i.pin_memory() for i in value])
        return self

    def cuda(self, device):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
            elif isinstance(value, list) and value and all(isinstance(i, torch.Tensor) for i in value):
                setattr(self, key, [i.to(device) for i in value])
        return self

    def to_dict(self):
        return self.__dict__.copy()


# Top-level function so it can be pickled.
def simple_custom_collate(batch):
    return SimpleCustomBatch(batch)
